#!/usr/bin/env python3
# b14_ampengfftmag.py
# Process LV events: compute amp/energy/frequency/magnitude metrics and save as EnhancedEvent JSON + pkl

import os
import shutil
import sqlite3
import pandas as pd
from obspy import read, read_inventory, UTCDateTime
from flovopy.core.enhanced import EnhancedStream, EnhancedEvent, VolcanoEventClassifier  # assumes EnhancedEvent has .write_to_db() added
from pprint import pprint

from flovopy.processing.sam import VSAM
from flovopy.analysis.asl import ASL, initial_source, make_grid, dome_location
import pickle

from flovopy.wrappers.MVOE_conversion_pipeline.db_backup import backup_db

from math import isnan

import gc

import multiprocessing
from functools import partial

import glob

from flovopy.utils.sysmon import log_system_status_csv


print('[STARTUP] modules loaded')

####################################################################
def stream_metrics_to_dataframe(st):
    rows = []

    for tr in st:
        metrics = {}
        metrics['id'] = tr.id
        metrics['starttime'] = tr.stats.starttime.isoformat()
        metrics['endtime'] = tr.stats.endtime.isoformat()       
        for key, value in getattr(tr.stats, 'metrics', {}).items():
            # Include only scalar values (not list/dict)
            if isinstance(value, (int, float, str, bool)):
                metrics[key] = value
        # Add basic trace info
        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df

def filter_traces_by_inventory(stream, inventory):
    inv_ids = set()
    for net in inventory:
        for sta in net:
            for cha in sta:
                inv_ids.add(f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}")

    # Filter manually based on full SEED id
    filtered_traces = [tr for tr in stream if tr.id in inv_ids]
    return stream.__class__(traces=filtered_traces)

'''
def process_row(rownum, r, numrows, TOP_DIR, source_coords, inventory, asl_config):
    start_time = UTCDateTime()
    print('\n', f'Processing row {rownum} of {numrows}')

    try: # try to process this event
        success = 0
        outdir = os.path.join(TOP_DIR, r['dfile']).replace('.cleaned','')
        if os.path.isdir(outdir):
            magcsv = os.path.join(outdir, 'magnitudes.csv')
            if os.path.isfile(magcsv):
                magdf = pd.read_csv(magcsv)
                if magdf['ME'].notna().any():
                    raise IOError(f'Already processed: {r["dfile"]}')

        os.makedirs(outdir, exist_ok=True)

        print(f"\n[INFO] Processing {r['public_id']} {r['dfile']}")

        mseed_path = os.path.join(r['dir'], r['dfile'])
        if not os.path.exists(mseed_path):
            raise IOError(f"[WARN] File not found: {mseed_path}")

        # Load waveform
        st = read(mseed_path)
        for tr in st:
            if tr.stats.channel[1]!='H':
                st.remove(tr)
        if len(st)==0:
            raise IOError('Stream has no traces')
        st.filter('highpass', freq=0.5, corners=2, zerophase=True)
        
        try:
            est = EnhancedStream(stream=st)
        except Exception as e:
            print('[ERR] cannot make EnhancedStream object')
            raise e

        for key in ['latitude', 'longitude', 'depth']:
            if not isnan(r[key]):
                source_coords[key] = r[key]

        # Compute metrics
        try:
            est, aefdf = est.ampengfftmag(inventory, source_coords, verbose=True, snr_min=3.0)
            est.write(os.path.join(outdir, 'enhanced_stream.mseed'), format='MSEED')
            aefdf.to_csv(os.path.join(outdir, 'magnitudes.csv'))
        except Exception as e:
            print('[ERR] ampengfftmag failed')
            raise e
            
        # Classify
        features = {}
        for tr in est:
            m = getattr(tr.stats, 'metrics', {})
            for key in ['peakf', 'meanf', 'skewness', 'kurtosis']:
                if key in m:
                    features[key] = m[key]
        #if trigger and 'duration' in trigger:
        #    features['duration'] = trigger['duration']

        try:
            classifier = VolcanoEventClassifier()
            label, score = classifier.classify(features)
            r['new_subclass'] = label
            r['new_score'] = score[label]  # the confidence score of the selected subclass
            print(f"[INFO] Event classified as '{label}' with score {score:.2f}")
        except Exception as e:
            print(f"[WARN] Classification failed: {e}")

        row_df = pd.DataFrame([r])
        row_df.to_csv(os.path.join(outdir, 'database_row.csv'), index=False)

        metrics_df = stream_metrics_to_dataframe(est)
        metrics_df.to_csv(os.path.join(outdir, 'stream_metrics.csv'))
        success = 1

        # === Do ASL? ===
        if r['subclass'] in 're':
            pass
        elif hasattr(r, 'new_subclass') and r['new_subclass'] in 're':
            pass
        else:
            raise ValueError('[WARN] subclass not valid for ASL')

        peakf = metrics_df['peakf'].median()
        filt_est = est.copy().select(component='Z')
        if len(filt_est)>=asl_config['min_stations']:
            filt_est.filter('bandpass', freqmin=peakf/1.5, freqmax=peakf*1.5, corners=2, zerophase=True)
            filt_est = filter_traces_by_inventory(filt_est, inventory)
        stations = list(set([tr.stats.station for tr in filt_est]))
        if len(stations)<asl_config['min_stations']:
            raise ValueError(f'[WARN]: Not enough stations for ASL after filtering {stations}')
        print(f'Stations remaining after filtering for ASL: {stations}')

        peakf = int(round(peakf))
        distance_cache_file = os.path.join(TOP_DIR, "grid_node_distances.pkl")
        ampcorr_cache_file = os.path.join(TOP_DIR, f"amplitude_corrections_Q{asl_config['Q']}_F{peakf}.pkl")             
                
        for tr in filt_est:
            tr.stats['units'] = 'm/s'

        vsamObj = VSAM(stream=filt_est, sampling_interval=1.0)
        if len(vsamObj.dataframes)==0:
            raise IOError('No dataframes in vsamObj')
        vsamObj.plot(equal_scale=True, outfile=os.path.join(outdir, "VSAM.png"))
        #vsamObj.write(outdir, ext="csv")

        # Initialize ASL object
        aslobj = ASL(vsamObj, asl_config['vsam_metric'], inventory, asl_config['gridobj'], asl_config['window_seconds'])

        # === Handle node distance caching ===
        if os.path.isfile(distance_cache_file):
            try:
                with open(distance_cache_file, "rb") as f:
                    aslobj.node_distances_km = pickle.load(f)
                print(f"[CACHE HIT] Loaded node distances from {distance_cache_file}")
            except Exception as e:
                print(f"[WARN] Failed to load node distances: {e}")
                print("[INFO] Recomputing node distances...")
                aslobj.compute_grid_distances()
                with open(distance_cache_file, "wb") as f:
                    pickle.dump(aslobj.node_distances_km, f)
        else:
            print("[INFO] Computing node distances from scratch...")
            aslobj.compute_grid_distances()
            with open(distance_cache_file, "wb") as f:
                pickle.dump(aslobj.node_distances_km, f)

        # === Compute amplitude corrections with internal caching ===
        print(f"[INFO] Ensuring amplitude corrections (Q={asl_config['Q']}, f={peakf}) are available...")
        aslobj.compute_amplitude_corrections(
            surfaceWaves=True,
            wavespeed_kms=asl_config['surfaceWaveSpeed_kms'],
            Q=asl_config['Q'],
            fix_peakf=peakf,
            cache_dir=os.path.dirname(ampcorr_cache_file)  # Use the same folder for caching
        )

        try:
            aslobj.fast_locate()
        except Exception as e:
            print('[ERR] ASL location failed')
            raise e

        aslobj.print_event()
        aslobj.save_event(outfile=os.path.join(outdir, f"event_Q{asl_config['Q']}_F{peakf}.qml" ))
        try:
            aslobj.plot(zoom_level=0, threshold_DR=0.0, scale=0.2, join=True, number=0,
                equal_size=False, add_labels=True,
                stations=[tr.stats.station for tr in est],
                outfile=os.path.join(outdir, f"map_Q{asl_config['Q']}_F{peakf}.png" )
                )
        except Exception as e:
            print('[ERR] Cannot plot ASL object')
            raise e
        success = 2
    except Exception as e:
        print(e)
    finally:
        duration = UTCDateTime() - start_time  # seconds as float
        gc.collect()
        return rownum, r['time'], success, float(duration)
'''
def needs_asl_reprocessing(outdir, subclass):
    if subclass not in 're':
        return False
    for qml in glob.glob(os.path.join(outdir, 'event*.qml')):
        try:
            with open(qml, 'r') as f:
                if "OriginQuality" not in f.read():
                    return True
        except Exception:
            return True
    return False

def process_row(rownum, r, numrows, TOP_DIR, source_coords, inventory, asl_config):
    print(f'\nProcessing row {rownum} of {numrows}')
    start_time = UTCDateTime()
    success = 0

    try:
        event_time = UTCDateTime(r['time'])
        ymdfolder = os.path.join(str(event_time.year), f"{event_time.month:02}", f"{event_time.day:02}")
        outdir = os.path.join(TOP_DIR, ymdfolder, r['dfile']).replace('.cleaned','')
        if not os.path.isdir(outdir):
            oldoutdir = os.path.join(TOP_DIR, r['dfile']).replace('.cleaned', '')
            if os.path.isidr(oldoutdir):
                shutil.move(oldoutdir, outdir)
            else:
                os.makedirs(outdir, exist_ok=True)

        # === Check if ASL should be re-run ===
        if r['subclass'] in 're' and not needs_asl_reprocessing(outdir, r['subclass']):
            print(f"[SKIP] ASL already completed with OriginQuality for: {r['dfile']}")
            return rownum, r['time'], 2, 0.0

        # === Check for magnitudes.csv with ME value (result = 1) ===
        magcsv = os.path.join(outdir, 'magnitudes.csv')
        if os.path.isfile(magcsv):
            magdf = pd.read_csv(magcsv)
            if magdf['ME'].notna().any():
                print(f"[SKIP] Magnitudes exist, skipping to ASL decision: {r['dfile']}")
                success = 1  # we allow partial continuation to ASL

        # === Proceed with waveform and metric processing ===
        mseed_path = os.path.join(r['dir'], r['dfile'])
        if not os.path.exists(mseed_path):
            print(f"[WARN] Missing waveform file: {mseed_path}")
            return rownum, r['time'], 0, 0.0

        st = read(mseed_path)
        st = st.select(channel="*H*")
        if len(st) == 0:
            print('[WARN] No H-component traces after filtering')
            return rownum, r['time'], 0, 0.0

        st.filter('highpass', freq=0.5, corners=2, zerophase=True)
        est = EnhancedStream(stream=st)

        # Update source_coords if event origin is present
        for key in ['latitude', 'longitude', 'depth']:
            if not isnan(r[key]):
                source_coords[key] = r[key]

        # If not already computed, compute magnitude metrics
        if success < 1:
            est, aefdf = est.ampengfftmag(inventory, source_coords, verbose=True, snr_min=3.0)
            est.write(os.path.join(outdir, 'enhanced_stream.mseed'), format='MSEED')
            aefdf.to_csv(os.path.join(outdir, 'magnitudes.csv'))

        # === Classification ===
        features = {}
        for tr in est:
            m = getattr(tr.stats, 'metrics', {})
            for key in ['peakf', 'meanf', 'skewness', 'kurtosis']:
                if key in m:
                    features[key] = m[key]
        try:
            classifier = VolcanoEventClassifier()
            label, score = classifier.classify(features)
            r['new_subclass'] = label
            r['new_score'] = score[label]
            print(f"[INFO] Event classified as '{label}' with score {score[label]:.2f}")
        except Exception as e:
            print(f"[WARN] Classification failed: {e}")


        pd.DataFrame([r]).to_csv(os.path.join(outdir, 'database_row.csv'), index=False)

        metrics_df = stream_metrics_to_dataframe(est)
        metrics_df.to_csv(os.path.join(outdir, 'stream_metrics.csv'))

        success = 1  # at least the core metric work succeeded

        # === Determine if ASL should be run ===
        if not (r.get('new_subclass') in 're' or r.get('subclass') in 're'):
            print(f"[INFO] Skipping ASL: subclass not in ['r', 'e']")
            return rownum, r['time'], success, float(UTCDateTime() - start_time)

        # === Prepare filtered stream for ASL ===
        peakf = metrics_df['peakf'].median()
        filt_est = est.copy().select(component='Z')

        if len(filt_est) < asl_config['min_stations']:
            raise ValueError(f"[WARN] Not enough Z-component traces: {len(filt_est)}")

        filt_est.filter(
            'bandpass',
            freqmin=peakf / 1.5,
            freqmax=peakf * 1.5,
            corners=2,
            zerophase=True
        )
        filt_est = filter_traces_by_inventory(filt_est, inventory)

        stations = list(set(tr.stats.station for tr in filt_est))
        if len(stations) < asl_config['min_stations']:
            raise ValueError(f"[WARN] Not enough stations for ASL after filtering: {stations}")
        print(f"[INFO] Stations used for ASL: {stations}")

        # === Prep for ASL ===
        peakf = int(round(peakf))
        for tr in filt_est:
            tr.stats['units'] = 'm/s'

        vsamObj = VSAM(stream=filt_est, sampling_interval=1.0)
        if len(vsamObj.dataframes) == 0:
            raise IOError('[ERR] No dataframes in VSAM object')
        vsamObj.plot(equal_scale=True, outfile=os.path.join(outdir, "VSAM.png"))

        # === ASL object ===
        aslobj = ASL(
            vsamObj,
            asl_config['vsam_metric'],
            inventory,
            asl_config['gridobj'],
            asl_config['window_seconds']
        )

        # === Distance cache ===
        distance_cache_file = os.path.join(TOP_DIR, "grid_node_distances.pkl")
        if os.path.isfile(distance_cache_file):
            try:
                with open(distance_cache_file, "rb") as f:
                    aslobj.node_distances_km = pickle.load(f)
                print(f"[CACHE HIT] Loaded node distances from {distance_cache_file}")
            except Exception as e:
                print(f"[WARN] Failed to load node distances: {e}")
                print("[INFO] Recomputing node distances...")
                aslobj.compute_grid_distances()
                with open(distance_cache_file, "wb") as f:
                    pickle.dump(aslobj.node_distances_km, f)
        else:
            print("[INFO] Computing grid distances...")
            aslobj.compute_grid_distances()
            with open(distance_cache_file, "wb") as f:
                pickle.dump(aslobj.node_distances_km, f)

        # === Amplitude correction cache ===
        ampcorr_cache_file = os.path.join(
            TOP_DIR, f"amplitude_corrections_Q{asl_config['Q']}_F{peakf}.pkl"
        )
        print(f"[INFO] Computing amplitude corrections (Q={asl_config['Q']}, f={peakf})...")
        aslobj.compute_amplitude_corrections(
            surfaceWaves=True,
            wavespeed_kms=asl_config['surfaceWaveSpeed_kms'],
            Q=asl_config['Q'],
            fix_peakf=peakf,
            cache_dir=os.path.dirname(ampcorr_cache_file)
        )

        # === Location and plotting ===
        try:
            aslobj.fast_locate()
        except Exception as e:
            print('[ERR] ASL location failed')
            raise e

        aslobj.print_event()
        aslobj.save_event(
            outfile=os.path.join(outdir, f"event_Q{asl_config['Q']}_F{peakf}.qml")
        )

        try:
            aslobj.plot(
                zoom_level=0,
                threshold_DR=0.0,
                scale=0.2,
                join=True,
                number=0,
                equal_size=False,
                add_labels=True,
                stations=[tr.stats.station for tr in est],
                outfile=os.path.join(outdir, f"map_Q{asl_config['Q']}_F{peakf}.png")
            )
        except Exception as e:
            print('[ERR] Cannot plot ASL object')
            raise e

        success = 2


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {r['dfile']} — {e}")

    finally:
        duration = UTCDateTime() - start_time
        gc.collect()
        return rownum, r['time'], success, float(duration)


def process_chunk(chunk_rows, start_idx, numrows, TOP_DIR, source_coords, inventory, asl_config):
    logfile = os.path.join(TOP_DIR, "system_monitor.csv")
    pid = os.getpid()
    proc_name = multiprocessing.current_process().name
    chunk_start_time = UTCDateTime()
    worker = f'{proc_name} (PID {pid}) at {chunk_start_time}'
    print(f"[INFO] Worker {worker} is processing rows {start_idx} to {start_idx + len(chunk_rows) - 1}")
    #print(f"\n[CHUNK START] Processing rows {start_idx} to {start_idx + len(chunk_rows) - 1} at {chunk_start_time}")

    results = []
    
    for i, row in enumerate(chunk_rows):
        rownum = start_idx + i
        # Do your processing here
        result = process_row(rownum, row, numrows, TOP_DIR, source_coords, inventory, asl_config)
        rownum, _, success, duration = result
        print(f"Row {rownum+1}/{numrows}")
        results.append(result)
        if success < 2:
            with open(os.path.join(TOP_DIR, 'failures.log'), 'a') as f:
                f.write(f"{rownum},{row['time']},{success}\n")

        # ETA estimation
        processed = rownum + 1
        elapsed = UTCDateTime() - chunk_start_time
        avg_per_row = elapsed / (i + 1)
        remaining = numrows - processed
        eta_seconds = avg_per_row * remaining
        eta_time = UTCDateTime() + eta_seconds

        print(f"[{rownum+1}/{numrows}] Took {duration:.1f}s | ETA: {str(eta_time)[11:19]} | Remaining ~{eta_seconds/60:.1f} min", flush=True)

        if rownum % 100 == 0:
            log_system_status_csv(logfile, rownum=rownum)
    print(f"[CHUNK DONE] Processed {len(chunk_rows)} rows in {(UTCDateTime() - chunk_start_time) / 60:.2f} minutes at {UTCDateTime()}")

    return results

def chunkify(lst, n):
    """Split list `lst` into `n` roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]



print('[STARTUP] functions loaded')
####################################################################
if __name__ == "__main__":

    DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
    if not backup_db(DB_PATH, __file__):
        exit()

    print('[STARTUP] backup of database created')

    # === USER CONFIG ===
    N = None  # Set to None for all
    STATIONXML_PATH = "/data/SEISAN_DB/CAL/MV.xml"
    scriptname = os.path.basename(__file__).replace('.py', '')     
    TOP_DIR = os.path.join('/data', scriptname)
    os.makedirs(TOP_DIR, exist_ok=True)

    # === Connect to DB ===
    conn = sqlite3.connect(DB_PATH)
    print('[STARTUP] connected to database')

    query_result_file = os.path.join(TOP_DIR, f'query_result.pkl')
    query_dataframe_pkl = os.path.join(TOP_DIR, f'query_dataframe.pkl')

    if os.path.isfile(query_dataframe_pkl):
        df = pd.read_pickle(query_dataframe_pkl)
    else:

        # === Try loading from .pkl if it exists ===
        if os.path.isfile(query_result_file):
            print(f"[INFO] Loading cached query result from {query_result_file}")
            df = pd.read_pickle(query_result_file)
        else:
            print("[INFO] Running query and saving result to cache.")
            query = '''
            SELECT e.public_id, o.latitude, o.longitude, o.depth, ec.time, ec.author, ec.source, ec.dfile, ec.mainclass, ec.subclass, mfs.dir
            FROM event_classifications ec
            JOIN events e ON ec.event_id = e.public_id
            LEFT JOIN origins o ON o.event_id = e.public_id
            JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
            LEFT JOIN magnitudes m ON m.event_id = e.public_id
            WHERE ec.subclass IS NOT NULL
            GROUP BY e.public_id
            '''
            if N:
                query += f" LIMIT {N}"


            df = pd.read_sql_query(query, conn)
            df.to_pickle(query_result_file)
        
        # sort dataframe
        df = df.sort_values(by="time")
        df.to_pickle(query_dataframe_pkl)

    #df=df[df['subclass']=='r']    
    print(f'got {len(df)} rows')

    print('[STARTUP] dataframe of database query results created')


    # === Load station inventory ===
    inventory = read_inventory(STATIONXML_PATH)
    print('[STARTUP] inventory loaded')

    #DEFAULT_SOURCE_LOCATION = {"latitude": 16.712, "longitude": -62.177, "depth": 3000.0}


    # === Set up ASL ===
    #source = initial_source(lat=dome_location['lat'], lon=dome_location['lon'])
    #grid_cache_file = os.path.join(TOP_DIR, "gridobj_cache.pkl")
    ASL_CONFIG = {
        'window_seconds':5, 
        'min_stations':5, 
        'Q':100, 
        'surfaceWaveSpeed_kms':1.5,
        'vsam_metric':'mean',
        'source': initial_source(lat=dome_location['lat'], lon=dome_location['lon']),
        'grid_cache_file':os.path.join(TOP_DIR, "gridobj_cache.pkl")
        }
    # === Try loading gridobj from pickle ===
    if os.path.isfile(ASL_CONFIG['grid_cache_file']):
        print(f"[INFO] Loading gridobj from cache: {ASL_CONFIG['grid_cache_file']}")
        with open(ASL_CONFIG['grid_cache_file'], 'rb') as f:
            ASL_CONFIG['gridobj'] = pickle.load(f)
    else:
        print("[INFO] Creating new gridobj...")
        ASL_CONFIG['gridobj'] = make_grid(
            center_lat=dome_location['lat'],
            center_lon=dome_location['lon'],
            node_spacing_m=50,
            grid_size_lat_m=6000,
            grid_size_lon_m=6000
        )
        with open(ASL_CONFIG['grid_cache_file'], 'wb') as f:
            pickle.dump(ASL_CONFIG['gridobj'], f)
        print(f"[INFO] Saved gridobj to {ASL_CONFIG['grid_cache_file']}")

    print('[STARTUP] ASL configured')


    # Get source coords
    source_coords = {
        'latitude':dome_location['lat'],
        'longitude':dome_location['lon'],
        'depth':3000
    }

    # === Iterate over rows as dicts ===
    #lod = []
    df = df.reset_index(drop=True)
    rows = df.to_dict(orient="records")
    numrows = len(rows)
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    row_chunks = [chunk for chunk in chunkify(rows, n_workers) if chunk]

    chunk_args = [
        (
            chunk,
            sum(len(c) for c in row_chunks[:i]),  # starting index
            numrows,
            TOP_DIR,
            source_coords,
            inventory,
            ASL_CONFIG
        )
        for i, chunk in enumerate(row_chunks)
    ]

    try:
        with multiprocessing.Pool(processes=n_workers) as pool:
            results_nested = pool.starmap(process_chunk, chunk_args)
    finally:
        df.to_csv(os.path.join(TOP_DIR, 'interrupted_results.csv'), index=False)


    # Flatten results
    results = [item for sublist in results_nested for item in sublist]

    # Insert results back into DataFrame
    for rownum, _, success, duration in results:
        df.at[rownum, 'result'] = success
        df.at[rownum, 'duration_seconds'] = duration


    print(df.head())
    print(df['duration_seconds'].describe())
    total_time = df['duration_seconds'].sum()
    print(f"[SUMMARY] Total processing time: {total_time/3600:.2f} hours")

    df.to_csv(os.path.join(TOP_DIR, 'final_results.csv'), index=False)
    #df.to_pickle(os.path.join(TOP_DIR, 'final_results.pkl'))


    print("\n[✓] Done processing all events.")

