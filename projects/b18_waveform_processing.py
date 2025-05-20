"""
b18_waveform_processing.py

This script processes Local Volcanic (LV) events by computing amplitude, energy, frequency, and 
magnitude metrics from cleaned MiniSEED waveform files. Events may also be classified using a 
volcano-seismic classifier and optionally located using an amplitude-based source location (ASL) method.

Each event waveform is enhanced into an `EnhancedStream`, and event-level metrics are saved to CSV. 
Classification results are stored alongside diagnostic plots. Events classified as rockfalls ('r') or 
explosions ('e') may be further processed using the ASL system to estimate source locations and generate 
QuakeML files and spatial plots.

Processing can be parallelized using Python’s multiprocessing module or run in single-threaded mode. 
Command-line arguments allow control over which processing steps are enabled.

Usage:
    python b18_waveform_processing.py [--no-energy] [--no-asl] [--single-core] [--dry-run]

Options:
    --no-energy     Skip waveform metric computation (amp, energy, frequency, magnitude)
    --no-asl        Skip ASL source location and plotting
    --single-core   Run in single-threaded mode (no multiprocessing)
    --dry-run       Run but do not save anything
"""
import os
import sqlite3
import pickle
import pandas as pd
import numpy as np
import multiprocessing
import argparse
from obspy import read, read_inventory, UTCDateTime
from flovopy.core.enhanced import EnhancedStream, VolcanoEventClassifier # EnhancedEvent, 
from flovopy.config_projects import get_config
from flovopy.analysis.asl import make_grid, dome_location, initial_source, ASL
from flovopy.utils.sysmon import log_system_status_csv
from db_backup import backup_db
from flovopy.processing.sam import VSAM

from math import isnan
import gc
import multiprocessing
import glob

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

def make_enhanced_stream(r, inventory, source_coords):
    mseed_path = os.path.join(r['dir'], r['dfile'])
    if not os.path.exists(mseed_path):
        print(f"[WARN] Missing waveform file: {mseed_path}")
        return None, None

    st = read(mseed_path)
    st = st.select(channel="*H*")
    if len(st) == 0:
        print('[WARN] No H-component traces after filtering')
        return None, None

    st.filter('highpass', freq=0.5, corners=2, zerophase=True)
    est = EnhancedStream(stream=st)

    # Update source_coords if event origin is present
    for key in ['latitude', 'longitude', 'depth']:
        if not isnan(r[key]):
            source_coords[key] = r[key]

    est, aefdf = est.ampengfftmag(inventory, source_coords, verbose=True, snr_min=3.0)
    return est, aefdf

    
def reclassify(est, r, dry_run, conn=None):
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
        
        if conn and not dry_run:
            cur = conn.cursor()
            cur.execute('''
                INSERT OR REPLACE INTO event_classifications (
                    event_id, dfile, mainclass, subclass, author, time, source, score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                r['public_id'], r['dfile'], 'LV', label,
                r.get('author', 'unknown'), r['time'], 'VolcanoEventClassifier', score[label]
            ))
            conn.commit()

        return True
    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
        return False
    
import uuid
def insert_metrics_to_db(est, aefdf, r, conn):
    cur = conn.cursor()
    event_id = r['public_id']
    dfile = r['dfile']

    smagse = []
    smagsl = []
    smagsd = []

    for tr in est:
        metrics = getattr(tr.stats, 'metrics', {})
        if not metrics:
            continue

        trace_id = tr.id
        starttime = tr.stats.starttime.isoformat()
        endtime = tr.stats.endtime.isoformat()
        source = 'ampengfftmag'

        # Insert aef_metrics
        cur.execute('''
            INSERT OR IGNORE INTO aef_metrics (
                event_id, trace_id, time, endtime, dfile, snr, peakamp, peaktime, energy, 
                peakf, meanf, ssam_json, spectrum_id, sgramdir, sgramdfile,
                band_ratio1, band_ratio2, skewness, kurtosis, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, trace_id, starttime, endtime, dfile,
            metrics.get('snr'), metrics.get('peakamp'),
            metrics.get('peaktime'), metrics.get('energy'),
            metrics.get('peakf'), metrics.get('meanf'),
            str(metrics.get('ssam_json')) if 'ssam_json' in metrics else None,
            metrics.get('spectrum_id'), metrics.get('sgramdir'), metrics.get('sgramdfile'),
            metrics.get('band_ratio1'), metrics.get('band_ratio2'),
            metrics.get('skewness'), metrics.get('kurtosis'), source
        ))

        # Insert amplitude
        amp_id = str(uuid.uuid4())
        cur.execute('''
            INSERT OR IGNORE INTO amplitudes (
                amplitude_id, event_id, generic_amplitude, unit, type, period, snr, waveform_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            amp_id, event_id, metrics.get('peakamp'), 'm/s', 'peak',
            metrics.get('period', 1.0), metrics.get('snr'), trace_id
        ))

        # Insert station_magnitudes
        if 'ME' in metrics:
            smag_id = str(uuid.uuid4())
            smagse.append(metrics['ME'])

            cur.execute('''
                INSERT OR IGNORE INTO station_magnitudes (
                    smag_id, event_id, station_code, mag, mag_type, amplitude_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                smag_id, event_id, tr.stats.station, metrics['ME'], 'ME', amp_id
            ))

        # Insert station_magnitudes
        if 'ML' in metrics:
            smag_id = str(uuid.uuid4())
            smagsl.append(metrics['ML'])

            cur.execute('''
                INSERT OR IGNORE INTO station_magnitudes (
                    smag_id, event_id, station_code, mag, mag_type, amplitude_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                smag_id, event_id, tr.stats.station, metrics['ML'], 'ML', amp_id
            ))

        # Insert station_magnitudes
        if 'ML' in metrics:
            smag_id = str(uuid.uuid4())
            smagsd.append(metrics['MD'])

            cur.execute('''
                INSERT OR IGNORE INTO station_magnitudes (
                    smag_id, event_id, station_code, mag, mag_type, amplitude_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                smag_id, event_id, tr.stats.station, metrics['MD'], 'MD', amp_id
            ))


    # Insert average magnitude into magnitudes table
    if smagse:
        avg_me = sum(smagse) / len(smagse)
        mag_id = str(uuid.uuid4())
        cur.execute('''
            INSERT OR IGNORE INTO magnitudes (
                mag_id, event_id, magnitude, mag_type, origin_id
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            mag_id, event_id, avg_me, 'ME', None  # or link to actual origin_id
        ))

    if smagsl:
        avg_ml = sum(smagsl) / len(smagsl)
        mag_id = str(uuid.uuid4())
        cur.execute('''
            INSERT OR IGNORE INTO magnitudes (
                mag_id, event_id, magnitude, mag_type, origin_id
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            mag_id, event_id, avg_ml, 'ML', None  # or link to actual origin_id
        ))  

    if smagsd:
        avg_md = sum(smagsd) / len(smagsd)
        mag_id = str(uuid.uuid4())
        cur.execute('''
            INSERT OR IGNORE INTO magnitudes (
                mag_id, event_id, magnitude, mag_type, origin_id
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            mag_id, event_id, avg_md, 'MD', None  # or link to actual origin_id
        ))        

    conn.commit()    
    
def recreate_asl_tables(cur):
    # === Drop existing ASL tables ===
    cur.executescript("""
        DROP TABLE IF EXISTS asl_grid_station_amplitudes;
        DROP TABLE IF EXISTS asl_grid;
        DROP TABLE IF EXISTS asl_node_distances;
        DROP TABLE IF EXISTS asl_results;
        DROP TABLE IF EXISTS asl_model;
        DROP TABLE IF EXISTS asl_grid_definition;
    """)

    # === Create new streamlined ASL tables ===
    cur.execute('''CREATE TABLE IF NOT EXISTS asl_grid_definition (
        grid_id INTEGER PRIMARY KEY AUTOINCREMENT,
        centerlat REAL,
        centerlon REAL,
        nlat INTEGER,
        nlon INTEGER,
        ndepth INTEGER DEFAULT 1,
        node_spacing_m REAL,
        grid_pickle_path TEXT,
        UNIQUE(centerlat, centerlon, node_spacing_m)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS asl_model (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        wavetype TEXT,
        wavespeed REAL,
        peakf REAL,
        window_seconds REAL DEFAULT 5.0,
        q REAL,
        correction_pickle_path TEXT,
        UNIQUE(wavetype, wavespeed, peakf, q)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS asl_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        model_id INTEGER,
        grid_id INTEGER,
        timestamp TEXT,
        window_seconds REAL,
        est_lat REAL,
        est_lon REAL,
        est_elev REAL DEFAULT 0.0,
        reduced_displacement REAL,
        reduced_velocity REAL,
        reduced_energy REAL,
        misfit REAL,
        FOREIGN KEY(grid_id) REFERENCES asl_grid_definition(grid_id),
        FOREIGN KEY(model_id) REFERENCES asl_model(model_id),
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    print("[✓] ASL tables recreated.")


def insert_asl_results_to_db(aslobj, event_id, grid_pickle_path, correction_pickle_path, conn):
    cur = conn.cursor()

    # === Insert or get grid_id ===
    grid = aslobj.gridobj
    cur.execute("""
        INSERT OR IGNORE INTO asl_grid_definition (
            centerlat, centerlon, nlat, nlon, ndepth, node_spacing_m, grid_pickle_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        grid.centerlat, grid.centerlon,
        grid.gridlat.shape[0], grid.gridlon.shape[1], 1,
        grid.spacing_m, grid_pickle_path
    ))
    cur.execute("""
        SELECT grid_id FROM asl_grid_definition
        WHERE centerlat=? AND centerlon=? AND node_spacing_m=?
    """, (grid.centerlat, grid.centerlon, grid.spacing_m))
    grid_id = cur.fetchone()[0]

    # === Insert or get model_id ===
    cur.execute("""
        INSERT OR IGNORE INTO asl_model (
            wavetype, wavespeed, peakf, window_seconds, q, correction_pickle_path
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        'surface' if aslobj.surfaceWaves else 'body',
        aslobj.wavespeed_kms, aslobj.peakf,
        aslobj.window_seconds, aslobj.Q, correction_pickle_path
    ))
    cur.execute("""
        SELECT model_id FROM asl_model
        WHERE wavetype=? AND wavespeed=? AND peakf=? AND q=?
    """, (
        'surface' if aslobj.surfaceWaves else 'body',
        aslobj.wavespeed_kms, aslobj.peakf, aslobj.Q
    ))
    model_id = cur.fetchone()[0]

    # === Insert results from each timestep ===
    for i in range(len(aslobj.source['t'])):
        cur.execute("""
            INSERT INTO asl_results (
                event_id, model_id, grid_id, timestamp, window_seconds,
                est_lat, est_lon, est_elev,
                reduced_displacement, reduced_velocity, reduced_energy, misfit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, model_id, grid_id,
            aslobj.source['t'][i].isoformat(), aslobj.window_seconds,
            aslobj.source['lat'][i], aslobj.source['lon'][i], 0.0,
            aslobj.source['DR'][i], None, None, aslobj.source['misfit'][i]
        ))

    conn.commit()
    print(f"[✓] ASL results written for event {event_id}")

def asl_sausage(peakf, filt_est, outdir, asl_config, inventory, TOP_DIR, dry_run, conn=None, event_id=None):

    ''' SCAFFOLD: determine these
    grid_pickle_path = os.path.join(TOP_DIR, "grid.pkl")
    correction_pickle_path = os.path.join(
        TOP_DIR,
        f"amplitude_corrections_Q{asl_config['Q']}_F{peakf}.pkl"
    )
    '''

    # === Prep for ASL ===
    peakf = int(round(peakf))
    for tr in filt_est:
        tr.stats['units'] = 'm/s'

    vsamObj = VSAM(stream=filt_est, sampling_interval=1.0)
    if len(vsamObj.dataframes) == 0:
        raise IOError('[ERR] No dataframes in VSAM object')
    if not dry_run:
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
            if not dry_run:
                with open(distance_cache_file, "wb") as f:
                    pickle.dump(aslobj.node_distances_km, f)
    else:
        print("[INFO] Computing grid distances...")
        aslobj.compute_grid_distances()
        if not dry_run:
            with open(distance_cache_file, "wb") as f:
                pickle.dump(aslobj.node_distances_km, f)

    # === Amplitude correction cache ===
    #ampcorr_cache_file = os.path.join(
    #    TOP_DIR, f"amplitude_corrections_Q{asl_config['Q']}_F{peakf}.pkl"
    #) # filename is made up internally in this function, which computes for whole inventory
    print(f"[INFO] Computing amplitude corrections (Q={asl_config['Q']}, f={peakf})...")
    aslobj.compute_amplitude_corrections(
        surfaceWaves=True,
        wavespeed_kms=asl_config['surfaceWaveSpeed_kms'],
        Q=asl_config['Q'],
        fix_peakf=peakf,
        cache_dir=TOP_DIR,
    ) # SCAFFOLD GET ALL THESE FROM als_config (Q, surfaceWaves)

    # === Location and plotting ===
    try:
        aslobj.fast_locate()
    except Exception as e:
        print('[ERR] ASL location failed')
        raise e

    aslobj.print_event()
    if not dry_run:
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
        
        # SCAFFOLD
        '''
        if conn and event_id and aslobj.located:
            from db.asl_writer import insert_asl_results_to_db  # adjust as needed
            insert_asl_results_to_db(
                aslobj,
                event_id=event_id,
                grid_pickle_path=grid_pickle_path,
                correction_pickle_path=correction_pickle_path,
                conn=conn
            )
        '''

    return


def process_row(rownum, r, numrows, TOP_DIR, source_coords, inventory, asl_config, compute_energy, run_asl, dry_run, conn=None):
    print(f'\nProcessing row {rownum} of {numrows}')

    start_time = UTCDateTime()
    progress = {'complete':0}

    try:
        #if r['subclass']!='r':
        #    raise ValueError('Sorry - only processing rockfalls at this time')
        event_time = UTCDateTime(r['time'])
        ymdfolder = os.path.join(str(event_time.year), f"{event_time.month:02}", f"{event_time.day:02}")
        outdir = os.path.join(TOP_DIR, ymdfolder, r['dfile']).replace('.cleaned','')

        # === Check if ASL should be re-run ===
        '''
        if r['subclass'] in 're':
            if not needs_asl_reprocessing(outdir, r['subclass']):
                print(f"[SKIP] ASL already completed with OriginQuality for: {r['dfile']}")
                return rownum, r['time'], 2, 0.0
        else:
            # === Check for magnitudes.csv with ME value (result = 1) ===
            magcsv = os.path.join(outdir, 'magnitudes.csv')
            if os.path.isfile(magcsv):
                magdf = pd.read_csv(magcsv)
                if magdf['ME'].notna().any():
                    print(f"[SKIP] Magnitudes exist, this event is complete: {r['dfile']}")
                    return rownum, r['time'], 1, float(UTCDateTime() - start_time)
        '''
        # === Proceed with waveform and metric processing ===
        if compute_energy:

            # Skip if already in DB
            if not dry_run and conn:
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT 1 FROM event_classifications 
                        WHERE event_id = ? AND dfile = ? AND source = ?
                    """, (r.get('public_id') or r.get('event_id'), r['dfile'], 'VolcanoEventClassifier'))
                    if cur.fetchone():
                        print(f"[SKIP] Already classified in DB: {r['dfile']}")
                        return rownum, r['time'], {'skipped': True}, float(UTCDateTime() - start_time)
                except Exception as e:
                    print(f"[WARN] Failed DB check for existing event: {e}")            

            est, aefdf = make_enhanced_stream(r, start_time, rownum, outdir, inventory, source_coords)
            if not isinstance(est, EnhancedStream):
                return rownum, r['time'], progress, float(UTCDateTime() - start_time)
            progress['enhanced'] = True
            if not dry_run:
                est.write(os.path.join(outdir, 'enhanced_stream.mseed'), format='MSEED')
                aefdf.to_csv(os.path.join(outdir, 'magnitudes.csv'))

            # === Classification ===
            progress['reclassified'] = reclassify(est, r, dry_run, conn)

            # save r (row dict)
            if not dry_run:
                pd.DataFrame([r]).to_csv(os.path.join(outdir, 'database_row.csv'), index=False)

            # save trace metrics via dataframe
            metrics_df = stream_metrics_to_dataframe(est)
            metrics_names = stream_metrics_to_dataframe(est).columns
            progress['energy'] = 'energy' in metrics_names
            progress['ME'] = 'ME' in metrics_names
            if not dry_run:
                metrics_df.to_csv(os.path.join(outdir, 'stream_metrics.csv'))
                if conn and isinstance(est, EnhancedStream) and aefdf is not None:
                    insert_metrics_to_db(est, aefdf, r, conn)

        # === Determine if ASL should be run ===
        if run_asl and (r.get('new_subclass') in 're' or r.get('subclass') in 're'):
            if not compute_energy:
                raise ValueError("ASL processing requires energy computation step to run first.")
        else:
            print(f"[INFO] Skipping ASL: subclass not in ['r', 'e']")
            return rownum, r['time'], progress, float(UTCDateTime() - start_time)

        # === Prepare filtered stream for ASL ===
        try:
            peakf = metrics_df['peakf'].median()
        except Exception as e:
            for tr in est:
                print(tr.id, tr.stats.metrics)
            print(f'metrics_df={metrics_df}')
            raise e
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
        asl_sausage(peakf, filt_est, outdir, asl_config, inventory, TOP_DIR, dry_run, conn=conn, event_id=r['event_id'])  # or r['public_id'])

        progress['asl'] = True


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {r['dfile']} — {e}")

    finally:
        duration = UTCDateTime() - start_time
        gc.collect()
        progress['complete'] = sum(progress.values())
        return rownum, r['time'], progress, float(duration)


def process_chunk(chunk_rows, start_idx, numrows, TOP_DIR, source_coords, inventory, asl_config, compute_energy, run_asl, dry_run, conn=None):
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
        result = process_row(rownum, row, numrows, TOP_DIR, source_coords, inventory, asl_config, compute_energy, run_asl, dry_run, conn=conn)
        rownum, _, progress, duration = result
        print(f"Row {rownum+1}/{numrows}")
        results.append(result)
        if not 'reclassified' in progress:
            if not dry_run:
                with open(os.path.join(TOP_DIR, 'failures.log'), 'a') as f:
                    f.write(f"{rownum},{row['time']},{progress['complete']}\n")

        # ETA estimation
        processed = rownum + 1
        elapsed = UTCDateTime() - chunk_start_time
        avg_per_row = elapsed / (i + 1)
        remaining = numrows - processed
        eta_seconds = avg_per_row * remaining
        eta_time = UTCDateTime() + eta_seconds

        print(f"[{rownum+1}/{numrows}] Took {duration:.1f}s | ETA: {str(eta_time)[11:19]} | Remaining ~{eta_seconds/60:.1f} min", flush=True)

        if not dry_run and rownum % 1000 == 0:
            log_system_status_csv(logfile, rownum=rownum)
    print(f"[CHUNK DONE] Processed {len(chunk_rows)} rows in {(UTCDateTime() - chunk_start_time) / 60:.2f} minutes at {UTCDateTime()}")

    return results

def chunkify(lst, n):
    """Split list `lst` into `n` roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    """
    Main entry point for LV waveform processing.

    Loads configuration, backs up the seismic database, queries or loads pre-cached event metadata, 
    initializes the seismic station inventory and ASL configuration, and processes each event using 
    either single-threaded or parallel execution.

    Depending on user-selected options, this function:
      - Computes amplitude, energy, and frequency-based metrics for each event
      - Applies a rule-based volcano event classifier
      - Optionally performs ASL to estimate source location and generate output products

    Results are saved to structured subdirectories beneath the configured waveform processing directory.
    """

    parser = argparse.ArgumentParser(description="Process LV events: compute amp/energy/frequency/magnitude and/or run ASL")
    parser.add_argument("--no-asl", action="store_true", help="Disable ASL processing")
    parser.add_argument("--no-energy", action="store_true", help="Disable energy/magnitude computation")
    parser.add_argument("--single-core", action="store_true", help="Disable multiprocessing (run single-core)")
    parser.add_argument("--dry-run", action="store_true", help="Test mode: no outputs are saved.")
    args = parser.parse_args()
    use_multiprocessing = not args.single_core
    compute_energy = not args.no_energy
    run_asl = not args.no_asl
    dry_run = args.dry_run

    config = get_config()
    dbfile = config['mvo_seiscomp_db']
    top_dir = config['enhanced_results']
    if not dry_run:
        if not backup_db(dbfile, __file__):
            exit()
    conn = sqlite3.connect(dbfile)
    try:
        conn.cursor().execute('''ALTER TABLE event_classifications ADD COLUMN score REAL''')
        conn.commit()
        print("[✓] Column 'score' added to event_classifications.")
    except sqlite3.OperationalError as e:
        if 'duplicate column name' in str(e).lower():
            print("[i] Column 'score' already exists.")
        else:
            raise
    conn.execute("PRAGMA foreign_keys = ON;")
    query_result_file = os.path.join(top_dir, 'query_result.pkl')
    query_dataframe_pkl = os.path.join(top_dir, 'query_dataframe.pkl')

    if os.path.isfile(query_dataframe_pkl):
        df = pd.read_pickle(query_dataframe_pkl)
    else:
        if os.path.isfile(query_result_file):
            df = pd.read_pickle(query_result_file)
        else:
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
            df = pd.read_sql_query(query, conn)
            if not dry_run:
                df.to_pickle(query_result_file)
        df = df.sort_values(by="time")
        if not dry_run:
            df.to_pickle(query_dataframe_pkl)

    print(f"[INFO] Loaded {len(df)} rows")

    # === Load inventory ===
    inventory = read_inventory(config['inventory'])

    # === Configure ASL ===
    source_coords = {
        'latitude': dome_location['lat'],
        'longitude': dome_location['lon'],
        'depth': 3000.0
    }

    asl_config = {
        'window_seconds': 5,
        'min_stations': 5,
        'Q': 100,
        'surfaceWaveSpeed_kms': 1.5,
        'vsam_metric': 'mean',
        'source': initial_source(lat=dome_location['lat'], lon=dome_location['lon']),
        'top_dir': top_dir
    }

    grid_cache_file = os.path.join(top_dir, "gridobj_cache.pkl")
    if os.path.isfile(grid_cache_file):
        with open(grid_cache_file, 'rb') as f:
            asl_config['gridobj'] = pickle.load(f)
    else:
        asl_config['gridobj'] = make_grid(
            center_lat=dome_location['lat'],
            center_lon=dome_location['lon'],
            node_spacing_m=50,
            grid_size_lat_m=6000,
            grid_size_lon_m=6000
        )
        if not dry_run:
            with open(grid_cache_file, 'wb') as f:
                pickle.dump(asl_config['gridobj'], f)

    # === Multiprocessing setup ===
    rows = df.reset_index(drop=True).to_dict(orient="records")
    numrows = len(rows)
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    row_chunks = [chunk for chunk in np.array_split(rows, n_workers) if chunk]

    chunk_args = [
        (
            chunk,
            sum(len(c) for c in row_chunks[:i]),
            numrows,
            top_dir,
            source_coords,
            inventory,
            asl_config,
            compute_energy,
            run_asl,
            dry_run,
            conn
        )
        for i, chunk in enumerate(row_chunks)
    ]

    results_nested = []
    if use_multiprocessing and n_workers > 1:
        try:
            with multiprocessing.Pool(processes=n_workers) as pool:
                results_nested = pool.starmap(process_chunk, chunk_args)
        finally:
            if not dry_run:
                df.to_csv(os.path.join(top_dir, 'interrupted_results.csv'), index=False)
    else:
        print("[INFO] Running in single-core mode")
        for args in chunk_args:
            results_nested.append(process_chunk(*args))    

    # === Collect results ===
    results = [item for sublist in results_nested for item in sublist]
    for rownum, _, progress_summary, duration in results:
        df.at[rownum, 'progress_summary'] = progress_summary
        df.at[rownum, 'duration_seconds'] = duration

    if not dry_run:
        df.to_csv(os.path.join(top_dir, 'final_results.csv'), index=False)
    print("\n[✓] Done processing all events.")

    if dry_run:
        print("\n[DRY-RUN SUMMARY]")
        print("No files were written or modified.")
        total_processed = len(results)
        total_success = sum(1 for _, _, p, _ in results if p.get('complete', 0))
        total_failures = total_processed - total_success
        print(f"Processed {total_processed} events in dry-run mode.")
        print(f"✓ {total_success} succeeded (would have saved outputs)")
        print(f"✗ {total_failures} failed")
        print("Use without '--dry-run' to save results.\n")

print('[STARTUP] functions loaded')
####################################################################

if __name__ == "__main__":
    main()