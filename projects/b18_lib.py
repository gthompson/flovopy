# Refactored b18.py (Full expanded, modular, working version)

import os
import shutil
import sqlite3
import glob
import pickle
import gc
import multiprocessing
from functools import partial
from math import isnan

import pandas as pd
from obspy import read, read_inventory, UTCDateTime

from flovopy.core.enhanced import EnhancedStream, VolcanoEventClassifier
from flovopy.processing.sam import VSAM
from flovopy.analysis.asl import ASL, initial_source, make_grid, dome_location
from flovopy.utils.sysmon import log_system_status_csv
from flovopy.wrappers.MVOE_conversion_pipeline.db_backup import backup_db

# --- Helper Functions ---

def make_enhanced_stream(r, inventory, source_coords):
    mseed_path = os.path.join(r['dir'], r['dfile'])
    if not os.path.exists(mseed_path):
        print(f"[WARN] Missing waveform file: {mseed_path}")
        return None, None
    st = read(mseed_path).select(channel="*H*")
    if len(st) == 0:
        print("[WARN] No H-component traces after filtering")
        return None, None
    st.filter('highpass', freq=0.5, corners=2, zerophase=True)
    est = EnhancedStream(stream=st)
    for key in ['latitude', 'longitude', 'depth']:
        if not isnan(r[key]):
            source_coords[key] = r[key]
    est, aefdf = est.ampengfftmag(inventory, source_coords, verbose=True, snr_min=3.0)
    return est, aefdf

def reclassify(est, r):
    features = {}
    for tr in est:
        m = getattr(tr.stats, 'metrics', {})
        for key in ['peakf', 'meanf', 'skewness', 'kurtosis']:
            if key in m:
                features[key] = m[key]
    classifier = VolcanoEventClassifier()
    label, score = classifier.classify(features)
    r['new_subclass'] = label
    r['new_score'] = score[label]
    print(f"[INFO] Classified as '{label}' with score {score[label]:.2f}")
    return True

def asl_sausage(peakf, filt_est, outdir, asl_config):
    for tr in filt_est:
        tr.stats['units'] = 'm/s'
    vsamObj = VSAM(stream=filt_est, sampling_interval=1.0)
    vsamObj.plot(equal_scale=True, outfile=os.path.join(outdir, "VSAM.png"))
    aslobj = ASL(
        vsamObj,
        asl_config['vsam_metric'],
        asl_config['inventory'],
        asl_config['gridobj'],
        asl_config['window_seconds']
    )
    distance_cache = os.path.join(asl_config['top_dir'], "grid_node_distances.pkl")
    if os.path.isfile(distance_cache):
        with open(distance_cache, "rb") as f:
            aslobj.node_distances_km = pickle.load(f)
    else:
        aslobj.compute_grid_distances()
        with open(distance_cache, "wb") as f:
            pickle.dump(aslobj.node_distances_km, f)
    aslobj.compute_amplitude_corrections(
        surfaceWaves=True,
        wavespeed_kms=asl_config['surfaceWaveSpeed_kms'],
        Q=asl_config['Q'],
        fix_peakf=peakf,
        cache_dir=asl_config['top_dir']
    )
    aslobj.fast_locate()
    aslobj.save_event(outfile=os.path.join(outdir, f"event_Q{asl_config['Q']}_F{peakf}.qml"))
    aslobj.plot(
        zoom_level=0,
        threshold_DR=0.0,
        scale=0.2,
        join=True,
        number=0,
        equal_size=False,
        add_labels=True,
        stations=[tr.stats.station for tr in filt_est],
        outfile=os.path.join(outdir, f"map_Q{asl_config['Q']}_F{peakf}.png")
    )

# --- Main Row Processing ---

def process_row(rownum, r, numrows, asl_config, inventory, source_coords):
    start_time = UTCDateTime()
    progress = {}
    try:
        if r['subclass'] != 'r':
            raise ValueError('Only processing rockfalls at this time')
        event_time = UTCDateTime(r['time'])
        ymdfolder = os.path.join(str(event_time.year), f"{event_time.month:02}", f"{event_time.day:02}")
        outdir = os.path.join(asl_config['top_dir'], ymdfolder, r['dfile']).replace('.cleaned', '')
        os.makedirs(outdir, exist_ok=True)

        est, aefdf = make_enhanced_stream(r, inventory, source_coords)
        if not isinstance(est, EnhancedStream):
            return rownum, r['time'], '', float(UTCDateTime() - start_time)
        est.write(os.path.join(outdir, 'enhanced_stream.mseed'), format='MSEED')
        aefdf.to_csv(os.path.join(outdir, 'magnitudes.csv'))
        progress['enhanced'] = True

        if reclassify(est, r):
            progress['reclassified'] = True

        pd.DataFrame([r]).to_csv(os.path.join(outdir, 'database_row.csv'), index=False)
        metrics_df = pd.DataFrame([tr.stats.metrics for tr in est if hasattr(tr.stats, 'metrics')])
        metrics_df.to_csv(os.path.join(outdir, 'stream_metrics.csv'), index=False)

        if r.get('new_subclass') not in 're':
            return rownum, r['time'], '|'.join(progress.keys()), float(UTCDateTime() - start_time)

        peakf = metrics_df['peakf'].median()
        filt_est = est.copy().select(component='Z')
        filt_est.filter('bandpass', freqmin=peakf/1.5, freqmax=peakf*1.5, corners=2, zerophase=True)
        asl_sausage(peakf, filt_est, outdir, asl_config)
        progress['asl'] = True

    except Exception as e:
        print(f"[FAIL] Row {rownum}: {e}")

    duration = UTCDateTime() - start_time
    return rownum, r['time'], '|'.join(progress.keys()), float(duration)

# --- Chunk Processor ---

def process_chunk(chunk_rows, start_idx, numrows, asl_config, inventory, source_coords):
    results = []
    for i, r in enumerate(chunk_rows):
        rownum = start_idx + i
        result = process_row(rownum, r, numrows, asl_config, inventory, source_coords)
        results.append(result)
    return results
