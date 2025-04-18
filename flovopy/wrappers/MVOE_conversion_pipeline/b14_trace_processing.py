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

# === USER CONFIG ===
DB_PATH = "/home/thompsong/public_html/seiscomp_like_test.sqlite"
TEST_MODE = True
N = 10  # Set to None for all
STATIONXML_PATH = "/data/SEISAN_DB/CAL/MV.xml"
OUTPUT_JSON_DIR = "/data/metrics_json"
OUTPUT_PKL_DIR = "/data/metrics_pkl"
DEFAULT_SOURCE_LOCATION = {"latitude": 16.712, "longitude": -62.177, "depth": 3000.0}       
TEST_DB_COPY = "/tmp/seiscomp_like_test.sqlite"

# === TEST MODE: Copy DB ===
if TEST_MODE:
    #if os.path.exists(TEST_DB_COPY):
    #    os.remove(TEST_DB_COPY)
    #shutil.copy(DB_PATH, TEST_DB_COPY)
    DB_PATH = TEST_DB_COPY
    print(f"[TEST MODE] Using temporary DB: {DB_PATH}")
DB_PATH = 'test.sqlite'
# === Ensure output dirs exist ===
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_PKL_DIR, exist_ok=True)

# === Connect to DB ===
conn = sqlite3.connect(DB_PATH)
# Set the row factory to sqlite3.Row to access columns by name
conn.row_factory = sqlite3.Row
cur = conn.cursor()

query = '''
SELECT e.public_id, o.time, o.latitude, o.longitude, o.depth, ec.dfile, e.creation_time, mfs.dir
FROM event_classifications ec
JOIN events e ON ec.event_id = e.public_id
LEFT JOIN origins o ON o.event_id = e.public_id
JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
WHERE ec.mainclass = 'LV'
GROUP BY e.public_id
'''
query = '''
SELECT e.public_id, o.time, o.latitude, o.longitude, o.depth, ec.author, ec.source, ec.dfile, ec.mainclass, ec.subclass, mfs.dir
FROM event_classifications ec
JOIN events e ON ec.event_id = e.public_id
LEFT JOIN origins o ON o.event_id = e.public_id
JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
WHERE ec.subclass IS NOT NULL
GROUP BY e.public_id
'''
if N:
    query += f" LIMIT {N}"

cur.execute(query)
rows = cur.fetchall()
#print(rows)

# === Load station inventory ===
inventory = read_inventory(STATIONXML_PATH)

# === Process each event ===
for row in rows:
    r = dict(row) #row dictionary

    if TEST_MODE:
        print("\n[TEST MODE] Row:")
        pprint(r)

    #try:
    print(f"\n[INFO] Processing {r['public_id']} {r['dfile']}")

    mseed_path = os.path.join(r['dir'], r['dfile'])
    if not os.path.exists(mseed_path):
        print(f"[WARN] File not found: {mseed_path}")
        continue

    # Load waveform
    st = read(mseed_path)
    for tr in st:
        if tr.stats.channel[1]!='H':
            st.remove(tr)
    st.filter('highpass', freq=0.5, corners=2, zerophase=True)
    est = EnhancedStream(stream=st)

    # Get source coords
    source_coords = {
        "latitude": r['latitude'] if r['latitude'] is not None else DEFAULT_SOURCE_LOCATION["latitude"],
        "longitude": r['longitude'] if r['longitude'] is not None else DEFAULT_SOURCE_LOCATION["longitude"],
        "depth": r['depth'] if r['depth'] is not None else DEFAULT_SOURCE_LOCATION["depth"]
    }

    # Compute metrics
    est, df = est.ampengfftmag(inventory, source_coords, verbose=True, snr_min=4.0)

    # Classify
    features = {}
    for tr in est:
        m = getattr(tr.stats, 'metrics', {})
        for key in ['peakf', 'meanf', 'skewness', 'kurtosis']:
            if key in m:
                features[key] = m[key]
    #if trigger and 'duration' in trigger:
    #    features['duration'] = trigger['duration']
    classifier = VolcanoEventClassifier()
    subclass, score = classifier.classify(features)
    print('Classification: ',subclass, score)
    print('Original: ', r['mainclass'], r['subclass'])

    # Convert to EnhancedEvent
    #eev = est.to_enhancedevent(event_id=r['public_id'], dfile=r['dfile'], origin_time=r['time'])
    continue
    if TEST_MODE:
        print(est)
        print(df)
        print(source_coords)
        for tr in est:
            print()
            print(tr.id, tr.stats.metrics.snr)
        est.plot(outfile=f'{r["dfile"].replace("cleaned", "png")}', equal_scale=False)
        #print(eev)
        continue
    continue
    # Write EnhancedEvent to DB
    eev.write_to_db(conn)

    # Save as JSON
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{event_id}.json")
    eev.to_json(json_path)

    # Save as pickle (strip waveform data)
    est.to_pickle(OUTPUT_PKL_DIR, remove_data=True, mseed_path=mseed_path)

    #except Exception as e:
    #    print(f"[ERROR] Failed to process {r['public_id']}: {e}")

print("\n[✓] Done processing all events.")
