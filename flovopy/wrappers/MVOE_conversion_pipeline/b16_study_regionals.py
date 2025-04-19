# b16_study_regionals
#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd
from obspy import read, read_inventory, UTCDateTime
from flovopy.core.enhanced import EnhancedStream
from pprint import pprint
from flovopy.processing.detection import picker_with_plot
from flovopy.wrappers.MVOE_conversion_pipeline.db_backup import backup_db
DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
if not backup_db(DB_PATH, __file__):
    exit()
TEST_MODE = False
N = None  # Set to None for all
STATIONXML_PATH = "/data/SEISAN_DB/CAL/MV.xml"
OUTPUT_JSON_DIR = "/data/metrics_json"
OUTPUT_PKL_DIR = "/data/metrics_pkl"
DEFAULT_SOURCE_LOCATION = {"latitude": 16.712, "longitude": -62.177, "depth": 3000.0}       

# === Ensure output dirs exist ===
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_PKL_DIR, exist_ok=True)

# === Connect to DB ===
conn = sqlite3.connect(DB_PATH)
# Set the row factory to sqlite3.Row to access columns by name
conn.row_factory = sqlite3.Row
cur = conn.cursor()

query = '''
SELECT e.public_id, ec.time, o.latitude, o.longitude, o.depth, ec.dfile, e.creation_time, mfs.dir
FROM event_classifications ec
JOIN events e ON ec.event_id = e.public_id
LEFT JOIN origins o ON o.event_id = e.public_id
JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
WHERE ec.mainclass = 'R'
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
lod = []
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
    picks=picker_with_plot(st)
    #if len(picks)<10:
    #    continue
    print(f'Got {len(picks)} picks from {len(st)} traces')
    est = EnhancedStream(stream=st)

    # Get source coords
    source_coords = {
        "latitude": r['latitude'] if r['latitude'] is not None else DEFAULT_SOURCE_LOCATION["latitude"],
        "longitude": r['longitude'] if r['longitude'] is not None else DEFAULT_SOURCE_LOCATION["longitude"],
        "depth": r['depth'] if r['depth'] is not None else DEFAULT_SOURCE_LOCATION["depth"]
    }

    # Compute metrics
    est, df = est.ampengfftmag(inventory, source_coords, verbose=True, snr_min=4.0)
    for tr in est:
        print(tr.id, tr.stats.metrics.peakamp)

    # Now we want to create a dataframe. index column should be time. columns should be trace IDs. two more columns should be number of picks and dfile.
    this_dict = {'time': r['time'], 'num_picks': len(picks), 'dfile': r['dfile']}
    for tr in est:
        this_dict[tr.id] = tr.stats.metrics.peakamp *1e6
    print(this_dict)
    lod.append(this_dict)

print("\n[✓] Done processing all events.")
df = pd.DataFrame(lod)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.sort_index(inplace=True)

print(df)
df.to_csv('all_regionals_station_corrections.csv')