#!/usr/bin/env python3
# update_event_waveform_links.py
# Populate event_classifications.dfile and insert into event_waveform_map by matching on event_classifications.time = mseed_file_status.time
import os
import shutil
import sqlite3
from pprint import pprint

# === USER CONFIG ===
DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
TEST_MODE = True
N = 1000  # Set to None for all
TEST_DB_COPY = DB_PATH.replace('.sqlite', '_test.sqlite')

# === TEST MODE: Copy DB ===
if TEST_MODE:
    #if os.path.exists(TEST_DB_COPY):
    #    os.remove(TEST_DB_COPY)
    #shutil.copy(DB_PATH, TEST_DB_COPY)
    DB_PATH = TEST_DB_COPY
    print(f"[TEST MODE] Using temporary DB: {DB_PATH}")


#DB_PATH = "/home/thompsong/public_html/seiscomp_like_test.sqlite"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

query = '''
SELECT ec.event_id, ec.time, mfs.dfile
FROM event_classifications ec
JOIN mseed_file_status mfs ON ec.time = mfs.time
WHERE ec.dfile IS NULL
'''

query = '''
SELECT ec.event_id, ec.time, ec.dfile
FROM event_classifications ec
'''

query = '''
SELECT mfs.dfile, mfs.dir, mfs.time
FROM mseed_file_status mfs
'''
"""
query = '''
SELECT ec.time, mfs.time
FROM event_classifications ec
JOIN mseed_file_status mfs ON ec.time = mfs.time
'''


query = '''
SELECT aef.time, aef.endtime, aef.peaktime
FROM aef_metrics aef
'''

query = '''
SELECT o.time
FROM origins o
'''
"""

if N:
    query += f" LIMIT {N}"

# Fetch matching time-based pairs where dfile is currently null
cur.execute("CREATE INDEX IF NOT EXISTS idx_event_classifications_time ON event_classifications(time);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_mseed_file_status_time ON mseed_file_status(time);")
conn.commit()

print('about to execute query')
cur.execute(query)
print('query executed on cursor')
matches = cur.fetchall()
print('fetchall successed')
print(f"[INFO] Found {len(matches)} time-based matches to update.")

updated = 0
inserted = 0

for row in matches:
    if TEST_MODE:
        pprint(row)
        continue
    event_id, ev_time, dfile = row

    # Update event_classifications.dfile
    cur.execute('''
    UPDATE event_classifications
    SET dfile = ?
    WHERE event_id = ? AND time = ? AND dfile IS NULL
    ''', (dfile, event_id, ev_time))
    updated += cur.rowcount

    # Insert into event_waveform_map
    cur.execute('''
    INSERT OR IGNORE INTO event_waveform_map (event_id, dfile)
    VALUES (?, ?)
    ''', (event_id, dfile))
    inserted += cur.rowcount

conn.commit()
conn.close()

print(f"[✓] Updated {updated} rows in event_classifications.")
print(f"[✓] Inserted {inserted} new rows into event_waveform_map.")