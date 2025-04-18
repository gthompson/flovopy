#!/usr/bin/env python3
# update_event_waveform_links.py
# Populate event_classifications.dfile and insert into event_waveform_map by matching on event_classifications.time ≈ mseed_file_status.time

import os
import shutil
import sqlite3
from obspy import UTCDateTime
from pprint import pprint

# === USER CONFIG ===
DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
TEST_MODE = True
N = 10  # Set to None for all
TEST_DB_COPY = DB_PATH.replace('.sqlite', '_test.sqlite')
TIME_TOLERANCE = 1.05  # seconds

# === TEST MODE: Copy DB ===
if TEST_MODE:
    #if os.path.exists(TEST_DB_COPY):
    #    os.remove(TEST_DB_COPY)
    #shutil.copy(DB_PATH, TEST_DB_COPY)
    DB_PATH = TEST_DB_COPY
    print(f"[TEST MODE] Using temporary DB: {DB_PATH}")

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Fetch event_classifications entries with NULL dfile
cur.execute("SELECT event_id, time FROM event_classifications WHERE dfile IS NULL")
event_classifications = cur.fetchall()

# Fetch all mseed_file_status entries
cur.execute("SELECT dfile, time FROM mseed_file_status")
mseed_file_status = cur.fetchall()

# Convert mseed_file_status times to UTCDateTime for comparison
mseed_times = [(dfile, UTCDateTime(time)) for dfile, time in mseed_file_status]

updated = 0
inserted = 0

for event_id, ec_time_str in event_classifications:
    ec_time = UTCDateTime(ec_time_str)
    match_found = False
    for dfile, mfs_time in mseed_times:
        if abs(ec_time - mfs_time) <= TIME_TOLERANCE:
            # Update event_classifications.dfile
            cur.execute("UPDATE event_classifications SET dfile = ? WHERE event_id = ?", (dfile, event_id))
            updated += 1
            # Insert into event_waveform_map
            cur.execute("INSERT OR IGNORE INTO event_waveform_map (event_id, dfile) VALUES (?, ?)", (event_id, dfile))
            inserted += 1
            match_found = True
            break  # Assuming one match is sufficient
    if TEST_MODE and match_found:
        pprint({'event_id': event_id, 'matched_dfile': dfile})

conn.commit()
conn.close()

print(f"[INFO] Updated {updated} rows in event_classifications.")
print(f"[INFO] Inserted {inserted} rows into event_waveform_map.")
