import os
import pandas as pd
import sqlite3
import uuid
from obspy.core import UTCDateTime
from flovopy.config_projects import get_config
from db_backup import backup_db

# === Load configuration
config = get_config()
DB_PATH = config['mvo_seiscomp_db']
BASE_DIR = config['enhanced_results']

# === Backup and connect to database
if not backup_db(DB_PATH, __file__):
    exit()

# --- Connect to database ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
try:
    cur.execute('''ALTER TABLE event_classifications ADD COLUMN score REAL''')
    conn.commit()
    print("[✓] Column 'score' added to event_classifications.")
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e).lower():
        print("[i] Column 'score' already exists.")
    else:
        raise

# --- Commit control ---
commit_interval = 100
event_counter = 0

# --- Walk through directories ---
for root, dirs, files in os.walk(BASE_DIR):
    if 'stream_metrics.csv' in files and 'database_row.csv' in files:

        stream_csv = os.path.join(root, 'stream_metrics.csv')
        row_csv = os.path.join(root, 'database_row.csv')

        try:
            stream_df = pd.read_csv(stream_csv)
            row_df = pd.read_csv(row_csv)
            if row_df.empty:
                continue
            r = row_df.iloc[0].to_dict()

            # Required values
            event_id = r.get('public_id') or r.get('event_id')
            dfile = r['dfile']
            subclass = r.get('new_subclass') or r.get('subclass')
            score = r.get('new_score') or None
            time = r.get('time') or UTCDateTime(os.path.basename(root).split('.')[0]).isoformat()

            # === Insert into event_classifications ===
            cur.execute("""
                INSERT OR IGNORE INTO event_classifications (
                    event_id, dfile, mainclass, subclass, author, time, source, score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id, dfile, 'LV', subclass, 'system', time, 'VolcanoEventClassifier', score
            ))

            smags = {'ME': [], 'ML': []}

            for _, row in stream_df.iterrows():
                trace_id = row['id']
                amp_id = str(uuid.uuid4())

                # Insert amplitude row
                if not pd.isna(row.get('peakamp')):
                    cur.execute("""
                        INSERT OR IGNORE INTO amplitudes (
                            amplitude_id, event_id, generic_amplitude, unit, type, period, snr, waveform_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        amp_id, event_id, row['peakamp'], 'm/s', 'peak', None, row.get('snr'), trace_id
                    ))
                else:
                    amp_id = None

                # Insert station magnitudes for ME and ML
                for mag_type in ['ME', 'ML']:
                    if not pd.isna(row.get(mag_type)):
                        smag_id = str(uuid.uuid4())
                        smags[mag_type].append(row[mag_type])
                        cur.execute("""
                            INSERT OR IGNORE INTO station_magnitudes (
                                smag_id, event_id, station_code, mag, mag_type, amplitude_id
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            smag_id, event_id, trace_id.split('.')[1], row[mag_type], mag_type, amp_id
                        ))

                # Insert into aef_metrics
                ssam_json = row.get('ssam') if 'ssam' in row else None
                cur.execute("""
                    INSERT OR IGNORE INTO aef_metrics (
                        event_id, trace_id, time, endtime, dfile, snr, peakamp, peaktime, energy,
                        peakf, meanf, ssam_json, spectrum_id, sgramdir, sgramdfile, band_ratio1,
                        band_ratio2, skewness, kurtosis, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id, trace_id, row.get('starttime'), row.get('endtime'), dfile,
                    row.get('snr'), row.get('peakamp'), row.get('peaktime'), row.get('energy'),
                    row.get('peakf'), row.get('meanf'), ssam_json, None, None, None,
                    row.get('band_ratio1'), row.get('band_ratio2'),
                    row.get('skewness'), row.get('kurtosis'), 'ampengfftmag'
                ))

            # Insert average magnitudes
            for mag_type, values in smags.items():
                if values:
                    mag_id = str(uuid.uuid4())
                    avg_mag = sum(values) / len(values)
                    cur.execute("""
                        INSERT OR IGNORE INTO magnitudes (
                            mag_id, event_id, magnitude, mag_type, origin_id
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        mag_id, event_id, avg_mag, mag_type, None
                    ))

            event_counter += 1

            # Periodic commit
            if event_counter % commit_interval == 0:
                conn.commit()
                print(f"[✓] Committed after {event_counter} events...")

            print(f"[✓] Inserted metrics for {event_id}")

        except Exception as e:
            print(f"[✗] Failed for {root}: {e}")
            conn.rollback()

# Final commit in case of leftovers
if event_counter % commit_interval != 0:
    conn.commit()
    print(f"[✓] Final commit after {event_counter} total events.")

conn.close()
