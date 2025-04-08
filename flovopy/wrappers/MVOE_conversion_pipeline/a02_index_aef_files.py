# 02_index_aef_files.py
'''
the AEF indexer:

    Walks the entire AEF directory tree, looking for .AEF files.

    Checks if each AEF file path already exists in the aef_files table.

    Skips already-indexed files (based on file path).

    Processes and adds only new or previously unindexed files.

    Stores file metadata, including trigger/average window, station-channel info, amplitude, energy, SSAM data, etc.

    Writes the parsed content to .json in a mirrored directory tree under your desired output root.

So, if you later drop more .AEF files into the source tree and rerun 02_index_aef_files.py, it'll just pick up the new ones.

Here’s the current schema for the aef_files table and related entries, based on the version created in our indexing code (02_index_aef_files.py):

aef_files:
----------

Stores metadata about each AEF file indexed.
Column	Type	Description
id	INTEGER PRIMARY KEY AUTOINCREMENT	Unique ID for each record
path	TEXT UNIQUE	Full path to the original AEF file
trigger_window	REAL	Trigger window duration in seconds (from metadata line)
average_window	REAL	Averaging window in seconds (from metadata line)
json_path	TEXT	Full path to the generated .json output
sfile_bgs_path	TEXT	Full path to linked S-file from BGS archive
sfile_mvo_path	TEXT	Full path to linked S-file from MVO archive
wav_file_path	TEXT	Full path to corresponding WAV file
parsed_successfully	INTEGER	1 if parsed OK, 0 otherwise
parsed_time	TEXT	ISO timestamp of indexing
error	TEXT	Error message if parsing failed

aef_metrics:
------------

Stores per-trace AEF values from within the file: amplitude, energy, peak frequency, SSAM.
Column	Type	Description
id	INTEGER PRIMARY KEY AUTOINCREMENT	
aef_file_id	INTEGER	Foreign key linking to aef_files.id
station	TEXT	Station code from AEF line
channel	TEXT	Channel code from AEF line
amplitude	REAL	Peak amplitude (in m/s)
energy	REAL	Energy (in J/kg)
max_frequency	REAL	Peak frequency in Hz
trace_id	TEXT	Original SEISAN-style NSLC ID
fixed_id	TEXT	MVO-corrected NSLC ID
from_embedded	INTEGER	1 if AEF line was embedded in S-file, 0 if separate file
frequency_bands	BLOB	Serialized list of frequency bin edges (usually 12 values)
ssam_percentages	BLOB	Serialized list of SSAM amplitude percentages
ssam_energies	BLOB	Serialized list of SSAM energy values


'''
# 02_index_aef_files.py

import os
import sqlite3
import json
from datetime import datetime
from flovopy.seisanio.core.aeffile import AEFfile
from tqdm import tqdm

def index_aef_files(conn, aef_dir, json_output_dir):
    cur = conn.cursor()

    for root, dirs, files in os.walk(aef_dir):
        dirs.sort()
        files.sort()
        for fname in tqdm(files, desc=f"Indexing AEF files in {root}"):
            if not fname.upper().endswith('.AEF'):
                continue

            full_path = os.path.join(root, fname)

            try:
                full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with invalid encoding: {full_path}")
                continue

            # Skip if already indexed
            cur.execute("SELECT 1 FROM aef_files WHERE path = ?", (full_path,))
            if cur.fetchone():
                continue

            try:
                aef = AEFfile(full_path)
                trigger = aef.trigger_window
                avg = aef.average_window

                # Write JSON file for trace metrics
                metrics_out = []
                for row in aef.aefrows:
                    metrics_out.append(row)
                rel_path = os.path.relpath(full_path, aef_dir)
                json_path = os.path.join(json_output_dir, os.path.splitext(rel_path)[0] + '.json')
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w') as f:
                    json.dump(metrics_out, f, indent=2, default=str)

                parsed_successfully = 1
                aef_error = None

            except Exception as e:
                trigger = avg = None
                parsed_successfully = 0
                json_path = None
                aef = None
                aef_error = str(e)

            cur.execute('''INSERT INTO aef_files (path, trigger_window, average_window, json_path, sfile_bgs_path,
                                                  sfile_mvo_path, wav_file_path, parsed_successfully, parsed_time, error)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (full_path, trigger, avg, json_path, None, None, None,
                         parsed_successfully, datetime.utcnow().isoformat(), aef_error))

            if parsed_successfully:
                aef_file_id = cur.lastrowid
                for row in aef.aefrows:
                    ssam_json = json.dumps(row['ssam']) if row.get('ssam') else None
                    cur.execute('''INSERT INTO aef_metrics (aef_file_id, trace_id, station, channel,
                                                           amplitude, energy, maxf, ssam_json)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                (aef_file_id, row.get('fixed_id', ''), row.get('station'), row.get('channel'),
                                 row.get('amplitude'), row.get('energy'), row.get('maxf'), ssam_json))

    conn.commit()

def main(args):
    if os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        index_aef_files(conn, args.aef_dir, args.json_output)
        conn.close()    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index and convert SEISAN AEF files to JSON + DB.")
    parser.add_argument("--aef_dir", required=True, help="Top-level AEF directory")
    parser.add_argument("--json_output", required=True, help="Output directory for JSON files")
    parser.add_argument("--db", required=True, help="SQLite database path")
    args = parser.parse_args()
    main(args)

