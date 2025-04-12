# 02_index_aef_files.py
"""
AEF File Indexer and Parser

This script walks a directory tree containing SEISAN-format AEF files, parses each file,
and populates an SQLite database with event metadata and trace-level metrics.

Workflow:
---------
- Recursively traverses the AEF directory tree, identifying `.AEF` files.
- Checks if each file has already been indexed in the `aef_files` table.
- Reprocesses files only if `aef_metrics` are missing.
- Parses each AEF file for trigger/average window settings and per-trace metrics.
- Writes trace-level metrics to JSON in a mirrored output directory tree.
- Populates or updates the following database tables:
    - `aef_files` — one row per AEF file
    - `aef_metrics` — one row per trace within each AEF file

AEF Database Schema:
--------------------

aef_files
~~~~~~~~~
Stores metadata for each AEF file processed.

| Column           | Type     | Description                                           |
|------------------|----------|-------------------------------------------------------|
| id               | INTEGER  | Primary key                                           |
| path             | TEXT     | Full path to original AEF file (unique)              |
| trigger_window   | REAL     | Trigger window duration (seconds)                    |
| average_window   | REAL     | Averaging window duration (seconds)                  |
| json_path        | TEXT     | Path to exported JSON summary                        |
| sfile_bgs_path   | TEXT     | Path to linked BGS S-file (optional)                 |
| sfile_mvo_path   | TEXT     | Path to linked MVO S-file (optional)                 |
| wav_file_path    | TEXT     | Path to corresponding WAV file (optional)            |
| parsed_successfully | INTEGER | 1 if parsed successfully, 0 otherwise              |
| parsed_time      | TEXT     | ISO timestamp of last parse                          |
| error            | TEXT     | Error message if parsing failed                      |

aef_metrics
~~~~~~~~~~~
Stores per-trace values extracted from each AEF file.

| Column            | Type     | Description                                          |
|-------------------|----------|------------------------------------------------------|
| id                | INTEGER  | Primary key                                          |
| aef_file_id       | INTEGER  | Foreign key to `aef_files.id`                        |
| trace_id          | TEXT     | Original SEISAN NSLC ID                              |
| fixed_id          | TEXT     | Corrected NSLC ID                                    |
| station           | TEXT     | Station code                                         |
| channel           | TEXT     | Channel code                                         |
| amplitude         | REAL     | Peak amplitude (m/s)                                 |
| energy            | REAL     | Energy (J/kg)                                        |
| maxf              | REAL     | Peak frequency (Hz)                                  |
| ssam_json         | TEXT     | JSON-encoded SSAM dictionary (optional)              |

Re-runnability:
---------------
The script is idempotent. It skips already-processed files with complete metadata,
and automatically reprocesses files missing `aef_metrics`.

Usage:
------
Run from CLI or pipeline wrapper using:

    python a02_index_aef_files.py --aef_dir /path/to/AEFs \
                                  --json_output /output/json_dir \
                                  --db /path/to/index.sqlite
"""

import os
import sqlite3
import json
from datetime import datetime
from flovopy.seisanio.core.aeffile import AEFfile
from tqdm import tqdm

def index_aef_files(conn, aef_dir, json_output_dir, verbose=False, filename_filter=None, limit=None):
    """
    Index SEISAN AEF files for metadata and conversion.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite database connection.
    aef_dir : str
        Root directory containing AEF files.
    json_output_dir : str
        Output directory to store converted JSON files.
    filename_filter : callable or str or None
        A function or string used to filter filenames. E.g., lambda f: 'MB' in f
    limit : int or None
        Stop after indexing this many files (useful for testing).
    """
    cur = conn.cursor()
    count = 0

    walker = os.walk(aef_dir)

    for root, dirs, files in walker:
        dirs.sort()
        files.sort()

        for fname in tqdm(files, desc=f"Indexing AEF files in {root}"):
            if limit is not None and count >= limit:
                return
            if isinstance(filename_filter, str):
                if filename_filter not in fname:
                    continue
            elif callable(filename_filter):
                if not filename_filter(fname):
                    continue
            print('\n\n')
            if not fname.upper().endswith('.AEF'):
                continue

            full_path = os.path.join(root, fname)

            try:
                encoded_path = full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with invalid encoding: {full_path}")
                # We log failed AEF files into a separate table with the specific error
                cur.execute(
                    '''INSERT INTO aef_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, 'UnicodeEncodeError')
                )              
                continue            

            # Check if already indexed in aef_files
            cur.execute("SELECT id FROM aef_files WHERE path = ?", (full_path,))
            existing_file = cur.fetchone()

            if existing_file:
                aef_file_id = existing_file[0]
                # Check if metrics already exist
                cur.execute("SELECT COUNT(*) FROM aef_metrics WHERE aef_file_id = ?", (aef_file_id,))
                metric_count = cur.fetchone()[0]
                if metric_count > 0:
                    print(f"[SKIP] Already has metrics: {full_path}")
                    continue
                else:
                    print(f"[INFO] Reprocessing for missing metrics: {full_path}")
            else:
                aef_file_id = None

            try:
                print(f"[INFO] Parsing AEF file: {full_path}")
                try:
                    aef = AEFfile(full_path)
                except Exception as e:
                    print(f"[ERROR] AEFfile parser failed")
                    raise e
                trigger = aef.trigger_window
                avg = aef.average_window
                filetime = aef.filetime.isoformat()
                network = aef.network

                print(f"[INFO] Extracted trigger={trigger}, average={avg}")

                metrics_out = []
                for row in aef.aefrows:
                    if verbose:
                        print(f"[DEBUG] Parsed row: {row}")
                    metrics_out.append(row)

                rel_path = os.path.relpath(full_path, aef_dir)
                json_path = os.path.join(json_output_dir, os.path.splitext(rel_path)[0] + '.json')
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w') as f:
                    json.dump(metrics_out, f, indent=2, default=str)

                parsed_successfully = 1

                aef_error = None
                last_op = ''
                if not existing_file:
                    last_op = 'insert aef_files'
                    print((full_path, filetime, network, trigger, avg, json_path, None,  None,
                                parsed_successfully, datetime.utcnow().isoformat(), aef_error))
                    cur.execute('''INSERT INTO aef_files (path, filetime, network, trigger_window, average_window, json_path, 
                                                        sfile_path, wav_file_path, parsed_successfully, parsed_time, error)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (full_path, filetime, network, trigger, avg, json_path, None,  None,
                                parsed_successfully, datetime.utcnow().isoformat(), aef_error))
                    aef_file_id = cur.lastrowid
                else:
                    last_op = 'update aef_files'
                    cur.execute('''UPDATE aef_files SET filetime=?, network=?, trigger_window=?, average_window=?, json_path=?,
                                parsed_successfully=?, parsed_time=?, error=? WHERE id=?''',
                                (filetime, network, trigger, avg, json_path, parsed_successfully, datetime.utcnow().isoformat(), aef_error, aef_file_id))

                if aef:
                    for row in aef.aefrows:
                        ssam_json = json.dumps(row.get('ssam')) if row.get('ssam') else None
                        last_op = 'insert aef_metrics'
                        cur.execute('''INSERT INTO aef_metrics (time, aef_file_id, sfile_path, trace_id,
                                                            amplitude, energy, maxf, ssam_json)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                    (filetime, aef_file_id, None, row.get('fixed_id', ''),
                                    row.get('amplitude'), row.get('energy'), row.get('maxf'), ssam_json))
                    print(f"[INFO] Inserted {len(aef.aefrows)} metrics for file {full_path}")
                else:
                    print(f"[WARN] No rows parsed from AEF file: {full_path}")
                count += 1

            except Exception as e: # If AEF file cannot be read or parsed
                print(f"[ERROR] Failed to parse {full_path}: {e} {last_op}")
                # We log failed AEF files into a separate table with the specific error, as well as below in the wav_files table
                cur.execute(
                    '''INSERT INTO aef_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, str(e))
                )                        

    conn.commit()

def main(args):
    if os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        index_aef_files(conn, args.aef_dir, args.json_output, limit=args.limit)
        conn.close()    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index and convert SEISAN AEF files to JSON + DB.")
    parser.add_argument("--aef_dir", required=True, help="Top-level AEF directory")
    parser.add_argument("--json_output", required=True, help="Output directory for JSON files")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--limit", type=int, default=None, help="stop after this number of files")
    args = parser.parse_args()
    print(args)
    #main(args)