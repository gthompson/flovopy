# 03_index_sfiles.py
'''
The script 03_index_sfiles.py will:

    Index all S-files in the specified REA directory.

    Track whether they were parsed successfully.

    Extract key metadata.

    Link up to 2 WAV files and 1 AEF file (if present).

    Support separate tables for the bgs and mvo archives based on the --archive argument.

03_index_sfiles.py will now:

    Parse AEF lines and trigger/average windows if present in the S-file (via your Sfile and AEFfile classes).

    Save extracted information (metadata, aefrows, trigger/average window) as a .json file per S-file, in a parallel directory structure.
'''


import os
import sqlite3
#import json
from datetime import datetime
from flovopy.seisanio.core.sfile import Sfile
#from flovopy.core.enhanced import EnhancedEvent
from tqdm import tqdm

def index_sfiles(conn, sfile_dir, archive_type, json_output_dir):
    assert archive_type in ["bgs", "mvo"], "archive_type must be 'bgs' or 'mvo'"

    cur = conn.cursor()
    table = f"sfiles_{archive_type}"

    for root, _, files in os.walk(sfile_dir):
        files.sort()
        for fname in tqdm(files, desc=f"Indexing S-files ({archive_type}) in {root}"):
            if not fname.endswith("S"):
                continue

            full_path = os.path.join(root, fname)
            try:
                full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with invalid encoding: {full_path}")
                continue

            # Skip if already indexed
            cur.execute(f"SELECT 1 FROM {table} WHERE path = ?", (full_path,))
            if cur.fetchone():
                continue

            # Try parsing the S-file
            try:
                s = Sfile(full_path, use_mvo_parser=True)
                d = s.to_dict()

                aef_file = s.aeffiles[0].path if s.aeffiles else None
                wavfile1 = d.get("wavfile1")
                wavfile2 = d.get("wavfile2")

                cur.execute(f'''
                    INSERT INTO {table} (path, event_id, parsed_successfully, parsed_time, error, trigger_window, average_window, wavfile1, wavfile2, aef_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    full_path,
                    d.get("id"),
                    1,
                    datetime.utcnow().isoformat(),
                    None,
                    s.aeffiles[0].trigger_window if s.aeffiles else None,
                    s.aeffiles[0].average_window if s.aeffiles else None,
                    wavfile1,
                    wavfile2,
                    aef_file
                ))

                # Save JSON dump of extra info (including AEF data)
                rel_path = os.path.relpath(full_path, sfile_dir)
                enh = s.to_enhancedevent()
                enh.save(json_output_dir, os.path.splitext(rel_path)[0])


            except Exception as e:
                cur.execute(f'''
                    INSERT INTO {table} (path, parsed_successfully, parsed_time, error)
                    VALUES (?, ?, ?, ?)
                ''', (
                    full_path, 0, datetime.utcnow().isoformat(), str(e)
                ))

    conn.commit()

def main(args):
    if os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        index_sfiles(conn, args.sfile_dir, args.archive, args.json_output)
        conn.close() 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index SEISAN S-files for BGS or MVO archive.")
    parser.add_argument("--sfile_dir", required=True, help="Top-level S-file directory")
    parser.add_argument("--json_output", required=True, help="Directory to save extra JSON info")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--archive", choices=["bgs", "mvo"], required=True, help="Which archive to index")
    args = parser.parse_args()
    main(args)
