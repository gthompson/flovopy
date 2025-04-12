'''
The script 03_index_sfiles.py will:

    Index all S-files in the specified REA directory.
    Track whether they were parsed successfully.
    Extract key metadata.
    Link up to 2 WAV files and 1 AEF file (if present).
    Support separate tables for the bgs and mvo archives based on the --archive argument.
    Parse AEF lines and trigger/average windows if present in the S-file (via your Sfile and AEFfile classes).
    Save extracted information (metadata, aefrows, trigger/average window) as a .json file per S-file, in a parallel directory structure.
'''

import os
import sqlite3
from datetime import datetime
from flovopy.seisanio.core.sfile import Sfile
from tqdm import tqdm
import json

def add_column_if_not_exists(conn, table, column, coltype):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cur.fetchall()]
    if column not in columns:
        print(f"[INFO] Adding column '{column}' to table '{table}'")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        conn.commit()

def index_sfiles(conn, sfile_dir, archive_type, json_output_dir, filename_filter=None, limit=None):

    """
    Index SEISAN S-files for metadata and conversion.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite database connection.
    sfile_dir : str
        Root directory containing S-files.
    archive_type : st
        bgs or mvo
    json_output_dir : str
        Output directory to store converted JSON files.
    include_subdirs : bool
        Whether to walk subdirectories under wav_dir.
    filename_filter : callable or str or None
        A function or string used to filter filenames. E.g., lambda f: 'MB' in f
    limit : int or None
        Stop after indexing this many files (useful for testing).
    """
    assert archive_type in ["bgs", "mvo"], "archive_type must be 'bgs' or 'mvo'"    
    cur = conn.cursor()
    count = 0

    walker = os.walk(sfile_dir)
    table = f"sfiles"
    #add_column_if_not_exists(conn, "aef_metrics", "sfile_path", "TEXT")
    #add_column_if_not_exists(conn, "sfiles", "", "TEXT")
    for root, dirs, files in walker:
        dirs.sort()
        files.sort()

        for fname in tqdm(files, desc=f"Indexing S-files ({archive_type}) in {root}"):
            if limit is not None and count >= limit:
                return
            if isinstance(filename_filter, str):
                if filename_filter not in fname:
                    continue
            elif callable(filename_filter):
                if not filename_filter(fname):
                    continue        

            if not '.S' in fname:
                continue

            full_path = os.path.join(root, fname)
            print(f'Processing {full_path}')
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
            if True: # try
                #print('0123456789'*8)
                #os.system(f'cat {full_path}')
                s = Sfile(full_path, use_mvo_parser=True)
                filetime = s.filetime.isoformat()

                aef_file = s.aeffileobj.path if s.aeffileobj else None
                dsn_file = s.dsnwavfileobj.path if s.dsnwavfileobj else None
                asn_file = s.asnwavfileobj.path if s.asnwavfileobj else None

                
                # Save JSON dump of extra info (including AEF data)
                rel_path = os.path.relpath(full_path, sfile_dir)
                enh = s.to_enhancedevent()
                qml_path, json_path = enh.save(json_output_dir, os.path.splitext(rel_path)[0])

                cur.execute(f'''
                    INSERT INTO {table} (path, filetime, event_id, mainclass, subclass, agency, last_action,
                    action_time, analyst, analyst_delay, dsnwavfile, asnwavfile, aeffile, qml_path, json_path, 
                    sfilearchive, parsed_successfully, parsed_time, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    full_path,
                    filetime,
                    s.eventobj.get("id"),
                    s.mainclass,
                    s.subclass,
                    s.agency,
                    s.last_action,
                    s.action_time.isoformat(),
                    s.analyst,
                    s.analyst_delay,
                    dsn_file,
                    asn_file,
                    aef_file,
                    qml_path,
                    json_path,
                    archive_type,
                    1,
                    datetime.utcnow().isoformat(),
                    None,

                ))

                # If AEF data was embedded in the S-file, insert metrics directly
                if s.aeffileobj:
                    for row in s.aeffileobj.aefrows:
                        ssam_json = json.dumps(row['ssam']) if row.get('ssam') else None
                        cur.execute('''
                            INSERT INTO aef_metrics (time, aef_file_id, sfile_path, trace_id,
                                                    amplitude, energy, maxf, ssam_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (filetime, None, full_path, row.get('fixed_id', ''),
                            row.get('amplitude'), row.get('energy'), row.get('maxf'), ssam_json, full_path))

                count += 1

            #except Exception as e:
            else:
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
        index_sfiles(conn, args.sfile_dir, args.archive, args.json_output, limit=args.limit)
        conn.close() 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index SEISAN S-files for BGS or MVO archive.")
    parser.add_argument("--sfile_dir", required=True, help="Top-level S-file directory")
    parser.add_argument("--json_output", required=True, help="Directory to save extra JSON info")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--archive", choices=["bgs", "mvo"], required=True, help="Which archive to index")
    parser.add_argument("--limit", type=int, default=None, help="stop after this number of files")
    args = parser.parse_args()
    print(args)
    #main(args)


'''
python 03_index_sfiles.py \
  --sfile_dir /data/SEISAN_DB/REA/MVOE_ \
  --json_output /data/SEISAN_DB/JSON/MVOE_SFILES \
  --db /home/thompsong/public_html/index_sfile_test.sqlite \
  --archive bgs \
  --verbose
'''