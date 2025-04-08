# 01_index_wav_files.py

import os
import sqlite3
from datetime import datetime
from obspy import read
from obspy.core.trace import Trace
from flovopy.core.mvo import correct_nslc_mvo, fix_trace_mvo
from obspy.core import Stream
from tqdm import tqdm

def index_wav_files(conn, wav_dir, mseed_output_dir):
    cur = conn.cursor()

    for root, dirs, files in os.walk(wav_dir):
        dirs.sort()
        files.sort()
        for fname in tqdm(files, desc=f"Indexing WAV files in {root}"):

            if fname.endswith('.png'):
                continue
            if 'dbspn' in root:
                continue
            full_path = os.path.join(root, fname)

            try:
                encoded_path = full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with invalid encoding: {encoded_path}")
                # We log failed WAV files into a separate table with the specific error
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, 'UnicodeEncodeError')
                )              
                continue

            fnamelower = fname.lower()
            if 'mvo' in fnamelower or 'asne' in fnamelower or 'dsne' in fnamelower or 'SPN' in fnamelower:
                print(f'Processing {fname}')
            else:
                print(f'Not processing {fname}')
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, 'Invalid filename')
                )                
                continue

            legacy = False
            if 'asne' in fnamelower or 'spn' in fnamelower:
                legacy = True # WAV file from the analog network
            

            # Skip if already indexed
            cur.execute("SELECT 1 FROM wav_files WHERE path = ?", (encoded_path,))
            if cur.fetchone():
                print(f'Skipping - already processed')
                continue

            # Try reading the WAV file
            try:
                print('- Reading')
                st = read(full_path)
                start_time = str(min(tr.stats.starttime for tr in st))
                end_time = str(max(tr.stats.endtime for tr in st))
                parsed_wav = 1
                wav_error = None

                # Fix trace IDs and write to MiniSEED in a parallel structure
                new_st = Stream()
                print('- Fixing ids')
                for tr in st:
                    original_id = tr.id
                    fix_trace_mvo(tr, legacy=False, netcode='MV')
                    fixed_id = tr.id
                    new_st += tr

                    # track all combinations of date, original_id, and fixed_id
                    cur.execute(
                        '''INSERT INTO trace_id_corrections (date, original_id, corrected_id) VALUES (?, ?, ?)''',
                        (tr.stats.starttime.date.isoformat(), original_id, fixed_id)
                    )
                    if tr.stats.endtime.date != tr.stats.starttime.date:
                        cur.execute(
                            '''INSERT INTO trace_id_corrections (date, original_id, corrected_id) VALUES (?, ?, ?)''',
                            (tr.stats.endtime.date.isoformat(), original_id, fixed_id)
                        )                        

                rel_path = os.path.relpath(full_path, wav_dir)
                #new_fname = os.path.splitext(rel_path)[0] + ".mseed"
                new_fname = rel_path + '.mseed'
                mseed_path = os.path.join(mseed_output_dir, new_fname)
                os.makedirs(os.path.dirname(mseed_path), exist_ok=True)
                print(f'- Writing to {mseed_path}')
                new_st.write(mseed_path, format="MSEED")

            except Exception as e: # If WAV file cannot be read or parsed
                start_time = end_time = None
                parsed_wav = 0
                wav_error = str(e)
                mseed_path = None
                # We log failed WAV files into a separate table with the specific error, as well as below in the wav_files table
                print(f'- Failed with error {e}')
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, str(e))
                )                

            cur.execute('''INSERT INTO wav_files (path, start_time, end_time, associated_sfile, used_in_event_id, parsed_successfully, parsed_time, error)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                        (encoded_path, start_time, end_time, None, None, parsed_wav, datetime.utcnow().isoformat(), wav_error))

    conn.commit()

def main(args):
    if os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        index_wav_files(conn, args.wav_dir, args.mseed_output)
        conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index and convert SEISAN WAV files to MiniSEED.")
    parser.add_argument("--wav_dir", required=True, help="Top-level WAV directory")
    parser.add_argument("--mseed_output", required=True, help="Output directory for MiniSEED files")
    parser.add_argument("--db", required=True, help="SQLite database path")
    args = parser.parse_args()
    main(args)
