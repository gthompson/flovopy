# 01_index_wav_files.py

import os
import sqlite3
from datetime import datetime
from obspy import read
from obspy.core.trace import Trace
from flovopy.core.mvo import correct_nslc_mvo, fix_trace_mvo
from flovopy.seisanio.core.wavfile import Wavfile
#from flovopy.seisanio.utils.helpers import filetime2spath, legacy_or_not
from obspy.core import Stream
from tqdm import tqdm

def index_wav_files(conn, wav_dir, mseed_output_dir, 
                    filename_filter=None, limit=None):
    """
    Index SEISAN WAV files for metadata and conversion.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite database connection.
    wav_dir : str
        Root directory containing WAV files.
    mseed_output_dir : str
        Output directory to store converted MiniSEED files.
    filename_filter : callable or str or None
        A function or string used to filter filenames. E.g., lambda f: 'MB' in f
    limit : int or None
        Stop after indexing this many files (useful for testing).
    """
    if limit:
        print(f'Will only process {limit} WAV files')
    cur = conn.cursor()
    count = 0

    walker = os.walk(wav_dir)

    for root, dirs, files in walker:
        dirs.sort()
        files.sort()

        for fname in tqdm(files, desc=f"Indexing WAV files in {root}"):
            if limit is not None and count >= limit:
                return
            if isinstance(filename_filter, str):
                if filename_filter not in fname:
                    continue
            elif callable(filename_filter):
                if not filename_filter(fname):
                    continue

            if fname.endswith('.png'):
                continue
            if 'dbspn' in root:
                continue
            full_path = os.path.join(root, fname)

            try:
                encoded_path = full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with invalid encoding: {full_path}")
                # We log failed WAV files into a separate table with the specific error
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, 'UnicodeEncodeError')
                )              
                continue

            # Skip if already indexed
            cur.execute("SELECT 1 FROM wav_files WHERE path = ?", (encoded_path,))
            if cur.fetchone():
                print(f'Skipping - already processed')
                continue

            # Try reading the WAV file
            try:
                print('- Reading')
                #st = read(full_path)
                wavfileobj = Wavfile(full_path)
                wavfileobj.read() # this applies trace ID corrections
                parsed_wav = 1
                wav_error = None

                # Fix trace IDs and write to MiniSEED in a parallel structure
                print('- Fixing ids')
                for tr in wavfileobj.st:

                    # track all combinations of date, original_id, and fixed_id
                    cur.execute(
                        '''INSERT INTO trace_id_corrections (date, original_id, corrected_id) VALUES (?, ?, ?)''',
                        (tr.stats.starttime.date.isoformat(), tr.stats.original_id, tr.id)
                    )
                    if tr.stats.endtime.date != tr.stats.starttime.date:
                        cur.execute(
                            '''INSERT INTO trace_id_corrections (date, original_id, corrected_id) VALUES (?, ?, ?)''',
                            (tr.stats.endtime.date.isoformat(), tr.stats.original_id, tr.id)
                        )                        

                rel_path = os.path.relpath(full_path, wav_dir)
                new_fname = rel_path + '.mseed'
                mseed_path = os.path.join(mseed_output_dir, new_fname)
                os.makedirs(os.path.dirname(mseed_path), exist_ok=True)
                print(f'- Writing to {mseed_path}')
                wavfileobj.st.write(mseed_path, format="MSEED")
                cur.execute('''INSERT INTO wav_files (path, filetime, start_time, end_time, associated_sfile, used_in_event_id, network, parsed_successfully, parsed_time, error)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (encoded_path, wavfileobj.filetime.isoformat(), wavfileobj.start_time, wavfileobj.end_time, None, None, wavfileobj.network, parsed_wav, datetime.utcnow().isoformat(), wav_error))                
                count += 1

            except Exception as e: # If WAV file cannot be read or parsed
                # We log failed WAV files into a separate table with the specific error, as well as below in the wav_files table
                print(f'- Failed with error {e}')
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, str(e))
                )                

    conn.commit()

def main(args):
    if os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        index_wav_files(conn, args.wav_dir, args.mseed_output, limit=args.limit)
        conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index and convert SEISAN WAV files to MiniSEED.")
    parser.add_argument("--wav_dir", required=True, help="Top-level WAV directory")
    parser.add_argument("--mseed_output", required=True, help="Output directory for MiniSEED files")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--limit", type=int, default=None, help="stop after this number of files")
    args = parser.parse_args()
    print(args)
    #main(args)
