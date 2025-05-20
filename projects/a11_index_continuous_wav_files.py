# a11_index_continuous_wav_files.py

import os
import sqlite3
from datetime import datetime
from tqdm import tqdm
from obspy import read

from flovopy.core.mvo import correct_nslc_mvo, fix_trace_mvo
from flovopy.seisanio.core.wavfile import Wavfile
from flovopy.config_projects import get_config

def nearest_sampling_rate(tr):
    return 75.0 if abs(tr.stats.sampling_rate - 75.0) < 5 else 100.0

def index_continuous_wav_files(conn, wav_dir, mseed_output_dir, filename_filter='DSNC_', limit=None):
    """
    Index SEISAN WAV files from continuous database for metadata and conversion.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite database connection.
    wav_dir : str
        Root directory containing WAV files.
    mseed_output_dir : str
        Output directory to store converted MiniSEED files.
    filename_filter : str
        Filter to identify files from continuous database (e.g. 'DSNC_').
    limit : int or None
        Stop after indexing this many files (for testing).
    """
    cur = conn.cursor()
    count = 0

    for root, dirs, files in os.walk(wav_dir):
        dirs.sort()
        files.sort()

        for fname in tqdm(files, desc=f"Indexing WAV files in {root}"):
            if limit is not None and count >= limit:
                return

            if not fname.endswith('.wav'):
                continue
            if filename_filter not in fname:
                continue
            if 'dbspn' in root:
                continue

            full_path = os.path.join(root, fname)
            try:
                encoded_path = full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with encoding error: {full_path}")
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (full_path, 'UnicodeEncodeError')
                )
                continue

            # Skip if already indexed
            cur.execute("SELECT 1 FROM wav_files WHERE path = ?", (encoded_path,))
            if cur.fetchone():
                continue

            try:
                wavfile = Wavfile(full_path)
                wavfile.read()  # Applies trace ID corrections

                # Track all combinations of date, original_id, and fixed_id
                for tr in wavfile.st:
                    cur.execute(
                        '''INSERT INTO trace_id_corrections (date, original_id, corrected_id) VALUES (?, ?, ?)''',
                        (tr.stats.starttime.date.isoformat(), tr.stats.original_id, tr.id)
                    )
                    if tr.stats.endtime.date != tr.stats.starttime.date:
                        cur.execute(
                            '''INSERT INTO trace_id_corrections (date, original_id, corrected_id) VALUES (?, ?, ?)''',
                            (tr.stats.endtime.date.isoformat(), tr.stats.original_id, tr.id)
                        )
                    nearest_sampling_rate(tr)

                # Write MiniSEED file to matching output structure
                rel_path = os.path.relpath(full_path, wav_dir)
                new_fname = rel_path + '.mseed'
                mseed_path = os.path.join(mseed_output_dir, new_fname)
                os.makedirs(os.path.dirname(mseed_path), exist_ok=True)
                wavfile.st.write(mseed_path, format="MSEED")

                cur.execute('''
                    INSERT INTO wav_files (
                        path, filetime, start_time, end_time, associated_sfile, used_in_event_id, 
                        network, parsed_successfully, parsed_time, error, file_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    encoded_path,
                    wavfile.filetime.isoformat(),
                    wavfile.start_time,
                    wavfile.end_time,
                    None,
                    None,
                    wavfile.network,
                    1,
                    datetime.utcnow().isoformat(),
                    None,
                    'continuous'
                ))
                count += 1

            except Exception as e:
                print(f"[ERROR] Failed on {full_path}: {e}")
                cur.execute(
                    '''INSERT INTO wav_processing_errors (path, error_message) VALUES (?, ?)''',
                    (encoded_path, str(e))
                )

    conn.commit()


def main():
    config = get_config()
    dbfile = config['mvo_seisan_index_db']
    wav_dir = os.path.join(config['seisan_top'], 'WAV', 'DSNC_')
    mseed_output_dir = os.path.join(config['seisan_top'], 'miniseed', 'DSNC_')

    conn = sqlite3.connect(dbfile)
    index_continuous_wav_files(conn, wav_dir, mseed_output_dir)
    conn.close()
    print(f"[✓] Finished indexing continuous WAV files.")

if __name__ == "__main__":
    main()