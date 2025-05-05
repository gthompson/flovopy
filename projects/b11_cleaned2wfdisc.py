# b11_cleanedwfdisc.py — Create index of cleaned MiniSEED waveform files

import sqlite3
import glob
import os
from obspy import read
from obspy.core.stream import Stream
from typing import Dict


def is_file_already_processed(conn: sqlite3.Connection, mseed_base: str) -> bool:
    """
    Check if a MiniSEED file (by basename) is already in the database.
    """
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM mseed_file_status WHERE dfile = ?", (mseed_base,))
    return cur.fetchone() is not None


def insert_miniseed_file(conn: sqlite3.Connection, cleaned_dict: Dict, stream: Stream, commit: bool = True) -> None:
    """
    Insert metadata and waveform entries for a cleaned MiniSEED file.
    """
    cur = conn.cursor()
    completed = ""

    try:
        cur.execute('''INSERT OR IGNORE INTO mseed_file_status 
                       (time, endtime, dir, dfile, network, format)
                       VALUES (?, ?, ?, ?, ?, ?)''', (
            cleaned_dict['time'],
            cleaned_dict['endtime'],
            cleaned_dict['dir'],
            cleaned_dict['dfile'],
            cleaned_dict['network'],
            cleaned_dict['format']
        ))
        completed = "mseed_file_status"

        for tracenum, tr in enumerate(stream):
            s = tr.stats
            cur.execute('''INSERT OR IGNORE INTO wfdisc 
                           (trace_id, time, endtime, dfile, tracenum, nsamp, samprate, calib, units)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                tr.id,
                s.starttime.isoformat(),
                s.endtime.isoformat(),
                cleaned_dict['dfile'],
                tracenum,
                s.npts,
                s.sampling_rate,
                getattr(s, 'calib', 1.0),
                getattr(s, 'units', 'counts')
            ))
        completed = "wfdisc"

    except Exception as e:
        print(f"[ERROR] Failed during {completed} insertion: {e}")

    finally:
        if commit:
            conn.commit()
        print(f"[INFO] Inserted: {cleaned_dict['dfile']}")


def main():
    from flovopy.config_projects import get_config
    from flovopy.core.enhanced import EnhancedEvent 
    from db_backup import backup_db
    config = get_config()
    dbfile = config['mvo_seiscomp_db']
    if not backup_db(dbfile, __file__):
        exit()      

    conn = sqlite3.connect(dbfile)
    conn.execute("PRAGMA foreign_keys = ON;")

    miniseed_files = sorted(glob.glob(os.path.join(config['miniseed_top'], "MVOE_", "*", "*", "*.cleaned")))
    total = len(miniseed_files)
    succeeded, failed, skipped = 0, 0, 0

    for i, mseed in enumerate(miniseed_files):
        dfile = os.path.basename(mseed)

        if is_file_already_processed(conn, dfile):
            skipped += 1
            continue

        try:
            st = read(mseed)
            if len(st) == 0:
                raise ValueError("No traces in stream")
            stations = [tr.stats.station for tr in st]
            network = "DSN" if all(sta.startswith("MB") for sta in stations) else "ASN"

            cleaned_dict = {
                'format': 'MSEED',
                'dir': os.path.dirname(mseed),
                'dfile': dfile,
                'network': network,
                'time': st[0].stats.starttime.isoformat(),
                'endtime': st[0].stats.endtime.isoformat()
            }

            insert_miniseed_file(conn, cleaned_dict, st, commit=False)
            succeeded += 1

        except Exception as e:
            print(f"[WARN] Failed to process {dfile}: {e}")
            failed += 1

        if i % 100 == 0 and i > 0:
            print(f"[Progress] {i}/{total} processed — Success: {succeeded}, Failed: {failed}, Skipped: {skipped}")
            conn.commit()

    conn.commit()
    conn.close()
    print(f"[DONE] Total: {total}, Succeeded: {succeeded}, Failed: {failed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()

