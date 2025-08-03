"""
audit_multiprocessing.py — Refactored SDS trace audit tool
- Uses multiprocessing and SQLite for speed and crash safety
- Imports shared logic from sds_utils.py
- Logs CPU temp and memory
- Exports ordered Excel workbook of trace metadata
"""

import os
import sqlite3
import pandas as pd
import multiprocessing as mp
from obspy import read, UTCDateTime
from flovopy.core.trace_utils import fix_id_wrapper
from flovopy.core.computer_health import log_memory_usage, get_cpu_temperature, start_cpu_logger
from sds_utils import (
    setup_database,
    discover_files,
    populate_file_log,
    print_progress,
    sqlite_to_excel
)

def process_files(file_chunk, db_path, cpu_id):
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    total = len(file_chunk)
    start_time = UTCDateTime()
    for i, filepath in enumerate(file_chunk):
        try:
            cursor.execute("SELECT status FROM file_log WHERE filepath = ?", (filepath,))
            if cursor.fetchone()[0] != 'pending':
                continue

            st = read(filepath, headonly=True)
            for tr in st:
                orig_sr = tr.stats.sampling_rate
                orig_start = tr.stats.starttime.isoformat()
                orig_end = tr.stats.endtime.isoformat()
                npts = tr.stats.npts

                orig_id, fixed_id = fix_id_wrapper(tr)

                cursor.execute("""
                    INSERT OR REPLACE INTO trace_metadata
                    (filepath, original_id, starttime, endtime, npts, sampling_rate, fixed_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (filepath, orig_id, orig_start, orig_end, npts, orig_sr, fixed_id))

            cursor.execute("""
                UPDATE file_log SET status='done', reason=NULL, timestamp=datetime('now')
                WHERE filepath = ?
            """, (filepath,))
            conn.commit()

        except Exception as e:
            cursor.execute("""
                UPDATE file_log SET status='failed', reason=?, timestamp=datetime('now')
                WHERE filepath = ?
            """, (str(e), filepath))
            conn.commit()

        if i % 10 == 0:
            print_progress(cpu_id, i + 1, total, start_time)
            log_memory_usage(f"[{cpu_id}] Memory after {i+1} files")
            temp = get_cpu_temperature()
            if temp:
                print(f"[{cpu_id}] CPU Temperature: {temp:.1f}°C")

    conn.close()

def run_audit(sds_root, db_path, n_processes=6, use_sds=True, filterdict=None, starttime=None, endtime=None):
    setup_database(db_path, mode="audit")
    file_list = discover_files(sds_root, use_sds=use_sds, filterdict=filterdict, starttime=starttime, endtime=endtime)
    populate_file_log(file_list, db_path)

    start_cpu_logger(interval_sec=60, log_path="cpu_temperature_log.csv")

    chunk_size = len(file_list) // n_processes + 1
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    args = [(chunk, db_path, str(i)) for i, chunk in enumerate(file_chunks)]

    with mp.Pool(processes=n_processes) as pool:
        pool.starmap(process_files, args)

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Audit SDS trace IDs using SQLite + multiprocessing")
    parser.add_argument("sds_root", help="Path to SDS archive")
    parser.add_argument("--db", default="audit.sqlite", help="Path to SQLite DB")
    parser.add_argument("--excel", default="audit.xlsx", help="Path to final Excel output")
    parser.add_argument("--nproc", type=int, default=6, help="Number of processes")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--nosds", action="store_true", help="Disable SDS parsing (use raw walk)")
    args = parser.parse_args()

    start = UTCDateTime(args.start) if args.start else None
    end = UTCDateTime(args.end) if args.end else None

    run_audit(args.sds_root, args.db, n_processes=args.nproc, use_sds=not args.nosds, starttime=start, endtime=end)
    sqlite_to_excel(args.db, args.excel)
    print(f"✅ Done. Results in {args.db} and {args.excel}")

if __name__ == '__main__':
    cli()


# python audit_mulitprocessing.py /data/SDS_Montserrat --db audit.sqlite --excel audit.xlsx --nproc 6 --start 2020-01-01 --end 2020-12-31