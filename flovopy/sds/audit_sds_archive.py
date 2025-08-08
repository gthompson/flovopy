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
    sqlite_to_excel,
    parse_sds_filename,
    safe_commit,
    safe_sqlite_exec
)
import atexit

def write_trace_metadata(cursor, filepath, orig_id, orig_start, orig_end, npts, orig_sr, fixed_id):
    """
    Write trace metadata to the SQLite database.
    """
    # Fallback: if fixed_id is None/"" use orig_id
    fixed_id = fixed_id or orig_id

    safe_sqlite_exec(cursor, """
        INSERT OR REPLACE INTO trace_metadata
        (filepath, original_id, starttime, endtime, npts, sampling_rate, fixed_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (filepath, orig_id, orig_start, orig_end, npts, orig_sr, fixed_id))

def process_files(file_chunk, db_path, cpu_id, speed):
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    total = len(file_chunk)
    start_time = UTCDateTime()
    commit_interval = 10  # or 100
    commit_counter = 0
    print(f'Processing {total} files')
    for i, filepath in enumerate(file_chunk):
        try:
            safe_sqlite_exec(cursor, "SELECT status FROM file_log WHERE filepath = ?", (filepath,))
            row = cursor.fetchone()
            if not row or row[0] != 'pending':
                continue


            orig_id = orig_start = orig_end = fixed_id = orig_sr = npts = None

            if speed == 1:
                st = read(filepath, headonly=True)
                for tr in st: # should only ever be one trace id in an SDSfile though
                    orig_sr = tr.stats.sampling_rate
                    orig_start = tr.stats.starttime.isoformat()
                    orig_end = tr.stats.endtime.isoformat()
                    npts = tr.stats.npts
                    orig_id, fixed_id = fix_id_wrapper(tr)
                    write_trace_metadata(cursor, filepath, orig_id, orig_start, orig_end, npts, orig_sr, fixed_id)
                    orig_id = orig_start = orig_end = fixed_id = orig_sr = npts = None

            elif speed == 2:
                # Fast mode: get Trace ID from filepath, assuming normal SDS naming
                try:
                    net, sta, loc, chan, typecode, yyyy, jjj = parse_sds_filename(filepath)
                    orig_id = f"{net}.{sta}.{loc}.{chan}"
                    orig_start = UTCDateTime(f"{yyyy}-01-01T00:00:00") + (int(jjj) - 1) * 86400
                    orig_end = orig_start + 86400 - 1/2000
                    # use orig_id as fixed_id to avoid blanks
                    write_trace_metadata(cursor, filepath, orig_id, orig_start.isoformat(), orig_end.isoformat(), 0, 0.0, orig_id)
                except Exception as e:
                    print(f"⚠️ Failed to parse SDS filename {filepath}: {e}")
                    continue
            else:
                raise ValueError("Invalid speed mode. Use 1 for normal, 2 for fast mode.")
            


            safe_sqlite_exec(cursor, """
                UPDATE file_log SET status='done', reason=NULL, timestamp=datetime('now')
                WHERE filepath = ?
            """, (filepath,))
            commit_counter += 1

        except Exception as e:
            safe_sqlite_exec(cursor, """
                UPDATE file_log SET status='failed', reason=?, timestamp=datetime('now')
                WHERE filepath = ?
            """, (str(e), filepath))
            commit_counter += 1

        if commit_counter >= commit_interval:
            safe_commit(conn)
            commit_counter = 0
            print_progress(cpu_id, i + 1, total, start_time)
            log_memory_usage(f"[{cpu_id}] Memory after {i+1} files")
            temp = get_cpu_temperature()
            if temp is not None:
                print(f"[{cpu_id}] CPU Temperature: {temp:.1f}°C")

    # Final commit
    if commit_counter > 0:
        safe_commit(conn)
    conn.close()

def run_audit(sds_root, db_path, n_processes=6, use_sds=True, filterdict=None, starttime=None, endtime=None, speed=1):
    setup_database(db_path, mode="audit")
    file_list = discover_files(sds_root, use_sds=use_sds, filterdict=filterdict, starttime=starttime, endtime=endtime)
    print(f'Found {len(file_list)} files at {sds_root}')
    populate_file_log(file_list, db_path, mode="audit")

    start_cpu_logger(interval_sec=60, log_path="cpu_temperature_log.csv")

    chunk_size = len(file_list) // n_processes + 1
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    args = [(chunk, db_path, str(i), speed) for i, chunk in enumerate(file_chunks)]

    with mp.Pool(processes=n_processes) as pool:
        pool.starmap(process_files, args)

def summarize_audit(db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM file_log WHERE status='done'")
        done = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM file_log WHERE status='failed'")
        failed = c.fetchone()[0]
        print(f"\n📊 Audit Summary:")
        print(f"  ✅ Completed: {done}")
        print(f"  ❌ Failed:    {failed}")

def load_trace_metadata(db_path):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM trace_metadata", conn, parse_dates=["starttime", "endtime"])



def export_grouped_summary(df, output_csv="trace_id_mapping_summary.csv"):
    """
    Export a grouped summary showing how each original_id maps to one or more fixed_id values.

    Parameters
    ----------
    df : pandas.DataFrame
        The audit DataFrame with columns: original_id, fixed_id, sampling_rate
    output_csv : str
        Path to the CSV file to write.
    """
    summary = (
        df.groupby("original_id")
        .agg(
            fixed_ids=("fixed_id", lambda x: sorted(set(x))),
            num_fixed_ids=("fixed_id", lambda x: len(set(x))),
            sampling_rates=("sampling_rate", lambda x: sorted(set(x)))
        )
        .reset_index()
    )

    # Convert list columns to strings
    summary["fixed_ids"] = summary["fixed_ids"].apply(lambda x: ", ".join(x))
    summary["sampling_rates"] = summary["sampling_rates"].apply(lambda x: ", ".join(str(s) for s in x))

    summary.to_csv(output_csv, index=False)
    print(f"📄 Grouped summary saved to {output_csv}. Total unique original_ids: {len(summary)}")


def compute_contiguous_ranges(df, output_csv="trace_segment_ranges.csv", gap_threshold=1.0, rate_tolerance=1.0):
    """
    Compute contiguous time ranges for each trace ID, with autosave protection.
    If fixed_id is blank/NaN, fall back to original_id.
    """
    from datetime import datetime
    import numpy as np

    # Create an effective ID for grouping
    eff = df["fixed_id"].replace("", pd.NA)
    df = df.copy()
    df["effective_id"] = eff.fillna(df["original_id"])

    records = []
    partial_csv = output_csv + ".partial"

    def autosave():
        if records:
            pd.DataFrame(records).to_csv(partial_csv, index=False)
            print(f"🛟 Autosaved contiguous ranges to {partial_csv} (may be partial)")
    atexit.register(autosave)

    # Group by effective_id instead of fixed_id
    for i, (trace_id, group) in enumerate(df.groupby("effective_id")):
        group = group.sort_values(by="starttime")
        segment_start = group.iloc[0]["starttime"]
        segment_end = group.iloc[0]["endtime"]
        segment_sr = group.iloc[0]["sampling_rate"]
        total_npts = group.iloc[0]["npts"]

        for j in range(1, len(group)):
            row = group.iloc[j]
            gap = (pd.to_datetime(row["starttime"]) - pd.to_datetime(segment_end)).total_seconds()
            sr_diff = abs(row["sampling_rate"] - segment_sr)

            if gap > gap_threshold or sr_diff > rate_tolerance:
                records.append({
                    # write out under 'fixed_id' to keep your existing column name
                    "fixed_id": trace_id,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "total_npts": total_npts,
                    "sampling_rate": segment_sr
                })
                segment_start = row["starttime"]
                total_npts = 0
                segment_sr = row["sampling_rate"] if row["sampling_rate"] is not None else 0

            segment_end = max(segment_end, row["endtime"])
            total_npts += row["npts"]

        records.append({
            "fixed_id": trace_id,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "total_npts": total_npts,
            "sampling_rate": segment_sr
        })

        if i % 100 == 0 and records:
            pd.DataFrame(records).to_csv(partial_csv, index=False)
            print(f"💾 Wrote partial ranges after {i} trace IDs")

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"📄 Contiguous ranges saved to {output_csv}. Total rows: {len(out_df)}")

    if os.path.exists(partial_csv):
        os.remove(partial_csv)

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
    parser.add_argument("--speed", type=int, choices=[1, 2], default=1, help="Speed mode: 1 for normal, 2 for fast SDS filename parsing")
    args = parser.parse_args()

    start = UTCDateTime(args.start) if args.start else None
    end = UTCDateTime(args.end) if args.end else None
    print(f"📁 Auditing SDS at: {args.sds_root}")
    print(f"🧠 Using {args.nproc} processes | Speed mode: {args.speed}")
    run_audit(args.sds_root, args.db, n_processes=args.nproc, use_sds=not args.nosds, starttime=start, endtime=end, speed=args.speed)
    summarize_audit(args.db)
    sqlite_to_excel(args.db, args.excel)
    df = load_trace_metadata(args.db)
    export_grouped_summary(df, output_csv=args.excel.replace('.xlsx', '_summary.csv'))
    compute_contiguous_ranges(
        df,
        output_csv=args.excel.replace('.xlsx', '_ranges.csv'),
        gap_threshold=1.0,
        rate_tolerance=1.0
    )
    print(f"✅ Done. Results in {args.db} and {args.excel}")

if __name__ == '__main__':
    cli()

# python Developer/flovopy_test/flovopy/sds/audit_multiprocessing.py /raid/newhome/thompsong/work/PROJECTS/MASTERING/seed/DSNC_SDS_from_Silvio_wrong_sampling_rate --db audit2.sqlite --excel audit2.xlsx --nproc 6 --speed 2 --start 1900-01-01 --end 2024-12-31 --nosds
# python audit_mulitprocessing.py /data/SDS_Montserrat --db audit.sqlite --excel audit.xlsx --nproc 6 --start 2020-01-01 --end 2020-12-31
# python audit_multiprocessing.py /data/SDS --db audit.sqlite --excel audit.xlsx --nproc 6 --start 2020-01-01 --end 2020-12-31 --speed 1
