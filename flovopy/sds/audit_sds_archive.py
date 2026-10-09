"""
audit_multiprocessing.py — Refactored SDS trace audit tool
- Uses multiprocessing and SQLite for speed and crash safety
- Imports shared logic from sds_utils.py
- Logs CPU temp and memory
- Exports ordered Excel workbook of trace metadata
"""

import os
import sys
import sqlite3
import pandas as pd
import multiprocessing as mp
from obspy import read, UTCDateTime
from flovopy.core.trace_utils import fix_id_wrapper
from flovopy.core.computer_health import log_memory_usage, get_cpu_temperature, start_cpu_logger
from flovopy.sds.sds_utils import (
    setup_database,
    discover_files,
    populate_file_log,
    print_progress,
    sqlite_to_excel,
    parse_sds_filename,
    safe_commit,
    safe_sqlite_exec,
    refresh_audit_file_log
)
import atexit

def _scan_file(task):
    """Worker: read a file only. Never open or write the SQLite database."""
    filepath, speed = task
    try:
        traces = []
        if speed == 1:
            st = read(filepath, headonly=True)
            for tr in st:
                original_id = tr.id
                _, fixed_id = fix_id_wrapper(tr)
                traces.append((filepath, original_id, tr.stats.starttime.isoformat(),
                               tr.stats.endtime.isoformat(), int(tr.stats.npts),
                               float(tr.stats.sampling_rate), fixed_id or original_id))
            if not traces:
                raise ValueError("No traces found in MiniSEED file")
        elif speed == 2:
            net, sta, loc, chan, _, yyyy, jjj = parse_sds_filename(filepath)
            original_id = f"{net}.{sta}.{loc}.{chan}"
            start = UTCDateTime(f"{yyyy}-01-01T00:00:00") + (int(jjj) - 1) * 86400
            end = start + 86400 - 1 / 2000
            traces.append((filepath, original_id, start.isoformat(), end.isoformat(),
                           0, 0.0, original_id))
        else:
            raise ValueError("Invalid speed mode")
        return filepath, traces, None
    except Exception as exc:
        return filepath, [], f"{type(exc).__name__}: {exc}"


def _commit_file(conn, filepath, traces, error):
    """Replace traces and status in ONE atomic transaction; propagate DB errors."""
    with conn:
        conn.execute("DELETE FROM trace_metadata WHERE filepath=?", (filepath,))
        if error is None:
            conn.executemany("""
                INSERT OR REPLACE INTO trace_metadata
                (filepath, original_id, starttime, endtime, npts, sampling_rate, fixed_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, traces)
            conn.execute("""UPDATE file_log SET status='done', reason=NULL,
                           timestamp=datetime('now') WHERE filepath=?""", (filepath,))
        else:
            conn.execute("""UPDATE file_log SET status='failed', reason=?,
                           timestamp=datetime('now') WHERE filepath=?""", (error, filepath))


def check_audit_database(db_path, selected_files=None):
    """Read-only consistency checks. Does not discard completed audit work."""
    with sqlite3.connect(db_path, timeout=60) as conn:
        orphan = conn.execute("""SELECT COUNT(*) FROM trace_metadata AS t
                    LEFT JOIN file_log AS f ON t.filepath=f.filepath
                    WHERE f.filepath IS NULL""").fetchone()[0]
        invalid_done = conn.execute("""SELECT COUNT(*) FROM file_log AS f
                    WHERE f.status='done' AND f.scan_speed=1
                    AND NOT EXISTS (SELECT 1 FROM trace_metadata AS t
                                    WHERE t.filepath=f.filepath AND t.npts>0)""").fetchone()[0]
        partial = conn.execute("""SELECT COUNT(*) FROM file_log AS f
                    WHERE f.status!='done' AND EXISTS
                    (SELECT 1 FROM trace_metadata AS t WHERE t.filepath=f.filepath)""").fetchone()[0]
        # A prior version silently swallowed failed SQL operations. We cannot prove
        # historical 'done' rows are complete, so report rather than auto-trust.
        print(f"DB consistency: orphan traces={orphan}, header-mode done without traces={invalid_done}, "
              f"non-done with traces={partial}")
        if invalid_done:
            print("WARNING: completed header scans without trace rows need rechecking. "
                  "Use --recheck-done to rescan them, or --recheck-all to revalidate all selected files.")
    return dict(orphan_traces=orphan, done_without_traces=invalid_done,
                incomplete_with_traces=partial)


def run_audit(sds_root, db_path, n_processes=6, use_sds=True, filterdict=None,
              starttime=None, endtime=None, speed=1, skip_unmatched=True,
              monitor_temperature=False, recheck_done=False, recheck_all=False):
    setup_database(db_path, mode="audit")
    file_list = discover_files(sds_root, use_sds=use_sds, filterdict=filterdict,
                               starttime=starttime, endtime=endtime)
    if not use_sds and skip_unmatched:
        file_list = [f for f in file_list if 'unmatched' not in f.lower()]
    print(f'Found {len(file_list)} files at {sds_root}')
    refresh_audit_file_log(file_list, db_path, speed=speed)
    check_audit_database(db_path)
    if monitor_temperature and sys.platform.startswith("linux"):
        start_cpu_logger(interval_sec=60, log_path="cpu_temperature_log.csv")
    selected = []
    with sqlite3.connect(db_path, timeout=60) as conn:
        conn.execute('PRAGMA busy_timeout=60000')
        # WAL benefits readers and safe resumption, but does not replace single writer.
        conn.execute('PRAGMA journal_mode=WAL')
        for filepath in file_list:
            row = conn.execute("SELECT status, scan_speed FROM file_log WHERE filepath=?", (filepath,)).fetchone()
            if row is None:
                continue
            status, scan_speed = row
            if recheck_all or (recheck_done and status == 'done') or status != 'done':
                selected.append(filepath)
        print(f"Files to scan: {len(selected)} (already complete: {len(file_list)-len(selected)})")
        if not selected:
            return
        start_time = UTCDateTime()
        done = failed = 0
        tasks = ((filepath, speed) for filepath in selected)
        # One process owns the DB; workers only return read-only scan results.
        with mp.Pool(processes=max(1, int(n_processes)), maxtasksperchild=200) as pool:
            for filepath, traces, error in pool.imap_unordered(_scan_file, tasks, chunksize=1):
                # A failed DB commit raises immediately; the file remains pending or
                # retains its old status and can be safely revisited on the next run.
                _commit_file(conn, filepath, traces, error)
                if error is None:
                    done += 1
                else:
                    failed += 1
                    if failed <= 10:
                        print(f"WARNING: {filepath}: {error}")
                if (done + failed) % 100 == 0:
                    print_progress('writer', done + failed, len(selected), start_time)
                    log_memory_usage(f"Writer after {done + failed} files")
        print(f"Scanned {done + failed}/{len(selected)} files: {done} successful, {failed} failed")
    check_audit_database(db_path)


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
    parser.add_argument("--coverage", default=None, help="Export sample-aware observed coverage intervals to CSV (requires --speed 1)")
    parser.add_argument("--recheck-done", action="store_true", help="Rescan completed files in the selected set")
    parser.add_argument("--recheck-all", action="store_true", help="Rescan all selected files, including previously completed ones")
    parser.add_argument("--cpu-temp", action="store_true",
                        help="Enable CPU temperature checks on Linux only (off by default; ignored on macOS)")
    args = parser.parse_args()
    if args.cpu_temp and not sys.platform.startswith("linux"):
        print("ℹ️ CPU temperature monitoring is unavailable on this platform; disabled.")

    start = UTCDateTime(args.start) if args.start else None
    end = UTCDateTime(args.end) if args.end else None
    print(f"📁 Auditing SDS at: {args.sds_root}")
    print(f"🧠 Using {args.nproc} processes | Speed mode: {args.speed}")
    run_audit(args.sds_root, args.db, n_processes=args.nproc, use_sds=not args.nosds, starttime=start, endtime=end, speed=args.speed, monitor_temperature=args.cpu_temp,
              recheck_done=args.recheck_done, recheck_all=args.recheck_all)
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
    if args.coverage:
        if args.speed != 1:
            parser.error("--coverage requires --speed 1; filename-only dates are not observed waveform coverage")
        from flovopy.sds.coverage import export_coverage
        export_coverage(args.db, args.coverage, sds_root=args.sds_root, canonical_only=True)
    print(f"✅ Done. Results in {args.db} and {args.excel}")

if __name__ == '__main__':
    cli()

# python Developer/flovopy_test/flovopy/sds/audit_multiprocessing.py /raid/newhome/thompsong/work/PROJECTS/MASTERING/seed/DSNC_SDS_from_Silvio_wrong_sampling_rate --db audit2.sqlite --excel audit2.xlsx --nproc 6 --speed 2 --start 1900-01-01 --end 2024-12-31 --nosds
# python audit_mulitprocessing.py /data/SDS_Montserrat --db audit.sqlite --excel audit.xlsx --nproc 6 --start 2020-01-01 --end 2020-12-31
# python audit_multiprocessing.py /data/SDS --db audit.sqlite --excel audit.xlsx --nproc 6 --start 2020-01-01 --end 2020-12-31 --speed 1
