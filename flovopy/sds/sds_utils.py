"""
sds_utils.py — Shared utilities for SDS tools like:
- write_sds_archive_multiprocessing.py
- audit_multiprocessing.py

This module contains:
- SQLite setup and safe execution
- File list discovery (SDS-aware or not)
- ETA estimation
- Resource logging
- File locking helpers (optional for write tools)
"""

import os
import sqlite3
import pandas as pd
from obspy import UTCDateTime
import re


def setup_database(db_path, mode="write"):
    """
    Sets up the SQLite database schema for either 'write' or 'audit' workflows.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    mode : str
        Either 'write' or 'audit'. Determines which tables are created.
    """
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # Common table: file_log
        if mode == "write":
            c.execute("""
            CREATE TABLE IF NOT EXISTS file_log (
                filepath TEXT PRIMARY KEY,
                status TEXT,
                reason TEXT,
                ntraces_in INTEGER,
                ntraces_out INTEGER,
                cpu_id TEXT,
                timestamp TEXT
            )
            """)
        elif mode == "audit":
            c.execute("""
            CREATE TABLE IF NOT EXISTS file_log (
                filepath TEXT PRIMARY KEY,
                status TEXT,
                reason TEXT,
                timestamp TEXT
            )
            """)

        # Write mode extras
        if mode == "write":
            c.execute("""
            CREATE TABLE IF NOT EXISTS trace_log (
                source_id TEXT,
                fixed_id TEXT,
                trace_id TEXT,
                filepath TEXT,
                station TEXT,
                sampling_rate REAL,
                starttime TEXT,
                endtime TEXT,
                reason TEXT,
                outputfile TEXT,
                status TEXT,
                cpu_id TEXT,
                timestamp TEXT,
                PRIMARY KEY (trace_id, filepath)
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS locks (
                filepath TEXT PRIMARY KEY,
                locked_by TEXT,
                locked_at TEXT
            )
            """)
            c.execute("""
            CREATE TABLE IF NOT EXISTS output_locks (
                filepath TEXT PRIMARY KEY,
                locked_by TEXT,
                locked_at TEXT
            )
            """)

        # Audit mode extras
        elif mode == "audit":
            c.execute("""
            CREATE TABLE IF NOT EXISTS trace_metadata (
                filepath TEXT,
                original_id TEXT,
                starttime TEXT,
                endtime TEXT,
                npts INTEGER,
                sampling_rate REAL,
                fixed_id TEXT,
                PRIMARY KEY (filepath, original_id, starttime)
            )
            """)

        conn.commit()

def populate_file_log(file_list, db_path, mode="write"):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        now = UTCDateTime().isoformat()

        if mode == "audit":
            entries = [(f, 'pending', None, now) for f in file_list]
            c.executemany("""
                INSERT OR IGNORE INTO file_log
                (filepath, status, reason, timestamp)
                VALUES (?, ?, ?, ?)""", entries)
        else:
            entries = [(f, 'pending', None, None, None, None, now) for f in file_list]
            c.executemany("""
                INSERT OR IGNORE INTO file_log
                (filepath, status, reason, ntraces_in, ntraces_out, cpu_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", entries)

        conn.commit()


def get_pending_file_list(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM file_log WHERE status IN ('pending', 'incomplete', 'failed')")
        return [row[0] for row in cursor.fetchall()]

def safe_commit(conn, retries=3, wait=1.0):
    import time
    for attempt in range(retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            print(f"⚠️ Commit error (attempt {attempt+1}): {e}")
            time.sleep(wait)
    print("❌ Commit failed after retries.")

def safe_sqlite_exec(cursor, sql, params=(), retries=3, wait=1.0):
    import time
    for attempt in range(retries):
        try:
            cursor.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            print(f"⚠️ SQLite error (attempt {attempt+1}): {e}")
            time.sleep(wait)
    print("❌ SQLite exec failed after retries.")


def try_lock_output_file(conn, filepath, cpu_id):
    try:
        conn.execute("""
            INSERT INTO output_locks (filepath, locked_by, locked_at)
            VALUES (?, ?, datetime('now'))
        """, (filepath, cpu_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def release_output_file_lock_safe(conn, filepath):
    try:
        conn.execute("DELETE FROM output_locks WHERE filepath = ?", (filepath,))
        conn.commit()
    except Exception as e:
        print(f"{UTCDateTime()}: ⚠️ Failed to release output lock for {filepath}: {e}", flush=True)


def remove_stale_locks(cursor, conn, max_age_minutes=2):
    try:
        cutoff = UTCDateTime() - max_age_minutes * 60
        safe_sqlite_exec(cursor, """
            DELETE FROM locks WHERE locked_at < ?
        """, (cutoff.strftime('%Y-%m-%d %H:%M:%S'),))
        safe_commit(conn)
    except Exception as e:
        print(f"{UTCDateTime()}: ⚠️ Failed to remove stale locks: {e}", flush=True)


def release_input_file_lock(cursor, conn, file_path):
    try:
        safe_sqlite_exec(cursor, "DELETE FROM locks WHERE filepath = ?", (file_path,))
        safe_commit(conn)
    except Exception as e:
        print(f"⚠️ Failed to release input file lock for {file_path}: {e}", flush=True)

def discover_files(sds_root, use_sds=True, filterdict=None, starttime=None, endtime=None):
    """Discover SDS day files without reading MiniSEED or importing SDSobj.

    When use_sds=True (default), require canonical paths relative to sds_root:
    YEAR/NET/STA/CHAN.D/NET.STA.LOC.CHAN.D.YEAR.JDAY.
    When False, allow matching filenames anywhere under root (legacy --nosds).
    Hidden directories are excluded in either mode.

    Time selection uses SDS filename *days*, not observed waveform coverage.
    """
    from pathlib import Path
    from fnmatch import fnmatchcase
    from datetime import datetime, timedelta, timezone

    root = Path(sds_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"SDS root not found: {root}")

    def _matches(value, key):
        if not filterdict or key not in filterdict or filterdict[key] is None:
            return True
        choices = filterdict[key]
        if isinstance(choices, str):
            choices = [choices]
        return any(fnmatchcase(value, str(c)) for c in choices)

    start = UTCDateTime(starttime) if starttime is not None else None
    end = UTCDateTime(endtime) if endtime is not None else None
    if start is not None and end is not None and end < start:
        raise ValueError("endtime precedes starttime")

    result = []
    for directory, dirs, filenames in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for name in sorted(filenames):
            if name.startswith('.'):
                continue
            parsed = parse_sds_filename(name, normalize_empty_loc=False)
            if parsed is None:
                continue
            net, sta, loc, chan, dtype, year, jday = parsed
            if dtype.upper() != 'D':
                continue
            try:
                file_day = datetime.strptime(f"{year}-{jday}", "%Y-%j").replace(tzinfo=timezone.utc)
                if file_day.year != int(year) or file_day.timetuple().tm_yday != int(jday):
                    continue
            except ValueError:
                continue
            path = Path(directory) / name
            if use_sds:
                parts = path.relative_to(root).parts
                if len(parts) != 5 or parts[:4] != (year, net, sta, f"{chan}.D"):
                    continue
            if not all((_matches(net, 'network'), _matches(sta, 'station'),
                        _matches(loc if loc else '--', 'location'),
                        _matches(chan, 'channel'))):
                continue
            day_start = UTCDateTime(file_day)
            # Intersect requested interval with nominal UTC file day.
            if start is not None and day_start + 86400 <= start:
                continue
            if end is not None and day_start > end:
                continue
            result.append(str(path))
    return result


def estimate_eta(start_time, done, total):
    now = UTCDateTime()
    elapsed = now - start_time
    if done == 0:
        return None, None
    rate = elapsed / done
    remaining = total - done
    eta = now + (rate * remaining)
    return rate, eta

def print_progress(cpu_id, done, total, start_time):
    rate, eta = estimate_eta(start_time, done, total)
    if eta:
        print(f"[{cpu_id}] Processed {done}/{total}. ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} UTC")

def remove_empty_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if dirpath == root_dir:
            continue
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"🧹 Removed empty directory: {dirpath}")
            except Exception as e:
                print(f"⚠️ Could not remove {dirpath}: {e}")

def sqlite_to_excel(sqlite_path, excel_path):
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for table in tables:
                if table == 'trace_metadata':
                    df = pd.read_sql("SELECT * FROM trace_metadata ORDER BY fixed_id, starttime", conn)
                else:
                    df = pd.read_sql(f"SELECT * FROM {table}", conn)
                df.to_excel(writer, sheet_name=table, index=False)
        conn.close()
        print(f"✅ Exported SQLite DB to {excel_path}")
    except Exception as e:
        print(f"❌ Excel export failed: {e}")

def parse_sds_dirname(dir_path):
    """
    Parse and extract components from an SDS directory path.
    Expected format: .../YEAR/NET/STA/CHAN.D

    Returns
    -------
    tuple or None
        (year, network, station, channel) if valid format, else None
    """
    parts = os.path.normpath(dir_path).split(os.sep)[-4:]
    if len(parts) != 4:
        return None

    year, net, sta, chanD = parts

    # Validate each component
    if not (year.isdigit() and len(year) == 4):
        return None
    if not re.match(r"^[A-Z0-9]{1,8}$", net):
        return None
    if not re.match(r"^[A-Z0-9]{1,8}$", sta):
        return None
    chan_match = re.match(r"^([A-Z0-9]{3})\.D$", chanD)
    if not chan_match:
        return None

    chan = chan_match.group(1)
    return year, net, sta, chan

def is_valid_sds_dir(dir_path):
    """
    Validate that a directory follows the SDS structure: YEAR/NET/STA/CHAN.D

    Returns
    -------
    bool
        True if valid SDS directory format, else False.
    """
    return parse_sds_dirname(dir_path) is not None


def parse_sds_filename(filename, normalize_empty_loc=True):
    """
    Parses an SDS-style MiniSEED filename and extracts its components.
    Supports location codes of 0–2 characters.
    Normalizes empty location codes ('') to '--' if normalize_empty_loc is True.
    
    Parameters
    ----------
    filename : str
        Filename to parse (can be a full path).
    normalize_empty_loc : bool
        If True, replaces empty location codes ('') with '--'.
        
    Returns
    -------
    tuple or None
        (network, station, location, channel, type, year, day), or None if no match.
    """
    filename = os.fspath(filename)
    if '/' in filename:
        filename = os.path.basename(filename)

    pattern = r"^([A-Z0-9]+)\.([A-Z0-9]+)\.([A-Z0-9\-]{0,2})\.([A-Z0-9]+)\.([A-Z])\.(\d{4})\.(\d{3})$"
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        groups = list(match.groups())
        if normalize_empty_loc and groups[2] == '':
            groups[2] = '--'
        return tuple(groups)
    return None

def is_valid_sds_filename(filename):
    """
    Validate SDS MiniSEED filename using parsing logic.
    Accepts only files matching NET.STA.LOC.CHAN.D.YEAR.DAY format
    with dtype == 'D' (daily MiniSEED).
    """
    parsed = parse_sds_filename(filename)
    if parsed is None:
        return False

    _, _, _, _, dtype, _, _ = parsed
    return dtype.upper() == 'D'

def convert_numpy_types(obj):
    """
    Converts NumPy types to native Python types for JSON or SQLite compatibility.
    """
    import numpy as np
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def setup_merge_tracking_db(db_path):
    """
    Initializes or updates a tracking SQLite DB for merge_sds_archives.

    Parameters
    ----------
    db_path : str
        Path to tracking SQLite DB.
    """
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS merge_log (
                filepath TEXT PRIMARY KEY,
                result TEXT,
                source_hash TEXT,
                dest_hash TEXT,
                ntraces_source INTEGER,
                ntraces_dest INTEGER,
                status TEXT,
                timestamp TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS restored_files (
                filepath TEXT PRIMARY KEY,
                backup_path TEXT,
                restored_at TEXT
            )
        """)
        conn.commit()

def restore_backup_file(dest_file, session_id):
    """
    If a backup file exists with the given session ID, move it back into place.
    """
    import shutil
    backup_path = dest_file + f".bak_{session_id}"
    if os.path.exists(backup_path):
        shutil.move(backup_path, dest_file)
        print(f"♻️ Restored backup: {dest_file} <- {backup_path}")
        return True
    return False


def write_csv(filename, headers, rows):
    """Write rows to a CSV using pandas DataFrame."""
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(filename, index=False)





