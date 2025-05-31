import sqlite3
import re
import os
from obspy import UTCDateTime
from datetime import datetime

def extract_datetime_from_filename(filename):
    match_pre2000 = re.search(r'(\d{2})(\d{2})-(\d{2})-(\d{2})(\d{2})', filename)
    if match_pre2000:
        try:
            return datetime.strptime('19' + ''.join(match_pre2000.groups()), '%Y%m%d%H%M')
        except ValueError:
            return datetime.strptime('20' + ''.join(match_pre2000.groups()), '%Y%m%d%H%M')

    match_post2000 = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})', filename)
    if match_post2000:
        return datetime.strptime(''.join(match_post2000.groups()), '%Y%m%d%H%M')

    return None

def create_tables(conn):
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS mseed_processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE,
            status TEXT,
            error_message TEXT,
            response_failures TEXT,
            processing_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS response_failures_summary (
            trace_id TEXT PRIMARY KEY,
            first_seen TEXT,
            last_seen TEXT,
            reason TEXT DEFAULT 'no matching response'
        )
    ''')
    conn.commit()

def create_gap_fill_table(conn):
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS gap_fills_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            date TEXT NOT NULL,
            num_event_files INTEGER DEFAULT 0,
            num_rbuffer_files INTEGER DEFAULT 0,
            total_filled_samples INTEGER DEFAULT 0,
            UNIQUE(trace_id, date)
        )
    ''')
    conn.commit()

def log_response_failure_summary(conn, trace_id, obs_time, reason="no matching response"):
    cur = conn.cursor()
    cur.execute('SELECT first_seen, last_seen FROM response_failures_summary WHERE trace_id = ?', (trace_id,))
    row = cur.fetchone()

    if row:
        first_seen, last_seen = row
        first_seen_dt = datetime.fromisoformat(first_seen)
        last_seen_dt = datetime.fromisoformat(last_seen)

        first_seen = min(first_seen_dt, obs_time).isoformat()
        last_seen = max(last_seen_dt, obs_time).isoformat()

        cur.execute('''
            UPDATE response_failures_summary
            SET first_seen = ?, last_seen = ?, reason = ?
            WHERE trace_id = ?
        ''', (first_seen, last_seen, reason, trace_id))
    else:
        cur.execute('''
            INSERT INTO response_failures_summary (trace_id, first_seen, last_seen, reason)
            VALUES (?, ?, ?, ?)
        ''', (trace_id, obs_time, obs_time, reason))
    conn.commit()

def log_gap_fill_summary(conn, trace_id, date_str, samples):
    cur = conn.cursor()
    cur.execute('''
        SELECT num_event_files, total_filled_samples FROM gap_fills_summary
        WHERE trace_id = ? AND date = ?
    ''', (trace_id, date_str))
    row = cur.fetchone()

    if row:
        event_files, total_samples = row
        cur.execute('''
            UPDATE gap_fills_summary
            SET num_event_files = ?, total_filled_samples = ?
            WHERE trace_id = ? AND date = ?
        ''', (event_files + 1, total_samples + samples, trace_id, date_str))
    else:
        cur.execute('''
            INSERT INTO gap_fills_summary (trace_id, date, num_event_files, total_filled_samples)
            VALUES (?, ?, ?, ?)
        ''', (trace_id, date_str, 1, samples))
    conn.commit()

def clear_old_tables(conn):
    cur = conn.cursor()
    #cur.execute('DELETE FROM mseed_processing_log')
    #cur.execute('DELETE FROM response_failures_summary')
    #cur.execute('DELETE FROM gap_fills_summary')
    delete_if_table_exists(conn, 'mseed_processing_log')
    delete_if_table_exists(conn, 'response_failures_summary')
    delete_if_table_exists(conn, 'gap_fills_summary')
    conn.commit()

def delete_if_table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table_name,))
    if cur.fetchone():
        cur.execute(f'DELETE FROM {table_name}')
        conn.commit()

def parse_log_file(log_path, conn):
    print(f"[INFO] Parsing log: {log_path}")
    #clear_old_tables(conn)
    #create_tables(conn)
    #create_gap_fill_table(conn)

    cur = conn.cursor()
    current_file = None
    current_start = None
    response_failures = set()
    error_message = None

    total_files = 0
    datetime_failures = 0
    error_files = 0
    partial_files = 0
    gap_filled = 0
    interpolated_gaps = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            match_file = re.match(r'\[\d+/\d+\] Reading: (.+\.mseed)', line)
            if match_file:
                total_files += 1
                if current_file:
                    cur.execute('''
                        INSERT OR REPLACE INTO mseed_processing_log
                        (filepath, status, error_message, response_failures)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        current_file,
                        "error" if error_message else "success" if not response_failures else "partial",
                        error_message,
                        ";".join(sorted(response_failures)) if response_failures else None
                    ))

                    for tid in response_failures:
                        if current_start:
                            log_response_failure_summary(conn, tid, current_start)

                    if error_message:
                        error_files += 1
                    elif response_failures:
                        partial_files += 1

                current_file = match_file.group(1)
                current_start = extract_datetime_from_filename(current_file)
                if current_start is None:
                    print(f"[WARN] Failed to extract datetime from: {current_file}")
                    datetime_failures += 1
                    continue
                response_failures = set()
                error_message = None
                continue

            match_resp = re.search(r'remove_response failed for ([\w\.]+)', line)
            if match_resp:
                response_failures.add(match_resp.group(1))
                continue

            match_gap = re.search(r'filling gaps for ([\w\.]+) \| (\d{4}-\d{2}-\d{2})T.*?(\d+) samples', line)
            if match_gap:
                trace_id = match_gap.group(1)
                date_str = match_gap.group(2)
                samples = int(match_gap.group(3))
                log_gap_fill_summary(conn, trace_id, date_str, samples)
                gap_filled += 1

            if "Interpolating small gap" in line:
                interpolated_gaps += 1

            if '[ERROR]' in line and not error_message:
                error_message = line

        if current_file:
            cur.execute('''
                INSERT OR REPLACE INTO mseed_processing_log
                (filepath, status, error_message, response_failures)
                VALUES (?, ?, ?, ?)
            ''', (
                current_file,
                "error" if error_message else "success" if not response_failures else "partial",
                error_message,
                ";".join(sorted(response_failures)) if response_failures else None
            ))

            for tid in response_failures:
                if current_start:
                    log_response_failure_summary(conn, tid, current_start)

            if error_message:
                error_files += 1
            elif response_failures:
                partial_files += 1

    conn.commit()
    print("[INFO] Done parsing log and updating tables.")
    print(f"[SUMMARY] Total files processed:         {total_files}")
    print(f"[SUMMARY] Datetime extraction failures:  {datetime_failures}")
    print(f"[SUMMARY] Files with response failures:  {partial_files}")
    print(f"[SUMMARY] Files with errors:             {error_files}")
    print(f"[SUMMARY] Files with gaps filled:        {gap_filled}")
    print(f"[SUMMARY] Interpolated small gaps:       {interpolated_gaps}")

def main(logfile, dbfile):
    conn = sqlite3.connect(dbfile)
    parse_log_file(logfile, conn)
    conn.close()

if __name__ == "__main__":
    import sys
    import sqlite3
    from flovopy.config_projects import get_config

    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="Parse MiniSEED processing log into index database.")
        parser.add_argument("logfile", help="Path to the stdout/stderr log file")
        parser.add_argument("dbfile", help="SQLite index database to update")
        args = parser.parse_args()
        main(args.logfile, args.dbfile)
    else:
        config = get_config()
        logfile = config['mseed_processing_log']
        dbfile = config['mvo_seisan_index_db']
        main(logfile, dbfile)

    # python parse_mseed_log.py miniseed2cleanedVEL.log /home/thompsong/public_html/index_mvoe4.sqlite
