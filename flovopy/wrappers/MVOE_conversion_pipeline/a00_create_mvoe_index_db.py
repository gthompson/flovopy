import os
import sqlite3
from datetime import datetime

def create_index_schema(conn):
    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS sfiles_bgs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        event_id TEXT,
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
        error TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS sfiles_mvo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        event_id TEXT,
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
        error TEXT
    )''')


    # Raw WAV file registry
    cur.execute('''CREATE TABLE IF NOT EXISTS wav_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        start_time TEXT,
        end_time TEXT,
        associated_sfile TEXT,
        used_in_event_id TEXT,
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
        error TEXT
    )''')

    # Mapping between S-files and WAVs
    cur.execute('''CREATE TABLE IF NOT EXISTS sfile_wav_map (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sfile_path TEXT,
        wav_path TEXT,
        FOREIGN KEY(sfile_path) REFERENCES sfiles(path),
        FOREIGN KEY(wav_path) REFERENCES wav_files(path)
    )''')

    # Table for AEF files
    cur.execute('''CREATE TABLE IF NOT EXISTS aef_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        trigger_window REAL,
        average_window REAL,
        json_path TEXT,
        sfile_bgs_path TEXT,
        sfile_mvo_path TEXT,
        wav_file_path TEXT,
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
        error TEXT
    )''')

    # Table for per-trace AEF metrics
    cur.execute('''CREATE TABLE IF NOT EXISTS aef_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aef_file_id INTEGER,
        trace_id TEXT,
        station TEXT,
        channel TEXT,
        amplitude REAL,
        energy REAL,
        maxf REAL,
        ssam_json TEXT,
        FOREIGN KEY(aef_file_id) REFERENCES aef_files(id)
    )''')

    # Processing log
    cur.execute('''CREATE TABLE IF NOT EXISTS processing_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sfile_path TEXT,
        wav_path TEXT,
        enhanced_event_saved INTEGER DEFAULT 0,
        enhanced_stream_saved INTEGER DEFAULT 0,
        catalog_inserted INTEGER DEFAULT 0,
        processing_time TEXT,
        error TEXT
    )''')

    # Trace ID corrections by day
    cur.execute('''CREATE TABLE IF NOT EXISTS trace_id_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        original_id TEXT,
        corrected_id TEXT
    )''')

    # WAV processing errors
    cur.execute('''CREATE TABLE IF NOT EXISTS wav_processing_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT,
        error_message TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )''')


    conn.commit()

def main(args):
    DB_FILENAME = args.db
    if not os.path.exists(DB_FILENAME):
        print(f"Creating new database: {DB_FILENAME}")
    else:
        print(f"Using existing database: {DB_FILENAME}")

    conn = sqlite3.connect(DB_FILENAME)
    create_index_schema(conn)
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create DB.")
    parser.add_argument("--db", required=True, help="SQLite database path")
    args = parser.parse_args()
    main(args)