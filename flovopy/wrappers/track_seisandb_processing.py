# seisan_processing_db.py
'''
sfiles: tracks the S-files discovered, whether they were parsed, and their resulting event IDs if known.

wav_files: stores each WAV file found in the WAV directory, its timespan, and any associated S-file or event.

sfile_wav_map: many-to-many mapping between S-files and WAVs.

processing_log: tracks attempts to turn the S-file and WAV combo into an EnhancedEvent, EnhancedStream, and insert into the main catalog.
'''
import sqlite3

def create_processing_schema(conn):
    cur = conn.cursor()

    # Raw S-file registry
    cur.execute('''CREATE TABLE IF NOT EXISTS sfiles (
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

    conn.commit()


def insert_stations_from_inventory(conn, inventory):
    cur = conn.cursor()
    for network in inventory:
        for station in network.stations:
            cur.execute('''INSERT OR REPLACE INTO stations (station_code, network_code, latitude, longitude, elevation)
                           VALUES (?, ?, ?, ?, ?)''',
                        (station.code, network.code, station.latitude, station.longitude, station.elevation))
    conn.commit()

def index_sfiles_and_wavs(conn, seisan_rea_dir, seisan_wav_dir):
    cur = conn.cursor()
    for root, dirs, files in os.walk(seisan_rea_dir):
        for file in files:
            if file.endswith('S'):  # basic S-file check
                sfile_path = os.path.join(root, file)
                try:
                    s = parse_sfile(sfile_path)
                    event_id = s.public_id if hasattr(s, 'public_id') else sfile_path  # fallback
                    wav_filename = s.wav_filename if hasattr(s, 'wav_filename') else None
                    if wav_filename:
                        for wav_root, _, wav_files in os.walk(seisan_wav_dir):
                            for wf in wav_files:
                                if wav_filename in wf:
                                    wav_path = os.path.join(wav_root, wf)
                                    cur.execute('''INSERT INTO sfile_wav_tracking (sfile_path, wav_file, event_id, processed)
                                                   VALUES (?, ?, ?, ?)''',
                                                (sfile_path, wav_path, event_id, 0))
                except Exception as e:
                    cur.execute('''INSERT INTO sfile_wav_tracking (sfile_path, wav_file, event_id, processed, error_msg)
                                   VALUES (?, NULL, NULL, 0, ?)''', (sfile_path, str(e)))
    conn.commit()