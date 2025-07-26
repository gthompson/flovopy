import os
import sqlite3
#from datetime import datetime

def create_index_schema(conn):
    cur = conn.cursor()

    ######################################################
    ###################### From WAV files ################
    ######################################################

    ''' Raw WAV file registry
        network is ASN or DSN
        associated_sfile is so we can map both ways later, if needed
    '''
    cur.execute('''CREATE TABLE IF NOT EXISTS wav_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        filetime TEXT,
        start_time TEXT,
        end_time TEXT,
        associated_sfile TEXT,
        used_in_event_id TEXT,
        network TEXT,
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
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
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(path) REFERENCES wav_files(path)
    )''')    

    ######################################################
    ###################### From AEF files ################
    ######################################################

    '''
    wav_file_path 
    '''

    # Table for AEF files
    cur.execute('''CREATE TABLE IF NOT EXISTS aef_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        filetime TEXT,
        network TEXT,
        trigger_window REAL,
        average_window REAL,
        json_path TEXT,
        sfile_path TEXT,
        wav_file_path TEXT,
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
        error TEXT,
        FOREIGN KEY(wav_file_path) REFERENCES wav_files(path),
        FOREIGN KEY(sfile_path) REFERENCES sfiles(path)        
    )''')

    # Table for per-trace AEF metrics (1 row per aefrow)
    cur.execute('''CREATE TABLE IF NOT EXISTS aef_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        aef_file_id INTEGER,
        sfile_path TEXT,
        trace_id TEXT,
        amplitude REAL,
        energy REAL,
        maxf REAL,
        ssam_json TEXT,
        FOREIGN KEY(aef_file_id) REFERENCES aef_files(id),
        FOREIGN KEY(sfile_path) REFERENCES sfiles(path)        
    )''')    

    # AEFfile processing errors
    cur.execute('''CREATE TABLE IF NOT EXISTS aef_processing_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT,
        error_message TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )''')        

    ######################################################
    ###################### From S-files ##################
    ######################################################

    cur.execute('''CREATE TABLE IF NOT EXISTS sfiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        filetime TEXT,
        event_id TEXT,
        mainclass TEXT,
        subclass TEXT,
        agency TEXT,
        last_action TEXT,
        action_time TEXT,
        analyst TEXT,
        analyst_delay REAL,
        dsnwavfile TEXT,
        asnwavfile TEXT,
        aeffile TEXT,
        qml_path TEXT,
        json_path TEXT,
        sfilearchive TEXT, 
        parsed_successfully INTEGER DEFAULT 0,
        parsed_time TEXT,
        error TEXT,
        FOREIGN KEY(dsnwavfile) REFERENCES wav_files(path),
        FOREIGN KEY(asnwavfile) REFERENCES wav_files(path)             
    )''')    

    """
    # Mapping between S-files and WAVs
    cur.execute('''CREATE TABLE IF NOT EXISTS sfile_wav_map (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sfile_path TEXT,
        wav_path TEXT,
        FOREIGN KEY(sfile_path) REFERENCES sfiles(path),
        FOREIGN KEY(wav_path) REFERENCES wav_files(path)
    )''')
    """

    # Processing log
    cur.execute('''CREATE TABLE IF NOT EXISTS processing_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sfile_path TEXT,
        wav_path TEXT,
        enhanced_event_saved INTEGER DEFAULT 0,
        enhanced_stream_saved INTEGER DEFAULT 0,
        catalog_inserted INTEGER DEFAULT 0,
        processing_time TEXT,
        error TEXT,
        FOREIGN KEY(sfile_path) REFERENCES sfiles(path),
        FOREIGN KEY(wav_path) REFERENCES wav_files(path)                
    )''')

    # S-file processing errors
    cur.execute('''CREATE TABLE IF NOT EXISTS sfile_processing_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT,
        error_message TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(path) REFERENCES sfiles(path)        
    )''')     

    conn.commit()

def main(DB_FILENAME):
    if not os.path.exists(DB_FILENAME):
        print(f"Creating new database: {DB_FILENAME}")
    else:
        print(f"Using existing database: {DB_FILENAME}")

    conn = sqlite3.connect(DB_FILENAME)
    create_index_schema(conn)
    conn.close()

if __name__ == "__main__":
    import os
    import sys
    import argparse
    from flovopy.config_projects import get_config
    if len(sys.argv) > 1:
        # Use argparse if there are any command-line arguments
        parser = argparse.ArgumentParser(description="Create database to index Seisan data.")
        parser.add_argument("--dbfile", required=True, help="SQLite database path")
        args = parser.parse_args()
        main(args.dbfile)
    else:
        # Fallback to config-based arguments
        config = get_config()
        dbfile = config['mvo_seisan_index_db']
        main(dbfile)
