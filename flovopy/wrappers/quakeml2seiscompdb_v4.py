# seiscomp_like_db.py

import sqlite3
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Origin, Magnitude, Pick, Arrival, Amplitude, StationMagnitude

def create_schema(conn):
    cur = conn.cursor()

    # Basic Event structure
    cur.execute('''CREATE TABLE IF NOT EXISTS events (
        public_id TEXT PRIMARY KEY,
        event_type TEXT,
        creation_time TEXT,
        preferred_origin_id TEXT,
        preferred_magnitude_id TEXT,
        region TEXT
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS origins (
        origin_id TEXT PRIMARY KEY,
        event_id TEXT,
        time TEXT,
        latitude REAL,
        longitude REAL,
        depth REAL,
        evaluation_mode TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS magnitudes (
        mag_id TEXT PRIMARY KEY,
        event_id TEXT,
        magnitude REAL,
        mag_type TEXT,
        origin_id TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(origin_id) REFERENCES origins(origin_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS picks (
        pick_id TEXT PRIMARY KEY,
        event_id TEXT,
        time TEXT,
        waveform_id TEXT,
        phase_hint TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS arrivals (
        arrival_id TEXT PRIMARY KEY,
        pick_id TEXT,
        origin_id TEXT,
        time_residual REAL,
        azimuth REAL,
        distance REAL,
        FOREIGN KEY(pick_id) REFERENCES picks(pick_id),
        FOREIGN KEY(origin_id) REFERENCES origins(origin_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS amplitudes (
        amplitude_id TEXT PRIMARY KEY,
        event_id TEXT,
        generic_amplitude REAL,
        unit TEXT,
        type TEXT,
        period REAL,
        snr REAL,
        waveform_id TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS station_magnitudes (
        smag_id TEXT PRIMARY KEY,
        event_id TEXT,
        station_code TEXT,
        mag REAL,
        mag_type TEXT,
        amplitude_id TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(amplitude_id) REFERENCES amplitudes(amplitude_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS stations (
        station_code TEXT PRIMARY KEY,
        network_code TEXT,
        latitude REAL,
        longitude REAL,
        elevation REAL
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS enhanced_metadata (
        event_id TEXT PRIMARY KEY,
        energy_mag REAL,
        event_class TEXT,
        location_quality TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS event_waveforms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        network TEXT,
        filepath TEXT,
        format TEXT DEFAULT 'MSEED',
        description TEXT,
        source_system TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS trace_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        trace_id TEXT,
        station TEXT,
        channel TEXT,
        metric_name TEXT,
        metric_value REAL,
        units TEXT,
        source TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS asl_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_id TEXT,
        q_factor REAL,
        wavespeed REAL,
        dsam_metric TEXT,
        window_seconds REAL,
        prefilter_low REAL,
        prefilter_high REAL,
        stations_used TEXT,
        reduced_displacement REAL,
        reduced_energy REAL,
        FOREIGN KEY(origin_id) REFERENCES origins(origin_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS trace_energy_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        trace_id TEXT,
        station TEXT,
        bsam REAL,
        energy REAL,
        site_correction REAL,
        source TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS meta_events (
        meta_event_id TEXT PRIMARY KEY,
        name TEXT,
        start_time TEXT,
        end_time TEXT,
        description TEXT,
        parent_meta_event_id TEXT,
        FOREIGN KEY(parent_meta_event_id) REFERENCES meta_events(meta_event_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS event_meta_mapping (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meta_event_id TEXT,
        event_id TEXT,
        FOREIGN KEY(meta_event_id) REFERENCES meta_events(meta_event_id),
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS event_classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        event_type TEXT,
        mainclass TEXT,
        subclass TEXT,
        author TEXT,
        timestamp TEXT,
        source TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS ssam_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        trace_id TEXT,
        station TEXT,
        channel TEXT,
        start_time TEXT,
        end_time TEXT,
        sampling_interval REAL,
        frequencies BLOB,
        amplitudes BLOB,
        spectrogram_file TEXT,
        source TEXT,
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    conn.commit()


def insert_event_data(conn, event: Event, waveform_files=None, trace_metrics=None):
    cur = conn.cursor()
    eid = event.resource_id.id
    cur.execute('INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?)', (
        eid,
        event.event_type if event.event_type else None,
        str(event.creation_info.creation_time) if event.creation_info else None,
        event.preferred_origin_id.id if event.preferred_origin_id else None,
        event.preferred_magnitude_id.id if event.preferred_magnitude_id else None,
        event.region or None
    ))

    for o in event.origins:
        cur.execute('INSERT OR IGNORE INTO origins VALUES (?, ?, ?, ?, ?, ?, ?)', (
            o.resource_id.id,
            eid,
            str(o.time),
            o.latitude,
            o.longitude,
            o.depth,
            o.evaluation_mode or None
        ))

    for m in event.magnitudes:
        cur.execute('INSERT OR IGNORE INTO magnitudes VALUES (?, ?, ?, ?, ?)', (
            m.resource_id.id,
            eid,
            m.mag,
            m.magnitude_type,
            m.origin_id.id if m.origin_id else None
        ))

    for p in event.picks:
        cur.execute('INSERT OR IGNORE INTO picks VALUES (?, ?, ?, ?, ?)', (
            p.resource_id.id,
            eid,
            str(p.time),
            p.waveform_id.id if p.waveform_id else None,
            p.phase_hint or None
        ))

    for a in event.arrivals:
        cur.execute('INSERT OR IGNORE INTO arrivals VALUES (?, ?, ?, ?, ?, ?)', (
            a.resource_id.id,
            a.pick_id.id,
            a.origin_id.id,
            a.time_residual,
            a.azimuth,
            a.distance
        ))

    for amp in event.amplitudes:
        cur.execute('INSERT OR IGNORE INTO amplitudes VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (
            amp.resource_id.id,
            eid,
            amp.generic_amplitude,
            amp.unit,
            amp.type,
            amp.period,
            amp.snr,
            amp.waveform_id.id if amp.waveform_id else None
        ))

    for sm in event.station_magnitudes:
        cur.execute('INSERT OR IGNORE INTO station_magnitudes VALUES (?, ?, ?, ?, ?, ?)', (
            sm.resource_id.id,
            eid,
            sm.station_magnitude_station,
            sm.mag,
            sm.magnitude_type,
            sm.amplitude_id.id if sm.amplitude_id else None
        ))

    cur.execute('INSERT OR IGNORE INTO enhanced_metadata VALUES (?, ?, ?, ?)', (
        eid,
        getattr(event, 'energy_mag', None),
        getattr(event, 'event_class', None),
        getattr(event, 'location_quality', None)
    ))

    if waveform_files:
        for wf in waveform_files:
            cur.execute('''INSERT INTO event_waveforms (event_id, network, filepath, format, description, source_system)
                           VALUES (?, ?, ?, ?, ?, ?)''', (
                eid,
                wf.get('network'),
                wf.get('filepath'),
                wf.get('format', 'MSEED'),
                wf.get('description'),
                wf.get('source_system')
            ))

    if trace_metrics:
        for tm in trace_metrics:
            cur.execute('''INSERT INTO trace_metrics (event_id, trace_id, station, channel, metric_name, metric_value, units, from)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                eid,
                tm.get('trace_id'),
                tm.get('station'),
                tm.get('channel'),
                tm.get('metric_name'),
                tm.get('metric_value'),
                tm.get('units'),
                tm.get('from')
            ))

    conn.commit()

def insert_catalog(conn, catalog: Catalog):
    for event in catalog:
        insert_event_data(conn, event)


# seisan_processing_db.py
'''
sfiles: tracks the S-files discovered, whether they were parsed, and their resulting event IDs if known.

wav_files: stores each WAV file found in the WAV directory, its timespan, and any associated S-file or event.

sfile_wav_map: many-to-many mapping between S-files and WAVs.

processing_log: tracks attempts to turn the S-file and WAV combo into an EnhancedEvent, EnhancedStream, and insert into the main catalog.
'''
import sqlite3
import os
from obspy import Catalog, UTCDateTime, read_inventory
from obspy.core.event import Event, Origin, Magnitude, Pick, Arrival, Amplitude, StationMagnitude
from flovopy.seisanio.core.sfile import Sfile, get_sfile_list
from datetime import datetime

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

# seiscomp_like_db.py

# seiscomp_like_db.py

import sqlite3
import os
from datetime import datetime
from glob import glob
from obspy import Catalog, UTCDateTime, read_inventory
from obspy.core.event import Event, Origin, Magnitude, Pick, Arrival, Amplitude, StationMagnitude
from flovopy.seisanio.core.sfile import Sfile, get_sfile_list
from obspy import read


def index_sfiles_and_wavs(
    conn,
    seisan_rea_dir,
    seisan_wav_dir,
    dbname="MVOE_",
    startdate=None,
    enddate=None
):
    """
    Index and cross-map Seisan S-files and WAV files in the given directories.

    Parameters:
    - conn: SQLite database connection
    - seisan_rea_dir: Path to SEISAN REA directory
    - seisan_wav_dir: Path to SEISAN WAV directory
    - dbname: SEISAN database name (e.g., "MVOE_")
    - startdate: UTCDateTime or datetime (inclusive)
    - enddate: UTCDateTime or datetime (inclusive)
    """
    cur = conn.cursor()

    # 1. Index S-files
    print("Indexing S-files...")
    sfiles = get_sfile_list(seisan_rea_dir, dbname, startdate, enddate)
    for sfile_path in sfiles:
        try:
            s = Sfile(sfile_path, fast_mode=False, use_mvo_parser=True)
            event_id = s.public_id if hasattr(s, 'public_id') else sfile_path
            wav1 = s.to_dict().get("wavfile1")
            wav2 = s.to_dict().get("wavfile2")
            parsed = 1
            error = None
        except Exception as e:
            event_id = None
            wav1, wav2 = None, None
            parsed = 0
            error = str(e)

        cur.execute('''INSERT OR IGNORE INTO sfiles (path, event_id, parsed_successfully, parsed_time, error)
                       VALUES (?, ?, ?, ?, ?)''',
                    (sfile_path, event_id, parsed, datetime.utcnow().isoformat(), error))

        for wavfile in [wav1, wav2]:
            if not wavfile:
                continue

            wavfound = False
            matched_wav_path = None

            # Try direct file check
            if os.path.isfile(wavfile):
                matched_wav_path = wavfile
                wavfound = True
            else:
                # Try fuzzy match
                altbase = os.path.basename(wavfile).split('.')[0][:-3]
                candidates = glob(os.path.join(seisan_wav_dir, '**', altbase + '*'), recursive=True)
                if len(candidates) == 1:
                    matched_wav_path = candidates[0]
                    wavfound = True

            if matched_wav_path:
                # Get WAV file metadata
                try:
                    st = read(matched_wav_path)
                    start_time = str(min(tr.stats.starttime for tr in st))
                    end_time = str(max(tr.stats.endtime for tr in st))
                    parsed_wav = 1
                    wav_error = None
                except Exception as e:
                    start_time = end_time = None
                    parsed_wav = 0
                    wav_error = str(e)

                cur.execute('''INSERT OR IGNORE INTO wav_files (path, start_time, end_time, associated_sfile, used_in_event_id, parsed_successfully, parsed_time, error)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                            (matched_wav_path, start_time, end_time, sfile_path, event_id, parsed_wav, datetime.utcnow().isoformat(), wav_error))

                # Link S-file to WAV
                cur.execute('''INSERT INTO sfile_wav_map (sfile_path, wav_path) VALUES (?, ?)''',
                            (sfile_path, matched_wav_path))

    # 2. Find WAVs without corresponding S-files
    print("Scanning for unmatched WAV files...")
    for root, _, files in os.walk(seisan_wav_dir):
        for fname in files:
            if not fname.lower().endswith(('.sac', '.mseed', '.miniseed')):
                continue

            full_path = os.path.join(root, fname)
            try:
                full_path.encode("utf-8")
            except UnicodeEncodeError:
                print(f"[WARN] Skipping file with invalid encoding: {full_path}")
                continue

            cur.execute("SELECT 1 FROM wav_files WHERE path = ?", (full_path,))
            if not cur.fetchone():
                # Not yet indexed
                try:
                    st = read(full_path)
                    start_time = str(min(tr.stats.starttime for tr in st))
                    end_time = str(max(tr.stats.endtime for tr in st))
                    parsed_wav = 1
                    wav_error = None
                except Exception as e:
                    start_time = end_time = None
                    parsed_wav = 0
                    wav_error = str(e)

                cur.execute('''INSERT INTO wav_files (path, start_time, end_time, associated_sfile, used_in_event_id, parsed_successfully, parsed_time, error)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                            (full_path, start_time, end_time, None, None, parsed_wav, datetime.utcnow().isoformat(), wav_error))

    conn.commit()



if __name__ == "__main__":
    dbfile = "seiscomp_like.sqlite"
    conn = sqlite3.connect(dbfile)
    create_schema(conn)
    print(f"Initialized schema in {dbfile}")
    conn.close()

    dbfile2 = "seisan_processing_tracker.sqlite"
    seisan_top = "/data/SEISAN_DB" 
    dbname = "MVOE_"
    startdate = UTCDateTime(2000,1,1)
    enddate = UTCDateTime(2009,1,1)
    conn2 = sqlite3.connect(dbfile2)
    create_processing_schema(conn2)
    print(f"Initialized processing schema in {dbfile2}")
    index_sfiles_and_wavs(conn2, seisan_top, dbname, startdate, enddate)
    conn2.close()
