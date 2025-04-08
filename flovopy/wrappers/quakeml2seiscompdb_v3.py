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
        from TEXT,
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
        from TEXT,
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

if __name__ == "__main__":
    dbfile = "seiscomp_like.sqlite"
    conn = sqlite3.connect(dbfile)
    create_schema(conn)
    print(f"Initialized schema in {dbfile}")

    # Example usage with a test catalog
    # from obspy import read_events
    # cat = read_events("events.xml")
    # insert_catalog(conn, cat)

    conn.close()
