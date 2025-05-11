import sqlite3

conn = sqlite3.connect('seiscomp_extended.db')
cur = conn.cursor()

# SeisComP standard or extended tables

# Basic Event structure
cur.execute('''CREATE TABLE IF NOT EXISTS event (
    public_id TEXT PRIMARY KEY,
    event_type TEXT,
    creation_time TEXT,
    preferred_origin_id TEXT,
    preferred_magnitude_id TEXT,
    region TEXT
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS origin (
    origin_id TEXT PRIMARY KEY,
    event_id TEXT,
    time TEXT,
    latitude REAL,
    longitude REAL,
    depth REAL,
    evaluation_mode TEXT,
    FOREIGN KEY(event_id) REFERENCES events(public_id)
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS magnitude (
    mag_id TEXT PRIMARY KEY,
    event_id TEXT,
    magnitude REAL,
    mag_type TEXT,
    origin_id TEXT,
    FOREIGN KEY(event_id) REFERENCES events(public_id),
    FOREIGN KEY(origin_id) REFERENCES origins(origin_id)
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS pick (
    pick_id TEXT PRIMARY KEY,
    event_id TEXT,
    time TEXT,
    waveform_id TEXT,
    phase_hint TEXT,
    FOREIGN KEY(event_id) REFERENCES events(public_id)
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS arrival (
    arrival_id TEXT PRIMARY KEY,
    pick_id TEXT,
    origin_id TEXT,
    time_residual REAL,
    azimuth REAL,
    distance REAL,
    FOREIGN KEY(pick_id) REFERENCES picks(pick_id),
    FOREIGN KEY(origin_id) REFERENCES origins(origin_id)
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS amplitude (
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

cur.execute('''CREATE TABLE IF NOT EXISTS station_magnitude (
    smag_id TEXT PRIMARY KEY,
    event_id TEXT,
    station_code TEXT,
    mag REAL,
    mag_type TEXT,
    amplitude_id TEXT,
    FOREIGN KEY(event_id) REFERENCES events(public_id),
    FOREIGN KEY(amplitude_id) REFERENCES amplitudes(amplitude_id)
)''')


cur.execute("""CREATE TABLE IF NOT EXISTS event_waveform_link (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES event(public_id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS waveform_segment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id TEXT NOT NULL
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS waveform_stream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_tag TEXT NOT NULL UNIQUE
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS waveform_qc_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id INTEGER,
    FOREIGN KEY(stream_id) REFERENCES waveform_stream(id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS stationxml_reference (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    network_code TEXT,
    station_code TEXT,
    location_code TEXT,
    channel_code TEXT,
    start_time TEXT,
    end_time TEXT,
    file_uri TEXT NOT NULL,
    checksum TEXT
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS event_classification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT,
    mainclass TEXT,
    subclass TEXT,
    author TEXT,
    time TEXT,
    source TEXT,
    dfile TEXT,
    FOREIGN KEY(event_id) REFERENCES event(public_id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS aef_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT,
    trace_id TEXT,
    time TEXT,
    endtime TEXT,
    snr REAL,
    peakamp REAL,
    peaktime TEXT,
    energy REAL,
    peakf REAL,
    meanf REAL,
    ssam_json TEXT,
    spectrum_id TEXT,
    spectrogram_id TEXT,
    band_ratio1 REAL,
    band_ratio2 REAL,
    skewness REAL,
    kurtosis REAL,
    source TEXT,
    UNIQUE(time, endtime, trace_id, source),
    FOREIGN KEY(event_id) REFERENCES event(public_id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS spectrum (
    spectrum_id TEXT PRIMARY KEY
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS spectrogram (
    spectrogram_id TEXT PRIMARY KEY
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS located_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    waveform_stream_id TEXT NOT NULL,
    time TEXT NOT NULL,
    endtime TEXT NOT NULL,
    distance REAL,
    source_lat REAL,
    source_lon REAL,
    source_elev REAL DEFAULT 0.0,
    location_assumed INTEGER DEFAULT 0 CHECK(location_assumed IN (0, 1)),
    site_correction_applied INTEGER DEFAULT 0 CHECK(site_correction_applied IN (0, 1)),
    reduced_displacement REAL,
    reduced_velocity REAL,
    local_magnitude REAL,
    surface_wave_magnitude REAL,
    duration_magnitude REAL,
    source_energy REAL,
    energy_magnitude REAL,
    FOREIGN KEY(event_id) REFERENCES event(public_id),
    UNIQUE(event_id, waveform_stream_id, time, endtime)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS asl_model (
    model_id INTEGER PRIMARY KEY
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS asl_grid_definition (
    grid_id INTEGER PRIMARY KEY
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS asl_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    grid_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    window_seconds REAL,
    est_lat REAL,
    est_lon REAL,
    est_elev REAL DEFAULT 0.0,
    reduced_displacement REAL,
    reduced_velocity REAL,
    reduced_energy REAL,
    misfit REAL,
    FOREIGN KEY(event_id) REFERENCES event(public_id),
    FOREIGN KEY(model_id) REFERENCES asl_model(model_id),
    FOREIGN KEY(grid_id) REFERENCES asl_grid_definition(grid_id),
    UNIQUE(event_id, timestamp, model_id, grid_id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS meta_events (
    meta_event_id INTEGER PRIMARY KEY,
    name TEXT,
    time TEXT,
    endtime TEXT,
    description TEXT,
    parent_meta_event_id INTEGER,
    FOREIGN KEY(parent_meta_event_id) REFERENCES meta_events(meta_event_id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS event_meta_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meta_event_id INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    FOREIGN KEY(meta_event_id) REFERENCES meta_events(meta_event_id),
    FOREIGN KEY(event_id) REFERENCES event(public_id),
    UNIQUE(meta_event_id, event_id)
)""")

cur.execute("""CREATE TABLE IF NOT EXISTS trace_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    waveform_stream_id TEXT NOT NULL,
    time TEXT NOT NULL,
    endtime TEXT NOT NULL,
    reltime REAL DEFAULT 0.0,
    metric TEXT NOT NULL,
    value REAL,
    units TEXT,
    source TEXT,
    FOREIGN KEY(event_id) REFERENCES event(public_id),
    UNIQUE(event_id, waveform_stream_id, time, metric, source)
)""")

conn.commit()
conn.close()