# Create extended SeisComP DB

import sqlite3
from obspy import read_inventory

def create_schema(conn):
    cur = conn.cursor()

    ###################################################
    #### tracking cleaned wav files and detections ####
    #### no concept of event here, just wav files  ####
    ###################################################

    ###############################
    # MiniSEED file status
    ###############################
    cur.execute('''CREATE TABLE IF NOT EXISTS mseed_file_status (
        mseed_id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        endtime TEXT,
        dir TEXT,
        dfile TEXT UNIQUE NOT NULL,
        network TEXT,
        format TEXT DEFAULT 'MSEED',
        detected INTEGER DEFAULT 0 CHECK(detected IN (0, 1)),
        classified INTEGER DEFAULT 0 CHECK(classified IN (0, 1)),
        located INTEGER DEFAULT 0 CHECK(located IN (0, 1)),
        quantified INTEGER DEFAULT 0 CHECK(quantified IN (0, 1)),
        comment TEXT
    )''')

    ###############################
    # wfdisc and detection tracking
    ###############################
    cur.execute('''CREATE TABLE IF NOT EXISTS wfdisc (
        wfid INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id TEXT,
        time TEXT,
        endtime TEXT,
        dfile TEXT,
        tracenum INTEGER,
        nsamp INTEGER,
        samprate REAL,
        calib REAL,
        units TEXT,
        comment TEXT,
        UNIQUE(dfile, trace_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS network_detection (
        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dfile TEXT,
        snr REAL,
        minchans INTEGER,
        algorithm TEXT,
        threshon REAL,
        threshoff REAL,
        sta_seconds REAL,
        lta_seconds REAL,
        pad_seconds REAL,
        freq_low REAL,
        freq_high REAL,
        criterion TEXT,
        ontime TEXT,
        duration TEXT,
        offtime TEXT,
        trace_ids TEXT,
        detection_quality REAL,
        comment TEXT,
        UNIQUE(dfile),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile)
    )''')  

    # Waveform-event mapping
    cur.execute('''CREATE TABLE IF NOT EXISTS event_waveform_map (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        dfile TEXT UNIQUE,
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile)      
    )''')


    #################################
    #### SeisComp-like db tables ####
    #################################

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

    #####################################
    #### Extended event schema       ####
    #### For classify, ASL, quantify ####
    #####################################

    cur.execute('''CREATE TABLE IF NOT EXISTS station_corrections (
        station_code TEXT UNIQUE,
        amplification REAL,
        FOREIGN KEY(station_code) REFERENCES stations(station_code)       
    )''')    

    # Trace metrics computed by ampengfft that do not required a location
    cur.execute('''CREATE TABLE IF NOT EXISTS aef_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        trace_id TEXT,
        time TEXT,
        endtime TEXT,
        dfile TEXT,
        snr REAL,
        peakamp REAL,
        peaktime TEXT,
        energy REAL,
        peakf REAL,
        meanf REAL,
        ssam_json TEXT,
        spectrum_id TEXT,
        sgramdir TEXT,
        sgramdfile TEXT,
        band_ratio1 REAL,
        band_ratio2 REAL,
        skewness REAL,
        kurtosis REAL,
        source TEXT,
        UNIQUE(dfile, trace_id, source),
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile),
        FOREIGN KEY(spectrum_id) REFERENCES event_spectra(spectrum_id)
    )''')  

    # Trace metrics - that do not need a location - and are not computed by ampengfft
    # Similar to wfmeas in CSS3.0
    cur.execute('''CREATE TABLE IF NOT EXISTS trace_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        dfile TEXT,
        trace_id TEXT,
        time TEXT,
        endtime TEXT,
        reltime REAL DEFAULT 0.0,
        metric TEXT,
        value REAL,
        units TEXT,
        source TEXT,
        UNIQUE(dfile, trace_id, metric),
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile)
    )''')

    # Magnitude model
    cur.execute('''CREATE TABLE IF NOT EXISTS magnitude_model (
        model_id TEXT PRIMARY KEY,
        wavetype TEXT,
        q REAL DEFAULT 23.0,
        c_earth REAL DEFAULT 2500.0,
        c_atmos REAL DEFAULT 340.0,
        rho_earth REAL DEFAULT 2000.0,
        rho_atmos REAL DEFAULT 1.2,
        boatwright_used INTEGER DEFAULT 0 CHECK(boatwright_used IN (0, 1)),
        mag_correction REAL DEFAULT 3.7,
        a REAL DEFAULT 1.6,
        b REAL DEFAULT -0.15,
        g REAL DEFAULT 0.0,
        station_correction_used INTEGER DEFAULT 0 CHECK(station_correction_used IN (0, 1))
    )''')
        
    # trace level ampengfftmag metrics that depend on a source location
    # 1 row per trace (and not broken into windows as for ASL)
    cur.execute('''CREATE TABLE IF NOT EXISTS located_metrics (
        event_id TEXT PRIMARY KEY,
        time TEXT,
        endtime TEXT,
        trace_id TEXT,
        dfile TEXT,
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
        mag_id TEXT,
        smag_id TEXT,
        UNIQUE(dfile, trace_id),
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile),
        FOREIGN KEY(mag_id) REFERENCES magnitudes(mag_id),
        FOREIGN KEY(smag_id) REFERENCES magnitudes(smag_id)
    )''') 

    # Classification per event - removed unique constraint on dfile
    cur.execute('''CREATE TABLE IF NOT EXISTS event_classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        dfile TEXT,
        mainclass TEXT,
        subclass TEXT,
        author TEXT,
        time TEXT,
        source TEXT,
        score REAL,
        FOREIGN KEY(event_id) REFERENCES events(public_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile)
    )''')

    #### SAM data ####
    cur.execute('''CREATE TABLE IF NOT EXISTS sam_timeseries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dfile TEXT,
        event_id TEXT,
        trace_id TEXT,
        time TEXT,
        endtime TEXT,
        sampling_rate REAL,
        sam_type TEXT,
        data BLOB,
        units TEXT,
        comments TEXT,
        UNIQUE(dfile, trace_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile),
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    #### Event Spectra ####
    cur.execute('''CREATE TABLE IF NOT EXISTS event_spectra (
        spectrum_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        dfile TEXT,
        trace_id TEXT,
        frequencies BLOB,
        amplitudes BLOB,
        time TEXT,
        endtime TEXT,
        source TEXT,
        UNIQUE(dfile, trace_id),
        FOREIGN KEY(dfile) REFERENCES mseed_file_status(dfile),
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    #### Amplitude Source Location ####
    # SCAFFOLD: Might need to break this up into two tables, one for the grid, another with grid_id and station attirbutes
    # ASL lookup grid
    cur.execute('''CREATE TABLE IF NOT EXISTS asl_grid (
        node_id INTEGER PRIMARY KEY AUTOINCREMENT,              
        grid_id INTEGER,
        node_lat REAL,
        node_lon REAL,
        node_elev REAL DEFAULT 0.0,
        UNIQUE(grid_id, node_lat, node_lon, node_elev),
        FOREIGN KEY(grid_id) REFERENCES asl_grid_definition(grid_id)           
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS asl_grid_station_amplitudes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id INTEGER,
        model_id INTEGER,                
        station_code TEXT,
        sta_lat REAL,
        sta_lon REAL,
        sta_elev REAL,
        distance REAL,
        geometric_spreading_correction REAL,
        inelastic_spreading_correction REAL,
        relative_amplitude REAL,
        UNIQUE(node_id, model_id, station_code),
        FOREIGN KEY(node_id) REFERENCES asl_grid(node_id),
        FOREIGN KEY(model_id) REFERENCES asl_model(model_id),  
        FOREIGN KEY(station_code) REFERENCES stations(station_code)                           
    )''')    

    # ASL model
    cur.execute('''CREATE TABLE IF NOT EXISTS asl_model (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        wavetype TEXT,
        wavespeed REAL,
        peakf REAL,
        window_seconds REAL DEFAULT 5.0,
        q REAL DEFAULT 23.0      
    )''')    

    # ASL grid definition
    cur.execute('''CREATE TABLE IF NOT EXISTS asl_grid_definition (
        grid_id INTEGER PRIMARY KEY AUTOINCREMENT,
        centerlat REAL,
        centerlon REAL,
        nlat INTEGER,
        nlon INTEGER,
        ndepth INTEGER DEFAULT 1,
        node_spacing_m REAL   
    )''')
  
    # ASL-based trajectory results (1 row per time step)
    cur.execute('''CREATE TABLE IF NOT EXISTS asl_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        model_id INTEGER,                
        grid_id INTEGER,                
        timestamp TEXT,
        window_seconds REAL,
        est_lat REAL,
        est_lon REAL,
        est_elev REAL DEFAULT 0.0,
        reduced_displacement REAL,
        reduced_velocity REAL,
        reduced_energy REAL,
        misfit REAL,
        FOREIGN KEY(grid_id) REFERENCES asl_grid_definiton(grid_id),
        FOREIGN KEY(model_id) REFERENCES asl_model(model_id),                
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')


    #####################################
    #### Extended event schema       ####
    #### meta_events                 ####
    #####################################

    cur.execute('''CREATE TABLE IF NOT EXISTS meta_events (
        meta_event_id INTEGER PRIMARY KEY,
        name TEXT,
        time TEXT,
        endtime TEXT,
        description TEXT,
        parent_meta_event_id TEXT,
        FOREIGN KEY(parent_meta_event_id) REFERENCES meta_events(meta_event_id)
    )''')

    cur.execute('''CREATE TABLE IF NOT EXISTS event_meta_mapping (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meta_event_id INTEGER,
        event_id TEXT,
        FOREIGN KEY(meta_event_id) REFERENCES meta_events(meta_event_id),
        FOREIGN KEY(event_id) REFERENCES events(public_id)
    )''')

    # Add indexes on common foreign key lookups (event_id or origin_id)
    cur.execute('CREATE INDEX IF NOT EXISTS idx_trace_metrics_event ON trace_metrics(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_spectra_eventid ON event_spectra(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_event_class_event ON event_classifications(event_id)')    
    '''
    cur.execute('CREATE INDEX IF NOT EXISTS idx_waveforms_event ON event_waveforms(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_energy_metrics_event ON trace_energy_metrics(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_aef_event ON aef_metrics(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_meta_mapping_event ON event_meta_mapping(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_meta_mapping_meta ON event_meta_mapping(meta_event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_asl_origin ON asl_metadata(origin_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_arrivals_origin ON arrivals(origin_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_arrivals_pick ON arrivals(pick_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_picks_event ON picks(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_magnitudes_event ON magnitudes(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_origins_event ON origins(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_amplitudes_event ON amplitudes(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_station_mags_event ON station_magnitudes(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_detect_file ON detections(file_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_detect_path ON detections(filepath)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_waveform_event ON event_waveforms(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_class_event ON event_classifications(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_amp_file ON amplitude_timeseries(file_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_amp_event ON amplitude_timeseries(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_spec_event ON event_spectra(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_traj_event ON event_trajectory(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_detections_ontime ON detections(ontime)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_aslgrid_gridid ON asl_lookup_grid(grid_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_amptimes_eventid ON amplitude_timeseries(event_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_traj_eventid ON event_trajectory(event_id)')
    '''

    conn.commit()

def insert_stations_from_inventory(conn, inventory):
    cur = conn.cursor()
    for network in inventory:
        for station in network.stations:
            cur.execute('''INSERT OR REPLACE INTO stations (station_code, network_code, latitude, longitude, elevation)
                           VALUES (?, ?, ?, ?, ?)''',
                        (station.code, network.code, station.latitude, station.longitude, station.elevation))
    conn.commit()


if __name__ == "__main__":
    from flovopy.config_projects import get_config
    from flovopy.core.enhanced import EnhancedEvent
    config = get_config()
    dbfile = config['mvo_seiscomp_db']
    conn = sqlite3.connect(dbfile)
    create_schema(conn)
    print(f"Initialized schema in {dbfile}")

    print('Reading inventory')
    inv = read_inventory(config['inventory'])
    print('Inserting inventory into db')
    insert_stations_from_inventory(conn, inv)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.commit()
    conn.close()
