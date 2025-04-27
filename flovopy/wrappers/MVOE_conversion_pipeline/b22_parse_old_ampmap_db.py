import os
import numpy as np
from obspy import UTCDateTime
from obspy.core.event import Event, Origin, OriginQuality, Amplitude
import scipy.io

def parse_event_from_mat(filepath, sample_rate=75.0):
    """
    Parse a .mat file into:
    - ObsPy Event
    - Event-level dictionary
    - List of Origin-level dictionaries
    - List of StationResidual dictionaries
    """
    basename = os.path.basename(filepath).replace('.mat', '')
    timestamp_part = basename.split('SP')[0]
    filetime = UTCDateTime.strptime(timestamp_part, "%Y-%m-%d-%H%M-%S")

    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    locs = mat_data['loc_fin']
    rms = mat_data.get('rms_fin', None)
    max_gap = mat_data.get('max_gap_fin', None)
    num_stations = mat_data.get('m_fin', None)
    amplitudes = mat_data.get('amp_fin', None)

    start_point = mat_data.get('start_point', 1)
    end_point = mat_data.get('end_point', None)
    time_step_samples = mat_data.get('time_step_v', 256)
    fft_length_samples = mat_data.get('FFT_v', 1024)

    n_windows = locs.shape[0]
    time_step_seconds = time_step_samples / sample_rate

    origin_base_time = filetime + (start_point - 1) / sample_rate

    event_id = basename  # use filename as event_id
    event = Event()

    origin_rows = []
    station_rows = []

    for i in range(n_windows):
        otime = origin_base_time + i * time_step_seconds
        lat, lon = locs[i]

        origin_id = f"{event_id}_win{i}"

        # Create Origin
        origin = Origin()
        origin.resource_id = origin_id
        origin.time = otime
        origin.latitude = lat
        origin.longitude = lon

        quality = OriginQuality()
        if rms is not None:
            quality.standard_error = float(rms[i])
        if max_gap is not None:
            quality.azimuthal_gap = float(max_gap[i])
        if num_stations is not None:
            quality.used_phase_count = int(num_stations[i])

        origin.origin_quality = quality
        event.origins.append(origin)

        # Create Amplitude
        if amplitudes is not None:
            amp = Amplitude()
            amp.generic_amplitude = float(amplitudes[i])
            amp.unit = "m"
            amp.method_id = "artificial::MATLAB2001_conversion"
            amp.time_window_start = otime
            amp.time_window_end = otime + (fft_length_samples / sample_rate)
            amp.magnitude_hint = "RD"
            amp.origin_id = origin.resource_id
            event.amplitudes.append(amp)

        # Capture Origin-level row
        origin_row = {
            'origin_id': origin_id,
            'event_id': event_id,
            'window_index': i,
            'origin_time': str(otime),
            'latitude': lat,
            'longitude': lon,
            'rms_error': float(rms[i]) if rms is not None else None,
            'azimuthal_gap': float(max_gap[i]) if max_gap is not None else None,
            'num_stations': int(num_stations[i]) if num_stations is not None else None,
            'amplitude': float(amplitudes[i]) if amplitudes is not None else None
        }
        origin_rows.append(origin_row)

        # Capture Station Residuals
        for varname in mat_data.keys():
            if varname.startswith('res_') and varname.endswith('_fin'):
                station_code = 'MB' + varname[4:6].upper()
                residual_array = mat_data[varname]
                if isinstance(residual_array, np.ndarray) and len(residual_array) > i:
                    residual = residual_array[i]
                    if not np.isnan(residual):
                        station_row = {
                            'id': f"{origin_id}_{station_code}",
                            'origin_id': origin_id,
                            'station_code': station_code,
                            'residual': float(residual),
                            'azimuth': None  # we could map later if needed
                        }
                        station_rows.append(station_row)

    # Capture Event-level row
    event_row = {
        'event_id': event_id,
        'filename': basename,
        'filetime': str(filetime),
        'start_time': str(origin_base_time),
        'end_time': str(origin_base_time + (n_windows - 1) * time_step_seconds),
        'num_origins': n_windows,
        'average_rms': np.nanmean(rms) if rms is not None else None,
        'max_amplitude': np.nanmax(amplitudes) if amplitudes is not None else None,
        'min_amplitude': np.nanmin(amplitudes) if amplitudes is not None else None,
        'start_point': start_point,
        'end_point': end_point,
        'fft_length_samples': fft_length_samples,
        'time_step_samples': time_step_samples,
        'sample_rate': sample_rate
    }

    return event, event_row, origin_rows, station_rows


if __name__ == '__main__':
    import scipy.io
    from obspy.core.event import Catalog

    folder_path = '/data/Montserrat/AMPMAPDB'

    import sqlite3
    import os
    import glob
    import pandas as pd
    from obspy.core.event import Catalog


    db_path = '/home/thompsong/Dropbox/matlab_events.db'
    catalog_path = '/home/thompsong/Dropbox/ampmap_events_catalog.xml'

    # Create Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Tables
    cursor.execute('PRAGMA foreign_keys = ON;')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS event (
        event_id TEXT PRIMARY KEY,
        filename TEXT,
        filetime TEXT,
        start_time TEXT,
        end_time TEXT,
        num_origins INTEGER,
        average_rms REAL,
        max_amplitude REAL,
        min_amplitude REAL,
        start_point INTEGER,
        end_point INTEGER,
        fft_length_samples INTEGER,
        time_step_samples INTEGER,
        sample_rate REAL
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS origin (
        origin_id TEXT PRIMARY KEY,
        event_id TEXT,
        window_index INTEGER,
        origin_time TEXT,
        latitude REAL,
        longitude REAL,
        rms_error REAL,
        azimuthal_gap REAL,
        num_stations INTEGER,
        amplitude REAL,
        FOREIGN KEY(event_id) REFERENCES event(event_id)
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS station_residual (
        id TEXT PRIMARY KEY,
        origin_id TEXT,
        station_code TEXT,
        residual REAL,
        azimuth REAL,
        FOREIGN KEY(origin_id) REFERENCES origin(origin_id)
    );
    ''')

    conn.commit()

    # --- Process Files ---
    catalog = Catalog()

    mat_files = sorted(glob.glob(os.path.join(folder_path, '*.mat')))
    for file in mat_files:
        try:
            event, event_row, origin_rows, station_rows = parse_event_from_mat(file)

            # Insert event
            cursor.execute('''
            INSERT OR IGNORE INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', tuple(event_row.values()))

            # Insert origins
            for origin in origin_rows:
                cursor.execute('''
                INSERT OR IGNORE INTO origin VALUES (?,?,?,?,?,?,?,?,?,?)
                ''', tuple(origin.values()))

            # Insert station residuals
            for station in station_rows:
                cursor.execute('''
                INSERT OR IGNORE INTO station_residual VALUES (?,?,?,?,?)
                ''', tuple(station.values()))

            catalog.events.append(event)

            print(f"[INFO] Processed {os.path.basename(file)} successfully.")

        except Exception as e:
            print(f"[ERROR] Problem processing {file}: {e}")

    conn.commit()
    conn.close()

    # Save backup QuakeML catalog
    catalog.write(catalog_path, format="QUAKEML")
    print(f"[DONE] Wrote catalog to {catalog_path}")

