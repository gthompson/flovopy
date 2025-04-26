# b12_qmljson2extseiscompdb.py Reload QML&JSON files that comprise 

import sqlite3
from obspy import Catalog, UTCDateTime, read_events, read_inventory
#from obspy.core.event import Event, Origin, Magnitude, Pick, Arrival, Amplitude, StationMagnitude
import glob
import os

def is_event_already_processed(conn, event_id):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM events WHERE public_id = ?", (event_id,))
    return cur.fetchone() is not None

def insert_obspy_event(event, conn, commit=True):
    """
    Insert an ObsPy Event object into the seiscomp-like database schema.
    
    Parameters
    ----------
    event : obspy.core.event.Event
        The ObsPy Event to insert.
    conn : sqlite3.Connection
        SQLite connection object.
    commit : bool
        Whether to commit changes at the end (default True).
    """
    cur = conn.cursor()
    eid = event.resource_id.id
    completed = ''

    try:

        # --- Insert event ---
        cur.execute('''INSERT OR IGNORE INTO events 
            (public_id, event_type, creation_time, preferred_origin_id, preferred_magnitude_id, region)
            VALUES (?, ?, ?, ?, ?, ?)''', (
            eid,
            str(event.event_type) if event.event_type else None,
            str(getattr(event.creation_info, 'creation_time', None)),
            event.preferred_origin_id.id if event.preferred_origin_id else None,
            event.preferred_magnitude_id.id if event.preferred_magnitude_id else None,
            getattr(event, 'region', None)
        ))
        completed = 'events'
        

        # --- Origins ---
        for o in event.origins:
            cur.execute('''INSERT OR IGNORE INTO origins 
                (origin_id, event_id, time, latitude, longitude, depth, evaluation_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?)''', (
                o.resource_id.id,
                eid,
                str(o.time) if o.time else None,
                getattr(o, 'latitude', None),
                getattr(o, 'longitude', None),
                getattr(o, 'depth', None),
                str(getattr(o, 'evaluation_mode', None))
            ))
            # --- Arrivals ---
            for a in o.arrivals:
                cur.execute('''INSERT OR IGNORE INTO arrivals 
                    (arrival_id, pick_id, origin_id, time_residual, azimuth, distance)
                    VALUES (?, ?, ?, ?, ?, ?)''', (
                    a.resource_id.id,
                    a.pick_id.id if a.pick_id else None,
                    a.origin_id.id if a.origin_id else None,
                    getattr(a, 'time_residual', None),
                    getattr(a, 'azimuth', None),
                    getattr(a, 'distance', None)
                ))

        completed = 'origins'

        # --- Magnitudes ---
        for m in event.magnitudes:
            cur.execute('''INSERT OR IGNORE INTO magnitudes 
                (mag_id, event_id, magnitude, mag_type, origin_id)
                VALUES (?, ?, ?, ?, ?)''', (
                m.resource_id.id,
                eid,
                getattr(m, 'mag', None),
                getattr(m, 'magnitude_type', None),
                m.origin_id.id if m.origin_id else None
            ))
        completed = 'magnitudes'

        # --- Picks ---
        for p in event.picks:
            cur.execute('''INSERT OR IGNORE INTO picks 
                (pick_id, event_id, time, waveform_id, phase_hint)
                VALUES (?, ?, ?, ?, ?)''', (
                p.resource_id.id,
                eid,
                str(p.time) if p.time else None,
                p.waveform_id.id if p.waveform_id else None,
                getattr(p, 'phase_hint', None)
            ))
        completed = 'picks'

        # --- Amplitudes ---
        for amp in event.amplitudes:
            cur.execute('''INSERT OR IGNORE INTO amplitudes 
                (amplitude_id, event_id, generic_amplitude, unit, type, period, snr, waveform_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                amp.resource_id.id,
                eid,
                getattr(amp, 'generic_amplitude', None),
                getattr(amp, 'unit', None),
                getattr(amp, 'type', None),
                getattr(amp, 'period', None),
                getattr(amp, 'snr', None),
                amp.waveform_id.id if amp.waveform_id else None
            ))
        completed = 'amplitudes'

        # --- Station Magnitudes ---
        for sm in event.station_magnitudes:
            cur.execute('''INSERT OR IGNORE INTO station_magnitudes 
                (smag_id, event_id, station_code, mag, mag_type, amplitude_id)
                VALUES (?, ?, ?, ?, ?, ?)''', (
                sm.resource_id.id,
                eid,
                getattr(sm, 'station_magnitude_station', None),
                getattr(sm, 'mag', None),
                getattr(sm, 'magnitude_type', None),
                sm.amplitude_id.id if sm.amplitude_id else None
            ))
        completed = 'station_magnitudes'
    except:
        pass
    finally:
        #print(completed)
        if commit:
            conn.commit()



def insert_catalog(conn, qml_files):
    succeeded = 0
    failed = 0
    for i,qml in enumerate(sorted(qml_files)):
        try:
            catalog = read_events(qml)
            print(catalog.events[0])
            insert_obspy_event(catalog.events[0], conn, commit=True)
            succeeded += 1
        except Exception as e:
            print(f"[WARN] Failed to read {qml}: {e}")    
            failed += 1
        if i%100:
            print(f'Success: {succeeded}, Failed: {failed}')

def insert_enhanced_metadata(conn, event_id, energy_mag=None, event_class=None, location_quality=None):
    cur = conn.cursor()
    cur.execute('''
        INSERT OR REPLACE INTO enhanced_metadata (event_id, energy_mag, event_class, location_quality)
        VALUES (?, ?, ?, ?)
    ''', (event_id, energy_mag, event_class, location_quality))

def insert_event_waveform(conn, event_id, network, filepath, start_time, end_time, trace_ids, 
                          format="MSEED", description="", source_system=""):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO event_waveforms 
        (event_id, network, filepath, start_time, end_time, trace_ids, format, description, source_system)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, network, filepath, start_time, end_time, trace_ids, format, description, source_system))

def insert_trace_metric(conn, event_id, trace_id, station, channel, metric_name, 
                        metric_value, units=None, source=None):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO trace_metrics 
        (event_id, trace_id, station, channel, metric_name, metric_value, units, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, trace_id, station, channel, metric_name, metric_value, units, source))

def insert_trace_energy_metric(conn, event_id, trace_id, station, bsam, energy, 
                               site_correction=None, source=None):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO trace_energy_metrics 
        (event_id, trace_id, station, bsam, energy, site_correction, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, trace_id, station, bsam, energy, site_correction, source))

def insert_asl_metadata(conn, origin_id, q_factor, wavespeed, dsam_metric, window_seconds,
                        prefilter_low, prefilter_high, stations_used, 
                        reduced_displacement, reduced_energy):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO asl_metadata 
        (origin_id, q_factor, wavespeed, dsam_metric, window_seconds,
         prefilter_low, prefilter_high, stations_used, reduced_displacement, reduced_energy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (origin_id, q_factor, wavespeed, dsam_metric, window_seconds,
          prefilter_low, prefilter_high, stations_used, reduced_displacement, reduced_energy))

def insert_event_classification(conn, event_id, event_type, mainclass, subclass, author, timestamp, source):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO event_classifications 
        (event_id, event_type, mainclass, subclass, author, timestamp, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, event_type, mainclass, subclass, author, timestamp, source))

def insert_aef_metric(conn, event_id, trace_id, start_time, end_time, amplitude, energy, maxf,
                      frequencies, amplitudes, ssam_json, source=None):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO aef_metrics 
        (event_id, trace_id, start_time, end_time, amplitude, energy, maxf,
         frequencies, amplitudes, ssam_json, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, trace_id, start_time, end_time, amplitude, energy, maxf,
          frequencies, amplitudes, ssam_json, source))

def insert_meta_event(conn, meta_event_id, name, start_time, end_time, description, parent_meta_event_id=None):
    cur = conn.cursor()
    cur.execute('''
        INSERT OR IGNORE INTO meta_events 
        (meta_event_id, name, start_time, end_time, description, parent_meta_event_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (meta_event_id, name, start_time, end_time, description, parent_meta_event_id))

def map_event_to_meta(conn, event_id, meta_event_id):
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO event_meta_mapping (meta_event_id, event_id)
        VALUES (?, ?)
    ''', (meta_event_id, event_id))


if __name__ == "__main__":

    from flovopy.core.enhanced import EnhancedEvent
    dbfile = "seiscomp_like5.sqlite"
    conn = sqlite3.connect(dbfile)
    create_schema(conn)
    print(f"Initialized schema in {dbfile}")

    print('Reading inventory')
    inv = read_inventory('/data/SEISAN_DB/CAL/MV.xml')
    print('Inserting inventory into db')
    insert_stations_from_inventory(conn, inv)

    # loop over all QML files
    QML_DIR = "/data/SEISAN_DB/json/MVOE_"
    qml_files = sorted(glob.glob(os.path.join(QML_DIR, "*", "*", "*.qml")))

    succeeded = 0
    failed = 0
    for i,qml in enumerate(sorted(qml_files)):
        eventbase = qml.replace('.qml','')
        ev = EnhancedEvent.load(eventbase)
        
        if not hasattr(ev, "event") or ev.event is None:
            print(f"[SKIP] No event found in {qml}")
            continue
        if is_event_already_processed(conn, ev.event_id):
            print(f"[SKIP] Already processed {ev.event_id}")
            continue

        try:
            insert_obspy_event(ev.event, conn, commit=False)
            #print(f"[INFO] Inserted {ev.event_id}")
            #if hasattr(ev, 'energy_mag'):
            #    insert_enhanced_metadata(conn, ev.event_id, ev.energy_mag, ev.event_class, ev.location_quality)

            if hasattr(ev.metrics, 'mainclass'):
                m = ev.metrics
                print(f'm={m}')
                filetime = m.filetime.isoformat()
                insert_event_classification(conn, ev.event_id, None, m.mainclass, m.subclass, m.analyst, filetime, None)
                if hasattr(m, 'aef_rows'):
                    for aef in m.aef_rows:
                        ssam_json = json.dumps(aef.ssam) if hasattr(aef,'ssam') else None
                        #insert_aef_metric(conn, ev.event_id, trace_id, start_time, end_time, amplitude, energy, maxf,
                        #      frequencies, amplitudes, ssam_json, source=None)
                        insert_aef_metric(conn, ev.event_id, trace_id, start_time, None, aef.amplitude, aef.energy, aef.maxf,
                            None, None, ssam_json, source=None)  
            if hasattr(ev, 'stream') or hasattr(ev, 'wav_paths'): 
                #insert_event_waveform(conn, ev.event_id, network, filepath, start_time, end_time, trace_ids, 
                #          format="MSEED", description="", source_system="")
                pass

            #for m in ev.trace_metrics:
            #    insert_trace_metric(conn, ev.event_id, **m)

            #for e in ev.trace_energy_metrics:
            #    insert_trace_energy_metric(conn, ev.event_id, **e)


            succeeded += 1

        except Exception as e:
            print(f"[WARN] Failed to insert {qml} ({getattr(ev, 'event_id', 'unknown')}): {e}")
            failed += 1
        if i%100==0:
            print(f'[Progress] {i} events processed: Success: {succeeded}, Failed: {failed}')
            conn.commit()
    conn.commit()
    print(f"[DONE] Total Success: {succeeded}, Failed: {failed}, out of {len(qml_files)} files.")
    conn.close()
