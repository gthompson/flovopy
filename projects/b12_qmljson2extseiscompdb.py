# b12_qmljson2extseiscompdb.py
# Load QuakeML and enhanced JSON metadata into extended SeisComP-style DB

import os
import glob
import json
import sqlite3
from obspy import read_events, UTCDateTime
from flovopy.core.enhanced import EnhancedEvent

def clear_seiscomp_tables(dbfile):
    tables = [
        "arrivals",
        "amplitudes",
        "station_magnitudes",
        "picks",
        "magnitudes",
        "origins",
        "events",
        "aef_metrics",
        "event_classifications",
        "event_waveform_map"
    ]

    conn = sqlite3.connect(dbfile)
    conn.execute("PRAGMA foreign_keys = OFF;")  # Disable temporarily to avoid constraint issues
    cur = conn.cursor()

    for table in tables:
        print(f"Deleting all rows from table: {table}")
        cur.execute(f"DELETE FROM {table};")

    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON;")  # Re-enable constraints
    conn.close()
    print("[DONE] All relevant tables cleared.")


def is_event_already_processed(conn, event_id):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM events WHERE public_id = ?", (event_id,))
    return cur.fetchone() is not None


def insert_obspy_event(event, conn):
    """Insert ObsPy Event object into extended SeisComP schema."""
    cur = conn.cursor()
    eid = event.resource_id.id

    try:
        # Insert event
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

        # Picks
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

        # Amplitudes
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

        # Origins
        for o in event.origins:
            orid = o.resource_id.id
            print(f'orid={orid}')
            cur.execute('''INSERT OR IGNORE INTO origins 
                (origin_id, event_id, time, latitude, longitude, depth, evaluation_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?)''', (
                #o.resource_id.id,
                orid,
                eid,
                str(o.time) if o.time else None,
                getattr(o, 'latitude', None),
                getattr(o, 'longitude', None),
                getattr(o, 'depth', None),
                str(getattr(o, 'evaluation_mode', None))
            ))

            for a in o.arrivals:
                cur.execute('''INSERT OR IGNORE INTO arrivals 
                    (arrival_id, pick_id, origin_id, time_residual, azimuth, distance)
                    VALUES (?, ?, ?, ?, ?, ?)''', (
                    a.resource_id.id,
                    a.pick_id.id if a.pick_id else None,
                    #a.origin_id.id if a.origin_id else None,
                    orid,
                    getattr(a, 'time_residual', None),
                    getattr(a, 'azimuth', None),
                    getattr(a, 'distance', None)
                ))

        # Station Magnitudes
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

        # Magnitudes
        for m in event.magnitudes:
            cur.execute('''INSERT OR IGNORE INTO magnitudes 
                (mag_id, event_id, magnitude, mag_type, origin_id)
                VALUES (?, ?, ?, ?, ?)''', (
                m.resource_id.id,
                eid,
                getattr(m, 'mag', None),
                getattr(m, 'magnitude_type', None),
                m.origin_id.id if m.origin_id else None
                #orid
            ))



    except Exception as e:
        print(f"[ERROR] Failed to insert ObsPy event {eid}: {e}")
        raise


def insert_json_metadata(conn, eid, metrics):
    """Insert aef_metrics and event_classifications from JSON sidecar."""
    cur = conn.cursor()

    try:
        dfile = None
        if metrics.get("aef_path"):
            dfile = os.path.basename(metrics.get("aef_path"))
        mainclass = metrics.get("mainclass")
        subclass = metrics.get("subclass")
        author = metrics.get("analyst")
        filetime = metrics.get("filetime")
        print(dfile, mainclass, subclass, author, filetime)

        # Insert classification
        cur.execute('''INSERT INTO event_classifications 
            (event_id, dfile, mainclass, subclass, author, time, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)''', (
            eid, dfile, mainclass, subclass, author,
            filetime if isinstance(filetime, str) else filetime.isoformat(),
            'seisan'
        ))
        print('- event classifications done')
        '''
        "event_id": "smi:local/f77de408-7840-4fe2-bf12-f51dbb5fe8a7",
        "sfile_path": "/data/SEISAN_DB/REA/MVOE_/2000/03/01-0039-53L.S200003",
        "wav_paths": [
            "/data/SEISAN_DB/WAV/MVOE_/2000/03/2000-03-01-0039-53S.MVO___019"
        ],
        "aef_path": "/data/SEISAN_DB/REA/MVOE_/2000/03/01-0039-53L.S200003",
        '''
        # Insert waveform mapping
        for wav in metrics.get("wav_paths", []):
            cur.execute('''INSERT OR IGNORE INTO event_waveform_map 
                (event_id, dfile)
                VALUES (?, ?)''', (
                eid, os.path.basename(wav)+'.cleaned'
            ))
        print('- event waveform map done')

        # Insert metrics (AEF)
        aefrows = metrics.get("aefrows", [])
        if aefrows:
            for aef in metrics.get("aefrows", []):
                trace_id = aef.get("fixed_id")
                peakamp = aef.get("amplitude")
                energy = aef.get("energy")
                peakf = aef.get("maxf")
                ssam_json = json.dumps(aef.get("ssam", {}))
                print('-- ', eid, trace_id, filetime, dfile,
                    peakamp, energy, peakf, ssam_json)
                cur.execute('''INSERT INTO aef_metrics 
                    (event_id, trace_id, time, dfile, peakamp, energy, peakf, ssam_json, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    eid, trace_id, filetime, dfile,
                    peakamp, energy, peakf, ssam_json, "seisan"
                ))
                print('-- done')

    except Exception as e:
        print(f"[ERROR] Failed to insert JSON metadata for {eid}: {e}")
        raise



def main():
    from flovopy.config_projects import get_config
    from flovopy.core.enhanced import EnhancedEvent  # noqa: F401, future use
    from db_backup import backup_db
    config = get_config()
    dbfile = config['mvo_seiscomp_db']
    if not backup_db(dbfile, __file__):
        exit()  

    # Uncomment to wipe existing data:
    # clear_seiscomp_tables(dbfile)    
    conn = sqlite3.connect(dbfile)

    conn.execute("PRAGMA foreign_keys = ON;")

    QML_DIR = os.path.join(config['json_top'], 'MVOE_')
    qml_files = sorted(glob.glob(os.path.join(QML_DIR, "*", "*", "*.qml")))

    succeeded = 0
    failed = 0

    for i, qml_path in enumerate(qml_files):
        print('\n')
        print(f'Processing {qml_path}')
        base = qml_path.replace(".qml", "")
        try:
            ev = EnhancedEvent.load(base)
            if not ev:
                print(f"[SKIP] No enhanced event in {qml_path}")
                continue
            print('- have enhanced event')

            if not hasattr(ev, "event") or ev.event is None:
                print(f"[SKIP] No event in {qml_path}")
                continue
            print('- have ObsPy event')

            if is_event_already_processed(conn, ev.event_id):
                print(f"[SKIP] Already inserted {ev.event_id}")
                continue

            print('- Trying to insert ObsPy event into database')
            insert_obspy_event(ev.event, conn)
            print('- ObsPy event succeeded')

            if hasattr(ev, "metrics"):
                print('- trying to insert metrics into database')
                insert_json_metadata(conn, ev.event_id, ev.metrics)
                print('- metrics succeeded')

            succeeded += 1

        except Exception as e:
            print(f"[WARN] Failed to insert {qml_path}: {e}")
            print('\nQuakeML file:\n')
            os.system(f'cat {qml_path}')
            print('\nJSON file:\n')
            os.system(f"cat {qml_path.replace('.qml','.json')}")
            failed += 1

        if i % 100 == 0:
            print(f"[Progress] {i}/{len(qml_files)} | Success: {succeeded}, Failed: {failed}")
            conn.commit()

    conn.commit()
    conn.close()
    print(f"[DONE] Inserted {succeeded} events, {failed} failed, from {len(qml_files)} QML files.")


if __name__ == "__main__":
    main()
