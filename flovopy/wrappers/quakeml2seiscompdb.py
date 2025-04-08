# seiscomp_like_db.py

import sqlite3
from obspy.core.event import Catalog, Event, Origin, Magnitude, Pick, Arrival
from datetime import datetime
from typing import Optional

class SeisComPLikeDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Basic event table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event (
                public_id TEXT PRIMARY KEY,
                preferred_origin_id TEXT,
                preferred_magnitude_id TEXT,
                type TEXT,
                description TEXT
            );
        """)

        # Origin table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS origin (
                public_id TEXT PRIMARY KEY,
                event_id TEXT,
                time TEXT,
                latitude REAL,
                longitude REAL,
                depth REAL,
                evaluation_mode TEXT,
                FOREIGN KEY(event_id) REFERENCES event(public_id)
            );
        """)

        # Magnitude table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS magnitude (
                public_id TEXT PRIMARY KEY,
                origin_id TEXT,
                mag REAL,
                magnitude_type TEXT,
                FOREIGN KEY(origin_id) REFERENCES origin(public_id)
            );
        """)

        # Pick table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pick (
                public_id TEXT PRIMARY KEY,
                time TEXT,
                waveform_id TEXT,
                phase_hint TEXT
            );
        """)

        # Arrival table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arrival (
                origin_id TEXT,
                pick_id TEXT,
                phase TEXT,
                time_residual REAL,
                azimuth REAL,
                distance REAL,
                FOREIGN KEY(origin_id) REFERENCES origin(public_id),
                FOREIGN KEY(pick_id) REFERENCES pick(public_id)
            );
        """)

        self.conn.commit()

    def insert_catalog(self, catalog: Catalog):
        for event in catalog:
            self.insert_event(event)

    def insert_event(self, event: Event):
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO event (public_id, preferred_origin_id, preferred_magnitude_id, type, description)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.resource_id.id,
            event.preferred_origin_id or None,
            event.preferred_magnitude_id or None,
            event.event_type or None,
            (event.event_descriptions[0].text if event.event_descriptions else None)
        ))

        for origin in event.origins:
            cursor.execute("""
                INSERT OR IGNORE INTO origin (public_id, event_id, time, latitude, longitude, depth, evaluation_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                origin.resource_id.id,
                event.resource_id.id,
                origin.time.isoformat() if origin.time else None,
                origin.latitude,
                origin.longitude,
                origin.depth,
                origin.evaluation_mode
            ))

        for magnitude in event.magnitudes:
            cursor.execute("""
                INSERT OR IGNORE INTO magnitude (public_id, origin_id, mag, magnitude_type)
                VALUES (?, ?, ?, ?)
            """, (
                magnitude.resource_id.id,
                magnitude.origin_id.id if magnitude.origin_id else None,
                magnitude.mag,
                magnitude.magnitude_type
            ))

        for pick in event.picks:
            waveform_id = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}.{pick.waveform_id.channel_code}" if pick.waveform_id else None
            cursor.execute("""
                INSERT OR IGNORE INTO pick (public_id, time, waveform_id, phase_hint)
                VALUES (?, ?, ?, ?)
            """, (
                pick.resource_id.id,
                pick.time.isoformat() if pick.time else None,
                waveform_id,
                pick.phase_hint
            ))

        for arrival in event.origins[0].arrivals if event.origins else []:
            cursor.execute("""
                INSERT INTO arrival (origin_id, pick_id, phase, time_residual, azimuth, distance)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.origins[0].resource_id.id,
                arrival.pick_id.id if arrival.pick_id else None,
                arrival.phase,
                arrival.time_residual,
                arrival.azimuth,
                arrival.distance
            ))

        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    from obspy.core.event import read_events
    db = SeisComPLikeDB("test_seiscomp_like.sqlite")
    cat = read_events("example_quakeml.xml")  # Replace with a valid file
    db.insert_catalog(cat)
    db.close()
