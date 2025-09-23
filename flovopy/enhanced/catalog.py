# flovopy/enhanced/catalog.py
from __future__ import annotations

import os
import json
import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from obspy.core.event import (
    Catalog, Event, Origin, Magnitude, Comment, CreationInfo, UTCDateTime,
    Amplitude, StationMagnitude, WaveformStreamID
)
from obspy.core.event.catalog import read_events
from obspy import read  # for optional waveform loading

from flovopy.enhanced.event import EnhancedEvent
from flovopy.enhanced.eventrate import EventRate, EventRateConfig
from flovopy.core.physics import magnitude2Eseismic, Eseismic2magnitude


# ---------- Enhanced Catalog ----------

class EnhancedCatalog(Catalog):
    """
    Extended ObsPy Catalog for volcano-seismic workflows.

    - Holds ObsPy Events (self itself is a Catalog)
    - Keeps parallel metadata as EnhancedEvent "records" (per-event extras, file paths, etc.)
    - Provides cached DataFrame views and convenience filters
    - Bridges to EventRate (regularly sampled series) via to_event_rate(...)
    """
    def __init__(
        self,
        events: Optional[List[Event]] = None,
        records: Optional[List[EnhancedEvent]] = None,
        triggerParams: Optional[Dict[str, Any]] = None,
        starttime: Optional[UTCDateTime] = None,
        endtime: Optional[UTCDateTime] = None,
        comments: Optional[List[str]] = None,
        description: str = "",
        resource_id: Optional[str] = None,
        creation_info: Optional[CreationInfo] = None,
    ):
        super().__init__(events=events or [])
        self.records: List[EnhancedEvent] = records or []
        self.triggerParams: Dict[str, Any] = triggerParams or {}
        self.starttime = starttime
        self.endtime = endtime
        self.comments = comments or []
        self.description = description
        self._df_cache: Optional[pd.DataFrame] = None

        if resource_id:
            self.resource_id = resource_id
        if creation_info:
            self.creation_info = creation_info

    # ---------- Cached DataFrame ----------
    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df_cache is None:
            self.update_dataframe()
        return self._df_cache

    def update_dataframe(self, force: bool = False) -> None:
        if force or self._df_cache is None:
            self._df_cache = self.to_dataframe()

    # ---------- Mutators ----------
    def addEvent(
        self,
        event: Event,
        *,
        stream=None,
        trigger: Optional[Dict[str, Any]] = None,
        classification: Optional[str] = None,
        event_type: Optional[str] = None,
        mainclass: Optional[str] = None,
        subclass: Optional[str] = None,
        author: str = "EnhancedCatalog",
        agency_id: str = "MVO",
        sfile_path: Optional[str] = None,
        wav_paths: Optional[List[str]] = None,
        aef_path: Optional[str] = None,
        trigger_window: Optional[float] = None,
        average_window: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an ObsPy Event with optional extras and create an EnhancedEvent record.
        """
        self.append(event)

        # Decorate event with simple QuakeML metadata
        if event_type:
            event.event_type = event_type
            event.event_type_certainty = "suspected"
        if mainclass:
            event.comments.append(Comment(text=f"mainclass: {mainclass}"))
        if subclass:
            event.comments.append(Comment(text=f"subclass: {subclass}"))
        if classification:
            event.comments.append(Comment(text=f"classification: {classification}"))
        if not event.creation_info:
            event.creation_info = CreationInfo(author=author, agency_id=agency_id, creation_time=UTCDateTime())

        enh = EnhancedEvent(
            obspy_event=event,
            metrics=metrics or {
                "classification": classification,
                "event_type": event.event_type,
                "mainclass": mainclass,
                "subclass": subclass,
            },
            sfile_path=sfile_path,
            wav_paths=wav_paths or ([s.path for s in stream] if stream is not None else []),
            aef_path=aef_path,
            trigger_window=trigger_window or (trigger.get("duration") if trigger else None),
            average_window=average_window or (trigger.get("average_window") if trigger else None),
            stream=stream,
        )
        self.records.append(enh)
        # bust cache
        self._df_cache = None

    # ---------- Views ----------
    def get_times(self) -> List[UTCDateTime]:
        return [ev.origins[0].time for ev in self if ev.origins]

    def plot_streams(self) -> None:
        for i, enh in enumerate(self.records):
            if enh.stream:
                print(f"\nEVENT #{i+1}  time: {enh.stream[0].stats.starttime}\n")
                enh.stream.plot(equal_scale=False)

    # ---------- Merging ----------
    def concat(self, other: "EnhancedCatalog") -> None:
        for ev in other:
            self.append(ev)
        self.records.extend(other.records)
        self._df_cache = None

    def merge(self, other: "EnhancedCatalog") -> None:
        existing = {e.resource_id.id for e in self if e.resource_id}
        for enh in other.records:
            rid = enh.event.resource_id.id if enh.event.resource_id else None
            if rid not in existing:
                self.append(enh.event)
                self.records.append(enh)
        self._df_cache = None

    # ---------- Summaries ----------
    def summary(self) -> None:
        df = self.to_dataframe()
        if df.empty:
            print("Empty catalog.")
            return
        print("\nEvent Type Summary:")
        print(df["event_type"].value_counts(dropna=False))
        print("\nMainclass Summary:")
        print(df["mainclass"].value_counts(dropna=False))
        print("\nSubclass Summary:")
        print(df["subclass"].value_counts(dropna=False))

    # ---------- Dataframe serialization ----------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Rows include: datetime, magnitude, lat, lon, depth, energy (if magnitude present),
        event_type, mainclass, subclass, classification, filename (first wav).
        """
        rows = []
        for enh in self.records:
            ev = enh.event
            row = {
                "datetime": ev.origins[0].time.datetime if ev.origins else None,
                "magnitude": ev.magnitudes[0].mag if ev.magnitudes else None,
                "latitude": ev.origins[0].latitude if ev.origins else None,
                "longitude": ev.origins[0].longitude if ev.origins else None,
                "depth": ev.origins[0].depth if ev.origins else None,
                "event_type": getattr(ev, "event_type", None),
                "classification": None,
                "mainclass": None,
                "subclass": None,
                "filename": enh.wav_paths[0] if enh.wav_paths else None,
            }
            # pull labels from comments if present
            for c in (ev.comments or []):
                txt = (c.text or "")
                if txt.startswith("mainclass:"):
                    row["mainclass"] = txt.split(":", 1)[-1].strip()
                elif txt.startswith("subclass:"):
                    row["subclass"] = txt.split(":", 1)[-1].strip()
                elif txt.startswith("classification:") and not row["classification"]:
                    row["classification"] = txt.split(":", 1)[-1].strip()

            if row["magnitude"] is not None:
                row["energy"] = magnitude2Eseismic(row["magnitude"])
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def export_csv(self, filepath: str) -> None:
        self.to_dataframe().to_csv(filepath, index=False)

    # ---------- Persistence ----------
    def save(self, outdir: str, outfile: str, net: str = "MV") -> None:
        """
        Write a catalog-wide QuakeML plus per-event JSON/QuakeML via EnhancedEvent.save.
        Also writes a small JSON summary with trigger params, description, etc.
        """
        self._write_events(outdir, net=net, xmlfile=outfile + ".xml")

        summary_json = os.path.join(outdir, outfile + "_meta.json")
        summary_vars = {
            "triggerParams": self.triggerParams,
            "comments": self.comments,
            "description": self.description,
            "starttime": self.starttime.strftime("%Y/%m/%d %H:%M:%S") if self.starttime else None,
            "endtime": self.endtime.strftime("%Y/%m/%d %H:%M:%S") if self.endtime else None,
        }
        os.makedirs(outdir, exist_ok=True)
        with open(summary_json, "w") as f:
            json.dump(summary_vars, f, indent=2, default=str)

    def _write_events(self, outdir: str, net: str = "MV", xmlfile: Optional[str] = None) -> None:
        if xmlfile:
            os.makedirs(outdir, exist_ok=True)
            self.write(os.path.join(outdir, xmlfile), format="QUAKEML")

        for enh in self.records:
            ev = enh.event
            if not ev.origins:
                continue
            t = ev.origins[0].time
            if not t:
                continue
            base_path = os.path.join(
                outdir, "WAV", net, t.strftime("%Y"), t.strftime("%m"),
                t.strftime("%Y%m%dT%H%M%S")
            )
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            enh.save(os.path.dirname(base_path), os.path.basename(base_path))

    # ---------- Filters (return new EnhancedCatalogs) ----------
    def filter_by_event_type(self, event_type: str) -> "EnhancedCatalog":
        recs = [r for r in self.records if getattr(r.event, "event_type", None) == event_type]
        return EnhancedCatalog(events=[r.event for r in recs], records=recs)

    def filter_by_mainclass(self, mainclass: str) -> "EnhancedCatalog":
        def has_mc(ev: Event) -> bool:
            return any((c.text or "").startswith("mainclass:") and mainclass in c.text for c in (ev.comments or []))
        recs = [r for r in self.records if has_mc(r.event)]
        return EnhancedCatalog(events=[r.event for r in recs], records=recs)

    def filter_by_subclass(self, subclass: str) -> "EnhancedCatalog":
        def has_sc(ev: Event) -> bool:
            return any((c.text or "").startswith("subclass:") and subclass in c.text for c in (ev.comments or []))
        recs = [r for r in self.records if has_sc(r.event)]
        return EnhancedCatalog(events=[r.event for r in recs], records=recs)

    def group_by(self, field: str) -> Dict[Any, "EnhancedCatalog"]:
        """
        Group records by a classification field: 'event_type', 'mainclass', or 'subclass'.
        """
        grouped: Dict[Any, List[EnhancedEvent]] = defaultdict(list)
        for r in self.records:
            val = None
            if field == "event_type":
                val = getattr(r.event, "event_type", None)
            else:
                for c in (r.event.comments or []):
                    txt = (c.text or "")
                    if txt.startswith(f"{field}:"):
                        val = txt.split(":", 1)[-1].strip()
                        break
            grouped[val].append(r)

        return {
            key: EnhancedCatalog(events=[rr.event for rr in recs], records=recs)
            for key, recs in grouped.items()
        }

    # ---------- Builders ----------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, load_waveforms: bool = False) -> "EnhancedCatalog":
        records: List[EnhancedEvent] = []

        for _, row in df.iterrows():
            ev = Event()

            # origin
            if not pd.isna(row.get("datetime")) and not pd.isna(row.get("latitude")):
                ev.origins = [Origin(
                    time=UTCDateTime(pd.to_datetime(row["datetime"])),
                    latitude=row["latitude"], longitude=row.get("longitude"),
                    depth=row.get("depth")
                )]

            # magnitude
            if not pd.isna(row.get("magnitude")):
                ev.magnitudes = [Magnitude(mag=row["magnitude"])]

            # labels
            if not pd.isna(row.get("event_type")):
                ev.event_type = row["event_type"]
            for key in ("mainclass", "subclass", "classification"):
                if key in row and not pd.isna(row[key]):
                    ev.comments.append(Comment(text=f"{key}: {row[key]}"))

            stream = None
            fn = row.get("filename")
            wav_paths = [fn] if isinstance(fn, str) else []

            if load_waveforms and isinstance(fn, str) and os.path.exists(fn):
                try:
                    stream = read(fn)
                except Exception:
                    stream = None

            records.append(EnhancedEvent(
                obspy_event=ev,
                stream=stream,
                wav_paths=wav_paths,
                metrics={}
            ))

        return cls(events=[r.event for r in records], records=records)

    @classmethod
    def load_dir(cls, catdir: str, *, load_waveforms: bool = False) -> "EnhancedCatalog":
        """
        Build EnhancedCatalog from per-event .qml + .json pairs created by EnhancedEvent.save().
        """
        qml_files = sorted(glob.glob(os.path.join(catdir, "**", "*.qml"), recursive=True))
        records: List[EnhancedEvent] = []

        for qml_file in qml_files:
            base = os.path.splitext(qml_file)[0]
            json_file = base + ".json"
            if not os.path.exists(json_file):
                continue

            try:
                ev = read_events(qml_file)[0]
                with open(json_file, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[WARN] Skipping {base}: {e}")
                continue

            wav_paths = meta.get("wav_paths", [])
            stream = None
            if load_waveforms and wav_paths and os.path.exists(wav_paths[0]):
                try:
                    stream = read(wav_paths[0])
                except Exception as e:
                    print(f"[WARN] Could not load waveform {wav_paths[0]}: {e}")

            records.append(EnhancedEvent(
                obspy_event=ev,
                metrics=meta.get("metrics", {}),
                sfile_path=meta.get("sfile_path"),
                wav_paths=wav_paths,
                aef_path=meta.get("aef_path"),
                trigger_window=meta.get("trigger_window"),
                average_window=meta.get("average_window"),
                stream=stream
            ))

        return cls(events=[r.event for r in records], records=records)

    # ---------- EventRate bridge ----------
    def to_event_rate(self, config: Optional[EventRateConfig] = None) -> EventRate:
        """
        Build a regularly sampled event-rate view from this catalog.
        """
        return EventRate.from_catalog(self, config=config)