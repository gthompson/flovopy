# flovopy/enhanced/catalog.py
from __future__ import annotations

import os
import json
import glob
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from obspy.core.event import (
    Catalog, Event, Origin, Magnitude, Comment, CreationInfo,
)
from obspy.core.event.catalog import read_events
from obspy import read, UTCDateTime  # for optional waveform loading
from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta
from flovopy.enhanced.eventrate import EventRate, EventRateConfig
from flovopy.core.physics import magnitude2Eseismic


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

        meta = EnhancedEventMeta(
            sfile_path=sfile_path,
            wav_paths=wav_paths or ([s.path for s in stream] if stream is not None else []),
            aef_path=aef_path,
            trigger_window=trigger_window or (trigger.get("duration") if trigger else None),
            average_window=average_window or (trigger.get("average_window") if trigger else None),
            metrics=metrics or {
                "classification": classification,
                "event_type": event.event_type,
                "mainclass": mainclass,
                "subclass": subclass,
            },
        )

        # Wrap and use the SAME object for both Catalog events and records
        enh = EnhancedEvent.wrap(event, meta=meta, stream=stream)
        self.append(enh)
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
            rid = enh.resource_id.id if enh.resource_id else None
            if rid not in existing:
                self.append(enh)          # enh is an Event
                self.records.append(enh)  # keep the same object in records
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
    def to_dataframe(self, origin_mode: str = "preferred") -> pd.DataFrame:
        """
        Build an event-level summary DataFrame.

        Parameters
        ----------
        origin_mode : {"preferred", "first", "all"}, default="preferred"
            Which origin(s) to export for each event:
            - "preferred" : one row per event using the preferred_origin
                            (falls back to the first origin if none set).
            - "first"     : one row per event using the first origin, if any.
            - "all"       : one row per *origin* (multi-rows per event).

        Returns
        -------
        df : pandas.DataFrame
            Columns: resource_id, datetime, latitude, longitude, depth, event_type,
            classification, mainclass, subclass, filename, magnitude, energy,
            DR, misfit, azgap, nsta, node_index, connectedness, connectedness_score,
            best_time_index, best_DR.
        """
        rows = []

        for ev in self.records:  # EnhancedEvent (subclass of ObsPy Event)
            rid = getattr(getattr(ev, "resource_id", None), "id", None)

            # Pull ASL series (if present) + summary metrics
            meta_src = (getattr(ev.meta, "asl_source", None)
                        or (ev.meta.metrics or {}).get("asl_series", {})
                        or {})
            best_idx = (ev.meta.metrics or {}).get("best_time_index")
            best_DR  = (ev.meta.metrics or {}).get("best_DR")
            connectedness = (ev.meta.metrics or {}).get("connectedness")
            conn_score = None
            try:
                if isinstance(connectedness, dict):
                    conn_score = connectedness.get("score", None)
            except Exception:
                pass

            # Helper to get per-time ASL values safely
            def asl_at(idx: int | None) -> dict:
                try:
                    if idx is None:
                        return {"DR": None, "misfit": None, "azgap": None, "nsta": None, "node_index": None}
                    DRv     = meta_src.get("DR", [])
                    misfitv = meta_src.get("misfit", [])
                    azgv    = meta_src.get("azgap", [])
                    nstav   = meta_src.get("nsta", [])
                    nodev   = meta_src.get("node_index", [])
                    # Bounds check
                    def pick(v):
                        return v[idx] if (hasattr(v, "__len__") and idx < len(v)) else None
                    return {
                        "DR": pick(DRv),
                        "misfit": pick(misfitv),
                        "azgap": pick(azgv),
                        "nsta": pick(nstav),
                        "node_index": pick(nodev),
                    }
                except Exception:
                    return {"DR": None, "misfit": None, "azgap": None, "nsta": None, "node_index": None}

            def _row_from_origin(o, idx_for_asl: int | None):
                row = {
                    "resource_id": rid,
                    "datetime": (o.time.datetime if getattr(o, "time", None) else None),
                    "latitude": getattr(o, "latitude", None),
                    "longitude": getattr(o, "longitude", None),
                    "depth": getattr(o, "depth", None),
                    "event_type": getattr(ev, "event_type", None),
                    "classification": None,
                    "mainclass": None,
                    "subclass": None,
                    "filename": (ev.meta.wav_paths[0] if getattr(ev.meta, "wav_paths", None) else None),
                    "magnitude": (ev.magnitudes[0].mag if ev.magnitudes else None),
                    "connectedness": connectedness,
                    "connectedness_score": conn_score,
                    "best_time_index": best_idx,
                    "best_DR": best_DR,
                }
                # labels from comments
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

                # attach ASL time-aligned values
                row.update(asl_at(idx_for_asl))
                return row

            if origin_mode == "all":
                # map each origin to its index; bounds checks handled in asl_at()
                for i, o in enumerate(ev.origins or []):
                    rows.append(_row_from_origin(o, i))
            else:
                origin = None
                idx_for_asl = None
                if origin_mode == "preferred":
                    try:
                        origin = ev.preferred_origin()
                    except Exception:
                        origin = getattr(ev, "preferred_origin", None)
                    # guard best_idx to be int and non-negative
                    if isinstance(best_idx, (int, np.integer)) and best_idx >= 0:
                        idx_for_asl = int(best_idx)
                if origin is None and ev.origins:
                    origin = ev.origins[0]
                    if idx_for_asl is None:
                        idx_for_asl = 0

                rows.append(_row_from_origin(origin or type("O", (), {})(), idx_for_asl))

        df = pd.DataFrame(rows)
        if not df.empty and "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def export_csv(
        self,
        filepath: str,
        *,
        df: Optional[pd.DataFrame] = None,
        origin_mode: str = "preferred"
    ) -> None:
        """
        Export the catalog to CSV.

        Parameters
        ----------
        filepath : str
            Path to output CSV file.
        df : pandas.DataFrame, optional
            If provided, write this DataFrame as-is. Otherwise,
            call `to_dataframe(origin_mode=origin_mode)`.
        origin_mode : {"preferred", "first", "all"}, default="preferred"
            Passed to `to_dataframe()` if `df` is not provided.

        Examples
        --------
        >>> # Export summary with preferred origins
        >>> cat.export_csv("events_summary.csv", origin_mode="preferred")

        >>> # Export summary with first origins
        >>> cat.export_csv("events_first_origin.csv", origin_mode="first")

        >>> # Export summary with all origins
        >>> cat.export_csv("events_all_origins.csv", origin_mode="all")

        >>> # Export trajectory dataframe instead
        >>> df_traj = cat.trajectory_dataframe()
        >>> cat.export_csv("trajectory.csv", df=df_traj)
        """
        (df if df is not None else self.to_dataframe(origin_mode=origin_mode)).to_csv(filepath, index=False)


    def trajectory_dataframe(self) -> pd.DataFrame:
        """
        Build a long-form DataFrame capturing *all origin trajectories*
        from ASL-derived events.

        Each row corresponds to a single origin within an event,
        preserving the full trajectory (time series of positions, DR, misfit, etc.)
        that `fast_locate()` produces.

        Returns
        -------
        df : pandas.DataFrame
            Columns include:
            - resource_id : str (event identifier)
            - datetime    : UTC datetime of origin
            - latitude    : float
            - longitude   : float
            - DR          : float, reduced displacement
            - misfit      : float, misfit at chosen node
            - azgap       : float, azimuthal gap
            - nsta        : int, usable station count
            - node_index  : int, chosen node index
            - energy      : float, if magnitude present
        """
        rows = []
        for enh in self.records:  # enh is an EnhancedEvent (subclass of Event)
            rid = getattr(getattr(enh, "resource_id", None), "id", None)
            meta_src = (
                getattr(enh.meta, "asl_source", None)
                or (enh.meta.metrics or {}).get("asl_series", {})
                or {}
            )

            # ASL trajectory arrays
            t_arr   = np.asarray(meta_src.get("t", []), dtype="datetime64[ns]")
            lat_arr = np.asarray(meta_src.get("lat", []), dtype=float)
            lon_arr = np.asarray(meta_src.get("lon", []), dtype=float)
            DR_arr  = np.asarray(meta_src.get("DR", []), dtype=float)
            misfit  = np.asarray(meta_src.get("misfit", []), dtype=float)
            azgap   = np.asarray(meta_src.get("azgap", []), dtype=float)
            nsta    = np.asarray(meta_src.get("nsta", []), dtype=int)
            nodes   = np.asarray(meta_src.get("node_index", []), dtype=int)

            for i in range(len(t_arr)):
                rows.append({
                    "resource_id": rid,
                    "datetime": pd.to_datetime(str(t_arr[i])) if t_arr.size else None,
                    "latitude": lat_arr[i] if i < lat_arr.size else None,
                    "longitude": lon_arr[i] if i < lon_arr.size else None,
                    "DR": DR_arr[i] if i < DR_arr.size else None,
                    "misfit": misfit[i] if i < misfit.size else None,
                    "azgap": azgap[i] if i < azgap.size else None,
                    "nsta": int(nsta[i]) if i < nsta.size else None,
                    "node_index": int(nodes[i]) if i < nodes.size else None,
                    "energy": magnitude2Eseismic(enh.magnitudes[0].mag)
                            if enh.magnitudes else None,
                })

        return pd.DataFrame(rows)
    
    def export_trajectory_csv(self, path: str) -> None:
        self.trajectory_dataframe().to_csv(path, index=False)

    # ---------- Persistence ----------
    def save(
        self,
        outdir: str,
        outfile: str,
        *,
        net: str = "MV",
        write_catalog_xml: bool = True,
        write_catalog_json: bool = False,
        write_individual_events: bool = True,
        write_summary_json: bool = True,
        csv_path: Optional[str] = None,
        # passthrough to per-event writes:
        write_event_quakeml: bool = True,
        write_event_obspy_json: bool = False,
        include_trajectory_in_sidecar: bool = True,
    ) -> None:
        os.makedirs(outdir, exist_ok=True)

        if write_catalog_xml:
            # Includes ALL origins for each event
            self.write(os.path.join(outdir, outfile + ".xml"), format="QUAKEML")

        if write_catalog_json:
            self.write(os.path.join(outdir, outfile + ".json"), format="JSON")

        if write_summary_json:
            summary_json = os.path.join(outdir, outfile + "_meta.json")
            summary_vars = {
                "triggerParams": self.triggerParams,
                "comments": self.comments,
                "description": self.description,
                "starttime": self.starttime.strftime("%Y/%m/%d %H:%M:%S") if self.starttime else None,
                "endtime": self.endtime.strftime("%Y/%m/%d %H:%M:%S") if self.endtime else None,
            }
            with open(summary_json, "w") as f:
                json.dump(summary_vars, f, indent=2, default=str)

        if write_individual_events:
            self._write_individual_events(
                outdir,
                net=net,
                write_quakeml=write_event_quakeml,
                write_obspy_json=write_event_obspy_json,
                include_trajectory_in_sidecar=include_trajectory_in_sidecar,
            )

        if csv_path:
            self.export_csv(csv_path)


    def _write_individual_events(
        self,
        outdir: str,
        *,
        net: str = "MV",
        write_quakeml: bool = True,
        write_obspy_json: bool = False,
        include_trajectory_in_sidecar: bool = True,
    ) -> None:
        for ev in self.records:  # ev is an EnhancedEvent (subclass of Event)
            if not ev.origins or not getattr(ev.origins[0], "time", None):
                continue
            t = ev.origins[0].time
            base = os.path.join(
                outdir, "WAV", net, t.strftime("%Y"), t.strftime("%m"),
                t.strftime("%Y%m%dT%H%M%S")
            )
            os.makedirs(os.path.dirname(base), exist_ok=True)
            ev.save(  # call EnhancedEvent.save()
                os.path.dirname(base),
                os.path.basename(base),
                write_quakeml=write_quakeml,
                write_obspy_json=write_obspy_json,
                include_trajectory_in_sidecar=include_trajectory_in_sidecar,
            )

    # ---------- Filters (return new EnhancedCatalogs) ----------
    def filter_by_event_type(self, event_type: str) -> "EnhancedCatalog":
        recs = [r for r in self.records if getattr(r, "event_type", None) == event_type]
        return EnhancedCatalog(events=[r for r in recs], records=recs)


    def filter_by_mainclass(self, mainclass: str) -> "EnhancedCatalog":
        def has_mc(ev: Event) -> bool:
            return any((c.text or "").startswith("mainclass:") and mainclass in (c.text or "")
                    for c in (ev.comments or []))
        recs = [r for r in self.records if has_mc(r)]
        return EnhancedCatalog(events=[r for r in recs], records=recs)


    def filter_by_subclass(self, subclass: str) -> "EnhancedCatalog":
        def has_sc(ev: Event) -> bool:
            return any((c.text or "").startswith("subclass:") and subclass in (c.text or "")
                    for c in (ev.comments or []))
        recs = [r for r in self.records if has_sc(r)]
        return EnhancedCatalog(events=[r for r in recs], records=recs)


    def group_by(self, field: str) -> Dict[Any, "EnhancedCatalog"]:
        """
        Group records by a classification field: 'event_type', 'mainclass', or 'subclass'.
        """
        grouped: Dict[Any, List[EnhancedEvent]] = defaultdict(list)
        for r in self.records:
            val = None
            if field == "event_type":
                val = getattr(r, "event_type", None)
            else:
                for c in (r.comments or []):
                    txt = (c.text or "")
                    if txt.startswith(f"{field}:"):
                        val = txt.split(":", 1)[-1].strip()
                        break
            grouped[val].append(r)

        return {
            key: EnhancedCatalog(events=[rr for rr in recs], records=recs)
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
                    latitude=row["latitude"],
                    longitude=row.get("longitude"),
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

            meta = EnhancedEventMeta(
                wav_paths=wav_paths,
                metrics={}
            )
            records.append(EnhancedEvent.wrap(ev, meta=meta, stream=stream))

        # EnhancedEvent IS an Event, so return the same objects for both lists.
        return cls(events=[r for r in records], records=records)
    


    @classmethod
    def load_dir(cls, catdir: str, *, load_waveforms: bool = False) -> "EnhancedCatalog":
        """
        Build EnhancedCatalog from per-event .qml + .json pairs created by EnhancedEvent.save().

        Notes
        -----
        - Uses EnhancedEvent.load(base_path) so that the sidecar (including `asl_source`)
        and all meta fields are reconstructed faithfully.
        - If `load_waveforms=True` and the sidecar contains `wav_paths`, the first path
        is read (if it exists) and attached as `enh.stream`.
        """
        qml_files = sorted(glob.glob(os.path.join(catdir, "**", "*.qml"), recursive=True))
        records: List[EnhancedEvent] = []

        for qml_file in qml_files:
            base = os.path.splitext(qml_file)[0]
            json_file = base + ".json"
            if not os.path.exists(json_file):
                continue

            try:
                enh = EnhancedEvent.load(base)  # brings back meta.asl_source, metrics, etc.
            except Exception as e:
                print(f"[WARN] Skipping {base}: {e}")
                continue

            # Optionally attach the first waveform listed in the sidecar
            if load_waveforms and enh.meta.wav_paths:
                fn = enh.meta.wav_paths[0]
                if isinstance(fn, str) and os.path.exists(fn):
                    try:
                        enh.stream = read(fn)
                    except Exception as e:
                        print(f"[WARN] Could not load waveform {fn}: {e}")

            records.append(enh)

        # EnhancedEvent is an Event, so use the same objects for both lists
        return cls(events=[r for r in records], records=records)

    # ---------- EventRate bridge ----------
    def to_event_rate(self, config: Optional[EventRateConfig] = None) -> EventRate:
        """
        Build a regularly sampled event-rate view from this catalog.
        """
        return EventRate.from_catalog(self, config=config)