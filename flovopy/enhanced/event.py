# flovopy/enhanced/event.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from obspy.core.event import (
    Event, Catalog, Comment, ResourceIdentifier,
    Origin, OriginQuality, CreationInfo, Magnitude
)
from obspy.core.utcdatetime import UTCDateTime
from obspy import read_events

# -------------------------
# Sidecar metadata container
# -------------------------
@dataclass
class EnhancedEventMeta:
    # File provenance
    sfile_path: Optional[str] = None
    wav_paths: List[str] = field(default_factory=list)
    aef_path: Optional[str] = None

    # Processing parameters
    trigger_window: Optional[float] = None
    average_window: Optional[float] = None

    # Arbitrary event-level metrics/classifications (flat JSON-serializable)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Optional per-trace summaries to persist alongside QuakeML
    # Each item is a flat dict (id, net, sta, loc, cha, starttime, distance_m, metrics{...}, spectral{...}, etc.)
    traces: List[Dict[str, Any]] = field(default_factory=list)

    asl_source: Dict[str, Any] = field(default_factory=dict)  # {"t":[], "lat":[], ...}

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "EnhancedEventMeta":
        return cls(
            sfile_path=d.get("sfile_path"),
            wav_paths=d.get("wav_paths", []) or [],
            aef_path=d.get("aef_path"),
            trigger_window=d.get("trigger_window"),
            average_window=d.get("average_window"),
            metrics=d.get("metrics", {}) or {},
            traces=d.get("traces", []) or [],
            asl_source=d.get("asl_source", {}) or {},   # ← add this
        )


# -------------------------
# EnhancedEvent subclassing ObsPy's Event
# -------------------------
class EnhancedEvent(Event):
    """
    An ObsPy Event with extra, non-QuakeML metadata stored in `self.meta`
    and convenience helpers for persistence and DB export.

    NOTE: Extra fields are NOT serialized into QuakeML; they’re saved to a
    JSON sidecar so QuakeML remains standards-compliant.
    """

    def __init__(self, *args, meta: Optional[EnhancedEventMeta] = None, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_meta", meta or EnhancedEventMeta())

    @property
    def meta(self) -> EnhancedEventMeta:
        return object.__getattribute__(self, "_meta")

    @meta.setter
    def meta(self, value: EnhancedEventMeta) -> None:
        object.__setattr__(self, "_meta", value)

    # -------- Convenience properties --------
    @property
    def event_id(self) -> str:
        rid = getattr(self, "resource_id", None)
        if isinstance(rid, ResourceIdentifier) and rid.id:
            return rid.id
        # Ensure the event always has a ResourceIdentifier (useful for DB keys)
        self.resource_id = self.resource_id or ResourceIdentifier()
        return self.resource_id.id or str(self.resource_id)

    @classmethod
    def wrap(cls, obspy_event: Event, *, meta: Optional[EnhancedEventMeta] = None, stream=None) -> "EnhancedEvent":
        ev = cls(meta=meta)
        # copy core attributes from obspy_event into our subclass
        for k, v in obspy_event.__dict__.items():
            setattr(ev, k, v)
        # attach ephemeral extras if you want
        ev.stream = stream
        return ev

    # --------- Sidecar persistence ---------
    def to_json(self) -> Dict[str, Any]:
        """
        JSON representation of enhanced pieces (not the QuakeML event itself).
        """
        return {
            "event_id": self.event_id,
            **self.meta.to_json_dict(),
        }

    def _extract_trajectory(self) -> list[dict]:
        """
        Compact origin trajectory for sidecar JSON.
        """
        traj = []
        for o in (self.origins or []):
            try:
                traj.append({
                    "time": (o.time.datetime.isoformat() if o.time else None),
                    "latitude": o.latitude,
                    "longitude": o.longitude,
                    "depth": o.depth,
                    "origin_id": getattr(o.resource_id, "id", None),
                })
            except Exception:
                continue
        # Stable order by time if present
        traj.sort(key=lambda r: r["time"] or "")
        return traj

    def save(
        self,
        outdir: str,
        base_name: str,
        *,
        write_quakeml: bool = True,
        write_obspy_json: bool = False,
        include_trajectory_in_sidecar: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        os.makedirs(outdir, exist_ok=True)
        qml_path = os.path.join(outdir, base_name + ".qml") if write_quakeml else None
        json_path = os.path.join(outdir, base_name + ".json")

        if write_quakeml:
            Catalog(events=[self]).write(qml_path, format="QUAKEML")

        # build sidecar payload
        payload = self.to_json()

        if include_trajectory_in_sidecar:
            # Prefer ASL trajectory if present
            src = getattr(self.meta, "asl_source", None) or (self.meta.metrics or {}).get("asl_series")
            if src:
                payload["asl_source"] = src
            else:
                # Fall back to building from self.origins
                payload["trajectory"] = self._extract_trajectory()

        # optionally also dump ObsPy JSON for the event
        if write_obspy_json:
            obsjson = os.path.join(outdir, base_name + ".obs.json")
            Catalog(events=[self]).write(obsjson, format="JSON")

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        return qml_path, json_path

    @classmethod
    def load(cls, base_path: str) -> "EnhancedEvent":
        """
        Load from `{base_path}.qml` + `{base_path}.json`.
        """
        qml_file = base_path + ".qml"
        json_file = base_path + ".json"
        if not os.path.exists(qml_file):
            raise FileNotFoundError(f"Missing QuakeML file: {qml_file}")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Missing JSON metadata file: {json_file}")

        cat = read_events(qml_file)
        if len(cat) == 0:
            raise ValueError(f"No events found in {qml_file}")
        obspy_event: Event = cat[0]

        with open(json_file, "r") as f:
            meta = EnhancedEventMeta.from_json_dict(json.load(f))

        # Wrap the ObsPy Event into our subclass
        # Easiest path: create new EnhancedEvent and copy attributes.
        ev = cls(meta=meta)
        # copy core attributes (ObsPy Event isn't a dataclass, so shallow copy fields)
        for attr, val in obspy_event.__dict__.items():
            setattr(ev, attr, val)
        return ev

    # --------- Stream / trace summaries integration ---------
    def attach_trace_summaries(self, traces: List[Dict[str, Any]]) -> None:
        """
        Attach a list of per-trace summaries (flat dicts).
        If you already have an EnhancedStream, expose a method there that returns these dicts.
        """
        self.meta.traces = traces or []

    def append_trace_summary(self, trace_summary: Dict[str, Any]) -> None:
        self.meta.traces.append(trace_summary)

    # --------- Lightweight classification tagging ---------
    def add_classification(self, label: str, **attrs: Any) -> None:
        """
        Store a classification label in sidecar metrics and add a human-readable
        comment into the ObsPy Event (visible in QuakeML viewers).
        """
        self.meta.metrics.setdefault("classifications", []).append({"label": label, **attrs})
        try:
            self.comments.append(Comment(text=f"[class] {label} {attrs}"))
        except Exception:
            pass

    # --------- DB export (trace metrics) ---------
    def write_to_db(self, conn) -> None:
        """
        Write per-trace metrics (if present in self.meta.traces) to DB.
        Table expected: aef_metrics(event_id, trace_id, network, station, location, channel,
                                    starttime, distance_m, snr, peakamp, energy,
                                    peakf, meanf, skewness, kurtosis)
        """
        traces = self.meta.traces or []
        if not traces:
            return

        cur = conn.cursor()
        rows = 0
        for tr in traces:
            metrics  = tr.get("metrics", {}) or {}
            spectral = tr.get("spectral", {}) or {}
            cur.execute(
                """
                INSERT OR REPLACE INTO aef_metrics 
                (event_id, trace_id, network, station, location, channel, starttime,
                 distance_m, snr, peakamp, energy, peakf, meanf, skewness, kurtosis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.event_id,
                    tr.get("id"),
                    tr.get("network"),
                    tr.get("station"),
                    tr.get("location"),
                    tr.get("channel"),
                    tr.get("starttime"),
                    tr.get("distance_m"),
                    metrics.get("snr") or metrics.get("snr_std"),
                    metrics.get("peakamp"),
                    metrics.get("energy"),
                    metrics.get("peakf") or spectral.get("peakF"),
                    metrics.get("meanf"),
                    metrics.get("skewness"),
                    metrics.get("kurtosis"),
                ),
            )
            rows += 1
        conn.commit()
        print(f"[✓] Inserted {rows} trace metrics for event {self.event_id}")

    @classmethod
    def from_asl(
        cls,
        aslobj,
        *,
        event_id: Optional[str] = None,
        meta: Optional[EnhancedEventMeta] = None,
        inventory=None,
        stream=None,
        title_comment: Optional[str] = None,
    ) -> "EnhancedEvent":
        if aslobj.source is None:
            raise ValueError("ASL object has no source – run locate()/fast_locate() first.")

        src = aslobj.source

        # ---- pull exactly what fast_locate() stores
        lat = np.asarray(src.get("lat", []), dtype=float)
        lon = np.asarray(src.get("lon", []), dtype=float)
        # time: fast_locate uses "t" (UTCDateTime array). Fallback to "time" if present.
        tvec = src.get("t", src.get("time", None))
        DR   = np.asarray(src.get("DR", []), dtype=float)       # already scaled by 1e7
        mis  = np.asarray(src.get("misfit", []), dtype=float)
        azg  = np.asarray(src.get("azgap", []), dtype=float)
        nsta = np.asarray(src.get("nsta", []), dtype=float)
        node = np.asarray(src.get("node_index", []), dtype=float)
        conn = src.get("connectedness", None)

        if lat.size == 0 or lon.size == 0:
            raise ValueError("ASL produced no lat/lon samples.")

        # best time = argmax DR (as fast_locate() does for preferred node/time)
        try:
            t_idx = int(np.nanargmax(DR)) if DR.size else 0
        except Exception:
            t_idx = 0

        # ---- build all Origins (trajectory), using fast_locate outputs
        origins: list[Origin] = []
        for i in range(lat.size):
            if not (np.isfinite(lat[i]) and np.isfinite(lon[i])):
                continue
            o = Origin()
            o.latitude = float(lat[i])
            o.longitude = float(lon[i])
            # time coercion
            if tvec is not None:
                try:
                    ti = tvec[i]
                    if isinstance(ti, UTCDateTime):
                        o.time = ti
                    elif isinstance(ti, (float, int)):
                        o.time = UTCDateTime(float(ti))
                    else:
                        o.time = UTCDateTime(str(ti))
                except Exception:
                    pass
            # quality scaffolding; distances are optional and can be filled later
            o.quality = OriginQuality()
            origins.append(o)

        if not origins:
            raise ValueError("No finite origins could be constructed from ASL source.")

        # ---- EnhancedEvent shell
        ev = cls(meta=meta or EnhancedEventMeta())
        if event_id:
            ev.resource_id = ResourceIdentifier(event_id)
        if title_comment:
            try:
                ev.comments.append(Comment(text=title_comment))
            except Exception:
                pass

        ev.origins = origins
        # preferred origin aligned to argmax(DR) index among the kept origins
        # map global index -> kept list index
        kept_mask = np.isfinite(lat) & np.isfinite(lon)
        try:
            kept_indices = np.flatnonzero(kept_mask)
            if kept_indices.size:
                local_idx = int(np.where(kept_indices == t_idx)[0][0]) if t_idx in kept_indices else 0
                ev.preferred_origin_id = ev.origins[local_idx].resource_id
        except Exception:
            pass

        # ---- sidecar metrics (pipeline params + series, keyed exactly like fast_locate())
        ev.meta.metrics.update({
            "Q": getattr(aslobj, "Q", None),
            "peakf": getattr(aslobj, "peakf", None),
            "wave_speed_kms": getattr(aslobj, "wave_speed_kms", None),
            "assume_surface_waves": getattr(aslobj, "assume_surface_waves", None),
            "vsam_metric": getattr(aslobj, "metric", None),
            "best_time_index": t_idx,
            "best_DR": float(DR[t_idx]) if (DR.size and np.isfinite(DR[t_idx])) else None,
            "connectedness": conn,
        })

        def _tolist(x):
            try:
                return np.asarray(x).tolist()
            except Exception:
                return None

        series = {
            "t":      _tolist(tvec),
            "lat":    _tolist(lat),
            "lon":    _tolist(lon),
            "DR":     _tolist(DR),
            "misfit": _tolist(mis),
            "azgap":  _tolist(azg),
            "nsta":   _tolist(nsta),
            "node_index": _tolist(node),
        }
        ev.meta.asl_source = series                  # canonical location
        ev.meta.metrics["asl_series"] = series       # optional legacy mirror

        # Optional waveform paths → meta.wav_paths (if available)
        if stream is not None:
            try:
                paths = [getattr(tr.stats, "path", None) for tr in stream if hasattr(tr.stats, "path")]
                ev.meta.wav_paths = [p for p in paths if p]
            except Exception:
                pass

        try:
            ev.creation_info = CreationInfo(author="flovopy.asl", version="1")
        except Exception:
            pass

        return ev
    

    def trajectory_dataframe(self) -> pd.DataFrame:
        """
        One row per time step from fast_locate():
        columns: event_id, time_index, datetime, lat, lon, DR, misfit, azgap, nsta, node_index, is_preferred_time
        """
        series = (self.meta.metrics or {}).get("asl_series", {})
        # fall back to origins-only if series missing
        if not series:
            rows = []
            for i, o in enumerate(self.origins or []):
                rows.append({
                    "event_id": getattr(getattr(self, "resource_id", None), "id", None),
                    "time_index": i,
                    "datetime": (o.time.datetime if getattr(o, "time", None) else None),
                    "lat": getattr(o, "latitude", None),
                    "lon": getattr(o, "longitude", None),
                    "DR": None, "misfit": None, "azgap": None, "nsta": None, "node_index": None,
                    "is_preferred_time": (o.resource_id == getattr(self, "preferred_origin_id", None)),
                })
            return pd.DataFrame(rows)

        def _aslist(k): 
            v = series.get(k)
            return v if isinstance(v, list) else []
        t  = _aslist("t")
        la = _aslist("lat")
        lo = _aslist("lon")
        DR = _aslist("DR")
        mi = _aslist("misfit")
        ag = _aslist("azgap")
        ns = _aslist("nsta")
        ni = _aslist("node_index")

        n = max(map(len, [t, la, lo, DR, mi, ag, ns, ni])) if any([t, la, lo, DR, mi, ag, ns, ni]) else 0
        def pad(v, n): return (v + [None]*max(0, n-len(v))) if v else [None]*n
        t, la, lo, DR, mi, ag, ns, ni = map(lambda v: pad(v, n), [t, la, lo, DR, mi, ag, ns, ni])

        best_idx = (self.meta.metrics or {}).get("best_time_index", None)
        eid = getattr(getattr(self, "resource_id", None), "id", None)

        rows = []
        for i in range(n):
            # best-effort datetime
            try:
                dt = pd.to_datetime(t[i]) if t[i] is not None else None
            except Exception:
                dt = None
            rows.append({
                "event_id": eid,
                "time_index": i,
                "datetime": dt,
                "lat": la[i], "lon": lo[i],
                "DR": DR[i], "misfit": mi[i], "azgap": ag[i], "nsta": ns[i], "node_index": ni[i],
                "is_preferred_time": (i == best_idx),
            })
        return pd.DataFrame(rows)