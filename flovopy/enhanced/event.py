# flovopy/enhanced/event.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

from obspy.core.event import Event, Catalog, Comment, ResourceIdentifier
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

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "EnhancedEventMeta":
        # tolerate missing keys for forward/backward compatibility
        return cls(
            sfile_path=d.get("sfile_path"),
            wav_paths=d.get("wav_paths", []) or [],
            aef_path=d.get("aef_path"),
            trigger_window=d.get("trigger_window"),
            average_window=d.get("average_window"),
            metrics=d.get("metrics", {}) or {},
            traces=d.get("traces", []) or [],
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

    def __init__(
        self,
        *args,
        meta: Optional[EnhancedEventMeta] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.meta: EnhancedEventMeta = meta or EnhancedEventMeta()

    # -------- Convenience properties --------
    @property
    def event_id(self) -> str:
        rid = getattr(self, "resource_id", None)
        if isinstance(rid, ResourceIdentifier) and rid.id:
            return rid.id
        # Ensure the event always has a ResourceIdentifier (useful for DB keys)
        self.resource_id = self.resource_id or ResourceIdentifier()
        return self.resource_id.id or str(self.resource_id)

    # --------- Sidecar persistence ---------
    def to_json(self) -> Dict[str, Any]:
        """
        JSON representation of enhanced pieces (not the QuakeML event itself).
        """
        return {
            "event_id": self.event_id,
            **self.meta.to_json_dict(),
        }

    def save(self, outdir: str, base_name: str) -> Tuple[str, str]:
        """
        Save QuakeML (.qml) and sidecar JSON (.json).
        Returns (qml_path, json_path).
        """
        os.makedirs(outdir, exist_ok=True)
        qml_path = os.path.join(outdir, base_name + ".qml")
        json_path = os.path.join(outdir, base_name + ".json")

        # QuakeML (standards-compliant)
        Catalog(events=[self]).write(qml_path, format="QUAKEML")

        # Sidecar JSON (all the extra bits)
        with open(json_path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)

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