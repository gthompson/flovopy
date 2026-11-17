"""
clock.py
--------

Correct legacy timestamps to UTC with an explicit, auditable table of clock
regimes.

Legacy observatory data often carry times from clocks that were set to local
time, changed zone part-way through a deployment, or were misread by software
(for example DOS ``mktime()`` with the default ``TZ``). Instead of scattering
offsets through processing code, describe each period once in a table and
apply it wherever a time is read.

Table columns (CSV or DataFrame):

``stream``        Semicolon-separated stream names the row applies to.
``regime``        Short label recorded with every corrected time.
``clock``         Name of the clock the raw times are on (documentation).
``start``/``end`` Half-open interval ``[start, end)`` on the raw clock;
                  blank means unbounded.
``seq_min``/``seq_max``
                  Optional inclusive bounds on a sortable sequence key
                  (file name, line number...). Use them where a clock was set
                  back and raw times repeat, so that file order decides.
``utc_offset_h``  Hours added to the raw time to obtain UTC.
``confidence``    Free text (e.g. high/medium/low).
``evidence``      Free text.

A lookup that matches no row is ``unresolved``; one that matches rows with
different offsets is ``ambiguous``. Both return no correction, so callers can
flag the record instead of guessing.

Main functionality
------------------
- dos_mktime_wall_clock: recover a wall-clock reading from a mktime() epoch.
- ClockRegimes: load, validate and apply a regime table.
"""

from __future__ import annotations

import math
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from obspy import UTCDateTime

__all__ = ["dos_mktime_wall_clock", "ClockRegimes", "REGIME_COLUMNS"]

REGIME_COLUMNS = ["stream", "regime", "clock", "start", "end", "seq_min", "seq_max",
                  "utc_offset_h", "confidence", "evidence"]


def dos_mktime_wall_clock(t, tz: str = "America/Los_Angeles") -> UTCDateTime:
    """Return the wall-clock reading that a C/DOS ``mktime()`` turned into ``t``.

    Microsoft C and Borland runtimes assume ``TZ=PST8PDT`` when ``TZ`` is unset,
    so an epoch written by DOS software is the wall clock interpreted as US
    Pacific time. Converting the epoch back to that zone recovers the reading
    (returned as a naive UTCDateTime). Whether that reading was UTC, local time
    or something else is a separate, data-specific question.
    """
    t = UTCDateTime(t)
    local = t.datetime.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
    return UTCDateTime(local.replace(tzinfo=None))


def _time_or_none(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return UTCDateTime(text) if text else None


def _text_or_none(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


class ClockRegimes:
    """A validated clock-regime table (see module docstring)."""

    def __init__(self, table: pd.DataFrame):
        missing = [c for c in ("stream", "regime", "start", "end", "utc_offset_h") if c not in table]
        if missing:
            raise ValueError(f"Clock-regime table is missing columns: {missing}")
        rows = []
        for i, row in table.reset_index(drop=True).iterrows():
            start, end = _time_or_none(row.get("start")), _time_or_none(row.get("end"))
            if start is not None and end is not None and end <= start:
                raise ValueError(f"Row {i} ({row['regime']}): end must follow start")
            offset = float(row["utc_offset_h"])
            if not math.isfinite(offset):
                raise ValueError(f"Row {i} ({row['regime']}): utc_offset_h must be finite")
            streams = {s.strip() for s in str(row["stream"]).split(";") if s.strip()}
            if not streams:
                raise ValueError(f"Row {i}: stream is empty")
            rows.append(dict(streams=streams, regime=str(row["regime"]).strip(),
                             clock=_text_or_none(row.get("clock")), start=start, end=end,
                             seq_min=_text_or_none(row.get("seq_min")),
                             seq_max=_text_or_none(row.get("seq_max")),
                             offset_s=offset * 3600.0,
                             confidence=_text_or_none(row.get("confidence")),
                             evidence=_text_or_none(row.get("evidence"))))
        self.rows = rows
        self.table = table.copy()

    @classmethod
    def read_csv(cls, path) -> "ClockRegimes":
        table = pd.read_csv(Path(path).expanduser(), dtype=str, comment="#",
                            keep_default_na=False, skipinitialspace=True)
        return cls(table)

    @property
    def streams(self) -> set:
        return set().union(*(r["streams"] for r in self.rows)) if self.rows else set()

    def candidates(self, stream: str, t, seq=None) -> list:
        """Rows whose stream, time window and (if given) sequence bounds match."""
        t = UTCDateTime(t)
        seq = None if seq is None else str(seq)
        out = []
        for r in self.rows:
            if stream not in r["streams"]:
                continue
            if r["start"] is not None and t < r["start"]:
                continue
            if r["end"] is not None and t >= r["end"]:
                continue
            if r["seq_min"] is not None or r["seq_max"] is not None:
                if seq is None:
                    out.append(r)  # cannot be excluded; may make the result ambiguous
                    continue
                if r["seq_min"] is not None and seq < r["seq_min"]:
                    continue
                if r["seq_max"] is not None and seq > r["seq_max"]:
                    continue
            out.append(r)
        return out

    def lookup(self, stream: str, t, seq=None) -> dict:
        """Return {regime, offset_s, confidence, status}; offset_s is None unless resolved."""
        if stream not in self.streams:
            raise KeyError(f"No clock regimes for stream {stream!r}")
        rows = self.candidates(stream, t, seq)
        if not rows:
            return dict(regime="unresolved", offset_s=None, confidence=None, status="unresolved")
        offsets = {r["offset_s"] for r in rows}
        if len(offsets) > 1:
            return dict(regime="|".join(r["regime"] for r in rows), offset_s=None,
                        confidence=None, status="ambiguous")
        # Several rows with the same offset (e.g. overlapping documentation rows) agree.
        r = rows[0]
        return dict(regime=r["regime"], offset_s=r["offset_s"], confidence=r["confidence"],
                    status="resolved")

    def to_utc(self, stream: str, t, seq=None):
        """Return (UTCDateTime or None, lookup dict)."""
        info = self.lookup(stream, t, seq)
        if info["offset_s"] is None:
            return None, info
        return UTCDateTime(t) + info["offset_s"], info

    def apply(self, frame: pd.DataFrame, stream: str, time_column: str, *,
              seq_column: str | None = None, stream_column: str | None = None,
              raw_prefix: str = "raw_") -> pd.DataFrame:
        """Return a copy of ``frame`` with ``time_column`` converted to UTC.

        The original value is kept in ``raw_prefix + time_column``; columns
        ``clock_regime``, ``clock_offset_s``, ``clock_confidence`` and
        ``clock_status`` record the rule. Unresolved or ambiguous rows get an
        empty time and must be handled by the caller. ``stream_column`` allows
        per-row stream names. Applying twice raises ValueError.
        """
        raw_column = raw_prefix + time_column
        if raw_column in frame:
            raise ValueError(f"{raw_column} already present; times look corrected already")
        out = frame.copy()
        out.insert(out.columns.get_loc(time_column) + 1, raw_column, out[time_column])
        values, regimes, offsets, confidence, status = [], [], [], [], []
        for idx, raw in out[raw_column].items():
            name = out.at[idx, stream_column] if stream_column else stream
            seq = out.at[idx, seq_column] if seq_column else None
            if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
                values.append(None); regimes.append("no_time"); offsets.append(math.nan)
                confidence.append(None); status.append("no_time")
                continue
            t, info = self.to_utc(name, pd.Timestamp(raw).isoformat(), seq)
            values.append(t)
            regimes.append(info["regime"])
            offsets.append(math.nan if info["offset_s"] is None else info["offset_s"])
            confidence.append(info["confidence"])
            status.append(info["status"])
        out[time_column] = [None if t is None else t.isoformat() for t in values]
        out["clock_regime"] = regimes
        out["clock_offset_s"] = offsets
        out["clock_confidence"] = confidence
        out["clock_status"] = status
        return out
