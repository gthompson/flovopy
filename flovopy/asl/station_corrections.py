# flovopy/asl/station_corrections.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional, Callable, Mapping, Any, Sequence

import numpy as np
import pandas as pd
from obspy import Stream, Trace

# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
# This module provides two ways to estimate multiplicative station gains ("hot" stations):
#   1) Table-based:   estimate_station_gains_from_table(event×station amplitudes DataFrame)
#   2) Eventwise:     estimate_station_gains(events, reduce_fn=...) from Streams directly
#
# After estimating gains, convert them into a **long/tidy interval table**:
#      start_time | end_time | seed_id | gain | method | n_events | ...
#
# Then apply them to new events with:
#   apply_interval_station_gains(st, gains_df)
#
# A high-level wrapper closes the loop from raw events → interval CSV:
#   build_station_gains_from_events(...)
#
# Notes
# -----
# * A "seed_id" is expected to be "NET.STA.LOC.CHA".
# * The application supports smart fallback lookups, e.g. if exact channel is
#   missing you can include rows for "NET.STA..CHA", "NET.STA.*.CHA", "NET.STA" etc.
# * All interval times are treated as UTC. Timestamps may be tz-aware or naive
#   on input; they are normalized to UTC in the output CSV.
# * New in this version:
#     - enforce_calibrated_trace(): attempt response removal or counts→SI fallback; else drop.
#     - robust_amplitude_metric(): spike-aware, band-limited median/RMS helpers for table building.
#     - estimate_station_gains_from_table(): now enforces min_events_per_station strictly.
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# 0) Utilities: robust amplitude & calibration enforcement
# --------------------------------------------------------------------------------------
def robust_amplitude_metric(
    tr: Trace,
    *,
    metric: str = "median_env",           # "median_env" | "rms" | "abs_median"
    band: Optional[Tuple[float, float]] = None,   # e.g., (0.5, 8.0)
    spike_guard: bool = True,
    spike_z: float = 8.0,                 # z-score threshold for spike fraction
) -> Tuple[float, dict]:
    """
    Compute a robust amplitude proxy with optional bandpass & spike guard.

    Returns
    -------
    value : float (np.nan on failure)
    info  : dict { 'spike_fraction', 'metric', 'band' }

    Notes
    -----
    - 'median_env' is a good default for heterogeneous data.
    - 'rms' can overweight transients; median-based options are safer for gains.
    """
    try:
        x = np.asarray(tr.data, float)
        if not np.isfinite(x).any():
            return np.nan, {"spike_fraction": np.nan, "metric": metric, "band": band}

        # Optional bandpass
        trw = tr.copy()
        if band is not None:
            lo, hi = map(float, band)
            if hi > 0 and hi > lo:
                trw.filter("bandpass", freqmin=lo, freqmax=hi, corners=2, zerophase=True)
        x = np.asarray(trw.data, float)

        # Spike guard: crude but effective
        if spike_guard:
            x_f = x[np.isfinite(x)]
            if x_f.size >= 8:
                mu = np.nanmedian(x_f)
                mad = np.nanmedian(np.abs(x_f - mu)) + 1e-12
                z = (x_f - mu) / (1.4826 * mad)
                spike_fraction = float(np.mean(np.abs(z) > spike_z))
            else:
                spike_fraction = np.nan
        else:
            spike_fraction = np.nan

        # Metric
        if metric == "median_env":
            # envelope ≈ |analytic signal|; use Hilbert amplitude proxy
            # fall back to |x| median if hilbert not available
            try:
                from scipy.signal import hilbert
                env = np.abs(hilbert(x_f))
                val = float(np.nanmedian(env))
            except Exception:
                val = float(np.nanmedian(np.abs(x_f)))
        elif metric == "rms":
            val = float(np.sqrt(np.nanmean(x**2)))
        elif metric == "abs_median":
            val = float(np.nanmedian(np.abs(x)))
        else:
            raise ValueError(f"Unknown metric '{metric}'")

        return val, {"spike_fraction": spike_fraction, "metric": metric, "band": band}
    except Exception:
        return np.nan, {"spike_fraction": np.nan, "metric": metric, "band": band}


def enforce_calibrated_trace(
    tr: Trace,
    *,
    inventory,
    prefer_velocity: bool = True,
    calib_fallback: Optional[Dict[str, float]] = None,  # seed_id -> counts_to_(m/s or m)
    mean_thresh: float = 1.0,
    max_thresh: float = 1.0,
) -> Tuple[Optional[Trace], dict]:
    """
    Return (trace_in_physical_units, info) or (None, info) if we must skip.

    Heuristic:
      - If tr.stats.metrics['abs_mean'] > 1.0 → definitely uncorrected (Counts).
      - Else if tr.stats.metrics['abs_max']  > 1.0 → likely spikes or uncorrected.
      - Try remove_response(). If that fails and calib_fallback has a factor for this
        seed_id, multiply by that factor; otherwise return None (drop trace).
    """
    info = {
        "seed_id": tr.id,
        "status": "ok",
        "reason": "",
        "response_applied": False,
        "fallback_applied": False,
        "units": "unknown",
    }

    m = getattr(tr.stats, "metrics", {})
    abs_mean = float(m.get("abs_mean", np.nan))
    abs_max  = float(m.get("abs_max",  np.nan))

    needs_cal = False
    if np.isfinite(abs_mean) and abs_mean > mean_thresh:
        needs_cal = True
    elif np.isfinite(abs_max) and abs_max > max_thresh:
        needs_cal = True

    tr_out = tr.copy()
    if needs_cal:
        # Try instrument correction first
        try:
            if inventory is not None:
                tr_out.remove_response(
                    inventory=inventory,
                    output="VEL" if prefer_velocity else "DISP",
                    water_level=60.0,
                )
                info["response_applied"] = True
                info["units"] = "m/s" if prefer_velocity else "m"
                return tr_out, info
        except Exception as e:
            info["reason"] = f"remove_response failed: {e}"

        # Counts→SI fallback if provided
        if calib_fallback:
            fac = calib_fallback.get(tr.id)
            if fac and np.isfinite(fac) and fac > 0:
                tr_out.data = tr_out.data * float(fac)
                info["fallback_applied"] = True
                info["units"] = "m/s" if prefer_velocity else "m"
                return tr_out, info

        # No way to calibrate → drop
        info["status"] = "drop"
        info["reason"] = info["reason"] or "no response & no fallback"
        return None, info

    # Looks already calibrated; keep as-is
    info["units"] = "m/s" if prefer_velocity else "m"
    return tr_out, info


# --------------------------------------------------------------------------------------
# 1) Table-based estimator from an event×station amplitude DataFrame
# --------------------------------------------------------------------------------------
def _infer_station_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic: numeric columns that aren’t obvious metadata."""
    non_numeric = {c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
    candidates = [c for c in df.columns if c not in non_numeric]
    # drop common numeric meta columns if present
    junk = {"num_picks", "year", "event_id", "dfile"}
    return [c for c in candidates if c not in junk]


def estimate_station_gains_from_table(
    df: pd.DataFrame,
    *,
    value_cols: Optional[Iterable[str]] = None,
    groupby_col: Optional[str] = None,
    min_events_per_station: int = 3,
    robust: bool = True,
    verbose: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Estimate multiplicative station gains from an event×station table of amplitudes.

    For each event row, compute ratios r_{e,s} = A_{e,s} / median_s(A_{e,s}).
    The station's gain is g_s = median_e(r_{e,s})  (or mean if robust=False).

    Apply later as: A_corrected = A / g_s

    Parameters
    ----------
    df : DataFrame
        One row per event, columns for station seed_ids and (optionally) metadata.
    value_cols : iterable of str, optional
        Columns to treat as stations. If None, inferred.
    groupby_col : str, optional
        If provided (e.g., 'year'), compute gains per group and return a MultiIndex Series.
    min_events_per_station : int
        Require at least this many finite ratios per station to report a gain.
    robust : bool
        Use median across events (True) or mean (False).
    verbose : bool

    Returns
    -------
    gains : pd.Series
        If no grouping: Index = station seed_id → gain
        If grouped:     MultiIndex (group, station) → gain
    stats : pd.DataFrame
        Per-station descriptive stats of the event-wise ratios, with n_events.
        If grouped, index is (group, station).
    """
    # Identify numeric columns defensively
    if value_cols is None:
        meta = {"time", "num_picks", "dfile", "year", "mseed_path", "event_id"}
        value_cols = [c for c in df.columns
                      if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

    # ensure numeric (anything non-numeric → NaN)
    df_numeric = df.copy()
    for c in value_cols:
        df_numeric[c] = pd.to_numeric(df_numeric[c], errors="coerce")

    def _per_block(block):
        mat = block[value_cols]
        # event-wise median across stations
        med = mat.median(axis=1, skipna=True)
        med = med.where(np.isfinite(med) & (med > 0))  # sanitize
        ratios = mat.div(med, axis=0)

        # aggregate across events (robust median by default)
        if robust:
            g = ratios.median(axis=0, skipna=True)
        else:
            g = ratios.mean(axis=0, skipna=True)

        n_evt = ratios.notna().sum(axis=0).astype(int)
        # strictly enforce minimum event count
        g = g.where(n_evt >= int(min_events_per_station))
        # diagnostics (include central tendency spread if you want)
        s = pd.DataFrame({"gain": g, "n_events": n_evt})
        return g, s

    if groupby_col:
        gains_blocks, stats_blocks = [], []
        for key, block in df_numeric.groupby(groupby_col):
            if verbose:
                print(f"[GAINS] Group {groupby_col}={key}: {len(block)} events")
            g, s = _per_block(block)
            g.name = key
            s.insert(0, groupby_col, key)
            gains_blocks.append(g)
            stats_blocks.append(s)

        gains = pd.concat(gains_blocks, axis=1).T  # rows = groups
        gains = gains.stack(dropna=True)
        gains.index.names = [groupby_col, "station"]
        stats = pd.concat(stats_blocks, axis=0)
        stats.set_index([groupby_col, stats.index], inplace=True)
    else:
        gains, stats = _per_block(df_numeric)
        gains.name = "gain"

    if verbose:
        kept = gains.dropna().shape[0]
        print(f"[GAINS] Produced {kept} station gains (min_events={min_events_per_station}).")

    return gains.dropna(), stats


# --------------------------------------------------------------------------------------
# 2) Eventwise estimator from Streams (no prebuilt table needed)
# --------------------------------------------------------------------------------------
def estimate_station_gains(
    events: Iterable[Tuple[Stream, dict]],
    *,
    reduce_fn: Callable[[str, Stream, dict], float],
    seed_ids: Optional[Iterable[str]] = None,
    min_events_per_station: int = 3,
) -> Dict[str, float]:
    """
    Estimate station gains directly from (Stream, meta) events.

    You supply reduce_fn(seed_id, stream, meta) → amplitude
    (ideally ASL-consistent, distance/Q-corrected). For each event we:
      - compute amplitudes for available stations
      - divide by the per-event median across stations
      - collect ratios per station
      - take the median across events to get g_station

    Returns
    -------
    dict : seed_id → gain
    """
    ratios_per_station: Dict[str, list] = {}

    for st, meta in events:
        ids: Sequence[str] = [tr.id for tr in st] if seed_ids is None else tuple(seed_ids)
        vals, ok_ids = [], []
        for sid in ids:
            try:
                a = float(reduce_fn(sid, st, meta))
            except Exception:
                a = np.nan
            if np.isfinite(a):
                vals.append(a); ok_ids.append(sid)

        if len(vals) < 2:
            continue

        vals = np.asarray(vals, float)
        med = np.nanmedian(vals)
        if not np.isfinite(med) or med == 0:
            continue

        for sid, a in zip(ok_ids, vals):
            r = a / med
            if np.isfinite(r):
                ratios_per_station.setdefault(sid, []).append(r)

    gains = {}
    for sid, rs in ratios_per_station.items():
        if len(rs) >= min_events_per_station:
            gains[sid] = float(np.median(rs))
    return gains


# --------------------------------------------------------------------------------------
# 3) Convert gains to a **long/tidy** interval table and persist
# --------------------------------------------------------------------------------------
def gains_series_to_interval_df(
    gains: pd.Series,
    *,
    groupby_col: Optional[str] = None,
    group_boundaries: Optional[Mapping[Any, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    default_open_interval: bool = False,
    method_label: str = "median_ratio",
    stats: Optional[pd.DataFrame] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Convert a Series of station gains into a tidy interval table:

      start_time | end_time | seed_id | gain | method | n_events | (optional stats…)

    Supported inputs
    ----------------
    1) Single-index Series: index = seed_id, values = gain
       - If default_open_interval=True → one open interval row per station (NaT, NaT).
       - Else, raises unless you provide group_boundaries for a synthetic single group.

    2) MultiIndex Series: index = (group, station), values = gain
       - If group_boundaries provided: use those for each group.
       - Else if groupby_col == 'year': infer [YYYY-01-01, (YYYY+1)-01-01).
       - Else: error (ambiguous interval).

    Returns
    -------
    DataFrame with columns: start_time, end_time, seed_id, gain, method, n_events (optional)
    """
    def _ensure_tz(ts: pd.Timestamp | None) -> pd.Timestamp:
        if ts is None or pd.isna(ts):
            return pd.NaT
        ts = pd.Timestamp(ts)
        if tz:
            if ts.tz is None:
                ts = ts.tz_localize(tz)
            else:
                ts = ts.tz_convert(tz)
        return ts

    def _lookup_n_events(idx) -> Optional[float]:
        if stats is None:
            return None
        try:
            val = stats.loc[idx, "n_events"] if "n_events" in stats.columns else stats.loc[idx, "count"]
            if hasattr(val, "item"):
                val = val.item()
            return float(val)
        except Exception:
            return None

    rows: list[dict] = []

    # Case 1: MultiIndex (group, station)
    if isinstance(gains.index, pd.MultiIndex) and len(gains.index.levels) == 2:
        lvl0, lvl1 = gains.index.names
        if groupby_col and lvl0 != groupby_col:
            gains.index = gains.index.set_names([groupby_col, "station"], level=[0, 1])

        groups = sorted(set(gains.index.get_level_values(0)))
        for gkey in groups:
            if group_boundaries and gkey in group_boundaries:
                start_ts, end_ts = group_boundaries[gkey]
                start_ts, end_ts = _ensure_tz(start_ts), _ensure_tz(end_ts)
            elif groupby_col == "year":
                year = int(gkey)
                start_ts = _ensure_tz(pd.Timestamp(f"{year}-01-01 00:00:00"))
                end_ts   = _ensure_tz(pd.Timestamp(f"{year+1}-01-01 00:00:00"))
            else:
                raise ValueError(
                    "Ambiguous intervals: provide group_boundaries or use groupby_col='year'."
                )

            block = gains.xs(gkey, level=0).dropna()
            for seed_id, gain in block.items():
                row = {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "seed_id": seed_id,
                    "gain": float(gain),
                    "method": method_label,
                }
                n_ev = _lookup_n_events((gkey, seed_id))
                if n_ev is not None:
                    row["n_events"] = n_ev
                rows.append(row)

    # Case 2: Single-index (station only)
    else:
        if not default_open_interval:
            raise ValueError(
                "Single-index gains provided but no interval definition. "
                "Use default_open_interval=True or pass group_boundaries."
            )
        for seed_id, gain in gains.dropna().items():
            rows.append({
                "start_time": pd.NaT,
                "end_time": pd.NaT,
                "seed_id": seed_id,
                "gain": float(gain),
                "method": method_label,
                "n_events": _lookup_n_events(seed_id),
            })

    out = pd.DataFrame(rows)
    prefer = ["start_time", "end_time", "seed_id", "gain", "method", "n_events"]
    cols = [c for c in prefer if c in out.columns] + [c for c in out.columns if c not in prefer]
    out = out[cols].sort_values(["seed_id", "start_time"], na_position="first").reset_index(drop=True)

    # Normalize timezone columns (string-safe for CSV)
    if "start_time" in out.columns:
        out["start_time"] = pd.to_datetime(out["start_time"], utc=True, errors="coerce")
    if "end_time" in out.columns:
        out["end_time"] = pd.to_datetime(out["end_time"], utc=True, errors="coerce")
    return out


def save_station_gains_df(df: pd.DataFrame, path: str) -> str:
    df_out = df.copy()
    df_out.to_csv(path, index=False)
    print(f"[GAINS] Wrote interval gains to {path} ({len(df_out)} rows)")
    return path


def load_station_gains_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ("seed_id", "gain"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df


# --------------------------------------------------------------------------------------
# 4) Apply gains (long/tidy interval table) to a Stream
# --------------------------------------------------------------------------------------
def _event_start_ts_utc(st: Stream) -> pd.Timestamp:
    if len(st) == 0:
        return pd.NaT
    return pd.to_datetime(st[0].stats.starttime.datetime, utc=True)


def _pick_interval_rows_for_time(gains_df: pd.DataFrame, t0_utc: pd.Timestamp) -> pd.DataFrame:
    """
    Filter rows whose [start_time, end_time) contains t0_utc.
    Rules:
      - start_time <= t < end_time
      - if end_time is NaT → treat as +∞ (open interval)
    Returns matching subset (could be empty).
    """
    g = gains_df.copy()
    if "start_time" not in g.columns:
        raise ValueError("gains_df must include 'start_time'")
    if "end_time" not in g.columns:
        g["end_time"] = pd.NaT

    # normalize tz
    g["start_time"] = pd.to_datetime(g["start_time"], utc=True, errors="coerce")
    g["end_time"]   = pd.to_datetime(g["end_time"],   utc=True, errors="coerce")

    if pd.isna(t0_utc):
        return g.iloc[0:0]

    contains = (g["start_time"] <= t0_utc) & (g["end_time"].isna() | (t0_utc < g["end_time"]))
    return g.loc[contains].copy()


def _fallback_candidates(seed_id: str) -> list[str]:
    """Generate station-scope fallback identifiers."""
    try:
        net, sta, loc, cha = seed_id.split(".")
    except ValueError:
        return [seed_id]
    return [
        f"{net}.{sta}.{loc}.{cha}",   # exact
        f"{net}.{sta}..{cha}",        # no loc
        f"{net}.{sta}.*.{cha}",       # wildcard loc
        f"{net}.{sta}.*.*",           # any loc/cha at station
        f"{net}.{sta}",               # station only
    ]


def apply_interval_station_gains(
    st: Stream,
    station_gains_df: pd.DataFrame,
    *,
    allow_station_fallback: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Divide each trace by the appropriate interval gain from a **long/tidy** gains table:
      start_time | end_time | seed_id | gain | …

    Matching strategy per trace:
      1) Find interval rows containing the event start time
      2) Look for exact seed_id
      3) If allow_station_fallback=True, try station-scope forms:
         NET.STA..CHA, NET.STA.*.CHA, NET.STA.*.*, NET.STA

    Special case
    ------------
    If `start_time` and `end_time` are present but **empty for all rows**, the
    table is treated as **time-agnostic** (global gains). If no interval matches
    are found, we fall back to using the entire table for matching.

    Returns
    -------
    dict with interval info and which traces were updated.
    """
    if len(st) == 0:
        return {"interval_start": None, "interval_end": None, "used": [], "missing": []}

    # --- basic schema checks (seed_id & gain required) ---
    for req in ("seed_id", "gain"):
        if req not in station_gains_df.columns:
            raise ValueError(f"gains_df must include '{req}'")

    df = station_gains_df.copy()

    # --- coerce time columns if present ---
    has_start = "start_time" in df.columns
    has_end = "end_time" in df.columns
    if has_start:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    if has_end:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", utc=True)

    # Are both time columns present and empty (all NaT)?
    time_agnostic = False
    if has_start and has_end:
        all_empty_start = df["start_time"].isna().all()
        all_empty_end = df["end_time"].isna().all()
        time_agnostic = bool(all_empty_start and all_empty_end)

    # --- try interval-based matching first (unless time-agnostic by design) ---
    if not time_agnostic:
        t0 = _event_start_ts_utc(st)
        block = _pick_interval_rows_for_time(df, t0)
    else:
        block = pd.DataFrame()  # force the fallback below if needed

    # --- fallback: if no interval match AND table is time-agnostic, use entire table ---
    if block.empty and time_agnostic:
        if verbose:
            print("[GAINS] No interval filtering (time-agnostic gains table). Applying global gains.")
        block = df.copy()
        interval_start = pd.NaT
        interval_end = pd.NaT
    else:
        # choose the latest starting interval row for each seed_id scope
        # (safe even if one of the time cols is missing)
        sort_cols = [c for c in ("start_time", "end_time") if c in block.columns]
        if sort_cols:
            block = block.sort_values(sort_cols, na_position="last")
        interval_start = block["start_time"].min() if "start_time" in block.columns else pd.NaT
        interval_end = (block["end_time"].max()
                        if "end_time" in block.columns and block["end_time"].notna().any()
                        else pd.NaT)

    if block.empty:
        if verbose:
            print("[GAINS] No matching interval; no station gains applied.")
        return {"interval_start": None, "interval_end": None, "used": [], "missing": [tr.id for tr in st]}

    # --- apply per-trace ---
    used, missing = [], []
    for tr in st:
        sid = tr.id
        cand = _fallback_candidates(sid) if allow_station_fallback else [sid]

        gain_val = None
        for s in cand:
            rows_s = block[block["seed_id"] == s]
            if not rows_s.empty:
                # latest wins (after sort); otherwise just take last
                gain_val = rows_s.iloc[-1]["gain"]
                break

        if gain_val is None or not np.isfinite(gain_val) or float(gain_val) <= 0.0:
            missing.append(sid)
            continue

        tr.data = tr.data / float(gain_val)
        used.append(sid)

    if verbose:
        if used:
            print(f"[GAINS] Applied gains to {len(used)} traces.")
        if missing:
            print(f"[GAINS] No gain for {len(missing)} traces (left unchanged).")

    return {
        "interval_start": interval_start,
        "interval_end": interval_end,
        "used": used,
        "missing": missing,
    }


# --------------------------------------------------------------------------------------
# 5) High-level workflow: events → interval gains CSV
# --------------------------------------------------------------------------------------
def build_station_gains_from_events(
    events: Iterable[Tuple[Stream, dict]],
    *,
    reduce_fn: Callable[[str, Stream, dict], float],
    groupby_col: Optional[str] = "year",
    min_events_per_station: int = 3,
    tz: str = "UTC",
    out_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    High-level wrapper: take regional-quake events, estimate station gains,
    and package them into a **long** interval table (optionally saved to CSV).

    Steps
    -----
    1) Build a rectangular event×station amplitude table via `reduce_fn`.
    2) Estimate gains per station (optionally per group, e.g. 'year').
    3) Convert to interval form (long/tidy).
    4) Save to CSV if `out_csv` given.

    Parameters
    ----------
    events : iterable of (Stream, meta) pairs
        meta can include `groupby_col` (e.g., 'year'); if missing and groupby_col is set,
        we will attempt to infer from the Stream start time.
    reduce_fn : callable
        Signature (seed_id, Stream, meta) -> amplitude (float). Should already incorporate
        your preferred distance/Q correction for a fair across-station comparison.
    groupby_col : str or None
        Common choice: 'year'. If None, compute one set of gains across all events.
    min_events_per_station : int
    tz : str
        Timezone label for output intervals (kept as UTC for CSV).
    out_csv : str or None
    verbose : bool

    Returns
    -------
    pd.DataFrame (long form): start_time, end_time, seed_id, gain, method, n_events
    """
    # 1) Build event×station table
    records: list[dict] = []
    for ev_id, (st, meta) in enumerate(events):
        row: dict[str, Any] = {"event_id": ev_id}
        # group key from meta or stream time if requested
        if groupby_col:
            gval = meta.get(groupby_col)
            if gval is None:
                # derive from stream start time (UTC year)
                t0 = _event_start_ts_utc(st)
                gval = int(t0.year) if pd.notna(t0) else None
            row[groupby_col] = gval

        # per-station amplitude via reduce_fn
        for tr in st:
            try:
                row[tr.id] = float(reduce_fn(tr.id, st, meta))
            except Exception:
                row[tr.id] = np.nan

        records.append(row)

    table = pd.DataFrame.from_records(records)
    if verbose:
        n_ev = len(table)
        n_cols = len([c for c in table.columns if c not in {"event_id", groupby_col}])
        print(f"[GAINS] Built event×station table: {n_ev} events × ~{n_cols} stations")

    # 2) Estimate gains
    gains, stats = estimate_station_gains_from_table(
        table,
        value_cols=_infer_station_columns(table),
        groupby_col=groupby_col,
        min_events_per_station=min_events_per_station,
        robust=True,
        verbose=verbose,
    )

    # 3) Convert to interval form
    interval_df = gains_series_to_interval_df(
        gains,
        groupby_col=groupby_col,
        stats=stats,
        tz=tz,
    )

    # 4) Save
    if out_csv:
        save_station_gains_df(interval_df, out_csv)

    return interval_df


# --------------------------------------------------------------------------------------
# 6) Simple stream-level apply (dict-based), still handy for ad-hoc work
# --------------------------------------------------------------------------------------
def apply_station_gains_to_stream(st: Stream, gains: Dict[str, float]) -> Stream:
    """
    In-place normalization of a Stream using a dict of seed_id → gain (divide).
    Falls back to station-only keys if available.
    """
    for tr in st:
        sid = tr.id
        g = gains.get(sid)
        if g is None:
            # fallback to station-only key
            try:
                net, sta, *_ = sid.split(".")
                g = gains.get(f"{net}.{sta}")
            except Exception:
                g = None
        if g and np.isfinite(g) and g > 0:
            tr.data = tr.data / float(g)
    return st