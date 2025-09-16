# flovopy/asl/station_corrections.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
# -------------------------
# 1) DataFrame-based gains
# -------------------------

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

    Returns
    -------
    gains : pd.Series
        If `groupby_col` is None: Index = station (seed_id) → gain
        If `groupby_col` is provided: MultiIndex (group, station) → gain
    stats : pd.DataFrame
        Descriptive stats of the ratio distribution per station (per group if provided).
    """
    df = df.copy()

    # Choose station columns if not provided: numeric columns that aren't obvious metadata
    if value_cols is None:
        non_numeric = {c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
        candidates = [c for c in df.columns if c not in non_numeric]
        # common metadata to drop if numeric:
        junk = {"num_picks", "dfile", "year"}
        value_cols = [c for c in candidates if c not in junk]
        if verbose:
            print(f"[GAINS] Inferred {len(value_cols)} station columns.")

    def _per_block(block: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        med = block[value_cols].median(axis=1, skipna=True)
        ratios = block[value_cols].div(med.replace(0, np.nan), axis=0)
        ratios = ratios.where((ratios > 0) & np.isfinite(ratios))

        desc = ratios.describe(percentiles=[.25, .5, .75]).T
        desc = desc[["mean", "std", "min", "25%", "50%", "75%", "max", "count"]]

        agg = (ratios.median(axis=0) if robust else ratios.mean(axis=0))
        counts = ratios.count(axis=0)
        agg[counts < min_events_per_station] = np.nan
        return agg, desc

    if groupby_col:
        gains_blocks = []
        stats_blocks = []
        for key, block in df.groupby(groupby_col):
            if verbose:
                print(f"[GAINS] Group {groupby_col}={key}: {len(block)} events")
            g, s = _per_block(block)
            g.name = key
            s.insert(0, groupby_col, key)
            gains_blocks.append(g)
            stats_blocks.append(s)
        gains = pd.concat(gains_blocks, axis=1).T  # rows = groups, cols = stations
        gains = gains.stack(dropna=True)           # MultiIndex (group, station) -> value
        gains.index.names = [groupby_col, "station"]
        stats = pd.concat(stats_blocks, axis=0)
        stats.set_index([groupby_col, stats.index], inplace=True)
    else:
        gains, stats = _per_block(df)
        gains.name = "gain"

    if verbose:
        kept = gains.dropna().shape[0]
        print(f"[GAINS] Produced {kept} station gains (min_events={min_events_per_station}).")

    return gains.dropna(), stats

# --------------------------------------
# 2) Stream-level application utilities
# --------------------------------------

def apply_station_gains_to_stream(st: Stream, gains: Dict[str, float]) -> Stream:
    """
    In-place normalization of a Stream using station/seed_id gains.
    A_corrected = A / g where possible; falls back to station-only if needed.
    """
    for tr in st:
        sid = tr.id
        g = gains.get(sid) or gains.get(tr.stats.station)
        if g and np.isfinite(g) and g > 0:
            tr.data = tr.data / g
    return st


# --------------------------------------------------
# 3) “Raw” estimator from reduced amplitudes per event
# --------------------------------------------------

def estimate_station_gains(
    events: Iterable[Tuple[Stream, dict]],
    *,
    seed_ids: Optional[Iterable[str]] = None,
    reduce_fn: Callable[[str, Stream, dict], float],
    min_events_per_station: int = 3,
) -> Dict[str, float]:
    """
    Generic estimator that doesn’t assume an existing table.
    You provide a `reduce_fn(seed_id, stream, meta) -> amplitude` that
    returns a distance/Q-corrected amplitude (your ASL-consistent measure).

    Returns a dict: seed_id → gain (divide by this later).
    """
    ratios_per_station: Dict[str, list] = {}

    for st, meta in events:
        ids = [tr.id for tr in st] if seed_ids is None else list(seed_ids)
        vals = []
        ok_ids = []
        for sid in ids:
            try:
                a = float(reduce_fn(sid, st, meta))
            except Exception:
                a = np.nan
            if np.isfinite(a):
                vals.append(a)
                ok_ids.append(sid)
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


from datetime import datetime
from typing import Mapping, Any

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

      start_time | end_time | seed_id | gain | method | n_events | (optional stats columns…)

    Supported inputs
    ----------------
    1) Single-index Series: index = seed_id, values = gain
       - If default_open_interval=True → one open interval row per station (NaT, NaT).
       - Else, raises unless you provide group_boundaries for a synthetic single group.

    2) MultiIndex Series: index = (group, station), values = gain
       - If group_boundaries provided: use those for each group.
       - Else if groupby_col == 'year': infer [YYYY-01-01, (YYYY+1)-01-01).
       - Else: error (ambiguous interval).

    Parameters
    ----------
    group_boundaries : mapping
        Dict: group_key -> (start_ts, end_ts). Timestamps must be tz-aware or naive;
        we cast to tz if provided.
    default_open_interval : bool
        For single-index gains: make start_time/end_time = NaT (open) if True.
    method_label : str
        Text to record in the 'method' column.
    stats : DataFrame or None
        If provided, try to add per-station (or per group, station) counts as 'n_events'
        and any columns that exist (mean, std, 50%, etc.). Non-matching indexes are ignored.

    Returns
    -------
    DataFrame with columns: start_time, end_time, seed_id, gain, method, n_events, …stats…
    """
    rows = []

    def _ensure_tz(ts: pd.Timestamp) -> pd.Timestamp:
        if ts is None or pd.isna(ts):
            return pd.NaT
        ts = pd.Timestamp(ts)
        if tz:
            # make tz-aware in given zone
            if ts.tz is None:
                ts = ts.tz_localize(tz)
            else:
                ts = ts.tz_convert(tz)
        return ts

    # Helper to extract n_events from stats if present
    def _lookup_n_events(idx) -> Optional[float]:
        if stats is None:
            return None
        try:
            val = stats.loc[idx, "count"]
            if hasattr(val, "item"):
                val = val.item()
            return float(val)
        except Exception:
            return None

    # Case 1: MultiIndex (group, station)
    if isinstance(gains.index, pd.MultiIndex) and len(gains.index.levels) == 2:
        lvl0, lvl1 = gains.index.names
        if groupby_col and lvl0 != groupby_col:
            # rename for consistency so we can look up boundaries
            gains.index = gains.index.set_names([groupby_col, "station"], level=[0, 1])

        groups = sorted(set(gains.index.get_level_values(0)))
        for gkey in groups:
            # Determine interval for this group
            if group_boundaries and gkey in group_boundaries:
                start_ts, end_ts = group_boundaries[gkey]
                start_ts, end_ts = _ensure_tz(start_ts), _ensure_tz(end_ts)
            elif groupby_col == "year":
                year = int(gkey)
                start_ts = _ensure_tz(pd.Timestamp(f"{year}-01-01 00:00:00"))
                end_ts   = _ensure_tz(pd.Timestamp(f"{year+1}-01-01 00:00:00"))
            else:
                raise ValueError(
                    "Ambiguous intervals: provide group_boundaries or use groupby_col='year' "
                    "so intervals can be inferred."
                )

            # Rows for this group
            block = gains.xs(gkey, level=0).dropna()
            for seed_id, gain in block.items():
                row = {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "seed_id": seed_id,
                    "gain": float(gain),
                    "method": method_label,
                }
                # add n_events if available
                n_ev = _lookup_n_events((gkey, seed_id))
                if n_ev is not None:
                    row["n_events"] = n_ev
                rows.append(row)

    # Case 2: Single index (station only)
    else:
        if default_open_interval:
            start_ts = pd.NaT
            end_ts   = pd.NaT
            for seed_id, gain in gains.dropna().items():
                row = {
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "seed_id": seed_id,
                    "gain": float(gain),
                    "method": method_label,
                }
                n_ev = _lookup_n_events(seed_id)
                if n_ev is not None:
                    row["n_events"] = n_ev
                rows.append(row)
        else:
            raise ValueError(
                "Single-index gains provided but no interval definition. "
                "Use default_open_interval=True or pass group_boundaries."
            )

    out = pd.DataFrame(rows)
    # canonical column order
    prefer = ["start_time", "end_time", "seed_id", "gain", "method", "n_events"]
    cols = [c for c in prefer if c in out.columns] + [c for c in out.columns if c not in prefer]
    return out[cols].sort_values(["seed_id", "start_time"], na_position="first").reset_index(drop=True)


def save_station_gains_df(df: pd.DataFrame, path: str) -> str:
    df_out = df.copy()
    # write timestamps as ISO8601; pandas handles tz-aware timestamps in CSV fine
    df_out.to_csv(path, index=False)
    print(f"[GAINS] Wrote interval gains to {path} ({len(df_out)} rows)")
    return path

def load_station_gains_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # parse datetimes; keep tz-naive as-is (will be treated as “always valid” intervals if NaT)
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # ensure schema
    for col in ("seed_id", "gain"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df

# Use them

# --- Interval helpers (use start_time / end_time consistently) ---

def _to_utc_ts(a):
    """Return a tz-aware pandas.Timestamp (UTC) or NaT."""
    ts = pd.to_datetime(a, utc=True, errors="coerce")
    return ts

def _normalize_interval_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'start_time' (required) / 'end_time' (optional) are tz-aware Timestamps (UTC).
    If end_time missing, (a) if there is another row for the same schema later, fill with next start_time,
    otherwise set to NaT (open-ended).
    """
    if "start_time" not in df.columns:
        raise ValueError("station_gains_df must have a 'start_time' column (datetime).")

    g = df.copy()
    g["start_time"] = _to_utc_ts(g["start_time"])  # normalize tz
    if "end_time" in g.columns:
        g["end_time"] = _to_utc_ts(g["end_time"])
    else:
        # leave open-ended as NaT; caller can choose the “latest wins” behavior
        g["end_time"] = pd.NaT

    # Stable ordering for interval picking
    g.sort_values(["start_time", "end_time"], inplace=True, na_position="last")
    g.reset_index(drop=True, inplace=True)
    return g

def _pick_interval_row_by_time(gains_df: pd.DataFrame, t0_utc: pd.Timestamp) -> pd.Series | None:
    """
    Pick the row whose [start_time, end_time) contains t0_utc.
    Rules:
      - If end_time is NaT: treat as open interval to +∞.
      - If multiple overlap, take the last (latest start_time).
    """
    g = _normalize_interval_table(gains_df)
    t0 = _to_utc_ts(t0_utc)
    if pd.isna(t0):
        return None

    # Contains if start <= t < end (or end is NaT)
    contains = (g["start_time"] <= t0) & (g["end_time"].isna() | (t0 < g["end_time"]))
    idx = np.flatnonzero(contains.values)
    if idx.size == 0:
        return None
    return g.iloc[idx[-1]]  # prefer the latest defined interval if overlaps exist

def _lookup_gain_in_row(row: pd.Series, seed_id: str, allow_station_fallback: bool = True) -> float | None:
    """
    Look gain up in a row:
      1) exact NET.STA.LOC.CHA
      2) station-scope fallbacks (if present as columns), e.g.:
         NET.STA..CHA, NET.STA.*.CHA, NET.STA.*.* , NET.STA
    """
    if seed_id in row.index and pd.notna(row[seed_id]):
        return float(row[seed_id])

    if not allow_station_fallback:
        return None

    try:
        net, sta, loc, cha = seed_id.split(".")
    except ValueError:
        return None

    candidates = [
        f"{net}.{sta}..{cha}",
        f"{net}.{sta}.*.{cha}",
        f"{net}.{sta}.*.*",
        f"{net}.{sta}",  # station-only
    ]
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return None

def apply_interval_station_gains(
    st: Stream,
    station_gains_df: pd.DataFrame,
    *,
    allow_station_fallback: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Divide each trace by the appropriate interval gain from `station_gains_df`.
    Expected columns: 'start_time', 'end_time' (datetimes), plus one column per seed-id (and/or fallbacks).

    Returns a small info dict with which interval was used and which traces were updated.
    """
    if len(st) == 0:
        return {"interval_start": None, "interval_end": None, "used_columns": [], "missing": []}

    event_ts = pd.to_datetime(st[0].stats.starttime.datetime, utc=True)
    row = _pick_interval_row_by_time(station_gains_df, event_ts)
    if row is None:
        if verbose:
            print("[GAINS] No matching interval; no station gains applied.")
        return {"interval_start": None, "interval_end": None, "used_columns": [], "missing": [tr.id for tr in st]}

    if verbose:
        s = row.get("start_time"); e = row.get("end_time")
        s_str = s.isoformat() if isinstance(s, pd.Timestamp) and not pd.isna(s) else "NaT"
        e_str = e.isoformat() if isinstance(e, pd.Timestamp) and not pd.isna(e) else "NaT"
        print(f"[GAINS] Applying interval gains: {s_str} → {e_str}")

    used, missing = [], []
    for tr in st:
        g = _lookup_gain_in_row(row, tr.id, allow_station_fallback=allow_station_fallback)
        if g is None or not np.isfinite(g) or g <= 0:
            missing.append(tr.id)
            continue
        tr.data = tr.data / float(g)
        used.append(tr.id)

    if verbose:
        if used:
            print(f"[GAINS] Applied gains to {len(used)} traces.")
        if missing:
            print(f"[GAINS] No gain for {len(missing)} traces (left unchanged).")

    return {
        "interval_start": row.get("start_time"),
        "interval_end": row.get("end_time"),
        "used_columns": used,
        "missing": missing,
    }