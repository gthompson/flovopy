"""
flovopy.asl.compare_runs
------------------------
Compare a baseline ASLConfig against one-change variants, write pairwise CSVs,
and produce rollups / winners.

This consolidates the old helpers from analyze_run_pairs.py and adds a clean
orchestrator entrypoint.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

# Public API
__all__ = [
    "cfg_variants_from",
    "compare_runs",
    "safe_compare",
    "load_all_event_comparisons",
    "add_composite_score",
    "summarize_variants",
    "per_event_winner",
    # scores
    "build_intrinsic_table",
    "add_baseline_free_scores",
    "summarize_absolute_runs",
    "per_event_winner_abs",
]


# =============================================================================
# Tiny geo + CSV utilities
# =============================================================================

def _gc_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _load_run_csv(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise IOError(f"{path} does not exist")

    df = pd.read_csv(path)

    # case-insensitive column map
    cmap = {c.lower(): c for c in df.columns}

    def getcol(name: str, default=np.nan) -> np.ndarray:
        key = name.lower()
        if key in cmap:
            return df[cmap[key]].to_numpy()
        return np.full(len(df), default)

    # parse time (t/time/utc/timestamp), normalize to UTC, round to seconds
    time_cols = [c for c in ("t", "time", "utc", "timestamp") if c in cmap]
    if time_cols:
        raw = df[cmap[time_cols[0]]]
        t_parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if t_parsed.notna().any():
            t = t_parsed.dt.round("s").to_numpy("datetime64[ns]")
            time_kind = "real"
        else:
            t = np.arange(len(df))
            time_kind = "index"
    else:
        t = np.arange(len(df))
        time_kind = "index"

    out = {
        "t": t,
        "time_kind": time_kind,
        "lat": getcol("lat").astype(float),
        "lon": getcol("lon").astype(float),
        "DR":   getcol("dr").astype(float),            # case-insensitive via cmap
        "misfit": getcol("misfit").astype(float),
        "azgap":  getcol("azgap").astype(float),
        "connectedness": float(np.nanmean(getcol("connectedness"))) if "connectedness" in cmap else np.nan,
        "tag": path.stem,
    }
    return out


def _align_two(
    A: dict, B: dict, *, return_mode: bool = False
) -> Tuple[dict, dict] | Tuple[dict, dict, str]:
    """
    Align two runs:
      - If both have real timestamps, intersect on rounded-to-second time.
      - If no overlap on time (or one/both lack times), fall back to index alignment.
    """
    def take(D: dict, idx: np.ndarray, common_vals: np.ndarray) -> dict:
        T = {}
        for k, v in D.items():
            if isinstance(v, np.ndarray) and v.shape[0] == D["t"].shape[0]:
                T[k] = v[idx]
            else:
                T[k] = v
        T["t"] = common_vals
        return T

    mode = "index"
    if A["time_kind"] == "real" and B["time_kind"] == "real":
        tA = A["t"].astype("datetime64[s]").astype("datetime64[ns]")
        tB = B["t"].astype("datetime64[s]").astype("datetime64[ns]")
        common = np.intersect1d(tA, tB)
        if common.size > 0:
            idxA = {v: i for i, v in enumerate(tA)}
            idxB = {v: i for i, v in enumerate(tB)}
            iA = np.array([idxA[v] for v in common], dtype=int)
            iB = np.array([idxB[v] for v in common], dtype=int)
            mode = "time"
        else:
            n = min(A["t"].shape[0], B["t"].shape[0])
            if n <= 0:
                raise ValueError("No overlapping samples between runs.")
            iA = np.arange(n, dtype=int)
            iB = np.arange(n, dtype=int)
            common = iA  # synthetic index
    else:
        n = min(A["t"].shape[0], B["t"].shape[0])
        if n <= 0:
            raise ValueError("No overlapping samples between runs.")
        iA = np.arange(n, dtype=int)
        iB = np.arange(n, dtype=int)
        common = iA

    A2 = take(A, iA, common)
    B2 = take(B, iB, common)
    return (A2, B2, mode) if return_mode else (A2, B2)


def compare_two_runs_csv(csvA: Path, csvB: Path, label: str = "(baseline vs alt)") -> Optional[dict]:
    csvA = Path(csvA); csvB = Path(csvB)
    if not csvA.is_file():
        print(f"{csvA} does not exist")
        return None
    if not csvB.is_file():
        print(f"{csvB} does not exist")
        return None

    try:
        A = _load_run_csv(csvA)
        B = _load_run_csv(csvB)
    except Exception as e:
        print(f"[compare] load error: {e}")
        return None

    try:
        A, B, mode = _align_two(A, B, return_mode=True)
    except Exception as e:
        print(f"[compare] align error: {e}")
        return None

    latA, lonA = A["lat"], A["lon"]
    latB, lonB = B["lat"], B["lon"]

    # amplitude-based weights (mean DR of the two runs)
    DRw = np.nanmean(np.vstack([A["DR"], B["DR"]]), axis=0)
    DRw = np.where(np.isfinite(DRw), DRw, 0.0)
    w = DRw / (DRw.sum() + 1e-12) if DRw.max() > 0 else np.ones_like(DRw) / max(1, DRw.size)

    # per-sample spatial separation
    sep = np.array([
        _gc_km(latA[i], lonA[i], latB[i], lonB[i])
        if np.isfinite(latA[i]) and np.isfinite(lonA[i]) and np.isfinite(latB[i]) and np.isfinite(lonB[i])
        else np.nan
        for i in range(len(latA))
    ], dtype=float)

    mask = np.isfinite(sep)
    mean_sep   = float(np.nanmean(sep[mask])) if mask.any() else np.nan
    median_sep = float(np.nanmedian(sep[mask])) if mask.any() else np.nan
    wmean_sep  = float(np.nansum(sep * w)) if mask.any() else np.nan

    # simple % within thresholds
    pct_1km = float(100.0 * np.nanmean(sep <= 1.0)) if mask.any() else np.nan
    pct_2km = float(100.0 * np.nanmean(sep <= 2.0)) if mask.any() else np.nan
    pct_5km = float(100.0 * np.nanmean(sep <= 5.0)) if mask.any() else np.nan

    # (optional) path correlation by index overlap
    m = np.isfinite(latA) & np.isfinite(lonA) & np.isfinite(latB) & np.isfinite(lonB)
    lat_r = float(np.corrcoef(latA[m], latB[m])[0, 1]) if np.count_nonzero(m) > 3 else np.nan
    lon_r = float(np.corrcoef(lonA[m], lonB[m])[0, 1]) if np.count_nonzero(m) > 3 else np.nan

    # misfit / azgap averages and deltas
    mean_misfit_A = float(np.nanmean(A["misfit"]))
    mean_misfit_B = float(np.nanmean(B["misfit"]))
    d_misfit = mean_misfit_B - mean_misfit_A

    mean_azgap_A = float(np.nanmean(A["azgap"]))
    mean_azgap_B = float(np.nanmean(B["azgap"]))
    d_azgap = mean_azgap_B - mean_azgap_A

    return {
        "runA_tag": A["tag"],
        "runB_tag": B["tag"],
        "label": label,
        "align_mode": mode,
        "n_overlap": int(A["t"].shape[0]),
        "mean_sep_km": mean_sep,
        "median_sep_km": median_sep,
        "amp_weighted_mean_sep_km": wmean_sep,
        "pct_within_1km": pct_1km,
        "pct_within_2km": pct_2km,
        "pct_within_5km": pct_5km,
        "lat_corr": lat_r,
        "lon_corr": lon_r,
        "mean_misfit_A": mean_misfit_A,
        "mean_misfit_B": mean_misfit_B,
        "delta_misfit_B_minus_A": d_misfit,
        "mean_azgap_A": mean_azgap_A,
        "mean_azgap_B": mean_azgap_B,
        "delta_azgap_B_minus_A": d_azgap,
        "connectedness_A": A["connectedness"],
        "connectedness_B": B["connectedness"],
        "delta_connectedness_B_minus_A": float(B["connectedness"] - A["connectedness"])
            if (np.isfinite(B["connectedness"]) and np.isfinite(A["connectedness"])) else np.nan,
    }


def safe_compare(summary_csv: Path | str, csvA: Path | str, csvB: Path | str, label: str):
    """
    Safe wrapper: check files exist, try compare, append a row to `summary_csv`.
    NOTE: Let the caller choose `summary_csv` name (encode baseline tag upstream).
    """
    csvA, csvB = Path(csvA), Path(csvB)
    if not csvA.exists():
        print(f"[skip] missing baseline: {csvA}")
        return None
    if not csvB.exists():
        print(f"[skip] missing variant:  {csvB}")
        return None
    try:
        row = compare_two_runs_csv(csvA, csvB, label)
    except Exception as e:
        print(f"[skip] compare failed ({label}): {e}")
        return None
    if row is None:
        return None

    out = Path(summary_csv)
    df = pd.DataFrame([row])
    if out.exists():
        try:
            df0 = pd.read_csv(out)
            df = pd.concat([df0, df], ignore_index=True)
        except Exception as e:
            print(f"[warn] could not read existing {out}: {e}")
    df.to_csv(out, index=False)
    print(f"[compare] appended to {out} ({row['align_mode']} alignment, n={row['n_overlap']})")
    return row

# =============================================================================
# Roll-up + scoring
# =============================================================================

def load_all_event_comparisons(root: Path | str) -> pd.DataFrame:
    """
    Crawl event folders and stack pairwise CSVs.
    Matches both legacy 'pairwise_run_comparisons.csv' and new 'pairwise_*.csv'.
    Adds event_id from folder name and 'variant' from label.
    """
    root = Path(root)
    rows = []
    patterns = ["pairwise_run_comparisons.csv", "pairwise_*.csv"]
    for pat in patterns:
        for csv in root.rglob(pat):
            try:
                df = pd.read_csv(csv)
                df["event_id"] = csv.parent.name
                rows.append(df)
            except Exception as e:
                print(f"[skip] {csv}: {e}")
    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["variant"] = out.get("label", "").astype(str)
    # guard expected columns
    for c in ["mean_sep_km", "delta_misfit_B_minus_A", "delta_azgap_B_minus_A"]:
        if c not in out.columns:
            out[c] = np.nan
    return out


def add_composite_score(
    df: pd.DataFrame, w_sep: float = 1.0, w_misfit: float = 0.5, w_azgap: float = 0.1
) -> pd.DataFrame:
    """
    Lower is better. Negative deltas are good if they reduce misfit/azgap.
    Uses global z-scores (switch to per-event if needed).
    """
    d = df.copy()
    for col in ["mean_sep_km", "delta_misfit_B_minus_A", "delta_azgap_B_minus_A"]:
        x = d[col].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        d[col + "_z"] = (x - mu) / sd

    d["score"] = (
        w_sep * d["mean_sep_km_z"]
        + w_misfit * d["delta_misfit_B_minus_A_z"]
        + w_azgap * d["delta_azgap_B_minus_A_z"]
    )
    return d


def summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant", dropna=False)

    def _se(x):
        x = pd.to_numeric(x, errors="coerce")
        n = x.notna().sum()
        if n <= 1:
            return np.nan
        return np.nanstd(x, ddof=1) / np.sqrt(n)

    agg = g.agg(
        n_events          = ("event_id", "nunique"),
        n_rows            = ("event_id", "size"),
        mean_sep_km_mean  = ("mean_sep_km", "mean"),
        mean_sep_km_med   = ("mean_sep_km", "median"),
        mean_sep_km_se    = ("mean_sep_km", _se),   # <— use guarded SE
        dmisfit_mean      = ("delta_misfit_B_minus_A", "mean"),
        dmisfit_med       = ("delta_misfit_B_minus_A", "median"),
        dazgap_mean       = ("delta_azgap_B_minus_A", "mean"),
        score_mean        = ("score", "mean"),
        score_med         = ("score", "median"),
    ).reset_index().sort_values("score_mean")
    return agg


def per_event_winner(df_scored: pd.DataFrame):
    """
    For each event, pick the variant with the lowest composite score.
    Skips events where score is all-NaN. Returns (winners_df, win_counts_df).
    """
    # Empty/column guards
    if df_scored is None or df_scored.empty or "score" not in df_scored.columns:
        empty_winners = pd.DataFrame(columns=["event_id", "variant", "score"])
        empty_counts  = pd.DataFrame(columns=["variant", "wins"])
        return empty_winners, empty_counts

    # Keep rows where 'score' is finite
    d = df_scored[np.isfinite(df_scored["score"])].copy()
    if d.empty:
        empty_winners = pd.DataFrame(columns=["event_id", "variant", "score"])
        empty_counts  = pd.DataFrame(columns=["variant", "wins"])
        return empty_winners, empty_counts

    # idxmin over groups -> positions into 'd'
    # Drop NaNs (events with all-NaN scores) and force int for iloc
    idx = d.groupby("event_id")["score"].idxmin()
    idx = idx.dropna()
    if hasattr(idx, "astype"):
        try:
            idx = idx.astype(int)
        except Exception:
            idx = pd.Series(idx).dropna().astype(int)

    if len(idx) == 0:
        empty_winners = pd.DataFrame(columns=["event_id", "variant", "score"])
        empty_counts  = pd.DataFrame(columns=["variant", "wins"])
        return empty_winners, empty_counts

    # Use iloc to avoid the .loc ambiguity you hit
    winners = d.iloc[idx][["event_id", "variant", "score"]].reset_index(drop=True)

    win_counts = (winners["variant"]
                  .value_counts()
                  .rename_axis("variant")
                  .reset_index(name="wins")
                  .sort_values("wins", ascending=False))

    return winners, win_counts

# =============================================================================
# Orchestrator + variant factory
# =============================================================================

def _products_dir_for(cfg, mseed_file: str | Path) -> Path:
    mseed_file = Path(mseed_file)
    event_dir = Path(cfg.output_base) / mseed_file.stem
    return event_dir / Path(cfg.outdir).name


def csv_for_run(cfg, mseed_file: str | Path) -> Optional[Path]:
    """Locate a CSV output for a config + event, if present."""
    pdir = _products_dir_for(cfg, mseed_file)
    tag = cfg.tag()
    candidates = [
        pdir / f"source_{tag}_refined.csv",
        pdir / f"source_{tag}.csv",
        pdir / f"{tag}_refined.csv",
        pdir / f"{tag}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def ensure_csv_for(
    cfg,
    mseed_file: str | Path,
    *,
    run_if_missing: bool,
    run_single_event=None,
    **kwargs,
) -> Optional[Path]:
    """Return path to run CSV, optionally auto-running if missing."""
    csv = csv_for_run(cfg, mseed_file)
    if (csv is None or not csv.exists()) and run_if_missing and run_single_event:
        try:
            _ = run_single_event(mseed_file=str(mseed_file), cfg=cfg, **kwargs)
            csv = csv_for_run(cfg, mseed_file)
        except Exception as e:
            print(f"  [run error] {Path(mseed_file).stem} · {cfg.tag()}: {e}")
            return None
    return csv


def cfg_variants_from(
    baseline,
    landgridobj=None,
    annual_station_corrections_df=None,
) -> Dict[str, "ASLConfig"]:
    """
    Return a dict of one-change variants based on a baseline config.
    Any optional dependency not provided (e.g., landgridobj) simply omits that variant.
    """
    variants: Dict[str, Optional["ASLConfig"]] = {
        "Q100":           replace(baseline, Q=100).build(),
        "Q10":            replace(baseline, Q=10).build(),
        "v1.0":           replace(baseline, speed=1.0).build(),
        "v3.0":           replace(baseline, speed=3.0).build(),
        "win1s":          replace(baseline, window_seconds=1.0).build(),
        "win10s":         replace(baseline, window_seconds=10.0).build(),
        "metric_median":  replace(baseline, sam_metric="median").build(),
        "metric_LP":      replace(baseline, sam_metric="LP").build(),
        "metric_VT":      replace(baseline, sam_metric="VT").build(),
        "no_stacorr":     replace(baseline, station_correction_dataframe=None).build(),
        "l2_engine":      replace(baseline, misfit_engine="l2").build(),
        "lin_engine":     replace(baseline, misfit_engine="lin").build(),
        "body":           replace(baseline, wave_kind="body").build(),
        "f5hz":           replace(baseline, peakf=5.0).build(),
        "f8hz":           replace(baseline, peakf=8.0).build(),
        "2d":             replace(baseline, dist_mode="2d").build(),
    }
    if annual_station_corrections_df is not None:
        variants["annual_stacorr"] = replace(
            baseline, station_correction_dataframe=annual_station_corrections_df
        ).build()
    if landgridobj is not None:
        variants["landgrid"] = replace(baseline, gridobj=landgridobj).build()
    # prune Nones
    return {k: v for k, v in variants.items() if v is not None}


def compare_runs(
    baseline_cfg,
    events: Iterable[str | Path],
    variants: Optional[Dict[str, "ASLConfig"]] = None,
    *,
    run_single_event=None,
    refine_sector: bool = False,
    topo_kw=None,
    run_if_missing_baseline: bool = True,
    run_if_missing_variants: bool = False,
    w_sep: float = 1.0,
    w_misfit: float = 0.5,
    w_azgap: float = 0.1,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compare baseline vs variants across events; append pairwise rows under each event
    as 'pairwise_{baseline_tag}_vs_variants.csv'. Then roll up and score.

    Returns: (scored_df, summary_df, win_counts_df) — each may be None if nothing found.
    """
    if variants is None:
        variants = {}

    base_tag = baseline_cfg.tag()

    for i, ev in enumerate(events, start=1):
        ev_key = Path(ev).stem
        print(f"\n[{i}] {ev_key}")
        event_dir = Path(baseline_cfg.output_base) / ev_key
        summary_csv = event_dir / f"pairwise_{base_tag}_vs_variants.csv"

        # Baseline (auto-run if missing)
        base_csv = ensure_csv_for(
            baseline_cfg,
            ev,
            run_if_missing=run_if_missing_baseline,
            run_single_event=run_single_event,
            refine_sector=refine_sector,
            topo_kw=topo_kw,
            station_gains_df=None,
            switch_event_ctag=True,
            mseed_units="m/s",
            reduce_time=True,
            debug=True,
        )
        if base_csv is None:
            print("  [skip] no baseline CSV; cannot compare.")
            continue

        # Variants
        for _, vcfg in variants.items():
            alt_csv = ensure_csv_for(
                vcfg,
                ev,
                run_if_missing=run_if_missing_variants,
                run_single_event=run_single_event,
                refine_sector=refine_sector,
                topo_kw=topo_kw,
                station_gains_df=None,
                switch_event_ctag=True,
                mseed_units="m/s",
                reduce_time=True,
                debug=True,
            )
            try:
                safe_compare(summary_csv, base_csv, alt_csv, label=vcfg.tag())
            except Exception as e:
                print(f"  [compare error] {vcfg.tag()}: {e}")
    
    # Roll-up from the baseline's output root
    ROOT = Path(baseline_cfg.output_base)
    allcmp = load_all_event_comparisons(ROOT)
    print(f"\nstacked rows: {len(allcmp)}, events: {allcmp['event_id'].nunique() if not allcmp.empty else 0}")

    if allcmp.empty:
        return None, None, None

    scored  = add_composite_score(allcmp, w_sep=w_sep, w_misfit=w_misfit, w_azgap=w_azgap)
    summary = summarize_variants(scored)
    winners, win_counts = per_event_winner(scored)
    return scored, summary, win_counts


#### New scoring functions
def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = 1.4826 * mad if mad > 0 else (np.nanstd(x) or 1.0)
    return (x - med) / denom

def event_scores(df_event, weights):
    # df_event: rows=runs, cols include ["mean_sep_km","mean_misfit","mean_azgap","roughness","connectedness"]
    cols_lower = ["mean_sep_km","mean_misfit","mean_azgap","roughness"]
    cols_higher = ["connectedness"]  # flip sign

    Z = {}
    for c in cols_lower:
        Z[c] = robust_scale(df_event[c].values)
    for c in cols_higher:
        Z[c] = -robust_scale(df_event[c].values)

    # cap extreme z’s
    for c in Z:
        Z[c] = np.clip(Z[c], -3, 3)

    score = np.zeros(len(df_event))
    for c, w in weights.items():  # e.g., {"mean_sep_km":1.0, "mean_misfit":0.5, "mean_azgap":0.1, "roughness":0.2, "connectedness":0.2}
        score += w * Z[c]
    return score

def medoid_distance(df_event, pairwise_dist):
    # pairwise_dist: (n_runs x n_runs) symmetric matrix per event
    dmean = np.nanmean(np.where(np.isfinite(pairwise_dist), pairwise_dist, np.nan), axis=1)
    return dmean  # lower is better


# -----------------------------------------------------------------------------
# Intrinsic metrics (baseline-free, per run)
# -----------------------------------------------------------------------------

def _pairwise_gc_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance (km) between successive samples; length n-1."""
    n = len(lat)
    if n <= 1:
        return np.zeros(0, dtype=float)
    seg = np.empty(n-1, dtype=float)
    for i in range(n-1):
        if (np.isfinite(lat[i]) and np.isfinite(lon[i]) and
            np.isfinite(lat[i+1]) and np.isfinite(lon[i+1])):
            seg[i] = _gc_km(lat[i], lon[i], lat[i+1], lon[i+1])
        else:
            seg[i] = np.nan
    return seg

def _chord_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Straight-line (km) from first to last valid positions; NaN if insufficient."""
    m = np.isfinite(lat) & np.isfinite(lon)
    if np.count_nonzero(m) < 2:
        return np.nan
    i0 = np.argmax(m)                      # first True index
    i1_candidates = np.where(m)[0]
    i1 = i1_candidates[-1]                 # last True index
    return _gc_km(lat[i0], lon[i0], lat[i1], lon[i1])

def _roughness_ratio(lat: np.ndarray, lon: np.ndarray) -> float:
    """
    Unitless tortuosity: (total path length / chord length) - 1.
    =0 for a perfectly straight path; larger is 'wigglier'. NaN-safe.
    """
    seg = _pairwise_gc_km(lat, lon)
    total = np.nansum(seg) if seg.size else np.nan
    chord = _chord_km(lat, lon)
    if not np.isfinite(total) or not np.isfinite(chord) or chord <= 0:
        return np.nan
    return (total / chord) - 1.0

def intrinsic_metrics_from_csv(path: str | Path) -> Optional[dict]:
    """
    Load one run CSV and compute absolute metrics that do not depend on a baseline.
    Returns a dict with keys:
      - tag, event_id
      - n_samples, valid_frac
      - mean_misfit, mean_azgap, connectedness
      - path_len_km, chord_len_km, roughness_ratio
    """
    try:
        R = _load_run_csv(path)
    except Exception as e:
        print(f"[intrinsic] load error {path}: {e}")
        return None

    lat = R["lat"]; lon = R["lon"]
    mis = R["misfit"]; azg = R["azgap"]

    seg = _pairwise_gc_km(lat, lon)
    path_len = float(np.nansum(seg)) if seg.size else np.nan
    chord = float(_chord_km(lat, lon))
    rough = float(_roughness_ratio(lat, lon))

    valid_xy = np.isfinite(lat) & np.isfinite(lon)
    n = int(len(lat))
    vfrac = float(np.count_nonzero(valid_xy)) / n if n > 0 else np.nan

    return {
        "tag": R["tag"],
        "event_id": Path(path).parents[1].name if len(Path(path).parents) >= 2 else "",
        "n_samples": n,
        "valid_frac": vfrac,
        "mean_misfit": float(np.nanmean(mis)),
        "mean_azgap":  float(np.nanmean(azg)),
        "connectedness": float(R.get("connectedness", np.nan)),
        "path_len_km": path_len,
        "chord_len_km": chord,
        "roughness_ratio": rough,
    }

def intrinsic_metrics_from_csv(path: str | Path) -> dict | None:
    p = Path(path)
    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    cmap = {c.lower(): c for c in df.columns}
    def has(c): return c.lower() in cmap

    # lat/lon
    lat = df[cmap["lat"]].to_numpy(float) if has("lat") else np.array([])
    lon = df[cmap["lon"]].to_numpy(float) if has("lon") else np.array([])

    # roughness from stepwise great-circle distances
    rough = np.nan
    if lat.size >= 2 and lon.size >= 2:
        with np.errstate(invalid="ignore"):
            dists = []
            for i in range(1, len(lat)):
                if np.isfinite(lat[i-1]) and np.isfinite(lon[i-1]) and np.isfinite(lat[i]) and np.isfinite(lon[i]):
                    dists.append(_gc_km(lat[i-1], lon[i-1], lat[i], lon[i]))
                else:
                    dists.append(np.nan)
            rough = safe_mean(dists)

    # misfit/azgap (may be missing)
    mis = df[cmap["misfit"]].to_numpy(float) if has("misfit") else np.array([])
    azg = df[cmap["azgap"]].to_numpy(float)  if has("azgap")  else np.array([])

    # connectedness (scalar summary)
    conn = np.nan
    if has("connectedness"):
        conn = safe_mean(df[cmap["connectedness"]].to_numpy(float))

    # tag
    tag = p.stem
    if tag.startswith("source_"):
        tag = tag[len("source_"):]
    if tag.endswith("_refined"):
        tag = tag[:-len("_refined")]

    return {
        "mean_misfit":   safe_mean(mis),
        "mean_azgap":    safe_mean(azg),
        "roughness":     rough,
        "connectedness": float(conn),
        "tag":           tag,
    }


def _ensure_and_get_csv(cfg, ev, *, run_if_missing, run_single_event, **kwargs) -> Optional[Path]:
    return ensure_csv_for(cfg, ev, run_if_missing=run_if_missing,
                          run_single_event=run_single_event, **kwargs)

def build_intrinsic_table(
    baseline_cfg,
    events: Iterable[str | Path],
    variants: Optional[Dict[str, "ASLConfig"]] = None,
    *,
    run_single_event=None,
    refine_sector: bool = False,
    topo_kw=None,
    run_if_missing_baseline: bool = True,
    run_if_missing_variants: bool = False,
) -> pd.DataFrame:
    """
    Ensure CSVs exist (optionally run), then compute intrinsic metrics for:
      baseline + all variants for each event.

    Returns a tidy DataFrame with one row per (event_id, tag).
    """
    if variants is None:
        variants = {}

    rows: List[dict] = []
    for ev in events:
        # baseline
        bcsv = _ensure_and_get_csv(
            baseline_cfg, ev,
            run_if_missing=run_if_missing_baseline,
            run_single_event=run_single_event,
            refine_sector=refine_sector, topo_kw=topo_kw,
            station_gains_df=None, switch_event_ctag=True,
            mseed_units="m/s", reduce_time=True, debug=True,
        )
        if bcsv is not None and bcsv.exists():
            r = intrinsic_metrics_from_csv(bcsv)
            if r: rows.append(r)

        # variants
        for _, vcfg in variants.items():
            vcsv = _ensure_and_get_csv(
                vcfg, ev,
                run_if_missing=run_if_missing_variants,
                run_single_event=run_single_event,
                refine_sector=refine_sector, topo_kw=topo_kw,
                station_gains_df=None, switch_event_ctag=True,
                mseed_units="m/s", reduce_time=True, debug=True,
            )
            if vcsv is not None and vcsv.exists():
                r = intrinsic_metrics_from_csv(vcsv)
                if r: rows.append(r)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Baseline-free (absolute) scoring
# -----------------------------------------------------------------------------

def robust_scale(x):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    z = np.zeros_like(x, dtype=float)
    if m.sum() == 0:
        return z  # no info → neutral score
    xm = x[m]
    med = np.median(xm)
    mad = np.median(np.abs(xm - med))
    if mad > 0:
        denom = 1.4826 * mad
    else:
        sd = np.std(xm)
        denom = sd if sd > 0 else 1.0
    z[m] = (xm - med) / denom
    return z

def add_baseline_free_scores(
    df_abs: pd.DataFrame,
    *,
    weights: Optional[Dict[str, float]] = None,
    cap: float = 3.0,
) -> pd.DataFrame:
    """
    Compute a robust composite `score_abs` per (event_id, tag) using *only* intrinsic metrics.
    Lower is better after sign conventions below.

    Default weights favor separation & misfit, lightly penalize azgap & roughness,
    and lightly reward connectedness and valid fraction.
    """
    if df_abs is None or df_abs.empty:
        return df_abs

    d = df_abs.copy()

    # Defaults chosen so terms are O(1) after robust scaling
    if weights is None:
        weights = {
            "mean_misfit":       0.8,   # lower better
            "mean_azgap":        0.2,   # lower better
            "roughness_ratio":   0.2,   # lower better
            "connectedness":    -0.2,   # higher better (note: negative weight)
            "valid_frac":       -0.2,   # higher better (note: negative weight)
        }

    # Build Z-scores per event (robust to outliers)
    out = []
    for ev, g in d.groupby("event_id", dropna=False):
        gg = g.copy()
        Z = {}

        # lower-is-better metrics
        for col in ("mean_misfit", "mean_azgap", "roughness_ratio"):
            if col in gg.columns:
                z = robust_scale(gg[col].values)
                Z[col] = np.clip(z, -cap, cap)
            else:
                Z[col] = np.full(len(gg), np.nan)

        # higher-is-better metrics (flip sign via negative weight)
        for col in ("connectedness", "valid_frac"):
            if col in gg.columns:
                z = robust_scale(gg[col].values)
                Z[col] = np.clip(z, -cap, cap)
            else:
                Z[col] = np.full(len(gg), np.nan)

        # composite
        score = np.zeros(len(gg))
        for col, w in weights.items():
            if col in Z:
                # If this is a higher-better metric, its negative weight makes lower score = better.
                score += w * Z[col]
        gg["score_abs"] = score
        out.append(gg)

    return pd.concat(out, ignore_index=True)

def summarize_absolute_runs(df_abs_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Summary across events for absolute scores. Lower is better.
    """
    if df_abs_scored is None or df_abs_scored.empty:
        return pd.DataFrame()

    g = df_abs_scored.groupby("tag")
    def _se(x):
        x = pd.to_numeric(x, errors="coerce")
        n = x.notna().sum()
        return (np.nanstd(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan

    return (g.agg(
                n_events=("event_id","nunique"),
                n_rows=("event_id","size"),
                score_abs_mean=("score_abs","mean"),
                score_abs_med =("score_abs","median"),
                score_abs_se  =("score_abs", _se),
                misfit_mean   =("mean_misfit","mean"),
                azgap_mean    =("mean_azgap","mean"),
                rough_mean    =("roughness_ratio","mean"),
                conn_mean     =("connectedness","mean"),
                vfrac_mean    =("valid_frac","mean"),
            )
            .reset_index()
            .sort_values("score_abs_mean"))

def per_event_winner_abs(df_abs_scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Winners by lowest absolute score per event (no baselines).
    Returns (winners_df, win_counts_df).
    """
    if df_abs_scored is None or df_abs_scored.empty or "score_abs" not in df_abs_scored.columns:
        W = pd.DataFrame(columns=["event_id","tag","score_abs"])
        C = pd.DataFrame(columns=["tag","wins"])
        return W, C

    d = df_abs_scored[np.isfinite(df_abs_scored["score_abs"])].copy()
    if d.empty:
        W = pd.DataFrame(columns=["event_id","tag","score_abs"])
        C = pd.DataFrame(columns=["tag","wins"])
        return W, C

    idx = d.groupby("event_id")["score_abs"].idxmin()
    idx = idx.dropna()
    try:
        idx = idx.astype(int)
    except Exception:
        idx = pd.Series(idx).dropna().astype(int)

    if len(idx) == 0:
        W = pd.DataFrame(columns=["event_id","tag","score_abs"])
        C = pd.DataFrame(columns=["tag","wins"])
        return W, C

    winners = d.iloc[idx][["event_id","tag","score_abs"]].reset_index(drop=True)
    win_counts = (winners["tag"].value_counts()
                  .rename_axis("tag").reset_index(name="wins")
                  .sort_values("wins", ascending=False))
    return winners, win_counts


# if we want to just discover what runs are available
def crawl_intrinsic_runs(root: str | Path) -> pd.DataFrame:
    """
    Walk output tree under `root` and compute intrinsic (baseline-free) metrics
    for every per-run CSV we can recognize. Returns one row per (event_id, tag).
    """
    root = Path(root)
    rows = []

    # heuristics: look for common per-run outputs, skip pairwise summaries
    name_ok = lambda p: (
        p.suffix.lower() == ".csv"
        and not p.name.startswith("pairwise_")
        and not p.name.endswith("_comparisons.csv")
    )
    patterns = ["**/source_*_refined.csv", "**/source_*.csv", "**/*_refined.csv", "**/*.csv"]

    seen = set()  # dedupe by (event_id, tag, path)
    for pat in patterns:
        for csv in root.glob(pat):
            if not name_ok(csv):
                continue

            # infer event_id (parent of products dir)
            # e.g., .../<event_id>/<products_dir>/<file.csv>
            try:
                event_id = csv.parent.parent.name
            except Exception:
                event_id = csv.parent.name

            # infer tag from filename
            nm = csv.stem  # without .csv
            if nm.startswith("source_"):
                tag = nm[len("source_"):]
            else:
                tag = nm
            if tag.endswith("_refined"):
                tag = tag[:-len("_refined")]

            key = (event_id, tag, str(csv))
            if key in seen:
                continue
            seen.add(key)

            r = intrinsic_metrics_from_csv(csv)
            if r:
                r["event_id"] = event_id
                r["tag"] = tag
                r["path"] = str(csv)
                rows.append(r)

    return pd.DataFrame(rows)


# safe reducers
def _finite1(x) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    return x[m], int(m.sum())

def safe_mean(x) -> float:
    x, n = _finite1(x)
    return float(np.nan if n == 0 else x.mean())

def safe_median(x) -> float:
    x, n = _finite1(x)
    return float(np.nan if n == 0 else np.median(x))

def crawl_intrinsic_runs(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    patterns = ["**/source_*_refined.csv", "**/source_*.csv", "**/*_refined.csv", "**/*.csv"]

    def is_per_run_csv(p: Path) -> bool:
        if not p.name.endswith(".csv"):
            return False
        if p.name.startswith("pairwise_"):
            return False
        if p.name.endswith("_comparisons.csv"):
            return False
        return True

    seen = set()
    for pat in patterns:
        for csv in root.glob(pat):
            if not is_per_run_csv(csv):
                continue

            # heuristics for event_id
            event_id = csv.parent.parent.name if csv.parent.parent != root else csv.parent.name

            r = intrinsic_metrics_from_csv(csv)
            if not r:
                continue
            r["event_id"] = event_id
            r["path"] = str(csv)

            key = (r["event_id"], r["tag"], r["path"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    return pd.DataFrame(rows)