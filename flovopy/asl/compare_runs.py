# flovopy.asl.compare_runs
# ------------------------
# Compare a baseline ASLConfig against one-change variants, write pairwise CSVs,
# and produce rollups / winners. Also supports baseline-free (absolute) scoring
# by computing intrinsic metrics per run and summarizing across events.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

# Public API
__all__ = [
    # pairwise (baseline vs variants)
    "tweak_config",
    "compare_runs",
    "safe_compare",
    "load_all_event_comparisons",
    "add_composite_score",
    "summarize_variants",
    "per_event_winner",
    # baseline-free (absolute)
    "build_intrinsic_table",
    "add_baseline_free_scores",
    "summarize_absolute_runs",
    "per_event_winner_abs",
    "crawl_intrinsic_runs",
]

# =============================================================================
# Tiny geo + safe math helpers
# =============================================================================

def _gc_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))

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

# =============================================================================
# CSV loader + alignment for pairwise comparisons
# =============================================================================

def _load_run_csv(path: str | Path) -> dict:
    """
    Load a per-run CSV into a dict with normalized columns.
    Tries to parse UTC timestamps; falls back to index-aligned time if not present.
    """
    path = Path(path)
    if not path.exists():
        raise IOError(f"{path} does not exist")

    df = pd.read_csv(path)

    # case-insensitive column map
    cmap = {c.lower(): c for c in df.columns}
    def has(name: str) -> bool:
        return name.lower() in cmap

    def getcol(name: str, default=np.nan) -> np.ndarray:
        key = name.lower()
        if key in cmap:
            return pd.to_numeric(df[cmap[key]], errors="coerce").to_numpy()
        # vectorized default
        return np.full(len(df), default, dtype=float)

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
        "lat": getcol("lat"),
        "lon": getcol("lon"),
        "DR": getcol("dr"),
        "misfit": getcol("misfit"),
        "azgap": getcol("azgap"),
        "connectedness": float(safe_mean(getcol("connectedness"))) if has("connectedness") else np.nan,
        "tag": path.stem,
        "path": str(path),
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
        # unify to second resolution but keep ns dtype for consistent keys
        tA = A["t"].astype("datetime64[s]").astype("datetime64[ns]")
        tB = B["t"].astype("datetime64[s]").astype("datetime64[ns]")
        common = np.intersect1d(tA, tB)
        if common.size > 0:
            idxA = {v: i for i, v in enumerate(tA)}
            idxB = {v: i for i, v in enumerate(tB)}
            iA = np.fromiter((idxA[v] for v in common), dtype=int, count=common.size)
            iB = np.fromiter((idxB[v] for v in common), dtype=int, count=common.size)
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
    """
    Build a single pairwise-comparison row between two per-run CSVs.
    """
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
    sep = np.empty(len(latA), dtype=float)
    sep[:] = np.nan
    for i in range(len(latA)):
        if np.isfinite(latA[i]) and np.isfinite(lonA[i]) and np.isfinite(latB[i]) and np.isfinite(lonB[i]):
            sep[i] = _gc_km(latA[i], lonA[i], latB[i], lonB[i])

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
    mean_misfit_A = float(safe_mean(A["misfit"]))
    mean_misfit_B = float(safe_mean(B["misfit"]))
    d_misfit = mean_misfit_B - mean_misfit_A

    mean_azgap_A = float(safe_mean(A["azgap"]))
    mean_azgap_B = float(safe_mean(B["azgap"]))
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
# Roll-up + scoring (pairwise mode)
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
                df["event_id"] = csv.parent.name   # event folder
                rows.append(df)
            except Exception as e:
                print(f"[skip] {csv}: {e}")
    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["variant"] = out.get("label", "").astype(str)

    # Ensure expected columns exist
    for c in ["mean_sep_km", "delta_misfit_B_minus_A", "delta_azgap_B_minus_A"]:
        if c not in out.columns:
            out[c] = np.nan

    return out

def add_composite_score(
    df: pd.DataFrame,
    *,
    w_sep: float = 1.0,
    w_misfit: float = 0.5,
    w_azgap: float = 0.1,
    w_conn: float = 0.0,     # set >0 to reward increased connectedness
    w_rough: float = 0.0,    # set >0 to penalize increased roughness (if present)
) -> pd.DataFrame:
    """
    Composite score for pairwise (baseline vs variant) rows.
    Lower is better. Negative deltas are good for misfit/azgap/roughness.
    Positive deltas are good for connectedness (so we subtract that term).

    We compute global z-scores per metric to put them on comparable scales.
    Missing/all-NaN columns are ignored gracefully.
    """
    d = df.copy()

    def _add_z(colname: str) -> str | None:
        if colname not in d.columns:
            return None
        x = pd.to_numeric(d[colname], errors="coerce").to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            # if all equal/NaN, this metric carries no information: return None
            return None
        zcol = colname + "_z"
        d[zcol] = (x - mu) / sd
        return zcol

    # Always-available trio in pairwise tables
    z_sep    = _add_z("mean_sep_km")
    z_dmis   = _add_z("delta_misfit_B_minus_A")
    z_dazgap = _add_z("delta_azgap_B_minus_A")

    # Optional: connectedness delta (higher is better → subtract with positive weight)
    z_dconn = _add_z("delta_connectedness_B_minus_A")

    # Optional: roughness delta (lower is better → add with positive weight)
    # Only useful if you extend your pairwise compare to compute it.
    z_drough = _add_z("delta_roughness_B_minus_A")

    # Build the score, skipping missing parts
    score = np.zeros(len(d), dtype=float)

    if z_sep   is not None: score += w_sep    * d[z_sep]
    if z_dmis  is not None: score += w_misfit * d[z_dmis]
    if z_dazgap is not None: score += w_azgap  * d[z_dazgap]

    # connectedness: reward increases (so subtract with +w_conn)
    if w_conn > 0 and z_dconn is not None:
        score += (-w_conn) * d[z_dconn]

    # roughness: penalize increases (so add with +w_rough)
    if w_rough > 0 and z_drough is not None:
        score += (w_rough) * d[z_drough]

    d["score"] = score
    return d

def summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    One line per variant: mean/median/SE of metrics + composite score.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def _se(x):
        x = pd.to_numeric(x, errors="coerce")
        n = x.notna().sum()
        return (np.nanstd(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan

    g = df.groupby("variant", dropna=False)
    agg = g.agg(
        n_events          = ("event_id", "nunique"),
        n_rows            = ("event_id", "size"),
        mean_sep_km_mean  = ("mean_sep_km", "mean"),
        mean_sep_km_med   = ("mean_sep_km", "median"),
        mean_sep_km_se    = ("mean_sep_km", _se),
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
    if df_scored is None or df_scored.empty or "score" not in df_scored.columns:
        return (pd.DataFrame(columns=["event_id", "variant", "score"]),
                pd.DataFrame(columns=["variant", "wins"]))

    d = df_scored[np.isfinite(df_scored["score"])].copy()
    if d.empty:
        return (pd.DataFrame(columns=["event_id", "variant", "score"]),
                pd.DataFrame(columns=["variant", "wins"]))

    # idxmin returns index labels; .loc works with labels
    idx = d.groupby("event_id")["score"].idxmin()
    idx = idx.dropna()
    winners = d.loc[idx, ["event_id", "variant", "score"]].reset_index(drop=True)

    win_counts = (winners["variant"]
                  .value_counts()
                  .rename_axis("variant")
                  .reset_index(name="wins")
                  .sort_values("wins", ascending=False))
    return winners, win_counts

# =============================================================================
# Orchestrator + variant factory (pairwise mode)
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

from itertools import product
from typing import Dict, Iterable, Optional

def tweak_config(
    baseline,
    *,
    changes: Optional[Iterable[dict]] = None,
    axes: Optional[Dict[str, Iterable]] = None,
    landgridobj=None,                           # backward-compat convenience
    annual_station_corrections_df=None,         # backward-compat convenience
    include_baseline: bool = False,             # optionally include the baseline itself
    dedupe: bool = True,                        # dedupe by tag() if collisions
) -> Dict[str, "ASLConfig"]:
    """
    Build a dict of ASLConfig variants derived from `baseline`.

    - `changes`: an iterable of override dicts, e.g. [{'Q':100}, {'speed':3.0}]
    - `axes`: a Cartesian sweep, e.g. {'speed':[1.0,3.0], 'Q':[10,100]}
      -> produces all combinations
    - Keys of the returned dict are each config's `tag()` string.
    """

    # Normalize inputs
    variant_specs: list[dict] = []
    if include_baseline:
        variant_specs.append({})

    if changes:
        variant_specs.extend(dict(c) for c in changes)

    if axes:
        keys = list(axes.keys())
        for values in product(*[axes[k] for k in keys]):
            variant_specs.append({k: v for k, v in zip(keys, values)})

    # Back-compat conveniences
    if landgridobj is not None:
        variant_specs.append({"gridobj": landgridobj})
    if annual_station_corrections_df is not None:
        variant_specs.append({"station_correction_dataframe": annual_station_corrections_df})

    # If nothing was requested, return empty dict
    if not variant_specs:
        return {}

    out: Dict[str, "ASLConfig"] = {}
    for spec in variant_specs:
        cfg = baseline.copy(**spec)  # your copy() auto-builds if needed
        key = cfg.tag()              # use canonical config tag as the label

        if key in out and dedupe:
            # Last one wins; collide silently or print if you prefer:
            # print(f"[tweak_config] duplicate tag {key}; overwriting")
            pass
        out[key] = cfg

    return out

def compare_runs(
    baseline_cfg,
    events: Iterable[str | Path],
    variants: Optional[Dict[str, "ASLConfig"] | Iterable["ASLConfig"]] = None,
    *,
    run_single_event=None,
    refine_sector: bool = False,
    topo_kw=None,
    run_if_missing_baseline: bool = True,
    run_if_missing_variants: bool = False,
    w_sep: float = 1.0,
    w_misfit: float = 0.5,
    w_azgap: float = 0.1,
    w_conn: float = 0.0,      # reward ↑ connectedness (use a small positive weight)
    w_rough: float = 0.0,     # penalize ↑ roughness (if you add it to pairwise)
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compare baseline vs variants across events; append pairwise rows under each event
    as 'pairwise_{baseline_tag}_vs_variants.csv'. Then roll up and score.

    Returns: (scored_df, summary_df, win_counts_df) — each may be None if nothing found.
    """
    # --- normalize variants to {tag: cfg} ---
    if variants is None:
        varmap: Dict[str, "ASLConfig"] = {}
    elif isinstance(variants, dict):
        # assume caller used tags/labels as keys already
        varmap = {str(k): v for k, v in variants.items()}
    else:
        varmap = {}
        for cfg in variants:
            try:
                varmap[cfg.tag()] = cfg
            except Exception:
                pass  # skip anything weird

    base_tag = baseline_cfg.tag()
    event_keys = []

    for i, ev in enumerate(events, start=1):
        ev_key = Path(ev).stem
        event_keys.append(ev_key)
        print(f"\n[{i}] {ev_key}")
        event_dir = Path(baseline_cfg.output_base) / ev_key
        event_dir.mkdir(parents=True, exist_ok=True)
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
        for vtag, vcfg in varmap.items():
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
            if alt_csv is None:
                print(f"  [skip] missing variant CSV: {vtag}")
                continue
            try:
                safe_compare(summary_csv, base_csv, alt_csv, label=vcfg.tag())
            except Exception as e:
                print(f"  [compare error] {vcfg.tag()}: {e}")

    # Roll-up from the baseline's output root
    ROOT = Path(baseline_cfg.output_base)
    allcmp = load_all_event_comparisons(ROOT)
    if allcmp.empty:
        print("\nstacked rows: 0, events: 0")
        return None, None, None

    # Filter to the specific pairwise files produced for this baseline & these events
    # (so we don't mix with any other baseline runs under the same OUTPUT_BASE)
    event_keys_set = set(event_keys)
    use = (allcmp["event_id"].astype(str).isin(event_keys_set)) & (
        allcmp.get("runA_tag", "").astype(str).str.contains(base_tag, na=False)
    )
    filtered = allcmp[use].copy()
    print(f"\nstacked rows: {len(filtered)}, events: {filtered['event_id'].nunique() if not filtered.empty else 0}")

    if filtered.empty:
        return None, None, None

    # Score (now includes optional connectedness / roughness)
    scored  = add_composite_score(
        filtered,
        w_sep=w_sep, w_misfit=w_misfit, w_azgap=w_azgap,
        w_conn=w_conn, w_rough=w_rough,
    )
    summary = summarize_variants(scored)
    winners, win_counts = per_event_winner(scored)
    return scored, summary, win_counts

# =============================================================================
# Intrinsic metrics (baseline-free, per run)
# =============================================================================

def _pairwise_gc_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance (km) between successive samples; length n-1."""
    n = len(lat)
    if n <= 1:
        return np.zeros(0, dtype=float)
    seg = np.empty(n-1, dtype=float)
    seg[:] = np.nan
    for i in range(n-1):
        if (np.isfinite(lat[i]) and np.isfinite(lon[i]) and
            np.isfinite(lat[i+1]) and np.isfinite(lon[i+1])):
            seg[i] = _gc_km(lat[i], lon[i], lat[i+1], lon[i+1])
    return seg

def _chord_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Straight-line (km) from first to last valid positions; NaN if insufficient."""
    m = np.isfinite(lat) & np.isfinite(lon)
    if np.count_nonzero(m) < 2:
        return np.nan
    idx = np.where(m)[0]
    i0 = idx[0]
    i1 = idx[-1]
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
    Returns a dict with keys at least:
      - tag, event_id
      - n_samples, valid_frac
      - mean_misfit, mean_azgap, connectedness
      - path_len_km, chord_len_km, roughness_ratio
    """
    p = Path(path)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[intrinsic] read error {p}: {e}")
        return None

    cmap = {c.lower(): c for c in df.columns}
    def has(c: str) -> bool: return c.lower() in cmap
    def col(c: str):
        return pd.to_numeric(df[cmap[c.lower()]], errors="coerce") if has(c) else pd.Series(dtype=float)

    # --- core columns ---
    lat = col("lat").to_numpy() if has("lat") else np.array([])
    lon = col("lon").to_numpy() if has("lon") else np.array([])
    mis = col("misfit").to_numpy() if has("misfit") else np.array([])
    azg = col("azgap").to_numpy() if has("azgap") else np.array([])
    dr  = col("dr").to_numpy() if has("dr") else np.array([])

    n_samples = int(len(df))
    if lat.size and lon.size:
        valid_xy = np.isfinite(lat) & np.isfinite(lon)
        valid_frac = float(valid_xy.mean()) if n_samples > 0 else np.nan
    else:
        valid_xy = np.zeros(n_samples, dtype=bool)
        valid_frac = np.nan

    # --- geometry metrics ---
    if lat.size and lon.size:
        seg = _pairwise_gc_km(lat, lon)
        path_len = float(np.nansum(seg)) if seg.size else np.nan
        chord    = float(_chord_km(lat, lon))
        rough    = float(_roughness_ratio(lat, lon))
    else:
        path_len = chord = rough = np.nan

    # --- misfit/azgap summary ---
    mean_misfit = safe_mean(mis)
    mean_azgap  = safe_mean(azg)

    # --- connectedness (read or compute) ---
    conn = np.nan
    if has("connectedness"):
        conn = safe_mean(col("connectedness").to_numpy())

    if not np.isfinite(conn):
        # pick selection mode based on available signal
        if dr.size == n_samples and np.isfinite(dr).any():
            sel = "dr"; dr_arg, mis_arg = dr, None
        elif mis.size == n_samples and np.isfinite(mis).any():
            sel = "misfit"; dr_arg, mis_arg = None, mis
        else:
            sel = "all"; dr_arg = mis_arg = None

        if lat.size >= 2 and lon.size >= 2:
            try:
                from flovopy.asl.utils import compute_spatial_connectedness
                c = compute_spatial_connectedness(
                    lat, lon,
                    dr=dr_arg, misfit=mis_arg,
                    select_by=sel,
                    top_frac=0.15, min_points=12, max_points=200,
                    fallback_if_empty=True,
                )
                conn = float(c.get("score", np.nan))
            except Exception as e:
                # keep it robust; don't fail the whole row
                print(f"[intrinsic] connectedness compute failed for {p.name}: {e}")
                conn = np.nan
        else:
            conn = np.nan

    # --- tag from filename ---
    tag = p.stem
    if tag.startswith("source_"):
        tag = tag[len("source_"):]
    if tag.endswith("_refined"):
        tag = tag[:-len("_refined")]

    # --- infer event_id from directory: .../<event_id>/<products_dir>/<file.csv> ---
    try:
        event_id = p.parent.parent.name
    except Exception:
        event_id = p.parent.name

    return {
        "event_id": event_id,
        "tag": tag,
        "path": str(p),
        "n_samples": n_samples,
        "valid_frac": valid_frac,
        "mean_misfit": mean_misfit,
        "mean_azgap":  mean_azgap,
        "connectedness": float(conn),
        "path_len_km": path_len,
        "chord_len_km": chord,
        "roughness_ratio": rough,
    }

# =============================================================================
# Baseline-free (absolute) scoring
# =============================================================================

def _robust_scale_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = np.zeros_like(x, dtype=float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return z  # no info → neutral
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

    Default weights favor misfit, lightly penalize azgap & roughness,
    and lightly reward connectedness and valid fraction (negative weights).
    """
    if df_abs is None or df_abs.empty:
        return df_abs

    d = df_abs.copy()

    if weights is None:
        weights = {
            "mean_misfit":       0.8,   # lower better
            "mean_azgap":        0.2,   # lower better
            "roughness_ratio":   0.2,   # lower better
            "connectedness":    -0.2,   # higher better (negative weight)
            "valid_frac":       -0.2,   # higher better (negative weight)
        }

    out = []
    # group per event_id for within-event normalization
    for ev, g in d.groupby("event_id", dropna=False):
        gg = g.copy()

        Z: Dict[str, np.ndarray] = {}
        # lower-is-better metrics
        for col in ("mean_misfit", "mean_azgap", "roughness_ratio"):
            arr = pd.to_numeric(gg.get(col, np.nan), errors="coerce").to_numpy()
            z = _robust_scale_vector(arr)
            Z[col] = np.clip(z, -cap, cap)

        # higher-is-better metrics (we'll apply negative weights)
        for col in ("connectedness", "valid_frac"):
            arr = pd.to_numeric(gg.get(col, np.nan), errors="coerce").to_numpy()
            z = _robust_scale_vector(arr)
            Z[col] = np.clip(z, -cap, cap)

        # composite score
        score = np.zeros(len(gg))
        for col, w in weights.items():
            if col in Z:
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

    def _se(x):
        x = pd.to_numeric(x, errors="coerce")
        n = x.notna().sum()
        return (np.nanstd(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan

    g = df_abs_scored.groupby("tag")
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
    winners = d.loc[idx, ["event_id","tag","score_abs"]].reset_index(drop=True)

    win_counts = (winners["tag"].value_counts()
                  .rename_axis("tag").reset_index(name="wins")
                  .sort_values("wins", ascending=False))
    return winners, win_counts

# =============================================================================
# Build intrinsic table (ensures CSVs exist, then compute intrinsic metrics)
# =============================================================================

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
        if bcsv is not None and Path(bcsv).exists():
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
            if vcsv is not None and Path(vcsv).exists():
                r = intrinsic_metrics_from_csv(vcsv)
                if r: rows.append(r)

    return pd.DataFrame(rows)

# =============================================================================
# Crawler for absolute mode (discover all runs already on disk)
# =============================================================================

def crawl_intrinsic_runs(root: str | Path, refined: bool = False) -> pd.DataFrame:
    """
    Walk output tree under `root` and compute intrinsic (baseline-free) metrics
    for every per-run CSV we can recognize. Returns one row per (event_id, tag).
    refined: if True, only use refined CSV files. If False, do not use any refined CSV files.
    """
    root = Path(root)
    rows: List[dict] = []

    def is_per_run_csv(p: Path) -> bool:
        if p.suffix.lower() != ".csv":
            return False
        nm = p.name
        if nm.startswith("pairwise_"):
            return False
        if nm.endswith("_comparisons.csv"):
            return False
        return True

    patterns = [
        #"**/source_*_refined.csv",
        "**/source_*.csv",
        #"**/*_refined.csv",
        #"**/*.csv",
    ]

    seen: set = set()
    for pat in patterns:
        for csv in root.glob(pat):
            if refined:
                if not 'refined' in str(csv):
                    continue
            elif 'refined' in str(csv):
                continue

            if not is_per_run_csv(csv):
                continue
            r = intrinsic_metrics_from_csv(csv)
            if not r:
                continue
            key = (r["event_id"], r["tag"], r.get("path", str(csv)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    return pd.DataFrame(rows)
