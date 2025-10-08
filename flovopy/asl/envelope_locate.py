"""
ASL (Amplitude Source Location) analysis tools
----------------------------------------------
Supports:
1) Per-event lag extraction (cross-correlation).
2) Event location on a Grid (constant speed).
3) Per-event optional fit: speed increasing linearly with Δd from dome.
4) Suite-level analysis: overall best constant / linear velocity.
5) Pairwise lag stability analysis and velocity vs Δd studies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable, List, Union
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import Stream, Inventory

from flovopy.processing.envelopes import (
    align_waveforms,
    align_waveforms_global,
    envelopes_stream,
    envelope_delays,
    locate_with_grid_from_delays,
)
from flovopy.asl.distances import compute_or_load_distances


# -------------------------- small helpers --------------------------

def _clean_sta(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()

def _lags_to_str(d: Dict[str, float]) -> str:
    return ", ".join(
        f"{k}:{v:+.2f}s"
        for k, v in sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )

def _resolve_dome_idx(grid, dome_location) -> int:
    if isinstance(dome_location, int):
        return int(dome_location)
    glon = np.asarray(grid.gridlon).ravel()
    glat = np.asarray(grid.gridlat).ravel()
    if isinstance(dome_location, dict) and {"lon", "lat"} <= set(dome_location):
        lon, lat = float(dome_location["lon"]), float(dome_location["lat"])
    elif isinstance(dome_location, (tuple, list)) and len(dome_location) == 2:
        lon, lat = float(dome_location[0]), float(dome_location[1])
    else:
        raise ValueError("dome_location must be node index, {'lon','lat'}, or (lon,lat).")
    return int(np.argmin((glon - lon) ** 2 + (glat - lat) ** 2))

def _collapse_distances_by_station(node_dists: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Collapse seed ids -> simple STA key; prefer first seen."""
    out: Dict[str, np.ndarray] = {}
    for seed_id, vec in node_dists.items():
        parts = str(seed_id).split(".")
        sta = parts[1] if len(parts) >= 2 else parts[0]
        if sta not in out:
            out[sta] = np.asarray(vec, float)
    return out

def _weighted_median(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if x.size == 0 or np.sum(w) <= 0:
        return np.nan
    idx = np.argsort(x); xs, ws = x[idx], w[idx]
    p = np.cumsum(ws) / np.sum(ws)
    j = int(np.searchsorted(p, 0.5))
    return float(xs[min(max(j, 0), xs.size - 1)])

def _linfit_fixed_intercept(x, y, b):
    """Least-squares slope with intercept fixed: y = b + m x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    den = float(np.sum(x * x))
    if den <= 0:
        return np.nan
    m = float(np.sum(x * (y - b)) / den)
    yhat = b + m * x
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    return m, r2


# -------------------------- per-event --------------------------

def process_event(
    st: Stream,
    gridobj,
    inventory: Inventory,
    cache_dir: str,
    dome_location: Union[int, Tuple[float, float], Dict[str, float]],
    *,
    event_idx: int = 0,
    # envelope / xcorr knobs
    smooth_s: float = 1.0,
    max_lag_s: float = 8.0,
    min_corr: float = 0.5,
    # locator knobs
    c_range: Tuple[float, float] = (0.1, 5.0),
    n_c: int = 80,
    min_delta_d_km: float = 0.5,
    min_abs_lag_s: float = 0.15,
    delta_d_weight: bool = True,
    c_phys: Tuple[float, float] = (0.5, 3.5),
    auto_ref: bool = True,
    # output / plotting
    topo_dem_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    # per-event linear speed fit
    linear_fit_intercept: Optional[float] = 0.5,  # km/s, None => free intercept
) -> Dict[str, object]:
    """
    Align traces (reference + global), compute pairwise delays, run grid locator.
    Also:
      • report Stage-A best speed at the dome (constant-c assumption),
      • fit per-event linear c(Δd)=b+mΔd using ref pairs and the Stage-A intercept.
    """
    if output_dir is None:
        raise ValueError("process_event: output_dir is required (for plots and per-event artifacts).")

    # per-event directories
    eventdir = Path(output_dir) / str(event_idx)
    eventdir.mkdir(parents=True, exist_ok=True)
    score_vs_c_png = str(eventdir / "score_vs_c.png")
    ccf_plot_dir   = str(eventdir / "ccf_plots")
    topo_png       = str(eventdir / "best_location_topo.png")

    # --- 1) reference-based alignment (kept for comparison/debug)
    aligned_st_ref, lags_ref = align_waveforms(
        st.copy(),
        max_lag_s=max_lag_s,
        smooth_s=smooth_s,
        decimate_to=None,
    )

    # --- 2) global alignment (used for locator)
    aligned_st_glb, lags_glb, delays = align_waveforms_global(
        st.copy(),
        max_lag_s=max_lag_s,
        min_corr=min_corr,
        smooth_s=smooth_s,
        decimate_to=None,
    )

    # --- 2b) CCFs for plots / debug
    env_st = envelopes_stream(st, smooth_s=smooth_s, decimate_to=None)
    _, ccfs = envelope_delays(env_st, max_lag_s=max_lag_s, min_corr=min_corr, return_ccfs=True)

    # --- 3) grid locator (Stage A @ dome + Stage B all nodes), writes score-vs-c PNG
    loc = locate_with_grid_from_delays(
        gridobj,
        cache_dir,
        inventory=inventory,
        delays=delays,                 # <- from global alignment
        stream=st,
        c_range=c_range,
        n_c=n_c,
        debug=True,
        ccfs=ccfs,
        plot_ccf_dir=ccf_plot_dir,
        plot_score_vs_c=score_vs_c_png,  # <- PNG
        dome_location=dome_location,      # ensures Stage-A anchored at dome if usable
        verbose=False,
        # Stage-A guards
        min_delta_d_km=min_delta_d_km,
        min_abs_lag_s=min_abs_lag_s,
        delta_d_weight=delta_d_weight,
        c_phys=c_phys,
        auto_ref=auto_ref,
        # topo overlay
        topo_map_out=topo_png,
        topo_dem_path=str(topo_dem_path) if topo_dem_path else None,
    )

    # --- 3a) extract dome-assumption best c from Stage-A scan
    dbg = loc.get("debug_info", {}) or {}
    scores_c: List[Dict[str, float]] = dbg.get("scores_c", []) or []
    c_at_dome_kms = np.nan
    score_at_dome = np.nan
    if scores_c:
        vals = np.array([s["score"] for s in scores_c], float)
        vals[~np.isfinite(vals)] = np.inf
        if np.isfinite(vals).any():
            k = int(np.argmin(vals))
            c_at_dome_kms = float(scores_c[k]["c"])
            score_at_dome = float(scores_c[k]["score"])

    # --- 3b) per-event linear speed-with-distance fit (dome-centered)
    # Using the Stage-A intercept 'a_best' and ref pairs from the Stage-A subset.
    linear_fit = None
    try:
        # distances collapsed per station, and dome node
        node_dists, _, _ = compute_or_load_distances(
            gridobj, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=True
        )
        dist_by_sta = _collapse_distances_by_station(node_dists)
        dome_idx = _resolve_dome_idx(gridobj, dome_location)

        # reconstruct ref→station τ using the same ref chosen by Stage-A
        ref = dbg.get("ref", None)
        if ref is None:
            # fall back: try to infer a plausible ref (zero lag) from lags_glb
            zeroers = [k for k, v in lags_glb.items() if abs(float(v)) < 1e-9]
            ref = zeroers[0] if zeroers else (sorted(lags_glb.keys())[0] if lags_glb else None)

        # rekey pair delays to station keys and build τ_ref[s]
        tau_ref: Dict[str, float] = {}
        if isinstance(delays, dict) and ref is not None:
            for (si, sj), (lag, corr) in delays.items():
                # delays keys may be seed IDs -> collapse to STA
                si_sta = _clean_sta(si.split(".")[1] if "." in si else si)
                sj_sta = _clean_sta(sj.split(".")[1] if "." in sj else sj)
                if si_sta == sj_sta:
                    continue
                if si_sta == ref:
                    tau_ref[sj_sta] = float(+lag)
                elif sj_sta == ref:
                    tau_ref[si_sta] = float(-lag)

        # Stage-A intercept from locator
        a_best = float(dbg.get("a_best")) if "a_best" in dbg else float(loc.get("intercept", np.nan))

        # Build Δd (km) w.r.t dome *between ref and station*, and apparent c_app = Δd / |τ - a_best|
        x_dd, y_capp = [], []
        for s, tau in tau_ref.items():
            if s not in dist_by_sta or ref not in dist_by_sta:
                continue
            d_s = float(dist_by_sta[s][dome_idx])
            d_r = float(dist_by_sta[ref][dome_idx])
            dd = abs(d_s - d_r)  # km
            if not (np.isfinite(dd) and dd > 0):
                continue
            denom = abs(float(tau) - a_best)
            if denom <= 0:
                continue
            capp = dd / denom
            if np.isfinite(capp) and 0.05 <= capp <= 7.0:
                x_dd.append(dd); y_capp.append(capp)

        if len(x_dd) >= 2:
            if linear_fit_intercept is None:
                # free intercept (simple polyfit)
                coeff = np.polyfit(np.asarray(x_dd), np.asarray(y_capp), deg=1)
                m_fit, b_fit = float(coeff[0]), float(coeff[1])
                yhat = b_fit + m_fit * np.asarray(x_dd)
                sse = float(np.sum((np.asarray(y_capp) - yhat) ** 2))
                sst = float(np.sum((np.asarray(y_capp) - np.mean(y_capp)) ** 2))
                r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
            else:
                b_fit = float(linear_fit_intercept)
                m_fit, r2 = _linfit_fixed_intercept(np.asarray(x_dd), np.asarray(y_capp), b_fit)
            linear_fit = dict(
                slope=m_fit,
                intercept=b_fit,
                r2=r2,
                n=len(x_dd),
            )
    except Exception:
        # keep linear_fit=None if anything goes sideways; we still return the core results
        pass

    # --- compact summary row for suite CSV
    summary_row = {
        "event_idx": int(event_idx),
        "c_at_dome_kms": float(c_at_dome_kms),           # Stage-A minimum @ dome
        "score_at_dome": float(score_at_dome),
        "c_at_bestnode_kms": float(loc.get("speed", np.nan)),
        "score_bestnode": float(loc.get("score", np.nan)),
        "n_pairs_used": int(loc.get("n_pairs", 0) or 0),
        "bestnode_lon": float(loc.get("lon", np.nan)),
        "bestnode_lat": float(loc.get("lat", np.nan)),
        "bestnode_elev_m": float(loc.get("elev_m", np.nan)),
        "lags_ref": _lags_to_str(lags_ref),
        "lags_global": _lags_to_str(lags_glb),
        "score_vs_c_png": score_vs_c_png,
        # per-event linear c(Δd) fit summary (if available)
        "lin_c_slope": float(linear_fit["slope"]) if linear_fit else np.nan,
        "lin_c_intercept": float(linear_fit["intercept"]) if linear_fit else np.nan,
        "lin_c_r2": float(linear_fit["r2"]) if linear_fit else np.nan,
        "lin_c_n": int(linear_fit["n"]) if linear_fit else 0,
    }

    return {
        "lags_ref": lags_ref,
        "lags_global": lags_glb,
        "delays": delays,                   # keep for any post-hoc analysis
        "locator_result": loc,
        "linear_fit": linear_fit,
        "summary_row": summary_row,
    }


# -------------------------- suite-level --------------------------

def summarize_suite(rows: List[Dict[str, object]], out_csv: str) -> pd.DataFrame:
    """Save per-event rows to CSV and return a DataFrame."""
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# -------------------------- pairwise stability --------------------------

def parse_lags_column(series: pd.Series) -> List[Dict[str, float]]:
    parsed: List[Dict[str, float]] = []
    for txt in series.fillna(""):
        ev: Dict[str, float] = {}
        for part in str(txt).split(","):
            if ":" not in part:
                continue
            sta, lag = part.strip().split(":")
            try:
                ev[_clean_sta(sta)] = float(str(lag).replace("s", ""))
            except ValueError:
                continue
        parsed.append(ev)
    return parsed

def compute_pairwise_diffs(lag_dicts: List[Dict[str, float]]) -> List[Dict[Tuple[str, str], float]]:
    diffs: List[Dict[Tuple[str, str], float]] = []
    for lagmap in lag_dicts:
        pairs: Dict[Tuple[str, str], float] = {}
        stations = sorted(lagmap.keys())
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                si, sj = stations[i], stations[j]
                pairs[(si, sj)] = lagmap[si] - lagmap[sj]
        diffs.append(pairs)
    return diffs

def summarize_pairwise(pairdiffs: List[Dict[Tuple[str, str], float]], method: str) -> pd.DataFrame:
    records = []
    all_pairs = sorted({p for ev in pairdiffs for p in ev})
    for (a, b) in all_pairs:
        vals = [ev[(a, b)] for ev in pairdiffs if (a, b) in ev]
        if not vals:
            continue
        arr = np.array(vals, float)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        keep = arr[np.abs(arr - med) <= 3 * mad] if mad > 0 else arr
        if len(keep) < 2:
            continue
        records.append(dict(
            pair=f"{a}-{b}",
            sta_a=a, sta_b=b,
            n=len(keep),
            median_s=float(np.median(keep)),
            mean_s=float(np.mean(keep)),
            std=float(np.std(keep)),
            mad=float(mad),
            min=float(np.min(keep)),
            max=float(np.max(keep)),
            method=method,
        ))
    return pd.DataFrame.from_records(records)


# -------------------------- speed estimation from stable pairs --------------------------

def estimate_speed_from_stable_pairs(
    *,
    grid,
    cache_dir: str,
    inventory: Inventory,
    dome_location: Union[int, Tuple[float, float], Dict[str, float]],
    stable_df: Optional[pd.DataFrame] = None,
    stable_pairs_csv: Optional[str] = None,
    # which lag stat
    use_value: str = "mean_s",        # tries this, then falls back to median_s/mean/median
    # weights
    weight_with: Optional[str] = "mad_scaled",  # or "std" or None
    uncert_floor_s: float = 0.05,
    # guards
    min_delta_d_km: float = 0.2,      # drop nearly coincident baselines
    min_abs_tau_s: float = 0.05,      # avoid 1/0 blow-ups
    c_bounds: Tuple[float, float] = (0.2, 7.0),
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Source fixed at dome. For each stable pair (a,b), compute c_ab=|Δd|/|Δτ|.
    Combine robustly to one speed with a bootstrap CI.
    Returns {'per_pair', 'speed_km_s', 'ci_68', 'n_pairs_used'}.
    """
    # load table
    if stable_df is None and stable_pairs_csv:
        stable_df = pd.read_csv(stable_pairs_csv)
    if stable_df is None or stable_df.empty:
        raise ValueError("No stable pairs provided.")

    df = stable_df.copy()

    # ensure station columns exist
    if not {"sta_a", "sta_b"}.issubset(df.columns):
        if "pair" not in df.columns:
            raise ValueError("stable_df must have 'sta_a'/'sta_b' or 'pair'.")
        ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
        ab.columns = ["sta_a", "sta_b"]
        df = pd.concat([df, ab], axis=1)

    df["sta_a"] = df["sta_a"].apply(_clean_sta)
    df["sta_b"] = df["sta_b"].apply(_clean_sta)

    # choose lag column
    if use_value not in df.columns:
        if use_value.endswith("_s"):
            base = use_value[:-2]
            if base in df.columns:
                df[use_value] = df[base]
            elif "mean_s" in df.columns:
                use_value = "mean_s"
            elif "median_s" in df.columns:
                use_value = "median_s"
            elif "mean" in df.columns:
                use_value = "mean"
            elif "median" in df.columns:
                use_value = "median"
            else:
                raise ValueError("No lag column found (mean_s/median_s/mean/median).")

    # distances & dome node
    node_dists, _, _ = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=True
    )
    dist_by_sta = _collapse_distances_by_station(node_dists)
    dome_idx = _resolve_dome_idx(grid, dome_location)

    # per-pair rows
    rows, rejects = [], []
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]), str(r["sta_b"])
        if a not in dist_by_sta or b not in dist_by_sta:
            rejects.append(("missing_station", a, b)); continue
        da = float(dist_by_sta[a][dome_idx]); db = float(dist_by_sta[b][dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            rejects.append(("nan_distance", a, b)); continue

        delta_d = abs(da - db)  # km
        tau = float(r[use_value])  # s
        if not np.isfinite(tau):
            rejects.append(("nan_tau", a, b)); continue
        if delta_d < min_delta_d_km:
            rejects.append(("small_delta_d", a, b)); continue
        if abs(tau) < min_abs_tau_s:
            rejects.append(("small_tau", a, b)); continue

        c = delta_d / abs(tau)
        if not (c_bounds[0] <= c <= c_bounds[1]):
            rejects.append(("c_bounds", a, b)); continue

        # simple weight: uncertainty if available -> inverse variance; else leverage
        if weight_with and (weight_with in r) and np.isfinite(r[weight_with]) and (r[weight_with] > 0):
            sig = max(float(r[weight_with]), uncert_floor_s)
            w = 1.0 / (sig ** 2)
        else:
            w = max(delta_d, 1e-6)

        rows.append(dict(
            sta_a=a, sta_b=b,
            delta_d_km=delta_d,
            delta_tau_s=abs(tau),
            c_pair_km_s=c,
            weight=float(w),
        ))

    per_pair = pd.DataFrame(rows)
    if verbose:
        print(f"[pairs] kept={len(per_pair)}  rejected={len(rejects)}")

    if per_pair.empty:
        return {"per_pair": per_pair, "speed_km_s": np.nan, "ci_68": (np.nan, np.nan), "n_pairs_used": 0}

    # robust center + bootstrap CI
    x = per_pair["c_pair_km_s"].values
    w = per_pair["weight"].values
    c_hat = _weighted_median(x, w)

    rng = np.random.default_rng(42)
    B = 500 if len(x) >= 6 else 200
    p = w / np.sum(w)
    boots = []
    for _ in range(B):
        idx = rng.choice(len(x), size=len(x), replace=True, p=p)
        boots.append(_weighted_median(x[idx], w[idx]))
    ci_lo, ci_hi = float(np.nanpercentile(boots, 16)), float(np.nanpercentile(boots, 84))

    return {
        "per_pair": per_pair.sort_values("delta_d_km").reset_index(drop=True),
        "speed_km_s": float(c_hat),
        "ci_68": (ci_lo, ci_hi),
        "n_pairs_used": int(per_pair.shape[0]),
    }


# -------------------------- plot helper --------------------------

def plot_speed_vs_distance(res_or_df, intercept: float = 0.5, cmax: float = 7.0, ax=None):
    """
    Scatter c_pair vs Δd with a linear fit whose intercept is fixed at `intercept`.
    Accepts either:
      - the dict returned by estimate_speed_from_stable_pairs (key 'per_pair'), or
      - the per_pair DataFrame itself.
    Returns {"slope","intercept","r2","n"}.
    """
    if isinstance(res_or_df, dict):
        if "per_pair" not in res_or_df:
            raise KeyError("Expected dict with key 'per_pair'.")
        df = res_or_df["per_pair"]
    else:
        df = res_or_df

    need = {"delta_d_km", "c_pair_km_s"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}; got {list(df.columns)}")

    x = df["delta_d_km"].to_numpy()
    y = df["c_pair_km_s"].to_numpy()
    msk = np.isfinite(x) & np.isfinite(y) & (y <= float(cmax))
    x = x[msk]; y = y[msk]
    if x.size < 2:
        raise ValueError(f"Not enough points after filtering (n={x.size}).")

    b = float(intercept)
    m_fit, r2 = _linfit_fixed_intercept(x, y, b)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=28, alpha=0.9, label=f"pairs (n={x.size})")
    xx = np.linspace(0.0, max(x) * 1.05, 200)
    ax.plot(xx, b + m_fit * xx, "r-", lw=2,
            label=f"fit (b={b:.2f}): y={m_fit:.2f}x+{b:.2f}, R²≈{r2:.3f}")
    ax.set_xlabel("Δd from dome between stations (km)")
    ax.set_ylabel("Apparent speed c_pair (km/s)")
    ax.set_ylim(0, float(cmax))
    ax.grid(alpha=0.3); ax.legend(loc="best")
    if ax is None:
        plt.tight_layout(); plt.show()

    return {"slope": float(m_fit), "intercept": b, "r2": float(r2), "n": int(x.size)}


def build_per_pair_from_stable(
    *,
    grid,
    inventory,
    cache_dir: str,
    dome_location,
    stable_df: pd.DataFrame,
    use_value: str = "mean_s",          # or "median_s"
    c_bounds: Tuple[float, float] = (0.2, 7.0),
    min_delta_d_km: float = 0.0,
    min_abs_tau_s: float = 0.0,
) -> pd.DataFrame:
    """
    Turn a 'stable pairs' table (from summarize_pairwise) into a per-pair table with:
      ['pair','sta_a','sta_b','delta_d_km','delta_tau_s','c_pair_km_s','n','std','mad',
       'w_inv_std','w_inv_mad','w_inv_n']
    Δd is computed using dome_location and the grid distances.
    """
    df = stable_df.copy()

    # ensure sta_a/sta_b
    if not {"sta_a","sta_b"}.issubset(df.columns):
        if "pair" not in df.columns:
            raise ValueError("stable_df must have 'sta_a'/'sta_b' or 'pair'")
        ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
        ab.columns = ["sta_a","sta_b"]
        df = pd.concat([df, ab], axis=1)
    df["sta_a"] = df["sta_a"].apply(_clean_sta)
    df["sta_b"] = df["sta_b"].apply(_clean_sta)

    # pick lag column
    if use_value not in df.columns:
        # fallbacks
        if use_value == "mean_s" and "mean" in df.columns:
            df["mean_s"] = df["mean"]
        elif use_value == "median_s" and "median" in df.columns:
            df["median_s"] = df["median"]
        if use_value not in df.columns:
            raise ValueError(f"Column '{use_value}' not found in stable_df")

    # distances & dome node
    node_dists, _, _ = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=True
    )
    dist_by_sta = _collapse_distances_by_station(node_dists)
    dome_idx = _resolve_dome_idx(grid, dome_location)

    rows = []
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]), str(r["sta_b"])
        if a not in dist_by_sta or b not in dist_by_sta:
            continue
        da = float(dist_by_sta[a][dome_idx])
        db = float(dist_by_sta[b][dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            continue
        delta_d = abs(da - db)
        tau = float(r[use_value])
        if not np.isfinite(tau):
            continue
        if (delta_d < min_delta_d_km) or (abs(tau) < min_abs_tau_s):
            continue

        c = delta_d / abs(tau) if abs(tau) > 0 else np.inf
        if not (c_bounds[0] <= c <= c_bounds[1]):
            continue

        n   = int(r["n"]) if "n" in r and np.isfinite(r["n"]) else 1
        std = float(r["std"]) if "std" in r and np.isfinite(r["std"]) else np.nan
        mad = float(r["mad"]) if "mad" in r and np.isfinite(r["mad"]) else np.nan

        # weights with floors to avoid infinities
        eps = 1e-6
        w_inv_std = 1.0 / max(std, eps) if np.isfinite(std) else 1.0
        w_inv_mad = 1.0 / max(mad, eps) if np.isfinite(mad) else 1.0
        w_inv_n   = 1.0 / max(n, 1)

        rows.append(dict(
            pair=f"{a}-{b}",
            sta_a=a, sta_b=b,
            delta_d_km=delta_d,
            delta_tau_s=abs(tau),
            c_pair_km_s=c,
            n=n, std=std, mad=mad,
            w_inv_std=w_inv_std,
            w_inv_mad=w_inv_mad,
            w_inv_n=w_inv_n,
        ))
    return pd.DataFrame(rows)


def _wls_fixed_intercept(x, y, w, intercept):
    """Weighted least squares with fixed intercept b: y = b + m x."""
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    b = float(intercept)
    num = np.sum(w * x * (y - b))
    den = np.sum(w * x * x)
    if den <= 0:
        return np.nan, np.nan
    m = num / den
    yhat = b + m * x
    # weighted R² (using weighted SST)
    ybar = np.sum(w * y) / np.sum(w)
    sse = np.sum(w * (y - yhat) ** 2)
    sst = np.sum(w * (y - ybar) ** 2)
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    return float(m), float(r2)


# --- Weighted least squares with optional fixed intercept ---

def _wls_fit(x, y, w=None, intercept=None):
    """
    Weighted least squares for y = b + m*x.
    - If intercept is None: fit both slope m and intercept b.
    - If intercept is a number: keep b fixed and fit only m.
    Returns (m, b, r2).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if w is None:
        w = np.ones_like(x, float)
    else:
        w = np.asarray(w, float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # mask invalid rows
    msk = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[msk], y[msk], w[msk]
    if x.size < 2 or np.sum(w) <= 0:
        return np.nan, np.nan, np.nan

    if intercept is None:
        # Fit both m and b: solve (X' W X) beta = X' W y
        X = np.column_stack([x, np.ones_like(x)])
        WX = X * w[:, None]
        Wy = y * w
        try:
            beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
            m_hat = float(beta[0])
            b_hat = float(beta[1])
        except Exception:
            return np.nan, np.nan, np.nan
    else:
        # Fixed intercept b: closed-form for m
        b_hat = float(intercept)
        num = np.sum(w * x * (y - b_hat))
        den = np.sum(w * x * x)
        if den <= 0:
            return np.nan, b_hat, np.nan
        m_hat = float(num / den)

    # Weighted R²
    yhat = b_hat + m_hat * x
    ybar = np.sum(w * y) / np.sum(w)
    sse = np.sum(w * (y - yhat) ** 2)
    sst = np.sum(w * (y - ybar) ** 2)
    r2  = float(1.0 - sse / sst) if sst > 0 else np.nan
    return m_hat, b_hat, r2

def plot_six_weighted_fits(
    per_pair_mean: pd.DataFrame,
    per_pair_median: pd.DataFrame,
    *,
    intercept: float = 0.5,
    cmax: float = 7.0,
    ax=None,
):
    """
    Plot apparent speed vs Δd scatter (using mean_s version) and overlay 6 WLS lines:
      (mean_s vs median_s) × (weights = 1/std, 1/mad, 1/n), all with fixed intercept.
    """
    # choose an axis
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        created = True

    # base scatter (mean version, just for visual reference)
    dm = per_pair_mean["delta_d_km"].to_numpy()
    cm = per_pair_mean["c_pair_km_s"].to_numpy()
    msk = np.isfinite(dm) & np.isfinite(cm) & (cm <= float(cmax))
    ax.scatter(dm[msk], cm[msk], s=22, alpha=0.7, label=f"pairs (mean_s) n={msk.sum()}")

    styles = {
        ("mean_s","w_inv_std"):   ("-",  "mean · 1/std"),
        ("mean_s","w_inv_mad"):   ("--", "mean · 1/mad"),
        ("mean_s","w_inv_n"):     (":",  "mean · 1/n"),
        ("median_s","w_inv_std"): ("-",  "median · 1/std"),
        ("median_s","w_inv_mad"): ("--", "median · 1/mad"),
        ("median_s","w_inv_n"):   (":",  "median · 1/n"),
    }

    def _fit_and_plot(pp: pd.DataFrame, weight_col: str, label: str, ls: str):
        x = pp["delta_d_km"].to_numpy()
        y = pp["c_pair_km_s"].to_numpy()
        w = pp[weight_col].to_numpy() if weight_col in pp.columns else np.ones_like(x)
        msk = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (y <= float(cmax)) & (w > 0)
        if msk.sum() < 2:
            return
        m, r2 = _wls_fixed_intercept(x[msk], y[msk], w[msk], intercept)
        if not np.isfinite(m):
            return
        xx = np.linspace(0.0, np.nanmax(x[msk]) * 1.05, 200)
        ax.plot(xx, intercept + m * xx, ls, lw=2, label=f"{label} (m={m:.2f}, R²={r2:.2f})")

    # 3 fits using mean_s-derived table
    _fit_and_plot(per_pair_mean,   "w_inv_std", "mean · 1/std", styles[("mean_s","w_inv_std")][0])
    _fit_and_plot(per_pair_mean,   "w_inv_mad", "mean · 1/mad", styles[("mean_s","w_inv_mad")][0])
    _fit_and_plot(per_pair_mean,   "w_inv_n",   "mean · 1/n",   styles[("mean_s","w_inv_n")][0])

    # 3 fits using median_s-derived table
    _fit_and_plot(per_pair_median, "w_inv_std", "median · 1/std", styles[("median_s","w_inv_std")][0])
    _fit_and_plot(per_pair_median, "w_inv_mad", "median · 1/mad", styles[("median_s","w_inv_mad")][0])
    _fit_and_plot(per_pair_median, "w_inv_n",   "median · 1/n",   styles[("median_s","w_inv_n")][0])

    ax.set_xlabel("Δd from dome between stations (km)")
    ax.set_ylabel("Apparent speed c_pair = |Δd| / |Δτ| (km/s)")
    ax.set_ylim(0, float(cmax))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if created:
        plt.tight_layout()
        plt.show()

def plot_speed_vs_distance(res_or_df, intercept=None, cmax=7.0, ax=None):
    """
    Scatter c_pair vs Δd; WLS fit with optional fixed intercept.
    - intercept=None  -> fit slope & intercept
    - intercept=0.5   -> fix intercept to 0.5 km/s
    """
    import matplotlib.pyplot as plt
    # accept dict or DataFrame
    df = res_or_df["per_pair"] if isinstance(res_or_df, dict) else res_or_df

    # columns & filtering
    need = {"delta_d_km", "c_pair_km_s"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing {missing} in per_pair DataFrame.")
    x = df["delta_d_km"].to_numpy()
    y = df["c_pair_km_s"].to_numpy()
    msk = np.isfinite(x) & np.isfinite(y) & (y <= float(cmax))
    x, y = x[msk], y[msk]
    if x.size < 2:
        raise ValueError("Not enough points to fit.")

    # Fit (unweighted here; pass weights if you want)
    m_fit, b_fit, r2 = _wls_fit(x, y, w=None, intercept=intercept)

    # Plot
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5)); created = True
    ax.scatter(x, y, s=28, alpha=0.9, label=f"pairs (n={x.size})")
    xx = np.linspace(0.0, np.nanmax(x) * 1.05, 200)
    yy = b_fit + m_fit * xx
    label = (f"fit free b: y={m_fit:.2f}x+{b_fit:.2f}, R²≈{r2:.3f}"
             if intercept is None else
             f"fit b={intercept:.2f}: y={m_fit:.2f}x+{b_fit:.2f}, R²≈{r2:.3f}")
    ax.plot(xx, yy, lw=2, label=label)
    ax.set_xlabel("Δd from dome between stations (km)")
    ax.set_ylabel("Apparent speed c_pair (km/s)")
    ax.set_ylim(0, float(cmax))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if created:
        plt.tight_layout()
        plt.show()
    return {"slope": m_fit, "intercept": b_fit, "r2": r2, "n": int(x.size)}

def plot_six_weighted_fits(
    per_pair_mean: pd.DataFrame,
    per_pair_median: pd.DataFrame,
    *,
    intercept: float | None = 0.5,   # None => free intercept
    cmax: float = 7.0,
    ax=None,
):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8)); created = True

    def _scatter_base(pp, label):
        x = pp["delta_d_km"].to_numpy()
        y = pp["c_pair_km_s"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & (y <= float(cmax))
        ax.scatter(x[m], y[m], s=18, alpha=0.55, label=label)

    _scatter_base(per_pair_mean,   f"pairs mean_s (n={len(per_pair_mean)})")
    _scatter_base(per_pair_median, f"pairs median_s (n={len(per_pair_median)})")

    def _fit_and_plot(pp, wcol, label, ls):
        if wcol not in pp.columns:
            return
        x = pp["delta_d_km"].to_numpy()
        y = pp["c_pair_km_s"].to_numpy()
        w = pp[wcol].to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (y <= float(cmax)) & (w > 0)
        if m.sum() < 2:
            return
        m_fit, b_fit, r2 = _wls_fit(x[m], y[m], w=w[m], intercept=intercept)
        if not np.isfinite(m_fit):
            return
        xx = np.linspace(0.0, np.nanmax(x[m]) * 1.05, 200)
        yy = b_fit + m_fit * xx
        if intercept is None:
            sub = f"free b={b_fit:.2f}"
        else:
            sub = f"b={intercept:.2f}"
        ax.plot(xx, yy, ls, lw=2, label=f"{label} ({sub}, m={m_fit:.2f}, R²={r2:.2f})")

    # mean_s weights
    _fit_and_plot(per_pair_mean,   "w_inv_std", "mean · 1/std", "-")
    _fit_and_plot(per_pair_mean,   "w_inv_mad", "mean · 1/mad", "--")
    _fit_and_plot(per_pair_mean,   "w_inv_n",   "mean · 1/n",   ":")
    # median_s weights
    _fit_and_plot(per_pair_median, "w_inv_std", "median · 1/std", "-")
    _fit_and_plot(per_pair_median, "w_inv_mad", "median · 1/mad", "--")
    _fit_and_plot(per_pair_median, "w_inv_n",   "median · 1/n",   ":")

    ax.set_xlabel("Δd from dome between stations (km)")
    ax.set_ylabel("Apparent speed c_pair (km/s)")
    ax.set_ylim(0, float(cmax))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if created:
        plt.tight_layout()
        plt.show()

################### UNTESTED BELOW HERE #####

# --- Build Δd and Δτ tables from the "stable pairs" DF (your CSV-derived table) ---

def build_tau_table_from_stable(
    grid, inventory, cache_dir, dome_location,
    stable_df: pd.DataFrame, use_value: str = "mean_s",
    min_delta_d_km: float = 0.0, min_abs_tau_s: float = 0.0
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ['pair','sta_a','sta_b','delta_d_km','delta_tau_s','std','mad','n'].
    """
    # ensure sta columns
    df = stable_df.copy()
    if not {"sta_a","sta_b"}.issubset(df.columns):
        ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
        ab.columns = ["sta_a","sta_b"]
        df = pd.concat([df, ab], axis=1)

    # distances from dome
    node_dists, _, _ = compute_or_load_distances(grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=True)
    def key(s): 
        parts = str(s).split(".")
        return parts[1] if len(parts) >= 2 else str(s)
    dist_by_sta = {}
    for seed, vec in node_dists.items():
        k = key(seed)
        dist_by_sta.setdefault(k, np.asarray(vec, float))
    glon = np.asarray(grid.gridlon).ravel()
    glat = np.asarray(grid.gridlat).ravel()
    if isinstance(dome_location, dict):
        lon, lat = float(dome_location["lon"]), float(dome_location["lat"])
    elif isinstance(dome_location, (tuple, list)) and len(dome_location) == 2:
        lon, lat = float(dome_location[0]), float(dome_location[1])
    else:
        lon, lat = float(dome_location.lon), float(dome_location.lat)  # fallback if you carry an object
    dome_idx = int(np.argmin((glon - lon)**2 + (glat - lat)**2))

    rows = []
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]).upper(), str(r["sta_b"]).upper()
        if a not in dist_by_sta or b not in dist_by_sta:
            continue
        da, db = float(dist_by_sta[a][dome_idx]), float(dist_by_sta[b][dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            continue
        dd = abs(da - db)
        tau = float(r[use_value]) if use_value in r and np.isfinite(r[use_value]) else np.nan
        if not np.isfinite(tau) or dd < min_delta_d_km or abs(tau) < min_abs_tau_s:
            continue
        rows.append(dict(
            pair=r.get("pair", f"{a}-{b}"), sta_a=a, sta_b=b,
            delta_d_km=dd, delta_tau_s=abs(tau),
            std=float(r["std"]) if "std" in r and np.isfinite(r["std"]) else np.nan,
            mad=float(r["mad"]) if "mad" in r and np.isfinite(r["mad"]) else np.nan,
            n=int(r["n"]) if "n" in r and np.isfinite(r["n"]) else np.nan,
        ))
    return pd.DataFrame(rows)

# --- Weighted least squares (optional fixed intercept on τ-models) ---

def _wls_fit_tau_constant(dd, tt, w=None, fix_intercept=None):
    # τ = a + dd / c  -> params θ = [a, s] with s = 1/c
    dd = np.asarray(dd, float); tt = np.asarray(tt, float)
    if w is None: w = np.ones_like(tt)
    w = np.asarray(w, float)
    m = np.isfinite(dd) & np.isfinite(tt) & np.isfinite(w) & (w > 0)
    dd, tt, w = dd[m], tt[m], w[m]
    if dd.size < 2: return np.nan, np.nan, np.nan  # (a, c, R2)
    if fix_intercept is None:
        X = np.column_stack([np.ones_like(dd), dd])
        y = tt
        W = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(W * X, W[:,0] * y, rcond=None)
        a, s = float(beta[0]), float(beta[1])
    else:
        a = float(fix_intercept)
        # minimize Σ w*(tt - (a + s*dd))^2 -> s = Σ w*dd*(tt-a) / Σ w*dd^2
        num = np.sum(w * dd * (tt - a)); den = np.sum(w * dd * dd)
        s = float(num / den) if den > 0 else np.nan
    c = 1.0 / s if np.isfinite(s) and s > 0 else np.nan
    yhat = (a + (dd * s)) if np.isfinite(s) else np.full_like(tt, np.nan)
    ybar = np.sum(w * tt) / np.sum(w)
    sse = np.sum(w * (tt - yhat) ** 2)
    sst = np.sum(w * (tt - ybar) ** 2)
    r2  = float(1.0 - sse / sst) if sst > 0 else np.nan
    return a, c, r2

def _wls_fit_tau_linear_c(dd, tt, w=None, fix_intercept=None, b_grid=(0.3, 2.0, 60), m_grid=(0.0, 0.8, 60)):
    """
    τ = a + (1/m)*ln(1 + (m/b)*dd); parameters: a (s), b (km/s), m (1/s).
    We do a coarse grid search on (b,m) and closed-form a as weighted median of residuals.
    """
    dd = np.asarray(dd, float); tt = np.asarray(tt, float)
    if w is None: w = np.ones_like(tt)
    w = np.asarray(w, float)
    msk = np.isfinite(dd) & np.isfinite(tt) & np.isfinite(w) & (w > 0)
    dd, tt, w = dd[msk], tt[msk], w[msk]
    if dd.size < 2: return np.nan, np.nan, np.nan, np.nan  # (a,b,m,R2)

    def wmedian(x, w):
        idx = np.argsort(x); xs, ws = x[idx], w[idx]
        p = np.cumsum(ws) / np.sum(ws)
        j = np.searchsorted(p, 0.5)
        return float(xs[min(j, xs.size-1)])

    b_lo, b_hi, b_n = b_grid
    m_lo, m_hi, m_n = m_grid
    b_vals = np.linspace(b_lo, b_hi, int(b_n))
    m_vals = np.linspace(m_lo, m_hi, int(m_n))
    best = dict(score=np.inf)

    for b in b_vals:
        for mpar in m_vals:
            if mpar == 0:
                model = dd / b
            else:
                model = (1.0 / mpar) * np.log1p((mpar / b) * dd)
            if fix_intercept is None:
                a_hat = wmedian(tt - model, w)
            else:
                a_hat = float(fix_intercept)
            resid = tt - (a_hat + model)
            # robust score with weights (MAD of weighted residuals)
            s = 1.4826 * np.median(np.abs(resid))
            score = s
            if np.isfinite(score) and score < best["score"]:
                best = dict(score=score, a=a_hat, b=b, m=mpar)

    # R² with the winning params (use standard WLS R²)
    b = best["b"]; mpar = best["m"]; a = best["a"]
    model = (dd / b) if mpar == 0 else (1.0 / mpar) * np.log1p((mpar / b) * dd)
    yhat = a + model
    ybar = np.sum(w * tt) / np.sum(w)
    sse = np.sum(w * (tt - yhat) ** 2)
    sst = np.sum(w * (tt - ybar) ** 2)
    r2  = float(1.0 - sse / sst) if sst > 0 else np.nan
    return a, b, mpar, r2

# --- Convenience: compute weights from the stable table ---

def _weights_from_stable(df, scheme: str):
    # scheme in {"inv_std2","inv_mad2","inv_n"}
    if scheme == "inv_std2":
        sig = df["std"].to_numpy()
        return 1.0 / np.clip(sig, 1e-6, np.inf) ** 2
    if scheme == "inv_mad2":
        sig = df["mad"].to_numpy()
        return 1.0 / np.clip(sig, 1e-6, np.inf) ** 2
    if scheme == "inv_n":
        n = df["n"].to_numpy()
        return 1.0 / np.clip(n, 1.0, np.inf)
    raise ValueError("Unknown scheme")

# --- One-stop plot comparing the 6 combinations on τ vs Δd and (optionally) the derived c_pair lines ---

def compare_constant_vs_linear_models(
    grid, inventory, cache_dir, dome_location, stable_df,
    fix_intercept: float | None = None,   # None => free a; or e.g. 0.0s to force through origin in τ-space
    cmax: float = 7.0,
    stats=("mean_s","median_s"),
    weight_schemes=("inv_std2","inv_mad2","inv_n"),
    show_cpair_overlay: bool = True,
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    colors = {("mean_s","inv_std2"): "-", ("mean_s","inv_mad2"): "--", ("mean_s","inv_n"): ":",
              ("median_s","inv_std2"): "-", ("median_s","inv_mad2"): "--", ("median_s","inv_n"): ":"}

    legends = []
    for stat in stats:
        tau_tbl = build_tau_table_from_stable(grid, inventory, cache_dir, dome_location, stable_df, use_value=stat)
        if tau_tbl.empty:
            continue
        dd = tau_tbl["delta_d_km"].to_numpy()
        tt = tau_tbl["delta_tau_s"].to_numpy()
        for scheme in weight_schemes:
            w = _weights_from_stable(tau_tbl, scheme)
            # constant-c fit
            a_c, c_hat, r2_c = _wls_fit_tau_constant(dd, tt, w=w, fix_intercept=fix_intercept)
            # linear-c fit
            a_l, b_hat, m_hat, r2_l = _wls_fit_tau_linear_c(dd, tt, w=w, fix_intercept=fix_intercept)

            # plot as c_pair lines if requested (convert model τ to apparent c_pair = Δd/τ)
            xx = np.linspace(0.0, np.nanmax(dd) * 1.05, 200)
            if show_cpair_overlay:
                # constant: τ = a + xx/c
                tau_line = a_c + xx / c_hat if np.isfinite(c_hat) else np.full_like(xx, np.nan)
                cpair = np.where(tau_line > 0, xx / tau_line, np.nan)
                ax.plot(xx, cpair, lw=2,
                        label=f"{stat}·{scheme} const: c≈{c_hat:.2f} km/s, R²τ={r2_c:.2f}",
                        alpha=0.8)
                # linear-c: τ = a + (1/m)ln(1 + (m/b)xx)
                if np.isfinite(m_hat) and np.isfinite(b_hat) and m_hat >= 0 and b_hat > 0:
                    if m_hat == 0:
                        tau_line2 = a_l + xx / b_hat
                    else:
                        tau_line2 = a_l + (1.0 / m_hat) * np.log1p((m_hat / b_hat) * xx)
                    cpair2 = np.where(tau_line2 > 0, xx / tau_line2, np.nan)
                    ax.plot(xx, cpair2, lw=2,
                            label=f"{stat}·{scheme} linear: b≈{b_hat:.2f}, m≈{m_hat:.2f}, R²τ={r2_l:.2f}",
                            alpha=0.8, linestyle=colors[(stat, scheme)])

    # scatter the observed c_pair for context
    mean_tbl = build_tau_table_from_stable(grid, inventory, cache_dir, dome_location, stable_df, use_value="mean_s")
    med_tbl  = build_tau_table_from_stable(grid, inventory, cache_dir, dome_location, stable_df, use_value="median_s")
    for tbl, nm in [(mean_tbl, "pairs mean_s"), (med_tbl, "pairs median_s")]:
        x = tbl["delta_d_km"].to_numpy()
        y = np.where(tbl["delta_tau_s"] > 0, x / tbl["delta_tau_s"].to_numpy(), np.nan)
        m = np.isfinite(x) & np.isfinite(y) & (y <= cmax)
        ax.scatter(x[m], y[m], s=20, alpha=0.55, label=f"{nm} (n={m.sum()})")

    ax.set_xlabel("Δd from dome between stations (km)")
    ax.set_ylabel("Apparent speed c_pair (km/s)")
    ax.set_ylim(0, cmax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


# --- Δd–Δτ + c_pair vs d_avg plotting helpers -------------------------------
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flovopy.asl.distances import compute_or_load_distances

def _clean_sta(x: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()

def _resolve_dome_idx(grid, dome_location) -> int:
    if isinstance(dome_location, int):
        return dome_location
    glon = np.asarray(grid.gridlon).ravel()
    glat = np.asarray(grid.gridlat).ravel()
    if isinstance(dome_location, dict) and {"lon","lat"} <= set(dome_location):
        lon, lat = float(dome_location["lon"]), float(dome_location["lat"])
    else:
        lon, lat = dome_location
    return int(np.argmin((glon - lon)**2 + (glat - lat)**2))

def make_pair_geometry_from_stable(
    *,
    grid, inventory, cache_dir,
    dome_location,
    stable_df: pd.DataFrame,
    use_value: str = "mean_s",              # or "median_s"
    use_elevation: bool = True,
    min_delta_d_km: float = 0.0,
    min_abs_tau_s: float = 0.0,
    c_bounds: Tuple[float,float] = (0.2, 7.0),
) -> pd.DataFrame:
    """
    Returns a DataFrame with per-pair geometry + stats for plotting:
      ['sta_a','sta_b','delta_d_km','delta_tau_s','c_pair_km_s','d_avg_km','n','std','mad']
    """
    df = stable_df.copy()

    # Ensure station cols exist
    if not {"sta_a","sta_b"}.issubset(df.columns):
        if "pair" not in df.columns:
            raise ValueError("stable_df must have sta_a/sta_b or a 'pair' column")
        ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
        ab.columns = ["sta_a","sta_b"]
        df = pd.concat([df, ab], axis=1)

    df["sta_a"] = df["sta_a"].apply(_clean_sta)
    df["sta_b"] = df["sta_b"].apply(_clean_sta)

    # Choose lag statistic column
    if use_value not in df.columns:
        fallback = ["mean_s","median_s","mean","median"]
        for c in fallback:
            if c in df.columns:
                use_value = c; break
        else:
            raise ValueError("No lag column found (need mean_s/median_s/mean/median).")

    # Distances
    node_dists, _, _ = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=use_elevation
    )
    def to_sta_key(seed_or_sta: str) -> str:
        p = str(seed_or_sta).split(".")
        return p[1] if len(p) >= 2 else str(seed_or_sta)

    dist_by_sta: Dict[str, np.ndarray] = {}
    for full_id, vec in node_dists.items():
        key = to_sta_key(full_id)
        dist_by_sta.setdefault(key, np.asarray(vec, float))

    dome_idx = _resolve_dome_idx(grid, dome_location)

    rows = []
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]), str(r["sta_b"])
        if a not in dist_by_sta or b not in dist_by_sta:
            continue
        da_vec, db_vec = dist_by_sta[a], dist_by_sta[b]
        if dome_idx >= da_vec.size or dome_idx >= db_vec.size:
            continue
        da = float(da_vec[dome_idx]); db = float(db_vec[dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            continue

        delta_d = abs(da - db)                                 # km
        tau = float(r[use_value]) if np.isfinite(r[use_value]) else np.nan
        if not np.isfinite(tau):
            continue
        if (delta_d < min_delta_d_km) or (abs(tau) < min_abs_tau_s):
            continue

        c_pair = delta_d / abs(tau) if abs(tau) > 0 else np.nan
        if not (np.isfinite(c_pair) and (c_bounds[0] <= c_pair <= c_bounds[1])):
            continue

        # average radial distance from dome
        d_avg = 0.5 * (da + db)

        rows.append(dict(
            sta_a=a, sta_b=b,
            delta_d_km=delta_d,
            delta_tau_s=abs(tau),
            c_pair_km_s=c_pair,
            d_avg_km=d_avg,
            n=int(r["n"]) if "n" in df.columns and np.isfinite(r["n"]) else np.nan,
            std=float(r["std"]) if "std" in df.columns and np.isfinite(r["std"]) else np.nan,
            mad=float(r["mad"]) if "mad" in df.columns and np.isfinite(r["mad"]) else np.nan,
            use_value=use_value,
        ))

    out = pd.DataFrame(rows)
    return out

def _wls_fit(x, y, w, fixed_intercept: Optional[float] = None):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if x.size < 2:
        return dict(m=np.nan, b=np.nan, r2=np.nan, n=int(x.size))
    if fixed_intercept is None:
        X = np.column_stack([x, np.ones_like(x)])
        W = np.diag(np.sqrt(w))
        beta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
        m_hat, b_hat = float(beta[0]), float(beta[1])
        yhat = m_hat * x + b_hat
    else:
        b_hat = float(fixed_intercept)
        # minimize sum w (y - (b + m x))^2 -> m = sum w x (y-b) / sum w x^2
        num = np.sum(w * x * (y - b_hat))
        den = np.sum(w * x * x)
        m_hat = float(num / den) if den > 0 else np.nan
        yhat = b_hat + m_hat * x
    # R^2 (unweighted, for readability)
    sse = float(np.sum((y - yhat)**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2  = float(1.0 - sse/sst) if sst > 0 else np.nan
    return dict(m=m_hat, b=b_hat, r2=r2, n=int(x.size))

def _weight_vector(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "1/std^2":
        sig = df["std"].to_numpy(dtype=float)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
        w = 1.0 / (sig**2)
    elif mode == "1/mad^2":
        sig = df["mad"].to_numpy(dtype=float)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
        w = 1.0 / (sig**2)
    elif mode == "1/n":
        n = df["n"].to_numpy(dtype=float)
        n = np.where(np.isfinite(n) & (n > 0), n, np.nan)
        w = 1.0 / n
    else:
        raise ValueError("Unknown weight mode")
    # replace missing weights with median of finite weights
    med = np.nanmedian(w)
    w = np.where(np.isfinite(w), w, med if np.isfinite(med) and med > 0 else 1.0)
    return w

def plot_delta_t_vs_delta_d_with_wls(
    df_mean: pd.DataFrame,
    df_median: pd.DataFrame,
    *,
    fixed_intercept_tau: Optional[float] = None,   # set to 0.0 to force through origin in τ-space
    title: str = "Δt vs Δd (WLS fits)",
):
    plt.figure(figsize=(9,5))
    # scatter
    plt.scatter(df_mean["delta_d_km"], df_mean["delta_tau_s"], s=28, alpha=0.75, label=f"mean_s (n={len(df_mean)})")
    plt.scatter(df_median["delta_d_km"], df_median["delta_tau_s"], s=28, alpha=0.75, label=f"median_s (n={len(df_median)})")

    combos = [
        ("mean_s",   df_mean,   "1/std^2",  "-",  "#1f77b4"),
        ("mean_s",   df_mean,   "1/mad^2",  "--", "#1f77b4"),
        ("mean_s",   df_mean,   "1/n",      ":",  "#1f77b4"),
        ("median_s", df_median, "1/std^2",  "-",  "#d62728"),
        ("median_s", df_median, "1/mad^2",  "--", "#d62728"),
        ("median_s", df_median, "1/n",      ":",  "#d62728"),
    ]
    legend_lines = []

    for stat, dfx, wmode, ls, col in combos:
        w = _weight_vector(dfx, wmode)
        fit = _wls_fit(dfx["delta_d_km"], dfx["delta_tau_s"], w, fixed_intercept=fixed_intercept_tau)
        # in τ-space, slope = 1/c_eff
        m, b, r2 = fit["m"], fit["b"], fit["r2"]
        xx = np.linspace(0, max(dfx["delta_d_km"].max(), df_mean["delta_d_km"].max())*1.05, 200)
        yy = b + m * xx
        line, = plt.plot(xx, yy, ls=ls, color=col,
                         label=f"{stat} · {wmode} (τ = {m:.2f}·Δd + {b:.2f},  c≈{(1/m):.2f} km/s, R²={r2:.2f})")
        legend_lines.append(line)

    plt.xlabel("Δd from dome between stations (km)")
    plt.ylabel("Δt between stations (s)")
    plt.title(title + ("" if fixed_intercept_tau is None else f"  [intercept fixed @ {fixed_intercept_tau:.2f}s]"))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_cpair_vs_davg_with_wls(
    df_mean: pd.DataFrame,
    df_median: pd.DataFrame,
    *,
    fixed_intercept_c: Optional[float] = None,     # set to e.g. 0.5 to force c(d)=b+m d through b
    cmax: float = 7.0,
    title: str = "c_pair vs average radial distance d_avg (WLS fits)",
):
    plt.figure(figsize=(9,5))
    # scatter
    plt.scatter(df_mean["d_avg_km"], df_mean["c_pair_km_s"], s=28, alpha=0.75, label=f"mean_s (n={len(df_mean)})")
    plt.scatter(df_median["d_avg_km"], df_median["c_pair_km_s"], s=28, alpha=0.75, label=f"median_s (n={len(df_median)})")

    combos = [
        ("mean_s",   df_mean,   "1/std^2",  "-",  "#1f77b4"),
        ("mean_s",   df_mean,   "1/mad^2",  "--", "#1f77b4"),
        ("mean_s",   df_mean,   "1/n",      ":",  "#1f77b4"),
        ("median_s", df_median, "1/std^2",  "-",  "#d62728"),
        ("median_s", df_median, "1/mad^2",  "--", "#d62728"),
        ("median_s", df_median, "1/n",      ":",  "#d62728"),
    ]
    for stat, dfx, wmode, ls, col in combos:
        w = _weight_vector(dfx, wmode)
        fit = _wls_fit(dfx["d_avg_km"], dfx["c_pair_km_s"], w, fixed_intercept=fixed_intercept_c)
        m, b, r2 = fit["m"], fit["b"], fit["r2"]
        xx = np.linspace(0, max(dfx["d_avg_km"].max(), df_mean["d_avg_km"].max())*1.05, 200)
        yy = b + m * xx
        plt.plot(xx, yy, ls=ls, color=col,
                 label=f"{stat} · {wmode} (c = {m:.2f}·d + {b:.2f}, R²={r2:.2f})")

    plt.xlabel("Average radial distance d_avg (km)")
    plt.ylabel("Apparent speed c_pair (km/s)")
    plt.ylim(0, cmax)
    plt.title(title + ("" if fixed_intercept_c is None else f"  [intercept fixed @ {fixed_intercept_c:.2f} km/s]"))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy.optimize import brentq, least_squares
from scipy.integrate import quad
from flovopy.asl.distances import compute_or_load_distances

# ---------- geometry helpers (radii from dome) ----------
def _sta_key(seed_or_sta: str) -> str:
    parts = str(seed_or_sta).split(".")
    return parts[1] if len(parts) >= 2 else str(seed_or_sta)

def _dome_idx(grid, dome):
    if isinstance(dome, int):
        return int(dome)
    glon = np.asarray(grid.gridlon).ravel()
    glat = np.asarray(grid.gridlat).ravel()
    if isinstance(dome, dict):
        lon, lat = float(dome["lon"]), float(dome["lat"])
    else:
        lon, lat = float(dome[0]), float(dome[1])
    return int(np.argmin((glon - lon)**2 + (glat - lat)**2))

def station_offsets_from_dome(grid, inventory, cache_dir, dome_location) -> Dict[str, float]:
    node_dists, _, _ = compute_or_load_distances(grid, cache_dir, inventory=inventory,
                                                 stream=None, use_elevation=True)
    dome = _dome_idx(grid, dome_location)
    by_sta = {}
    for seed, vec in node_dists.items():
        k = _sta_key(seed)
        by_sta.setdefault(k, float(vec[dome]))
    return by_sta  # km

# ---------- ray integrals in a linear gradient ----------
def _range_time_for_p(p: float, v0: float, g: float) -> Tuple[float, float]:
    """
    One-way X(p), T(p) for v(z)=v0+g z. Uses numerical quadrature; symmetric path (down then up).
    p in [0, 1/v0). If g==0, reduces to constant-velocity straight ray.
    """
    if g <= 0:  # constant v
        # straight ray, sin(i)=p v0 ; tan(i)=p v0 / sqrt(1-(p v0)^2)
        # for a surface-to-surface path, choose any depth scale -> use geometry via p only:
        # We’ll let the root-finder handle the mapping via X.
        pass
    if p <= 0 or p >= 1.0/max(v0, 1e-9):
        return np.nan, np.nan

    # turning depth: p v(z_t) = 1  -> z_t = (1/p - v0)/g
    if g > 0:
        zt = max((1.0/p - v0)/g, 0.0)
    else:
        # no turning in zero gradient; we won't hit this branch because we go through g==0 below
        return np.nan, np.nan

    # integrands
    def v(z): return v0 + g*z
    def dx_dz(z): 
        vv = v(z)
        s = 1.0 - (p*vv)**2
        return (p*vv)/np.sqrt(max(s, 1e-15))
    def dt_dz(z):
        vv = v(z)
        s = 1.0 - (p*vv)**2
        return 1.0/(vv*np.sqrt(max(s, 1e-15)))

    X_half = quad(dx_dz, 0.0, zt, limit=200, epsabs=1e-8, epsrel=1e-8)[0]
    T_half = quad(dt_dz, 0.0, zt, limit=200, epsabs=1e-8, epsrel=1e-8)[0]
    return 2.0*X_half, 2.0*T_half  # km, s

def _travel_time_for_offset(R: float, v0: float, g: float) -> float:
    """Find p so that X(p)=R, then return T(p). Handles g≈0 gracefully."""
    if R <= 0:
        return 0.0
    if g <= 0:  # constant v
        # straight ray at constant v0; time = R / v0
        return R / max(v0, 1e-6)

    # root-find p in (0, 1/v0) for X(p)-R=0
    p_lo = 1e-8
    p_hi = (1.0 / v0) * (1.0 - 1e-6)

    def f(p):
        X, _ = _range_time_for_p(p, v0, g)
        return X - R

    # insure bracket
    f_lo, f_hi = f(p_lo), f(p_hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo>0 or f_hi<0:
        # expand a bit; if still broken, fall back to constant-v approximation near v0
        try:
            T_guess = R / max(v0, 1e-6)
            return T_guess
        except Exception:
            return np.nan

    p_star = brentq(f, p_lo, p_hi, xtol=1e-10, rtol=1e-10, maxiter=200)
    _, T = _range_time_for_p(p_star, v0, g)
    return T

# ---------- objective over pairwise lags ----------
def _build_pairs(stable_df: pd.DataFrame, use_value: str) -> pd.DataFrame:
    df = stable_df.copy()
    if not {"sta_a","sta_b"}.issubset(df.columns):
        if "pair" in df.columns:
            ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
            ab.columns = ["sta_a","sta_b"]
            df = pd.concat([df, ab], axis=1)
        else:
            raise ValueError("stable_df needs 'sta_a'/'sta_b' or 'pair'.")
    # keep only needed columns; coerce to float
    df = df[["sta_a","sta_b",use_value,"std","mad","n"]].copy()
    for c in [use_value,"std","mad","n"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[use_value]).reset_index(drop=True)

def fit_v0_g_from_pairs(
    grid, inventory, cache_dir, dome_location,
    stable_df: pd.DataFrame,
    use_value: str,           # "mean_s" or "median_s"
    weight_mode: str = "1/std",  # "1/std", "1/mad", "1/n"
    v0_prior: Optional[Tuple[float,float]]=None,
):
    # station radii (km) from dome
    R_by_sta = station_offsets_from_dome(grid, inventory, cache_dir, dome_location)

    pairs = _build_pairs(stable_df, use_value)
    # compute offsets for each pair end
    Ra = pairs["sta_a"].map(R_by_sta).to_numpy(float)
    Rb = pairs["sta_b"].map(R_by_sta).to_numpy(float)
    m = np.isfinite(Ra) & np.isfinite(Rb)
    pairs = pairs.loc[m].copy()
    Ra, Rb = Ra[m], Rb[m]

    # weights
    if weight_mode == "1/std" and "std" in pairs.columns:
        w = 1.0 / np.clip(pairs["std"].to_numpy(float), 1e-6, np.inf)
    elif weight_mode == "1/mad" and "mad" in pairs.columns:
        w = 1.0 / np.clip(pairs["mad"].to_numpy(float), 1e-6, np.inf)
    elif weight_mode == "1/n" and "n" in pairs.columns:
        w = 1.0 / np.clip(pairs["n"].to_numpy(float), 1.0, np.inf)
    else:
        w = np.ones(len(pairs), float)

    dtaus_obs = pairs[use_value].to_numpy(float)  # seconds, signed

    # residual function for least_squares
    def residual(theta):
        v0, g = theta
        v0 = max(v0, 0.2)
        g  = max(g,  0.0)
        Ta = np.array([_travel_time_for_offset(R, v0, g) for R in Ra])
        Tb = np.array([_travel_time_for_offset(R, v0, g) for R in Rb])
        r  = (Ta - Tb - dtaus_obs) * np.sqrt(w)
        if v0_prior is not None:
            mu, sigma = v0_prior
            r_prior = (v0 - mu) / max(sigma, 1e-6)
            r = np.concatenate([r, np.array([r_prior])])
        return r

    # initial guess from your simple line fits (free-intercept on c vs R)
    # Use v0≈ 0.8–1.2 and a small gradient
    x0 = np.array([1.0, 0.05])  # v0=1.0 km/s, g=0.05 s^-1 (i.e., 0.05 km/s per km)
    ls = least_squares(residual, x0, bounds=([0.2, 0.0], [5.0, 1.0]), xtol=1e-10, ftol=1e-10)

    # covariance approx (Gauss-Newton)
    # J ~ (∂resid/∂theta); least_squares provides it in ls.jac
    if ls.jac is not None and ls.jac.size:
        JTJ = ls.jac.T @ ls.jac
        try:
            cov = np.linalg.inv(JTJ)
            sig = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            cov = None; sig = (np.nan, np.nan)
    else:
        cov = None; sig = (np.nan, np.nan)

    v0_fit, g_fit = ls.x
    return {
        "v0": float(v0_fit),
        "g": float(g_fit),
        "sigma": (float(sig[0]), float(sig[1])),
        "success": bool(ls.success),
        "cost": float(ls.cost),
        "n_pairs": int(len(pairs)),
        "pairs": pairs.assign(Ra=Ra, Rb=Rb, w=w, dtau_obs=dtaus_obs)
    }

# ---------- convenience: run all 6 combos + plot ----------
def run_six_and_plot(
    grid, inventory, cache_dir, dome_location,
    stable_df: pd.DataFrame,
    cmax=7.0, v0_prior: Optional[Tuple[float,float]]=None,
):
    combos = [
        ("mean_s","1/std"), ("mean_s","1/mad"), ("mean_s","1/n"),
        ("median_s","1/std"), ("median_s","1/mad"), ("median_s","1/n"),
    ]
    fits = []
    for use_value, wmode in combos:
        try:
            fit = fit_v0_g_from_pairs(grid, inventory, cache_dir, dome_location,
                                      stable_df, use_value=use_value, weight_mode=wmode, v0_prior=v0_prior)
            fit["label"] = f"{use_value} · {wmode} (v0={fit['v0']:.2f}, g={fit['g']:.3f})"
            fits.append(fit)
        except Exception as e:
            print(f"[WARN] {use_value}/{wmode} failed: {e}")

    # Plot apparent speed vs offset using model-predicted times (for reference)
    # For each fit, compute model speed curve c_model(R) = R / T(R; v0,g)
    Rs = np.linspace(0, max(1.0, np.nanmax([np.nanmax(f["pairs"][["Ra","Rb"]].to_numpy()) for f in fits if len(f["pairs"])])), 200)
    plt.figure(figsize=(9,5))
    for k, fit in enumerate(fits):
        v0, g = fit["v0"], fit["g"]
        Tmod = np.array([_travel_time_for_offset(R, v0, g) if R>0 else 0.0 for R in Rs])
        cmod = np.divide(Rs, Tmod, out=np.full_like(Rs, np.nan), where=Tmod>0)
        style = "-" if "mean_s" in fit["label"] else "--"
        plt.plot(Rs, np.clip(cmod, 0, cmax), style, lw=2, label=fit["label"])
    plt.ylim(0, cmax)
    plt.xlabel("Offset from dome R (km)")
    plt.ylabel("Model apparent speed R/T (km/s)")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.title("Linear-gradient 1D model fits (six weight/stat combos)")
    plt.tight_layout()
    plt.show()

    return fits

import numpy as np

def _range_time_linear_profile(R, v0, g):
    """
    One-way horizontal range X/2 and time T/2 for a linear profile v(z)=v0+g z,
    expressed *at the turning ray* via u0 = p v0.
    We expose them here via a helper that takes R directly.

    Returns: T (one-way travel time) for surface-to-surface symmetric path of
    half-range R/2; X is just R. Handles edge cases robustly.

    Notes:
    • In a linear profile the *maximum* surface-to-surface range for a turning
      ray is R_max = 2/g. Guard for R > R_max.
    • ε-clipping keeps logs and square roots numerically safe.
    """
    R = float(R)
    g = float(g)
    v0 = float(v0)
    if R <= 0:
        return 0.0
    if g <= 0:
        # constant-velocity fallback
        return R / max(v0, 1e-9)

    # linear-profile turning-ray geometry implies: g R / 2 ∈ (0, 1)
    x = g * R / 2.0
    # Guard for geometry out of bounds (no real turning ray). Clamp slightly inside.
    if not np.isfinite(x) or x >= 1.0:
        x = min(max(x, 0.0), 1.0 - 1e-12)

    # To avoid catastrophic cancellation, form s = sqrt(1 - x^2) with safe clip
    s = np.sqrt(max(1.0 - x*x, 1e-16))

    # One-way travel time (closed form). Use the numerically stable log form:
    # T = (2/g) * ln( (1 + x) / s )
    # (equivalent to -(2/g) * ln( (1 - s) / (x + 1e-300) ), but better-conditioned)
    T = (2.0 / g) * np.log((1.0 + x) / s)

    return T  # seconds

def _travel_time_for_offset(R: float, v0: float, g: float) -> float:
    """Stable travel time for linear gradient; includes constant-v fallback."""
    return _range_time_linear_profile(R, v0, g)

import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple
from scipy.optimize import least_squares
from flovopy.asl.distances import compute_or_load_distances

def fit_v0_beta_from_pairs(
    *,
    grid,
    inventory,
    cache_dir: str,
    dome_location,
    stable_df: pd.DataFrame,           # needs sta_a, sta_b, and a lag col (mean_s/median_s)
    use_value: Literal["mean_s","median_s"]="mean_s",
    depth_proxy: Literal["avg","min"]="avg",  # avg radial distance or closest-station distance
    weight_by: Literal["inv_std","inv_mad","inv_n","none"]="inv_std",
    c_bounds: Tuple[float,float]=(0.2, 7.0),  # keep only physical apparent speeds for plotting QA (not used in cost)
    v0_prior: float = 0.9,                    # km/s initial guess
    beta_prior: float = 0.02,                 # km/s per km of proxy (start nonzero!)
    bounds_v0: Tuple[float,float]=(0.2, 3.5),
    bounds_beta: Tuple[float,float]=(0.0, 0.2),  # non-negative trend
    min_delta_d_km: float = 0.0,
    min_abs_tau_s: float = 0.0,
    verbose: bool = True,
):
    """
    Fit Δτ ≈ Δd / (v0 + beta * φ(d̂)), with φ(d̂) = d_avg or d_min.
    Returns best-fit v0, beta, diagnostics, and a per_pair table augmented with the proxy.
    """
    df = stable_df.copy()

    # Ensure station cols
    if not {"sta_a","sta_b"}.issubset(df.columns):
        if "pair" in df.columns:
            ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
            ab.columns = ["sta_a","sta_b"]
            df = pd.concat([df, ab], axis=1)
        else:
            raise ValueError("stable_df must have sta_a/sta_b or pair column")

    # Choose lag column
    if use_value not in df.columns:
        if use_value == "mean_s" and "mean" in df.columns:
            df["mean_s"] = df["mean"]
        elif use_value == "median_s" and "median" in df.columns:
            df["median_s"] = df["median"]
        else:
            raise ValueError(f"lag column {use_value!r} not in stable_df")

    # Compute radial distances of each station from dome node
    node_dists, _, _ = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=True
    )

    def to_sta_key(s):
        parts = str(s).split(".")
        return parts[1] if len(parts)>=2 else str(s)

    # collapse to first seen per station
    dist_by_sta = {}
    for full_id, vec in node_dists.items():
        k = to_sta_key(full_id)
        dist_by_sta.setdefault(k, np.asarray(vec, float))

    # resolve dome node index
    glon = np.asarray(grid.gridlon).ravel()
    glat = np.asarray(grid.gridlat).ravel()
    if isinstance(dome_location, int):
        dome_idx = int(dome_location)
    else:
        if isinstance(dome_location, dict):
            lon, lat = float(dome_location["lon"]), float(dome_location["lat"])
        else:
            lon, lat = float(dome_location[0]), float(dome_location[1])
        dome_idx = int(np.argmin((glon-lon)**2 + (glat-lat)**2))

    # Build per-pair table with Δd, Δτ, and proxy φ(d̂)
    rows=[]
    rej=0
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]), str(r["sta_b"])
        if a not in dist_by_sta or b not in dist_by_sta:
            rej += 1; continue
        da = float(dist_by_sta[a][dome_idx])
        db = float(dist_by_sta[b][dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            rej += 1; continue

        delta_d = abs(da - db)  # km
        tau = float(r[use_value])
        if not np.isfinite(tau) or delta_d < min_delta_d_km or abs(tau) < min_abs_tau_s:
            rej += 1; continue

        # proxy distance
        if depth_proxy == "avg":
            dhat = 0.5*(da + db)
        else:  # "min"
            dhat = min(da, db)

        # weight choices
        if weight_by == "inv_std" and "std" in df.columns and np.isfinite(r["std"]) and r["std"]>0:
            w = 1.0 / float(r["std"])
        elif weight_by == "inv_mad" and "mad" in df.columns and np.isfinite(r["mad"]) and r["mad"]>0:
            w = 1.0 / float(r["mad"])
        elif weight_by == "inv_n" and "n" in df.columns and r["n"]>0:
            w = 1.0 / float(r["n"])
        else:
            w = 1.0

        rows.append(dict(
            sta_a=a, sta_b=b, delta_d_km=delta_d, delta_tau_s=tau,
            d_proxy_km=dhat, weight=w
        ))

    per_pair = pd.DataFrame(rows)
    if verbose:
        print(f"[build] kept {len(per_pair)} pairs, rejected {rej}")

    if per_pair.empty:
        return {"per_pair": per_pair, "ok": False, "reason": "no_pairs"}

    xΔ = per_pair["delta_d_km"].to_numpy(float)
    τ  = np.abs(per_pair["delta_tau_s"].to_numpy(float))  # use absolute lag
    φ  = per_pair["d_proxy_km"].to_numpy(float)
    w  = per_pair["weight"].to_numpy(float)
    w  = np.clip(w, np.percentile(w,5), np.percentile(w,95))  # tame extremes
    w  = w / (w.max() if w.max()>0 else 1.0)

    # residuals: r = sqrt(w) * (τ_obs - Δd/(v0 + β φ))
    def residuals(p):
        v0, beta = p
        denom = v0 + beta * φ
        # keep denom positive; if not, heavily penalize
        bad = denom <= 0
        pred = np.empty_like(τ)
        pred[~bad] = xΔ[~bad] / denom[~bad]
        pred[bad]  = 1e6  # big penalty
        return np.sqrt(w) * (τ - pred)

    lb = [bounds_v0[0], bounds_beta[0]]
    ub = [bounds_v0[1], bounds_beta[1]]
    p0 = [float(v0_prior), float(beta_prior)]  # non-zero beta start

    res = least_squares(residuals, p0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
    v0_hat, beta_hat = map(float, res.x)

    # simple diagnostics
    denom = v0_hat + beta_hat * φ
    pred  = xΔ / denom
    resid = τ - pred
    sse   = float(np.sum((np.sqrt(w)*resid)**2))
    r2    = 1.0 - (np.sum(resid**2) / np.sum((τ - np.mean(τ))**2))

    # optional: filter for unphysical apparent speed if you need a QA plot
    c_app = xΔ / τ
    keep  = (c_app >= c_bounds[0]) & (c_app <= c_bounds[1]) & np.isfinite(c_app)

    out = {
        "ok": True,
        "v0_km_s": v0_hat,
        "beta_km_s_per_km": beta_hat,
        "cost": sse,
        "r2": float(r2),
        "n": int(len(xΔ)),
        "per_pair": per_pair.assign(pred_tau_s=pred, resid_s=resid, c_app_km_s=c_app, keep=keep),
        "message": res.message,
        "success": bool(res.success),
    }
    if verbose:
        print(f"[fit] v0={v0_hat:.3f} km/s, beta={beta_hat:.4f} (km/s)/km, R²={r2:.3f}  (n={len(xΔ)})")
    return out


# ====== Build per-pair table with distance proxies (avg/min radial) ======
import numpy as np
import pandas as pd
import re
from typing import Optional, Tuple, Dict
from flovopy.asl.distances import compute_or_load_distances

def _clean_sta(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()

def build_per_pair_from_stable(
    *,
    grid,
    inventory,
    cache_dir: str,
    dome_location,
    stable_df: pd.DataFrame,
    use_value: str = "mean_s",              # or "median_s"
    c_bounds: Tuple[float, float] = (0.2, 7.0),
    min_delta_d_km: float = 0.0,
    min_abs_tau_s: float = 0.0,
    use_elevation: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    From a stable pairwise table (one row per station pair with statistics on Δτ),
    build a per-pair table with:
      delta_d_km, delta_tau_s, c_pair_km_s, d_avg_km, d_min_km, n, std, mad.
    Distances are measured from the dome node to stations on the grid.
    """
    df = stable_df.copy()

    # Ensure sta_a/sta_b exist (derive from 'pair' if needed)
    if not {"sta_a", "sta_b"}.issubset(df.columns):
        if "pair" not in df.columns:
            raise ValueError("stable_df needs sta_a/sta_b or 'pair'")
        ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
        ab.columns = ["sta_a", "sta_b"]
        df = pd.concat([df, ab], axis=1)

    df["sta_a"] = df["sta_a"].apply(_clean_sta)
    df["sta_b"] = df["sta_b"].apply(_clean_sta)

    # Choose lag column (seconds)
    if use_value not in df.columns:
        if use_value.endswith("_s"):
            base = use_value[:-2]
            if base in df.columns:
                df[use_value] = df[base]
            elif "mean_s" in df.columns:
                use_value = "mean_s"
            elif "median_s" in df.columns:
                use_value = "median_s"
            elif "mean" in df.columns:
                use_value = "mean"
            elif "median" in df.columns:
                use_value = "median"
            else:
                raise ValueError("No suitable lag column found (mean_s/median_s/mean/median).")

    # Distances: node → station
    node_dists, _, _ = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=use_elevation
    )
    def to_sta_key(s: str) -> str:
        parts = str(s).split(".")
        return parts[1] if len(parts) >= 2 else str(s)

    dist_by_sta: Dict[str, np.ndarray] = {}
    for full_id, vec in node_dists.items():
        key = to_sta_key(full_id)
        if key not in dist_by_sta:
            dist_by_sta[key] = np.asarray(vec, float)

    # Resolve dome node index
    def _resolve_dome_idx(grid, dome):
        if isinstance(dome, int):
            return int(dome)
        glon = np.asarray(grid.gridlon).ravel()
        glat = np.asarray(grid.gridlat).ravel()
        if isinstance(dome, dict) and {"lon", "lat"} <= set(dome):
            lon, lat = float(dome["lon"]), float(dome["lat"])
        elif isinstance(dome, (tuple, list)) and len(dome) == 2:
            lon, lat = float(dome[0]), float(dome[1])
        else:
            raise ValueError("dome_location must be node index, {'lon','lat'}, or (lon,lat).")
        return int(np.argmin((glon - lon) ** 2 + (glat - lat) ** 2))

    dome_idx = _resolve_dome_idx(grid, dome_location)

    rows = []
    rejects = 0
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]), str(r["sta_b"])
        if a not in dist_by_sta or b not in dist_by_sta:
            rejects += 1; continue
        da = float(dist_by_sta[a][dome_idx]); db = float(dist_by_sta[b][dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            rejects += 1; continue

        delta_d = abs(da - db)                  # km
        d_avg   = 0.5 * (abs(da) + abs(db))     # avg radial distance
        d_min   = min(abs(da), abs(db))         # distance of closer station
        tau     = float(r[use_value])

        if not np.isfinite(tau) or abs(tau) < min_abs_tau_s or delta_d < min_delta_d_km:
            rejects += 1; continue

        c = delta_d / abs(tau)                  # km/s
        if not (c_bounds[0] <= c <= c_bounds[1]):
            rejects += 1; continue

        rows.append(dict(
            pair=f"{a}-{b}", sta_a=a, sta_b=b,
            delta_d_km=delta_d, delta_tau_s=abs(tau), c_pair_km_s=c,
            d_avg_km=d_avg, d_min_km=d_min,
            n=int(r["n"]) if "n" in r and np.isfinite(r["n"]) else np.nan,
            std=float(r["std"]) if "std" in r and np.isfinite(r["std"]) else np.nan,
            mad=float(r["mad"]) if "mad" in r and np.isfinite(r["mad"]) else np.nan,
            source_stat=use_value
        ))

    out = pd.DataFrame(rows)
    if verbose:
        print(f"[build] kept {len(out)} pairs, rejected {rejects}")
    return out


# ====== Fit constant-gradient model: c = v0 + beta * d_proxy ======
from typing import Literal, Union
from scipy.optimize import curve_fit

def fit_v0_beta_from_pairs(
    per_pair: pd.DataFrame,
    *,
    x_col: Literal["d_min_km", "d_avg_km"] = "d_min_km",
    y_col: str = "c_pair_km_s",
    weight_mode: Literal["inv_std","inv_mad","inv_n",None] = "inv_std",
    v0_prior: Optional[float] = None,
    beta_bounds: Tuple[float, float] = (0.0, 0.10),   # (km/s)/km
    intercept_bounds: Tuple[float, float] = (0.2, 3.5), # km/s, geologically sane surface speed
    verbose: bool = True,
) -> Dict[str, Union[float,int]]:
    """
    Weighted, bounded LS using curve_fit on y = v0 + beta * x.
    x is a depth-proxy distance from the dome (d_min or d_avg).
    """
    # data
    if x_col not in per_pair or y_col not in per_pair:
        raise KeyError(f"per_pair must have columns '{x_col}' and '{y_col}'")
    x = per_pair[x_col].to_numpy(float)
    y = per_pair[y_col].to_numpy(float)

    # weights → sigma for curve_fit (sigma ~ per-point std dev)
    if weight_mode == "inv_std":
        sig = per_pair["std"].to_numpy(float)
    elif weight_mode == "inv_mad":
        # convert MAD → ~σ using 1.4826
        sig = per_pair["mad"].to_numpy(float) * 1.4826
    elif weight_mode == "inv_n":
        n   = per_pair["n"].to_numpy(float)
        # variance ~ 1/n  ⇒ sigma ~ 1/sqrt(n)
        sig = np.where(np.isfinite(n) & (n > 0), 1.0 / np.sqrt(n), np.nan)
    elif weight_mode is None:
        sig = np.ones_like(x) * np.nan  # unweighted
    else:
        raise ValueError("weight_mode must be 'inv_std','inv_mad','inv_n', or None")

    # Clean & guard sigma
    sig = np.asarray(sig, float)
    if np.all(~np.isfinite(sig)):
        sig = None  # unweighted
    else:
        # Replace zeros/nans with median to avoid blowing weights up
        med = np.nanmedian(sig)
        if not np.isfinite(med) or med <= 0:
            sig = None
        else:
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, med)

    # Initial guess for v0: short-baseline proxy (smallest x), or weighted median of y
    if v0_prior is None:
        try:
            k = np.argsort(x)[:max(10, int(0.2*len(x)))]
            v0_prior = float(np.nanmedian(y[k]))
        except Exception:
            v0_prior = float(np.nanmedian(y))
    beta0 = 0.01  # mild positive gradient as a start

    # Model
    def model(x, v0, beta):
        return v0 + beta * x

    # Bounds
    bounds = ([intercept_bounds[0], beta_bounds[0]],
              [intercept_bounds[1], beta_bounds[1]])

    # Fit
    p0 = [float(np.clip(v0_prior, *intercept_bounds)), float(np.clip(beta0, *beta_bounds))]
    popt, pcov = curve_fit(model, x, y, p0=p0, sigma=sig, absolute_sigma=True if sig is not None else False,
                           bounds=bounds, maxfev=10000)

    v0, beta = float(popt[0]), float(popt[1])

    # Diagnostics (R²)
    yhat = model(x, v0, beta)
    sse  = float(np.sum((y - yhat)**2))
    sst  = float(np.sum((y - np.mean(y))**2))
    r2   = float(1.0 - sse/sst) if sst > 0 else np.nan

    if verbose:
        mode_label = {None:"unweighted","inv_std":"1/std","inv_mad":"1/mad","inv_n":"1/n"}[weight_mode]
        print(f"[fit] v0={v0:.3f} km/s, beta={beta:.4f} (km/s)/km, R²={r2:.3f}  (n={x.size})  [{mode_label}, x={x_col}]")

    return {"v0": v0, "beta": beta, "r2": r2, "n": int(x.size)}