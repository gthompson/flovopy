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