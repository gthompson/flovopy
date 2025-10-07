"""
ASL (Amplitude Source Location) analysis tools
----------------------------------------------
Supports:
1. Per-event lag extraction (cross-correlation).
2. Event location on a Grid (constant speed).
3. Optional fit: velocity increasing with distance (dome-centered).
4. Suite-level analysis: overall best constant / linear velocity.
5. Pairwise lag stability analysis and velocity vs Δd studies.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from obspy import read
from flovopy.processing.envelopes import (
    align_waveforms, align_waveforms_global,
    envelopes_stream, envelope_delays, locate_with_grid_from_delays
)
from flovopy.asl.distances import compute_or_load_distances
import re
import matplotlib.pyplot as plt

# ----------------- Event-level -----------------
def process_event(
    st,
    gridobj,
    inventory,
    cache_dir,
    dome_location,
    *,
    event_idx=0,
    smooth_s=1.0,
    max_lag_s=8.0,
    min_corr=0.5,
    c_range=(0.1, 5.0),
    n_c=80,
    min_delta_d_km=0.5,
    min_abs_lag_s=0.15,
    delta_d_weight=True,
    c_phys=(0.5, 3.5),
    auto_ref=True,
    topo_dem_path=None,
    output_dir=None,
):
    # prep per-event output folder & plot paths (so the locator can write PNGs)
    if output_dir is None:
        raise ValueError("process_event: output_dir is required to save score-vs-c plots.")
    eventdir       = Path(output_dir) / str(event_idx)
    eventdir.mkdir(parents=True, exist_ok=True)
    score_vs_c_png = str(eventdir / "score_vs_c.png")
    ccf_plot_dir   = str(eventdir / "ccf_plots")
    topo_png       = str(eventdir / "best_location_topo.png")

    # --- 1) reference-based alignment (kept for parity/debug)
    aligned_st_ref, lags_ref = align_waveforms(
        st.copy(),
        max_lag_s=max_lag_s,
        smooth_s=smooth_s,
        decimate_to=None
    )
    # optional check pass (not needed, but harmless)
    _, lags_ref_check = align_waveforms(
        aligned_st_ref.copy(),
        max_lag_s=max_lag_s,
        smooth_s=smooth_s,
        decimate_to=None
    )

    # --- 2) global alignment (used for locator)
    aligned_st_global, lags_global, delays = align_waveforms_global(
        st.copy(),
        max_lag_s=max_lag_s,
        min_corr=min_corr,
        smooth_s=smooth_s,
        decimate_to=None
    )

    # --- 2b) CCFs for plots / debug
    env_st = envelopes_stream(st, smooth_s=smooth_s, decimate_to=None)
    _, ccfs = envelope_delays(
        env_st,
        max_lag_s=max_lag_s,
        min_corr=min_corr,
        return_ccfs=True
    )

    # --- 3) grid locator with Stage-A guards + PNGs
    result = locate_with_grid_from_delays(
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
        plot_score_vs_c=score_vs_c_png,  # <- writes score-vs-c PNG
        dome_location=dome_location,
        verbose=False,
        # Stage-A debias/guards (same as notebook)
        min_delta_d_km=min_delta_d_km,
        min_abs_lag_s=min_abs_lag_s,
        delta_d_weight=delta_d_weight,
        c_phys=c_phys,
        auto_ref=auto_ref,
        # optional topo overlay
        topo_map_out=topo_png,
        topo_dem_path=str(topo_dem_path) if topo_dem_path else None,
    )

    # compact CSV-friendly lags strings (so parse_lags_column() can read later)
    def _lags_to_str(d):
        return ", ".join(f"{k}:{v:+.2f}s" for k,v in sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True))

    summary_row = {
        "event_idx": int(event_idx),
        "c_at_bestnode_kms": float(result.get("speed", np.nan)),
        "score_bestnode": float(result.get("score", np.nan)),
        "n_pairs_used": int(result.get("n_pairs", 0) or 0),
        "bestnode_lon": float(result.get("lon", np.nan)),
        "bestnode_lat": float(result.get("lat", np.nan)),
        "bestnode_elev_m": float(result.get("elev_m", np.nan)),
        "lags_ref": _lags_to_str(lags_ref),
        "lags_global": _lags_to_str(lags_global),
        "score_vs_c_png": score_vs_c_png,
    }

    return {
        "lags_ref": lags_ref,
        "lags_global": lags_global,
        "locator_result": result,
        "summary_row": summary_row,
    }

# ----------------- Suite-level -----------------
def summarize_suite(rows, out_csv):
    """Save per-event rows to CSV and print summary stats."""
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# ----------------- Pairwise lag stability -----------------
def parse_lags_column(series):
    parsed = []
    for txt in series.fillna(""):
        ev = {}
        for part in str(txt).split(","):
            if ":" not in part: continue
            sta, lag = part.strip().split(":")
            try:
                ev[sta.strip()] = float(lag.replace("s",""))
            except ValueError:
                continue
        parsed.append(ev)
    return parsed

def compute_pairwise_diffs(lag_dicts):
    diffs = []
    for lagmap in lag_dicts:
        pairs = {}
        stations = sorted(lagmap.keys())
        for i in range(len(stations)):
            for j in range(i+1, len(stations)):
                si, sj = stations[i], stations[j]
                pairs[(si,sj)] = lagmap[si] - lagmap[sj]
        diffs.append(pairs)
    return diffs

def summarize_pairwise(pairdiffs, method):
    records = []
    all_pairs = sorted({p for ev in pairdiffs for p in ev})
    for (a,b) in all_pairs:
        vals = [ev[(a,b)] for ev in pairdiffs if (a,b) in ev]
        if not vals: continue
        arr = np.array(vals,float)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        keep = arr[np.abs(arr-med) <= 3*mad] if mad>0 else arr
        if len(keep)<2: continue
        records.append(dict(pair=f"{a}-{b}", sta_a=a, sta_b=b,
                            n=len(keep), median_s=np.median(keep),
                            mean_s=np.mean(keep), std=np.std(keep),
                            mad=mad, min=np.min(keep), max=np.max(keep),
                            method=method))
    return pd.DataFrame(records)

# ----------------- Speed estimation -----------------
def estimate_speed_from_stable_pairs(
    *,
    grid,
    cache_dir: str,
    inventory,
    stable_df: Optional[pd.DataFrame] = None,
    stable_pairs_csv: Optional[str] = None,
    dome_location=None,
    use_value: str = "mean_s",
    weight_with: Optional[str] = "std",   # or "mad_scaled"
    min_delta_d_km: float = 0.2,
    min_abs_tau_s: float = 0.05,
    c_bounds: Tuple[float, float] = (0.2, 7.0),
    uncert_floor_s: float = 0.05,
    verbose: bool = True,
):
    # if only CSV given
    if stable_df is None and stable_pairs_csv:
        stable_df = pd.read_csv(stable_pairs_csv)
    if stable_df is None or stable_df.empty:
        raise ValueError("No stable pairs provided.")

    # then continue as before...
    """Compute per-pair apparent speeds and weighted median."""
    node_dists, coords, meta = compute_or_load_distances(
        grid, cache_dir, inventory=inventory, stream=None, use_elevation=True
    )
    def to_sta_key(s): return s.split(".")[1] if "." in s else s
    dist_by_sta = {to_sta_key(k):v for k,v in node_dists.items()}
    dome_idx = np.argmin((np.ravel(grid.gridlon)-dome_location["lon"])**2 +
                         (np.ravel(grid.gridlat)-dome_location["lat"])**2)

    rows=[]
    for _,r in stable_df.iterrows():
        a,b=r["sta_a"],r["sta_b"]
        if a not in dist_by_sta or b not in dist_by_sta: continue
        da,db=dist_by_sta[a][dome_idx],dist_by_sta[b][dome_idx]
        delta_d=abs(da-db)
        tau=float(r[use_value])
        if tau==0: continue
        c=delta_d/abs(tau)
        if c<c_bounds[0] or c>c_bounds[1]: continue
        rows.append(dict(sta_a=a,sta_b=b,delta_d_km=delta_d,
                         delta_tau_s=abs(tau),c_pair_km_s=c))
    return pd.DataFrame(rows)

def estimate_speed_from_stable_pairs(
    *,
    grid,
    cache_dir: str,
    inventory,
    stable_df: Optional[pd.DataFrame] = None,
    stable_pairs_csv: Optional[str] = None,
    dome_location=None,
    use_value: str = "mean_s",        # tries this, then falls back
    weight_with: Optional[str] = "mad_scaled",  # or "std" or None
    min_delta_d_km: float = 0.2,      # guard tiny baselines
    min_abs_tau_s: float = 0.05,      # guard near-zero lags
    c_bounds: Tuple[float, float] = (0.2, 7.0),
    uncert_floor_s: float = 0.05,
    verbose: bool = True,
):
    # 0) Load stable pairs
    if stable_df is None and stable_pairs_csv:
        stable_df = pd.read_csv(stable_pairs_csv)
    if stable_df is None or stable_df.empty:
        raise ValueError("No stable pairs provided.")

    df = stable_df.copy()

    # 1) Ensure sta_a/sta_b exist (derive from 'pair' if needed), and clean station codes
    if not {"sta_a", "sta_b"}.issubset(df.columns):
        if "pair" not in df.columns:
            raise ValueError("stable_df must have 'sta_a'/'sta_b' or a 'pair' column.")
        ab = df["pair"].astype(str).str.split("-", n=1, expand=True)
        ab.columns = ["sta_a", "sta_b"]
        df = pd.concat([df, ab], axis=1)

    def _clean_sta(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()

    df["sta_a"] = df["sta_a"].apply(_clean_sta)
    df["sta_b"] = df["sta_b"].apply(_clean_sta)

    # 2) Pick lag column (seconds)
    if use_value not in df.columns:
        # fallbacks
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

    # 3) Distances from dome to each station (collapse to one vector per STA)
    node_dists, coords, meta = compute_or_load_distances(
        grid, cache_dir, inventory=inventory, stream=None, use_elevation=True
    )

    def _to_sta_key(seed_or_sta: str) -> str:
        parts = str(seed_or_sta).split(".")
        return parts[1] if len(parts) >= 2 else str(seed_or_sta)

    dist_by_sta: Dict[str, np.ndarray] = {}
    for full_id, vec in node_dists.items():
        key = _to_sta_key(full_id)
        dist_by_sta.setdefault(key, np.asarray(vec, float))

    # 4) Resolve dome node index from dict / (lon,lat) / int
    def _resolve_dome_idx(grid, dome):
        if isinstance(dome, int):
            return dome
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

    # 5) Build per-pair rows with guards and weights
    rows, rejects = [], []
    for _, r in df.iterrows():
        a, b = str(r["sta_a"]), str(r["sta_b"])
        if a not in dist_by_sta or b not in dist_by_sta:
            rejects.append(("missing_station", a, b)); continue
        da = float(dist_by_sta[a][dome_idx]); db = float(dist_by_sta[b][dome_idx])
        if not (np.isfinite(da) and np.isfinite(db)):
            rejects.append(("nan_distance", a, b)); continue

        delta_d = abs(da - db)          # km
        tau = float(r[use_value])       # s
        if not np.isfinite(tau):
            rejects.append(("nan_tau", a, b)); continue
        if delta_d < min_delta_d_km:
            rejects.append(("small_delta_d", a, b)); continue
        if abs(tau) < min_abs_tau_s:
            rejects.append(("small_tau", a, b)); continue

        c = delta_d / abs(tau)          # km/s
        if not (c_bounds[0] <= c <= c_bounds[1]):
            rejects.append(("c_out_of_bounds", a, b)); continue

        # weights from uncertainty column if available
        if weight_with and (weight_with in r) and np.isfinite(r[weight_with]) and (r[weight_with] > 0):
            sig = max(float(r[weight_with]), uncert_floor_s)
            w = 1.0 / (sig ** 2)
        else:
            # simple geometric leverage if no uncertainty
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

    # 6) Robust weighted median for a single speed + bootstrap CI
    def _wmedian(x, w):
        x, w = np.asarray(x, float), np.asarray(w, float)
        if x.size == 0:
            return np.nan
        idx = np.argsort(x); xs, ws = x[idx], w[idx]
        p = np.cumsum(ws) / np.sum(ws)
        j = int(np.searchsorted(p, 0.5))
        return float(xs[min(max(j, 0), xs.size - 1)])

    x = per_pair["c_pair_km_s"].values
    w = per_pair["weight"].values
    c_hat = _wmedian(x, w)

    rng = np.random.default_rng(42)
    B = 500 if len(x) >= 6 else 200
    p = w / np.sum(w)
    boots = []
    for _ in range(B):
        idx = rng.choice(len(x), size=len(x), replace=True, p=p)
        boots.append(_wmedian(x[idx], w[idx]))
    ci_lo, ci_hi = float(np.nanpercentile(boots, 16)), float(np.nanpercentile(boots, 84))

    return {
        "per_pair": per_pair.sort_values("delta_d_km").reset_index(drop=True),
        "speed_km_s": float(c_hat),
        "ci_68": (ci_lo, ci_hi),
        "n_pairs_used": int(per_pair.shape[0]),
    }

# ----------------- Plot helper -----------------
def plot_speed_vs_distance(res_or_df, intercept=0.5, cmax=7.0, ax=None):
    """
    Scatter c_pair vs Δd with a linear fit whose intercept is fixed at `intercept`.
    Accepts either:
      - the dict returned by estimate_speed_from_stable_pairs (with key 'per_pair'), or
      - the per_pair DataFrame itself.

    Columns required in the DataFrame: ['delta_d_km','c_pair_km_s'].
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Accept dict result or DataFrame directly
    if isinstance(res_or_df, dict):
        if "per_pair" not in res_or_df:
            raise KeyError("Expected a dict with key 'per_pair' (DataFrame).")
        df = res_or_df["per_pair"]
    else:
        df = res_or_df

    # Column sanity
    need = {"delta_d_km", "c_pair_km_s"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Present: {list(df.columns)}")

    # Filter non-physical/high outliers and non-finite
    x = df["delta_d_km"].to_numpy()
    y = df["c_pair_km_s"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y) & (y <= float(cmax))
    x, y = x[m], y[m]

    if x.size < 2:
        raise ValueError(f"Not enough points to fit after filtering (kept {x.size}).")

    # Fixed-intercept linear fit: y = b + m x with b fixed
    b = float(intercept)
    num = np.sum(x * (y - b))
    den = np.sum(x * x)
    m_fit = num / den if den > 0 else np.nan

    # Diagnostics
    yhat = b + m_fit * x
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2  = float(1.0 - sse / sst) if sst > 0 else np.nan

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=28, alpha=0.9, label=f"pairs (n={x.size})")
    xx = np.linspace(0.0, max(x) * 1.05, 200)
    ax.plot(xx, b + m_fit * xx, "r-", lw=2,
            label=f"fit (b={b:.2f} km/s): y={m_fit:.2f}x+{b:.2f},  R²≈{r2:.3f}")
    ax.set_xlabel("Δd from dome between stations (km)")
    ax.set_ylabel("Apparent speed c_pair (km/s)")
    ax.set_ylim(0, float(cmax))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if ax is None:
        plt.tight_layout()
        plt.show()

    return {"slope": m_fit, "intercept": b, "r2": r2, "n": int(x.size)}