from pathlib import Path
import numpy as np
import pandas as pd
import math

# --- tiny geo helper ---
def _gc_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(a)))

def _load_run_csv(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise IOError(f"{path} does not exist")

    df = pd.read_csv(path)

    # case-insensitive column map
    cmap = {c.lower(): c for c in df.columns}
    def getcol(name, default=np.nan):
        key = name.lower()
        if key in cmap:
            return df[cmap[key]].to_numpy()
        return np.full(len(df), default)

    # parse time (t/time/utc/timestamp), normalize to UTC, round to seconds
    time_cols = [c for c in ("t", "time", "utc", "timestamp") if c in cmap]
    if time_cols:
        raw = df[cmap[time_cols[0]]]
        t_parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        # count real timestamps
        n_real = t_parsed.notna().sum()
        if n_real > 0:
            t = t_parsed.dt.round("S").to_numpy("datetime64[ns]")
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
        "DR": getcol("DR").astype(float) if "dr" in cmap else getcol("dr").astype(float),
        "misfit": getcol("misfit").astype(float),
        "azgap": getcol("azgap").astype(float),
        "connectedness": float(np.nanmean(getcol("connectedness"))) if "connectedness" in cmap else np.nan,
        "tag": path.stem,
    }
    return out

# --- aligner that falls back to index if time-intersection is empty ---
def _align_two(A, B, return_common: bool = False):
    """
    Align two runs:
      - If both have real timestamps, intersect on rounded-to-second time.
      - If no overlap on time (or one/both lack times), fall back to index alignment.
    Returns (A2, B2) or (A2, B2, common) if return_common=True.
    """
    def take(D, idx, common_vals):
        T = {}
        for k, v in D.items():
            if isinstance(v, np.ndarray) and v.shape[0] == D["t"].shape[0]:
                T[k] = v[idx]
            else:
                T[k] = v
        T["t"] = common_vals
        return T

    used_time = False
    if A["time_kind"] == "real" and B["time_kind"] == "real":
        # normalize/round to seconds for intersection
        tA = A["t"].astype("datetime64[s]").astype("datetime64[ns]")
        tB = B["t"].astype("datetime64[s]").astype("datetime64[ns]")
        common = np.intersect1d(tA, tB)
        if common.size > 0:
            idxA = {v: i for i, v in enumerate(tA)}
            idxB = {v: i for i, v in enumerate(tB)}
            iA = np.array([idxA[v] for v in common], dtype=int)
            iB = np.array([idxB[v] for v in common], dtype=int)
            used_time = True
        else:
            # fall back to index alignment
            n = min(A["t"].shape[0], B["t"].shape[0])
            if n <= 0:
                return None
            iA = np.arange(n, dtype=int)
            iB = np.arange(n, dtype=int)
            common = iA  # synthetic index
    else:
        # index alignment
        n = min(A["t"].shape[0], B["t"].shape[0])
        if n <= 0:
            return None
        iA = np.arange(n, dtype=int)
        iB = np.arange(n, dtype=int)
        common = iA

    A2 = take(A, iA, common)
    B2 = take(B, iB, common)
    if return_common:
        return A2, B2, common, ("time" if used_time else "index")
    return A2, B2

def compare_two_runs_csv(csvA, csvB, label="(baseline vs alt)"):
    if not csvA.is_file:
        print(f'{csvA} does not exist')
        return None
    if not csvB.is_file:
        print(f'{csvB} does not exist')
        return None
    print('both CSV files exist')
    def _debug_run_info(label, R):
        print(f"[{label}] kind={R['time_kind']}, n={R['t'].shape[0]}")
        if R["time_kind"] == "real" and R["t"].size:
            print("   first:", R["t"][0], " last:", R["t"][-1])

    # In compare_two_runs_csv just after loading:
    try:
        A = _load_run_csv(csvA); 
        B = _load_run_csv(csvB); 
    except:
        return None
    try:
        aligned = _align_two(A, B)
    except:
        _debug_run_info("A", A)
        _debug_run_info("B", B)
    if aligned is None:
        raise ValueError("No overlapping samples (time or index) between runs.")
    A, B = aligned

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

    # (optional) path correlation on lat/lon by index overlap
    m = np.isfinite(latA) & np.isfinite(lonA) & np.isfinite(latB) & np.isfinite(lonB)
    lat_r = float(np.corrcoef(latA[m], latB[m])[0,1]) if np.count_nonzero(m) > 3 else np.nan
    lon_r = float(np.corrcoef(lonA[m], lonB[m])[0,1]) if np.count_nonzero(m) > 3 else np.nan

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
        "align_mode": "time" if (A["time_kind"]=="real" and B["time_kind"]=="real") else "index",
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
        "delta_connectedness_B_minus_A": float(B["connectedness"] - A["connectedness"]) \
            if (np.isfinite(B["connectedness"]) and np.isfinite(A["connectedness"])) else np.nan,
    }



def safe_compare(summary_csv, csvA, csvB, label):
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
    out = Path(summary_csv)
    df = pd.DataFrame([row])
    if out.exists():
        df0 = pd.read_csv(out)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[compare] appended to {out} ({row['align_mode']} alignment, n={row['n_overlap']})")
    return row

######################################################

from pathlib import Path
import pandas as pd
import numpy as np

def load_all_event_comparisons(root: Path) -> pd.DataFrame:
    """
    Crawl event folders under `root` and stack `pairwise_run_comparisons.csv`.
    Returns a tidy DF with event_id inferred from folder name.
    """
    rows = []
    for csv in root.rglob("pairwise_run_comparisons.csv"):
        try:
            df = pd.read_csv(csv)
            df["event_id"] = csv.parent.name            # the event folder name
            rows.append(df)
        except Exception as e:
            print(f"[skip] {csv}: {e}")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # normalize label text to a short key
    out["variant"] = out["label"].astype(str)
    # guard presence of expected columns
    for c in ["mean_sep_km","delta_misfit_B_minus_A","delta_azgap_B_minus_A"]:
        if c not in out.columns: out[c] = np.nan
    return out

def add_composite_score(df: pd.DataFrame,
                        w_sep=1.0, w_misfit=0.5, w_azgap=0.1) -> pd.DataFrame:
    """
    Lower is better. Negative deltas are good if they reduce misfit/azgap.
    """
    d = df.copy()
    # z-score each metric for comparability (event-wise optional)
    # here: global z-scores; switch to per-event z if events differ strongly in scale
    for col in ["mean_sep_km","delta_misfit_B_minus_A","delta_azgap_B_minus_A"]:
        x = d[col].to_numpy(dtype=float)
        mu, sd = np.nanmean(x), np.nanstd(x) if np.nanstd(x)>0 else 1.0
        d[col+"_z"] = (x - mu)/sd
    d["score"] = (
        w_sep    * d["mean_sep_km_z"] +
        w_misfit * d["delta_misfit_B_minus_A_z"] +
        w_azgap  * d["delta_azgap_B_minus_A_z"]
    )
    return d

def summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    One line per variant: mean±SE of core metrics and composite score,
    plus 'wins' (how often variant beats baseline the most for an event).
    """
    g = df.groupby("variant", dropna=False)
    agg = g.agg(
        n_events          = ("event_id", "nunique"),
        n_rows            = ("event_id", "size"),
        mean_sep_km_mean  = ("mean_sep_km", "mean"),
        mean_sep_km_med   = ("mean_sep_km", "median"),
        mean_sep_km_se    = ("mean_sep_km", lambda x: np.nanstd(x)/np.sqrt(max(1,(x.notna().sum())))),
        dmisfit_mean      = ("delta_misfit_B_minus_A", "mean"),
        dmisfit_med       = ("delta_misfit_B_minus_A", "median"),
        dazgap_mean       = ("delta_azgap_B_minus_A", "mean"),
        score_mean        = ("score", "mean"),
        score_med         = ("score", "median"),
    ).reset_index().sort_values("score_mean")
    return agg

def per_event_winner(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, pick the variant with the lowest composite score.
    """
    # keep only the best per (event_id)
    idx = df_scored.groupby("event_id")["score"].idxmin()
    winners = df_scored.loc[idx, ["event_id","variant","score"]]
    win_counts = winners.groupby("variant").size().rename("wins").reset_index()
    return winners, win_counts.sort_values("wins", ascending=False)

