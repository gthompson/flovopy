# flovopy/asl/compare.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: keep geometry centralized
# Expecting functions like: haversine_km(lat1, lon1, lat2, lon2)
# If your distances module exposes a different name, tweak the import here.
from flovopy.asl import distances as _dist

__all__ = [
    # data model
    "SourceTrack",
    # loaders
    "from_csv", "from_quakeml", "from_asl",
    # alignment + metrics
    "align_tracks", "compare_tracks",
    "intrinsic_metrics",
    # scoring
    "add_composite_score", "add_baseline_free_scores",
    # plotting
    "plot_source_comparison",
    # legacy wrappers (diagnostics compat)
    "compare_asl_sources", "plot_asl_source_comparison", "extract_asl_diagnostics",
]

# --- immediately after: from flovopy.asl import distances as _dist

if not hasattr(_dist, "haversine_km"):
    # Fallback adapter if your distances module uses a different name
    def _haversine_km(lat1, lon1, lat2, lon2):
        # vectorized haversine (km)
        R = 6371.0
        φ1, λ1 = np.radians(lat1), np.radians(lon1)
        φ2, λ2 = np.radians(lat2), np.radians(lon2)
        dφ, dλ = φ2 - φ1, λ2 - λ1
        a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
        return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))
else:
    _haversine_km = _dist.haversine_km

# ---------------------------
# Small data model
# ---------------------------
@dataclass
class SourceTrack:
    t: np.ndarray                  # np.datetime64[s] or int index
    lat: np.ndarray                # float
    lon: np.ndarray                # float
    DR: Optional[np.ndarray] = None
    misfit: Optional[np.ndarray] = None
    azgap: Optional[np.ndarray] = None
    nsta: Optional[np.ndarray] = None
    tag: str = ""
    path: Optional[str] = None

# ---------------------------
# Time helpers
# ---------------------------
def _to_datetime_seconds(x: pd.Series | np.ndarray) -> np.ndarray:
    """
    Parse to np.datetime64[s].
    Accepts: datetime-like, ObsPy UTCDateTime, numpy datetime64, or epoch seconds.
    Otherwise, falls back to a synthetic index (0,1,2...).
    """
    arr = np.asarray(x)
    # numpy datetime64
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]")
    # ObsPy UTCDateTime or python datetimes can be handled by pandas
    try:
        ts = pd.to_datetime(arr, utc=True, errors="coerce")
        if ts.notna().any():
            return ts.dt.round("s").to_numpy("datetime64[s]")
    except Exception:
        pass
    # numeric → treat as epoch seconds
    try:
        a = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(a).any():
            return (a.astype("float64") * 1e9).astype("datetime64[ns]").astype("datetime64[s]")
    except Exception:
        pass
    # fallback: synthetic index
    n = len(arr)
    return np.arange(n, dtype="timedelta64[s]").astype("datetime64[s]")

def _canon_array(v, dtype=float) -> np.ndarray:
    return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=dtype)

# ---------------------------
# Loaders
# ---------------------------
def from_csv(path: str | Path, tag: Optional[str] = None) -> SourceTrack:
    p = Path(path)
    df = pd.read_csv(p)

    cmap = {c.lower(): c for c in df.columns}
    def col(name, default=None):
        k = name.lower()
        if k in cmap: return df[cmap[k]]
        return pd.Series(default, index=df.index)

    # time: choose first available of t/time/utc/timestamp
    for cand in ("t", "time", "utc", "timestamp"):
        if cand in cmap:
            t = _to_datetime_seconds(df[cmap[cand]])
            break
    else:
        t = np.arange(len(df), dtype="timedelta64[s]").astype("datetime64[s]")

    lat   = _canon_array(col("lat"))
    lon   = _canon_array(col("lon"))
    DR    = _canon_array(col("DR"))    if "dr" in cmap else None
    mis   = _canon_array(col("misfit")) if "misfit" in cmap else None
    az    = _canon_array(col("azgap")) if "azgap" in cmap else None
    nsta  = _canon_array(col("nsta"))  if "nsta" in cmap else None

    return SourceTrack(t=t, lat=lat, lon=lon, DR=DR, misfit=mis, azgap=az, nsta=nsta,
                       tag=tag or p.stem, path=str(p))

def from_quakeml(qml_path: str | Path, tag: Optional[str] = None) -> SourceTrack:
    from obspy import read_events
    p = Path(qml_path)
    cat = read_events(str(p))
    ev = cat.events[0]
    lat, lon, tt = [], [], []
    for org in ev.origins:
        if org.latitude is None or org.longitude is None: continue
        lat.append(float(org.latitude))
        lon.append(float(org.longitude))
        try:
            tt.append(pd.to_datetime(org.time.datetime, utc=True).to_datetime64().astype("datetime64[s]"))
        except Exception:
            tt.append(np.datetime64("NaT"))
    t = np.array(tt, dtype="datetime64[s]")
    return SourceTrack(t=t, lat=np.array(lat, float), lon=np.array(lon, float),
                       tag=tag or p.stem, path=str(p))

def from_asl(aslobj, tag: Optional[str] = None) -> SourceTrack:
    s = getattr(aslobj, "source", {})
    # time can be UTCDateTime list or numeric seconds; coerce
    t_raw = s.get("t", [])
    if len(t_raw) and hasattr(t_raw[0], "timestamp"):
        t = pd.to_datetime([x.datetime for x in t_raw], utc=True).to_numpy("datetime64[s]")
    else:
        # assume already seconds since epoch
        t = _to_datetime_seconds(np.asarray(t_raw))
    def arr(k): 
        v = s.get(k, [])
        return np.asarray(v, float) if v is not None else None
    return SourceTrack(
        t=t, lat=arr("lat"), lon=arr("lon"), DR=arr("DR"),
        misfit=arr("misfit"), azgap=arr("azgap"), nsta=arr("nsta"),
        tag=tag or getattr(aslobj, "tag", "asl")
    )

# ---------------------------
# Alignment
# ---------------------------
def align_tracks(A: SourceTrack, B: SourceTrack) -> Tuple[SourceTrack, SourceTrack, str]:
    """
    Intersect on timestamps (to the second) if possible; else align by index length.
    Returns (A2, B2, mode) with mode in {"time","index"}.
    """
    tA = A.t.astype("datetime64[s]"); tB = B.t.astype("datetime64[s]")
    mode = "index"
    if np.issubdtype(tA.dtype, np.datetime64) and np.issubdtype(tB.dtype, np.datetime64):
        common = np.intersect1d(tA, tB)
        if common.size:
            idxA = {v: i for i, v in enumerate(tA)}
            idxB = {v: i for i, v in enumerate(tB)}
            iA = np.fromiter((idxA[v] for v in common), dtype=int, count=common.size)
            iB = np.fromiter((idxB[v] for v in common), dtype=int, count=common.size)
            mode = "time"
        else:
            n = min(len(tA), len(tB))
            iA = np.arange(n); iB = np.arange(n); common = iA
    else:
        n = min(len(tA), len(tB))
        iA = np.arange(n); iB = np.arange(n); common = iA

    def take(T: SourceTrack, idx: np.ndarray, new_t) -> SourceTrack:
        def take_if(x): 
            return None if x is None else np.asarray(x)[idx]
        return SourceTrack(
            t=np.asarray(new_t),
            lat=np.asarray(T.lat)[idx],
            lon=np.asarray(T.lon)[idx],
            DR=take_if(T.DR),
            misfit=take_if(T.misfit),
            azgap=take_if(T.azgap),
            nsta=take_if(T.nsta),
            tag=T.tag, path=T.path
        )
    A2 = take(A, iA, common if mode=="time" else A.t[iA])
    B2 = take(B, iB, common if mode=="time" else B.t[iB])
    return A2, B2, mode

# ---------------------------
# Pairwise metrics
# ---------------------------
def _series_sep_km(latA, lonA, latB, lonB) -> np.ndarray:
    """Per-sample great-circle separation (km) for aligned arrays."""
    out = np.full(len(latA), np.nan, float)
    m = np.isfinite(latA) & np.isfinite(lonA) & np.isfinite(latB) & np.isfinite(lonB)
    if not m.any(): return out
    out[m] = _dist.haversine_km(latA[m], lonA[m], latB[m], lonB[m])
    return out

def compare_tracks(A: SourceTrack, B: SourceTrack) -> dict:
    A2, B2, mode = align_tracks(A, B)
    sep = _series_sep_km(A2.lat, A2.lon, B2.lat, B2.lon)
    loc_stats = {
        "mean": float(np.nanmean(sep)) if np.isfinite(sep).any() else np.nan,
        "p90":  float(np.nanpercentile(sep, 90)) if np.isfinite(sep).any() else np.nan,
        "max":  float(np.nanmax(sep)) if np.isfinite(sep).any() else np.nan,
        "series": sep,
    }
    def summ(name):
        a = getattr(A2, name, None); b = getattr(B2, name, None)
        if a is None or b is None: return {}
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.shape != b.shape: return {"shape_mismatch": (a.shape, b.shape)}
        d = b - a
        return {
            "mean_diff": float(np.nanmean(d)),
            "median_diff": float(np.nanmedian(d)),
            "p90_abs_diff": float(np.nanpercentile(np.abs(d), 90)),
            "max_abs_diff": float(np.nanmax(np.abs(d))),
        }
    out = {
        "align_mode": mode,
        "location_km": loc_stats,
        "DR":     summ("DR"),
        "azgap":  summ("azgap"),
        "nsta":   summ("nsta"),
    }
    # optional connectedness delta if either track stashes a scalar value
    for k in ("connectedness",):
        if hasattr(A, k) and hasattr(B, k):
            try:
                out[k] = {"A": float(getattr(A, k)), "B": float(getattr(B, k))}
            except Exception:
                pass
    return out

# ---------------------------
# Absolute (baseline-free) metrics
# ---------------------------
def _series_sep_km(latA, lonA, latB, lonB) -> np.ndarray:
    out = np.full(len(latA), np.nan, float)
    m = np.isfinite(latA) & np.isfinite(lonA) & np.isfinite(latB) & np.isfinite(lonB)
    if not m.any(): 
        return out
    out[m] = _haversine_km(latA[m], lonA[m], latB[m], lonB[m])
    return out

def _pairwise_len_km(lat: np.ndarray, lon: np.ndarray) -> float:
    n = len(lat)
    if n <= 1:
        return np.nan
    m = np.isfinite(lat) & np.isfinite(lon)
    idx = np.where(m)[0]
    if idx.size < 2:
        return np.nan
    tot = 0.0
    for i in range(idx.size - 1):
        i0, i1 = idx[i], idx[i+1]
        tot += float(_haversine_km(lat[i0], lon[i0], lat[i1], lon[i1]))
    return tot

def _chord_km(lat: np.ndarray, lon: np.ndarray) -> float:
    m = np.isfinite(lat) & np.isfinite(lon)
    if np.count_nonzero(m) < 2:
        return np.nan
    idx = np.where(m)[0]
    i0, i1 = idx[0], idx[-1]
    return float(_haversine_km(lat[i0], lon[i0], lat[i1], lon[i1]))

def _roughness_ratio(lat: np.ndarray, lon: np.ndarray) -> float:
    L = _pairwise_len_km(lat, lon); C = _chord_km(lat, lon)
    if not np.isfinite(L) or not np.isfinite(C) or C <= 0: return np.nan
    return (L / C) - 1.0

def intrinsic_metrics(track: SourceTrack, *, connectedness: float | None = None) -> dict:
    lat, lon = np.asarray(track.lat, float), np.asarray(track.lon, float)
    mis = None if track.misfit is None else np.asarray(track.misfit, float)
    azg = None if track.azgap  is None else np.asarray(track.azgap,  float)

    n = len(lat)
    valid_xy = np.isfinite(lat) & np.isfinite(lon)
    valid_frac = float(valid_xy.mean()) if n else np.nan

    mean_misfit = float(np.nanmean(mis)) if mis is not None else np.nan
    mean_azgap  = float(np.nanmean(azg)) if azg is not None else np.nan

    # use provided connectedness if caller computed it elsewhere
    conn = connectedness if connectedness is not None else np.nan

    return {
        "tag": track.tag,
        "path": track.path,
        "n_samples": int(n),
        "valid_frac": valid_frac,
        "mean_misfit": mean_misfit,
        "mean_azgap":  mean_azgap,
        "connectedness": float(conn) if np.isfinite(conn) else np.nan,
        "path_len_km": _pairwise_len_km(lat, lon),
        "chord_len_km": _chord_km(lat, lon),
        "roughness_ratio": _roughness_ratio(lat, lon),
    }

# ---------------------------
# Scoring
# ---------------------------
def add_composite_score(df: pd.DataFrame, *,
                        w_sep: float = 1.0,
                        w_misfit: float = 0.5,
                        w_azgap: float = 0.1,
                        w_conn: float = 0.0,
                        w_rough: float = 0.0) -> pd.DataFrame:
    d = df.copy()

    def add_z(col):
        if col not in d.columns: return None
        x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        mu, sd = np.nanmean(x), np.nanstd(x)
        if not np.isfinite(sd) or sd == 0: return None
        zc = col + "_z"; d[zc] = (x - mu) / sd; return zc

    z_sep   = add_z("mean_sep_km")
    z_dmis  = add_z("delta_misfit_B_minus_A")
    z_daz   = add_z("delta_azgap_B_minus_A")
    z_dconn = add_z("delta_connectedness_B_minus_A")
    z_drough= add_z("delta_roughness_B_minus_A")

    score = np.zeros(len(d))
    if z_sep  : score += w_sep   * d[z_sep]
    if z_dmis : score += w_misfit* d[z_dmis]
    if z_daz  : score += w_azgap * d[z_daz]
    if w_conn > 0 and z_dconn:  score += (-w_conn) * d[z_dconn]
    if w_rough> 0 and z_drough: score += ( w_rough) * d[z_drough]
    d["score"] = score
    return d

def add_baseline_free_scores(df_abs: pd.DataFrame,
                             *, weights: Optional[Dict[str, float]] = None,
                             cap: float = 3.0) -> pd.DataFrame:
    if df_abs is None or df_abs.empty: return df_abs
    d = df_abs.copy()
    if weights is None:
        weights = {
            "mean_misfit":       0.8,
            "mean_azgap":        0.2,
            "roughness_ratio":   0.2,
            "connectedness":    -0.2,
            "valid_frac":       -0.2,
        }

    def robust_z(x):
        x = pd.to_numeric(x, errors="coerce").to_numpy(float)
        m = np.isfinite(x)
        if not m.any(): return np.zeros_like(x)
        xm = x[m]; med = np.median(xm); mad = np.median(np.abs(xm - med))
        denom = (1.4826 * mad) if mad > 0 else (np.std(xm) if np.std(xm) > 0 else 1.0)
        z = np.zeros_like(x); z[m] = np.clip((xm - med)/denom, -cap, cap)
        return z

    out = []
    for ev, g in d.groupby("event_id", dropna=False):
        gg = g.copy()
        Z = {col: robust_z(gg.get(col)) for col in
             ("mean_misfit","mean_azgap","roughness_ratio","connectedness","valid_frac")}
        score = np.zeros(len(gg))
        for col, w in weights.items():
            if col in Z: score += w * Z[col]
        gg["score_abs"] = score
        out.append(gg)
    return pd.concat(out, ignore_index=True)

# ---------------------------
# Plotting
# ---------------------------
def plot_source_comparison(A: SourceTrack, B: SourceTrack, *,
                           title="ASL comparison", show=True, outfile=None,
                           figsize=(12, 9)):
    A2, B2, _ = align_tracks(A, B)
    n = min(len(A2.lat), len(B2.lat))
    idx = np.arange(n)
    sep = _series_sep_km(A2.lat[:n], A2.lon[:n], B2.lat[:n], B2.lon[:n])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1.1], hspace=0.35, wspace=0.25)

    axA = fig.add_subplot(gs[0,0])
    sc1 = axA.scatter(A2.lon[:n], A2.lat[:n], c=idx, s=20, label=A.tag or "A", alpha=0.85)
    sc2 = axA.scatter(B2.lon[:n], B2.lat[:n], c=idx, s=20, marker="x", label=B.tag or "B", alpha=0.85)
    axA.set_xlabel("Longitude"); axA.set_ylabel("Latitude")
    axA.set_title("Location tracks (color = time index)"); axA.legend(loc="best")

    axB = fig.add_subplot(gs[0,1])
    tt = A2.t[:n] if A2.t.shape == B2.t.shape else idx
    if A.DR is not None: axB.plot(tt, np.asarray(A2.DR, float)[:n], label=f"{A.tag}-DR")
    if B.DR is not None:
        dr2 = np.asarray(B2.DR, float)[:n]
        if A.DR is not None:
            m = np.errstate(divide="ignore", invalid="ignore")
            with m:
                rel = np.abs(dr2 - np.asarray(A2.DR,float)[:n]) / np.maximum(np.abs(dr2), np.abs(np.asarray(A2.DR,float)[:n]))
            dr2 = np.where(rel < 1e-3, np.asarray(A2.DR,float)[:n], dr2)
        axB.plot(tt, dr2, label=f"{B.tag}-DR")
    axB.set_title("DR vs time"); axB.set_xlabel("time"); axB.set_ylabel("DR"); axB.legend(loc="best")

    axC = fig.add_subplot(gs[1,0])
    axC.plot(idx, sep)
    axC.set_title(f"Per-step location difference (km) — mean={np.nanmean(sep):.2f}, max={np.nanmax(sep):.2f}")
    axC.set_xlabel("index"); axC.set_ylabel("km")

    axD = fig.add_subplot(gs[1,1])
    if A.nsta is not None: axD.plot(tt, np.asarray(A2.nsta,float)[:n], label=f"{A.tag}-nsta")
    if B.nsta is not None: axD.plot(tt, np.asarray(B2.nsta,float)[:n], label=f"{B.tag}-nsta")
    axD.set_title("Usable station count vs time"); axD.set_xlabel("time"); axD.set_ylabel("nsta"); axD.legend(loc="best")

    fig.suptitle(title)
    if outfile: fig.savefig(outfile, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)



# ---- Legacy wrappers (diagnostics.py compatibility) ----

def compare_asl_sources(asl1, asl2, *, atol=1e-8, rtol=1e-5):
    """
    Wrapper to compare two ASL objects (legacy API).
    Returns a dict similar to the old diagnostics.compare_asl_sources.
    """
    A = from_asl(asl1, tag=getattr(asl1, "tag", "ASL-1"))
    B = from_asl(asl2, tag=getattr(asl2, "tag", "ASL-2"))
    A2, B2, mode = align_tracks(A, B)
    sep = _series_sep_km(A2.lat, A2.lon, B2.lat, B2.lon)
    out = {
        "time_equal": (A2.t.shape == B2.t.shape) and np.all(A2.t == B2.t),
        "location_km": {
            "mean": float(np.nanmean(sep)) if np.isfinite(sep).any() else np.nan,
            "p90":  float(np.nanpercentile(sep, 90)) if np.isfinite(sep).any() else np.nan,
            "max":  float(np.nanmax(sep)) if np.isfinite(sep).any() else np.nan,
            "series": sep,
        },
    }
    def _summ(a, b):
        if a is None or b is None: 
            return {}
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.shape != b.shape: 
            return {"shape_mismatch": (a.shape, b.shape)}
        d = b - a
        return {
            "mean_diff": float(np.nanmean(d)),
            "median_diff": float(np.nanmedian(d)),
            "p90_abs_diff": float(np.nanpercentile(np.abs(d), 90)),
            "max_abs_diff": float(np.nanmax(np.abs(d))),
        }
    out["DR"]    = _summ(A2.DR,    B2.DR)
    out["azgap"] = _summ(A2.azgap, B2.azgap)
    out["nsta"]  = _summ(A2.nsta,  B2.nsta)
    return out

def plot_asl_source_comparison(asl1, asl2, *, title="ASL comparison", show=True, outfile=None, figsize=(12,9)):
    """
    Wrapper around plot_source_comparison to accept ASL objects (legacy API).
    """
    A = from_asl(asl1, tag=getattr(asl1, "tag", "ASL-1"))
    B = from_asl(asl2, tag=getattr(asl2, "tag", "ASL-2"))
    return plot_source_comparison(A, B, title=title, show=show, outfile=outfile, figsize=figsize)

# from diagnostics.py
def extract_asl_diagnostics(topdir: str, outputcsv: str, timestamp: bool = True) -> pd.DataFrame:
    """
    Scan ASL event directories for QuakeML + extract origin/amplitude diagnostics.
    """
    import glob
    from obspy import read_events, UTCDateTime

    all_dirs = sorted(glob.glob(str(Path(topdir) / "*")))
    lod = []

    for thisdir in all_dirs:
        qml = glob.glob(str(Path(thisdir) / "event*Q*.qml"))
        if not (len(qml) == 1 and Path(qml[0]).is_file()):
            continue
        try:
            cat = read_events(qml[0])
            ev = cat.events[0]
            for i, origin in enumerate(ev.origins):
                r = {
                    "qml_path": qml[0],
                    "time": origin.time.isoformat() if origin.time else None,
                    "latitude": float(origin.latitude) if origin.latitude is not None else None,
                    "longitude": float(origin.longitude) if origin.longitude is not None else None,
                    "depth_km": (float(origin.depth) / 1000.0) if origin.depth is not None else None,
                    "amplitude": ev.amplitudes[i].generic_amplitude if i < len(ev.amplitudes) else None,
                }
                oq = origin.quality
                if oq:
                    r.update({
                        "azimuthal_gap": getattr(oq, "azimuthal_gap", None),
                        "station_count": getattr(oq, "used_station_count", None),
                        "misfit": getattr(oq, "standard_error", None),
                        "min_dist_km": getattr(oq, "minimum_distance", None),
                        "max_dist_km": getattr(oq, "maximum_distance", None),
                        "median_dist_km": getattr(oq, "median_distance", None),
                    })
                lod.append(r)
        except Exception as e:
            print(f"[DIAG:WARN] Could not parse {qml[0]}: {e}")

    df = pd.DataFrame(lod)
    if timestamp:
        base, ext = os.path.splitext(outputcsv)
        outputcsv = f"{base}_{int(UTCDateTime().timestamp)}{ext or '.csv'}"
    Path(outputcsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputcsv, index=False)
    print(f"[DIAG] Saved ASL diagnostics → {outputcsv}")
    return df