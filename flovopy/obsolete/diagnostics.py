from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from obspy import read_events, UTCDateTime
import matplotlib.pyplot as plt

def extract_asl_diagnostics(topdir: str, outputcsv: str, timestamp: bool = True) -> pd.DataFrame:
    """
    Scan ASL event directories for QuakeML + extract origin/amplitude diagnostics.
    """
    all_dirs = sorted(glob.glob(os.path.join(topdir, "*")))
    lod = []

    for thisdir in all_dirs:
        qml = glob.glob(os.path.join(thisdir, "event*Q*.qml"))
        if not (len(qml) == 1 and os.path.isfile(qml[0])):
            continue
        try:
            cat = read_events(qml[0])
            ev = cat.events[0]
            for i, origin in enumerate(ev.origins):
                r = {
                    "qml_path": qml[0],
                    "time": origin.time.isoformat() if origin.time else None,
                    "latitude": origin.latitude,
                    "longitude": origin.longitude,
                    "depth_km": (origin.depth or 0) / 1000.0 if origin.depth is not None else None,
                    "amplitude": ev.amplitudes[i].generic_amplitude if i < len(ev.amplitudes) else None,
                }
                oq = origin.quality
                if oq:
                    r.update({
                        "azimuthal_gap": oq.azimuthal_gap,
                        "station_count": oq.used_station_count,
                        "misfit": oq.standard_error,
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
    df.to_csv(outputcsv, index=False)
    print(f"[DIAG] Saved ASL diagnostics → {outputcsv}")
    return df



def _to_array(v):
    """
    Convert various time containers to numeric arrays:
    - list/tuple of UTCDateTime  -> float seconds
    - numpy object array of UTCDateTime -> float seconds
    - otherwise np.asarray(v)
    """
    import numpy as _np

    # Already (numeric) ndarray
    if isinstance(v, _np.ndarray) and v.dtype != object:
        return v

    # Numpy object array of UTCDateTime?
    if isinstance(v, _np.ndarray) and v.dtype == object and v.size:
        first = v.flat[0]
        if hasattr(first, "timestamp"):  # ObsPy UTCDateTime-compatible
            return _np.array([x.timestamp for x in v.ravel()], dtype=float).reshape(v.shape)
        return v  # leave as is if not UTCDateTime-like

    # Plain Python list/tuple?
    if isinstance(v, (list, tuple)) and len(v) and hasattr(v[0], "timestamp"):
        return _np.array([x.timestamp for x in v], dtype=float)

    # Fallback
    return _np.asarray(v)
def _per_step_dist_km(lat1, lon1, lat2, lon2):
    """Return per-step great-circle distance (km) between two tracks."""
    # fast vectorized haversine
    R = 6371.0
    φ1, λ1 = np.radians(lat1), np.radians(lon1)
    φ2, λ2 = np.radians(lat2), np.radians(lon2)
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))

def compare_asl_sources(asl1, asl2, *, atol=1e-8, rtol=1e-5):
    """
    Compare two ASL 'source' dicts with meaningful metrics.

    - Ignores 'misfit' (different backends define it differently).
    - Reports spatial divergence in km (mean/max and per-step series).
    - Reports summary diffs for DR, azgap, nsta.

    Returns
    -------
    dict with keys like:
      {
        "time_equal": bool,
        "location_km": {"mean":..., "p90":..., "max":..., "series": np.ndarray},
        "DR": {"mean_diff":..., "max_abs_diff":...},
        "azgap": {...},
        "nsta": {...}
      }
    """
    s1 = getattr(asl1, "source", None)
    s2 = getattr(asl2, "source", None)
    if s1 is None or s2 is None:
        return {"error": "one or both ASL objects have no source"}

    # Basic key sanity (but don't fail if extra keys exist)
    needed = {"t", "lat", "lon", "DR", "azgap", "nsta"}
    missing1 = needed - set(s1.keys())
    missing2 = needed - set(s2.keys())
    if missing1 or missing2:
        return {"error": f"missing keys: asl1:{missing1}, asl2:{missing2}"}

    # Time
    t1 = _to_array(s1["t"])
    t2 = _to_array(s2["t"])
    same_time = (t1.shape == t2.shape) and np.allclose(t1, t2, atol=atol, rtol=rtol, equal_nan=True)

    # Spatial divergence (km)
    lat1, lon1 = np.asarray(s1["lat"], float), np.asarray(s1["lon"], float)
    lat2, lon2 = np.asarray(s2["lat"], float), np.asarray(s2["lon"], float)
    if lat1.shape != lat2.shape or lon1.shape != lon2.shape:
        return {"error": f"lat/lon shape mismatch: {lat1.shape, lon1.shape} vs {lat2.shape, lon2.shape}"}
    dist_km = _per_step_dist_km(lat1, lon1, lat2, lon2)
    loc_stats = {
        "mean": float(np.nanmean(dist_km)),
        "p90": float(np.nanpercentile(dist_km, 90)),
        "max": float(np.nanmax(dist_km)),
        "series": dist_km,  # keep series for plotting if caller wants
    }

    # Helper to summarize numeric deltas
    def _summ(vk):
        a = np.asarray(s1[vk], float)
        b = np.asarray(s2[vk], float)
        if a.shape != b.shape:
            return {"shape_mismatch": (a.shape, b.shape)}
        d = b - a
        return {
            "mean_diff": float(np.nanmean(d)),
            "median_diff": float(np.nanmedian(d)),
            "p90_abs_diff": float(np.nanpercentile(np.abs(d), 90)),
            "max_abs_diff": float(np.nanmax(np.abs(d))),
        }

    out = {
        "time_equal": bool(same_time),
        "location_km": loc_stats,
        "DR": _summ("DR"),
        "azgap": _summ("azgap"),
        "nsta": _summ("nsta"),
    }
    # Optionally copy over connectedness if present
    for k in ("connectedness",):
        if k in s1 and k in s2:
            out[k] = {"asl1": float(np.asarray(s1[k]).mean()), "asl2": float(np.asarray(s2[k]).mean())}
    return out

def plot_asl_source_comparison(
    asl1, asl2, *,
    title="ASL comparison",
    show=True,
    outfile=None,
    figsize=(12, 9),
):
    """
    Visual comparison of two ASL results:
      - Panel A: lat/lon tracks (colored by time index)
      - Panel B: DR vs time
      - Panel C: per-step location difference (km)
      - Panel D (optional): nsta vs time

    Saves to `outfile` if given.
    """
    s1, s2 = asl1.source, asl2.source
    t1 = _to_array(s1["t"]); t2 = _to_array(s2["t"])
    lat1, lon1 = np.asarray(s1["lat"], float), np.asarray(s1["lon"], float)
    lat2, lon2 = np.asarray(s2["lat"], float), np.asarray(s2["lon"], float)

    # Align by index; if times differ, just index by ordinal
    n = min(len(lat1), len(lat2))
    idx = np.arange(n)
    dist_km = _per_step_dist_km(lat1[:n], lon1[:n], lat2[:n], lon2[:n])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1.1], hspace=0.35, wspace=0.25)

    # A) Map-ish scatter (lat/lon) colored by index
    axA = fig.add_subplot(gs[0, 0])
    sc1 = axA.scatter(lon1[:n], lat1[:n], c=idx, s=20, label="ASL-1", alpha=0.85)
    sc2 = axA.scatter(lon2[:n], lat2[:n], c=idx, s=20, marker="x", label="ASL-2", alpha=0.85)
    axA.set_xlabel("Longitude"); axA.set_ylabel("Latitude")
    axA.set_title("Location tracks (color = time index)")
    axA.legend(loc="best")

    # B) DR vs time
    from matplotlib.ticker import ScalarFormatter

    axB = fig.add_subplot(gs[0, 1])
    tt = t1[:n] if t1.shape == t2.shape and np.allclose(t1, t2, equal_nan=True) else idx

    dr1 = np.asarray(s1["DR"], float)[:n]
    dr2 = np.asarray(s2["DR"], float)[:n]

    # relative difference
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(dr1 - dr2) / np.maximum(np.abs(dr1), np.abs(dr2))
    mask = (rel_diff < 1e-3)  # 0.1%

    # force overlap where difference is tiny
    dr2_plot = dr2.copy()
    dr2_plot[mask] = dr1[mask]

    axB.plot(tt, dr1, label="ASL-1")
    axB.plot(tt, dr2_plot, label="ASL-2")

    axB.set_title("DR vs time")
    axB.set_xlabel("time" if tt is idx else "UTC sec")
    axB.set_ylabel("DR (scaled)")
    axB.legend(loc="best")

    # force absolute scale from 0 to max
    ymin, ymax = 0.0, max(np.nanmax(dr1), np.nanmax(dr2))
    axB.set_ylim(ymin, ymax*1.1)

    # turn off scientific offset and use plain numbers
    axB.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    # annotate maximum relative difference in %
    max_diff_pct = np.nanmax(rel_diff) * 100
    axB.text(
        0.98, 0.02,
        f"max Δ = {max_diff_pct:.3f}%",
        ha="right", va="bottom",
        transform=axB.transAxes,
        fontsize=9, color="gray"
    )

    # C) per-step distance (km)
    axC = fig.add_subplot(gs[1, 0])
    axC.plot(idx, dist_km)
    axC.set_title(f"Per-step location difference (km)  — mean={np.nanmean(dist_km):.2f}, max={np.nanmax(dist_km):.2f}")
    axC.set_xlabel("index"); axC.set_ylabel("km")

    # D) nsta vs time (if present)
    axD = fig.add_subplot(gs[1, 1])
    axD.plot(tt, np.asarray(s1.get("nsta", np.full(n, np.nan)), float)[:n], label="ASL-1")
    axD.plot(tt, np.asarray(s2.get("nsta", np.full(n, np.nan)), float)[:n], label="ASL-2")
    axD.set_title("Usable station count vs time")
    axD.set_xlabel("time" if tt is idx else "UTC sec"); axD.set_ylabel("nsta")
    axD.legend(loc="best")

    fig.suptitle(title)
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

