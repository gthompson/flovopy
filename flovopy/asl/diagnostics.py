from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from obspy import read_events, UTCDateTime

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


def compare_asl_sources(asl1, asl2, atol=1e-8, rtol=1e-5):
    """
    Compare the 'source' dicts from two ASL objects in a meaningful way.
    Focuses on lat/lon/DR/azgap/nsta. Ignores misfit differences.
    
    Returns True if identical, else a dict of differences with summary stats.
    """
    import numpy as np

    from obspy.geodetics import gps2dist_azimuth

    s1, s2 = getattr(asl1, "source", None), getattr(asl2, "source", None)
    if s1 is None or s2 is None:
        return {"missing": "one or both ASL objects have no source"}

    diffs = {}
    common_keys = set(s1.keys()) & set(s2.keys())

    for k in common_keys:
        if k == "misfit":  # skip direct misfit comparison
            continue

        v1, v2 = s1[k], s2[k]

        # helper: convert UTCDateTime -> float
        def to_arr(v):
            if isinstance(v, (list, tuple)) and len(v) and hasattr(v[0], "timestamp"):
                return np.array([x.timestamp for x in v], dtype=float)
            return np.asarray(v)

        a1, a2 = to_arr(v1), to_arr(v2)
        if a1.shape != a2.shape:
            diffs[k] = f"Shape mismatch: {a1.shape} vs {a2.shape}"
            continue

        if np.issubdtype(a1.dtype, np.number) and np.issubdtype(a2.dtype, np.number):
            if not np.allclose(a1, a2, atol=atol, rtol=rtol, equal_nan=True):
                delta = a2 - a1
                if k in ("lat", "lon"):
                    # compute km distances for each timestep
                    dists_km = []
                    for lat1, lon1, lat2, lon2 in zip(s1["lat"], s1["lon"], s2["lat"], s2["lon"]):
                        dist, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
                        dists_km.append(dist / 1000.0)
                    diffs["location_km"] = {
                        "mean": float(np.nanmean(dists_km)),
                        "max": float(np.nanmax(dists_km)),
                    }
                else:
                    diffs[k] = {
                        "mean_diff": float(np.nanmean(delta)),
                        "max_abs_diff": float(np.nanmax(np.abs(delta))),
                    }
        else:
            if not (a1 == a2).all():
                diffs[k] = "Non-numeric mismatch"

    return True if not diffs else diffs