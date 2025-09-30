#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform

# Optional: PyGMT overlay (set PLOT_OVERLAY=False if you don’t want/need it)
PLOT_OVERLAY = True
try:
    import pygmt
except Exception:
    PLOT_OVERLAY = False

# -------------------------- CONFIG --------------------------
ASC_XYZ   = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/original/DEM_10m_1999_xyz.asc")
OUT_DIR   = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/auto_crs_fit")
DEFAULT_REGION = (-62.255, -62.135, 16.66, 16.84)  # lonmin, lonmax, latmin, latmax (Montserrat)
SCALE_FACTORS = [1.0, 0.4, 0.3048]  # raw, 10m->4m mis-scale guess, feet->meters
NODATA = -32768.0
# ------------------------------------------------------------

def _median_step(vals: np.ndarray) -> float:
    vals = np.unique(vals)
    steps = np.diff(np.sort(vals))
    return float(np.median(steps)) if steps.size else 1.0

def _grid_xyz(e: np.ndarray, n: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.unique(e)
    y_vals = np.unique(n)
    nx, ny = x_vals.size, y_vals.size
    x_index = np.searchsorted(x_vals, e)
    y_index = np.searchsorted(y_vals, n)
    Z = np.full((ny, nx), np.nan, dtype=float)
    Z[y_index, x_index] = z
    # Make north-up (row 0 = north)
    if y_vals[1] > y_vals[0]:
        Z = np.flipud(Z)
        y_vals = y_vals[::-1]
    return x_vals, y_vals, Z

def _affine_from_axes(x_vals: np.ndarray, y_vals: np.ndarray) -> Affine:
    dx = _median_step(x_vals)
    dy = _median_step(y_vals)
    x0 = float(x_vals.min())
    y0 = float(y_vals.max())
    return Affine.translation(x0, y0) * Affine.scale(dx, -dy)

def _write_geotiff(out_tif: Path, Z: np.ndarray, transform: Affine, crs: CRS) -> None:
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": Z.shape[0], "width": Z.shape[1], "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": NODATA,
        "tiled": False, "compress": "LZW", "predictor": 1, "interleave": "pixel",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(Z), Z, NODATA).astype("float32"), 1)

def reproject_to_wgs84(src_tif: Path, dst_tif: Path, dst_res: Optional[float] = None) -> None:
    with rasterio.open(src_tif) as src:
        dst_crs = CRS.from_epsg(4326)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height,
                       dtype="float32", nodata=profile.get("nodata", NODATA),
                       tiled=False, compress="LZW", predictor=1, interleave="pixel")
        data = src.read(1)
    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_tif, "w", **profile) as dst:
        out = np.empty((height, width), dtype="float32")
        reproject(
            data, out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform,   dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.bilinear,
            src_nodata=profile["nodata"], dst_nodata=profile["nodata"],
        )
        dst.write(out, 1)

def bbox_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    # (xmin, xmax, ymin, ymax)
    ixmin, ixmax = max(a[0], b[0]), min(a[1], b[1])
    iymin, iymax = max(a[2], b[2]), min(a[3], b[3])
    iw = max(0.0, ixmax - ixmin); ih = max(0.0, iymax - iymin)
    inter = iw * ih
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1.0
    return inter / union

def candidate_crs_list() -> List[CRS]:
    # 1) UTM zone 20N (WGS84)
    c1 = CRS.from_proj4("+proj=utm +zone=20 +datum=WGS84 +units=m +no_defs")
    # 2) UTM zone 20N (NAD27)
    c2 = CRS.from_proj4("+proj=utm +zone=20 +datum=NAD27 +units=m +no_defs")
    # 3) British West Indies Grid, Clarke 1880, lon0=-62, k=0.9995, FE=400000
    c3 = CRS.from_proj4(
        "+proj=tmerc +lat_0=0 +lon_0=-62 +k=0.9995 +x_0=400000 +y_0=0 "
        "+a=6378249.145 +b=6356514.86955 +units=m +no_defs"
    )
    # 4) Same as #3 but International 1924
    c4 = CRS.from_proj4(
        "+proj=tmerc +lat_0=0 +lon_0=-62 +k=0.9995 +x_0=400000 +y_0=0 "
        "+a=6378388 +b=6356911.946 +units=m +no_defs"
    )
    return [c1, c2, c3, c4]

def detect_elevation_units(z: np.ndarray) -> str:
    zmax = np.nanmax(z)
    if zmax > 2000:  # very rough heuristic
        return "feet"
    return "meters"

def scale_xy_about_lower_left(x: np.ndarray, y: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray]:
    # scale relative to the dataset’s lower-left corner (prevents artificial shifts)
    x0, y0 = float(np.min(x)), float(np.min(y))
    return x0 + s * (x - x0), y0 + s * (y - y0)

def test_combo(e: np.ndarray, n: np.ndarray, z: np.ndarray, scale: float, crs: CRS, label: str) -> Tuple[float, Path, Path]:
    # Scale XY about lower-left (0 shift in origin)
    xs, ys = scale_xy_about_lower_left(e, n, scale)
    # Grid and transform
    x_vals, y_vals, Z = _grid_xyz(xs, ys, z)
    T = _affine_from_axes(x_vals, y_vals)
    # Write native & WGS84, compute overlap
    native_tif = OUT_DIR / f"native_{label}.tif"
    wgs84_tif  = OUT_DIR / f"wgs84_{label}.tif"
    _write_geotiff(native_tif, Z, T, crs)
    reproject_to_wgs84(native_tif, wgs84_tif)
    with rasterio.open(wgs84_tif) as src:
        b = src.bounds
    dem_region = (b.left, b.right, b.bottom, b.top)
    score = bbox_overlap(dem_region, DEFAULT_REGION)
    print(f"[{label}] WGS84 bounds {np.round(dem_region, 6).tolist()}  overlap={score:.3f}")
    return score, native_tif, wgs84_tif

def plot_overlay(wgs84_tif: Path, png_out: Path):
    if not PLOT_OVERLAY:
        print("PyGMT not available; skipping overlay plot.")
        return
    fig = pygmt.Figure()
    fig.grdimage(
        grid=str(wgs84_tif),
        region=DEFAULT_REGION,
        projection="M12c",
        cmap="geo",
        shading=True,
        frame=["af",  "+tDEM + Coast"],
    )
    '''
    fig.coast(
        region=DEFAULT_REGION,
        projection="M12c",
        shorelines="1p,black",
        resolution="f",
        water="skyblue",
        land="white",
    )
    '''
    fig.savefig(png_out)
    print(f"✅ Wrote overlay plot: {png_out}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load XYZ
    arr = np.loadtxt(ASC_XYZ, dtype=float)
    if arr.shape[1] < 3:
        raise ValueError("Expected 3 columns: easting northing elevation")
    e, n, z = arr[:, 0], arr[:, 1], arr[:, 2]

    # Elevation units check (convert to meters if likely feet)
    units = detect_elevation_units(z)
    if units == "feet":
        print("⚠️ Elevations look like feet; converting to meters.")
        z = z * 0.3048

    best = None  # (score, label, native_tif, wgs84_tif)
    for s in SCALE_FACTORS:
        for i, crs in enumerate(candidate_crs_list(), 1):
            label = f"s{s}_{i}"
            score, native_tif, wgs84_tif = test_combo(e, n, z, s, crs, label)
            if (best is None) or (score > best[0]):
                best = (score, label, native_tif, wgs84_tif, crs, s)

    if best is None:
        raise RuntimeError("No combination produced a valid raster.")
    score, label, native_tif, wgs84_tif, crs, s = best
    print("\n🏁 Best combination:")
    print(f"  scale={s}, CRS={crs.to_string()}")
    print(f"  native: {native_tif}")
    print(f"  wgs84 : {wgs84_tif}")
    print(f"  overlap score: {score:.3f}")

    # Optional overlay
    if PLOT_OVERLAY:
        plot_overlay(wgs84_tif, OUT_DIR / f"overlay_best_{label}.png")

if __name__ == "__main__":
    main()