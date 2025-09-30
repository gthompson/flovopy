#!/usr/bin/env python3
"""
remove_islands_from_dem.py

Remove small “islands” (positive-elevation blobs) from a coastal DEM.
Keeps the largest landmass and discards other components smaller than a
user-set minimum area.

Usage:
  python remove_islands_from_dem.py \
      --in /path/to/input.tif \
      --out /path/to/cleaned.tif \
      --min-km2 0.2 \
      --sea-level 0

Notes
- Works for GeoTIFFs in geographic (lon/lat) or projected CRS.
- “Area” is computed from pixel size; for lon/lat we scale by cos(mean_lat).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from scipy import ndimage

def _pixel_meters(transform: Affine, crs: CRS, lat_hint: float | None) -> tuple[float, float]:
    """Return (dx_m, dy_m) for one pixel."""
    dx = abs(transform.a)
    dy = abs(transform.e)
    if crs and crs.is_geographic:
        # degrees -> meters using mean latitude
        lat = 0.0 if lat_hint is None else float(lat_hint)
        m_per_deg_lat = 111_132.0  # near-average
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
        return dx * m_per_deg_lon, dy * m_per_deg_lat
    else:
        # already meters
        return dx, dy

def remove_islands(
    dem_in: Path,
    dem_out: Path,
    sea_level: float = 0.0,
    min_island_km2: float = 0.2,
    keep_largest_always: bool = True,
    set_removed_to: float = 0.0,    # set small islands to sea level (or use np.nan)
) -> Path:
    dem_in = Path(dem_in); dem_out = Path(dem_out)
    with rasterio.open(dem_in) as src:
        z = src.read(1, masked=False).astype("float32")  # keep original nodata values in profile
        prof = src.profile.copy()
        tr = src.transform
        crs = src.crs
        # mean latitude (for degree->meter scaling)
        ny = src.height
        # center of top/bottom rows in world y
        y0 = tr.f + tr.e * (0 + 0.5)
        y1 = tr.f + tr.e * (ny - 1 + 0.5)
        lat_hint = (y0 + y1) / 2.0

    # build land mask (≥ sea_level) and finite
    finite = np.isfinite(z)
    land = finite & (z >= sea_level)

    if not np.any(land):
        raise RuntimeError("No land pixels found at/above the sea_level threshold.")

    # connectivity (8-neighbors)
    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]], dtype=np.uint8)

    labels, nlab = ndimage.label(land, structure=structure)
    if nlab == 1 and min_island_km2 <= 0:
        # nothing to do
        dem_out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dem_out, "w", **prof) as dst:
            dst.write(z, 1)
        return dem_out

    # compute pixel size and convert area threshold to pixels
    dx_m, dy_m = _pixel_meters(tr, crs, lat_hint)
    pix_area_m2 = max(dx_m * dy_m, 1e-6)
    min_pix = int(np.ceil((min_island_km2 * 1_000_000.0) / pix_area_m2))

    # component sizes
    sizes = np.bincount(labels.ravel())
    # label 0 is background; find largest land label
    if nlab > 0:
        land_sizes = sizes[1:]
        largest_label = 1 + int(np.argmax(land_sizes))
    else:
        largest_label = 0  # shouldn't happen

    # mask of components to keep
    keep = np.zeros_like(land, dtype=bool)
    if keep_largest_always and largest_label > 0:
        keep |= (labels == largest_label)
    # keep any other land components >= min_pix
    for lbl in range(1, nlab + 1):
        if lbl == largest_label and keep_largest_always:
            continue
        if sizes[lbl] >= min_pix:
            keep |= (labels == lbl)

    # “remove” = land but not kept
    drop = land & (~keep)

    cleaned = z.copy()
    # set removed land to sea (or NaN)
    if np.isnan(set_removed_to):
        cleaned[drop] = np.nan
    else:
        cleaned[drop] = float(set_removed_to)

    # write output
    dem_out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_out, "w", **prof) as dst:
        dst.write(cleaned.astype("float32"), 1)

    # quick report
    km2_dropped = sizes[1:].sum() - sizes[largest_label] if keep_largest_always else 0
    km2_dropped *= pix_area_m2 / 1_000_000.0
    print(f"✓ Saved cleaned DEM → {dem_out}")
    print(f"   Pixel ~ {dx_m:.1f}×{dy_m:.1f} m; min area = {min_island_km2:.3f} km² (~{min_pix} px)")
    print(f"   Components: {nlab}; dropped-small-land ≈ {km2_dropped:.3f} km² (if only-largest kept)")
    return dem_out

def parse_args():
    p = argparse.ArgumentParser(description="Remove small offshore 'islands' from a DEM.")
    p.add_argument("--in",  dest="infile",  required=True, type=Path)
    p.add_argument("--out", dest="outfile", required=True, type=Path)
    p.add_argument("--sea-level", type=float, default=0.0, help="Threshold separating sea/land (m).")
    p.add_argument("--min-km2",   type=float, default=0.2, help="Minimum island area to keep (km²).")
    p.add_argument("--keep-largest", action="store_true", help="Always keep the largest landmass.")
    p.add_argument("--remove-to-nan", action="store_true",
                   help="Set removed pixels to NaN instead of sea level.")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    remove_islands(
        dem_in=a.infile,
        dem_out=a.outfile,
        sea_level=a.sea_level,
        min_island_km2=a.min_km2,
        keep_largest_always=a.keep_largest,
        set_removed_to=(np.nan if a.remove_to_nan else 0.0),
    )