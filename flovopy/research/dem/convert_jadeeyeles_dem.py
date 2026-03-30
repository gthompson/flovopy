#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

# --- INPUT / OUTPUT ---
GRD = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/original/dem_MVO_v3_10m_jadeeyles.grd")
OUT_DIR = GRD.parent
TIF_UTM  = OUT_DIR / "dem_MVO_v3_10m_utm20.tif"     # EPSG:32620
TIF_WGS  = OUT_DIR / "dem_MVO_v3_10m_wgs84.tif"     # EPSG:4326

# --- ASSUMED CRS ---
CRS_UTM = CRS.from_epsg(32620)   # WGS84 / UTM zone 20N
CRS_WGS = CRS.from_epsg(4326)    # WGS84 geographic

NODATA = -32768.0  # adjust if needed

def write_geotiff(path, data, transform, crs, nodata=NODATA):
    profile = {
        "driver": "GTiff",
        "width":  data.shape[1],
        "height": data.shape[0],
        "count":  1,
        "dtype":  "float32",
        "crs":    crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": False,
        "compress": "LZW",
        "predictor": 1,
        "interleave": "pixel",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)

def pick_xy_dims(da: xr.DataArray):
    """
    Return (ydim, xdim) names and their coordinate arrays (ycoord, xcoord).
    Prefer conventional names; fall back to monotonicity checks.
    """
    dims = list(da.dims)
    if da.ndim != 2:
        raise ValueError(f"Expected 2D grid, got {da.ndim}D with dims={dims}")

    # Heuristics by name
    name_map = {"lon": "x", "longitude": "x", "x": "x", "easting": "x",
                "lat": "y", "latitude": "y", "y": "y", "northing": "y"}
    role = {}
    for d in dims:
        key = d.lower()
        role[d] = name_map.get(key, None)

    ydim = next((d for d in dims if role.get(d) == "y"), None)
    xdim = next((d for d in dims if role.get(d) == "x"), None)

    # If names are ambiguous, decide by axis extents: x spans ~east-west (bigger range near easting),
    # y spans ~north-south, and UTM y typically ~1.8e6..2.0e6 here, while x ~ 640..680 km.
    if xdim is None or ydim is None:
        d0, d1 = dims
        c0 = np.asarray(da.coords[d0].values, float)
        c1 = np.asarray(da.coords[d1].values, float)
        # Guess: the dimension with values closer to longitude/easting range is x
        # Use value ranges (not perfect, but robust enough here)
        r0 = np.nanmax(c0) - np.nanmin(c0)
        r1 = np.nanmax(c1) - np.nanmin(c1)
        # Prefer that the longer physical span is x for UTM (east-west extent often larger on small islands)
        if r0 >= r1:
            xdim, ydim = d0, d1
        else:
            xdim, ydim = d1, d0

    xcoord = np.asarray(da.coords[xdim].values, float)
    ycoord = np.asarray(da.coords[ydim].values, float)
    return ydim, xdim, ycoord, xcoord

def main():
    # 1) Read GRD with xarray
    da = xr.open_dataarray(GRD)

    # 2) Identify dims/coords robustly
    ydim, xdim, ycoord, xcoord = pick_xy_dims(da)
    Z = np.asarray(da.values, dtype=np.float32)

    # 3) Ensure north-up array: row 0 corresponds to MAX Y (so affine 'e' is negative)
    # If y increases from south->north (ascending), flip rows
    if ycoord[0] < ycoord[-1]:
        Z = Z[::-1, :]
        ycoord = ycoord[::-1]

    # 4) Build affine transform (top-left origin). Use median spacing to tolerate small rounding.
    if xcoord.size < 2 or ycoord.size < 2:
        raise ValueError("Not enough coordinate values to determine grid spacing.")
    dx = float(np.median(np.diff(xcoord)))
    dy = float(np.median(np.diff(ycoord)))
    xmin = float(xcoord.min())
    ymax = float(ycoord.max())
    transform_utm = Affine(dx, 0, xmin, 0, -abs(dy), ymax)  # e must be negative for north-up

    # 5) Write UTM GeoTIFF
    write_geotiff(TIF_UTM, Z, transform_utm, CRS_UTM)
    with rasterio.open(TIF_UTM) as src:
        b = src.bounds
        print(f"✅ Wrote UTM GeoTIFF: {TIF_UTM}")
        print(f"   bounds (m): left={b.left:.2f} right={b.right:.2f} bottom={b.bottom:.2f} top={b.top:.2f}")
        print(f"   size: {src.width}x{src.height}  dx≈{abs(src.transform.a):.3f} m  dy≈{abs(src.transform.e):.3f} m")
        print(f"   orientation check: e (transform.e) = {src.transform.e:.6f} (should be negative)")

    # 6) Reproject UTM -> WGS84
    with rasterio.open(TIF_UTM) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, CRS_WGS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=CRS_WGS,
            transform=dst_transform,
            width=dst_w,
            height=dst_h,
            nodata=profile.get("nodata", NODATA),
            dtype="float32",
            tiled=False, compress="LZW", predictor=1, interleave="pixel",
        )
        data = src.read(1)

    with rasterio.open(TIF_WGS, "w", **profile) as dst:
        out = np.empty((dst_h, dst_w), dtype="float32")
        reproject(
            data, out,
            src_transform=transform_utm, src_crs=CRS_UTM,
            dst_transform=dst_transform,  dst_crs=CRS_WGS,
            resampling=Resampling.bilinear,
            src_nodata=profile["nodata"], dst_nodata=profile["nodata"],
        )
        dst.write(out, 1)

    with rasterio.open(TIF_WGS) as src:
        b = src.bounds
        print(f"✅ Wrote WGS84 GeoTIFF: {TIF_WGS}")
        print(f"   bounds (deg): lon[{b.left:.6f}, {b.right:.6f}]  lat[{b.bottom:.6f}, {b.top:.6f}]")
        print(f"   size: {src.width}x{src.height}")
        print(f"   quick sanity: lat_top={b.top:.6f} > lat_bot={b.bottom:.6f} (north-up expected)")

if __name__ == "__main__":
    main()