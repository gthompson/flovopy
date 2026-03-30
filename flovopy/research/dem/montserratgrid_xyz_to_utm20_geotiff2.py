#!/usr/bin/env python3
"""
Convert Montserrat-grid XYZ DEM to GeoTIFF in UTM20N, and transform Montserrat-grid points to WGS84.

Usage examples
--------------
# DEM path → UTM20N GeoTIFF (Variant A: no scale ppm)
python xyz_to_utm20_geotiff.py   --xyz DEM_10m_1999_xyz.asc   --out dem_utm20.tif   --variant A   --dtype float32   --dst_res 10

# Also transform a points CSV (E,N,H,Site) from Montserrat grid to WGS84/UTM20N
python xyz_to_utm20_geotiff.py   --points_in mont_points.csv   --points_out mont_points_wgs84_utm20.csv   --variant A

If results are offset consistently by a few–tens of metres, try Variant B (adds +towgs84 scale=9.9976 ppm):
python xyz_to_utm20_geotiff.py --variant B ...

Inputs
------
- XYZ is assumed to be a regular grid in a local Montserrat Transverse Mercator based on Clarke 1880,
  with parameters as provided (False Easting 400000, lon0=-62°, lat0=0°, k0=0.9995).

- Points CSV must include headers: E,N,H,Site  (delimiter auto-detected).

Outputs
-------
- GeoTIFF written first in the source (Montserrat) CRS, then reprojected to EPSG:32620 (UTM20N, WGS84).
- Points CSV augmented with lon/lat (EPSG:4326) and UTM20N columns.
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


def build_montserrat_crs(variant: str = "A") -> CRS:
    """
    Construct the Montserrat 'local grid' CRS as a custom PROJ string.

    Parameters from field notes:
      - Projection: Transverse Mercator
      - lat_0=0, lon_0=-62, k=0.9995
      - x_0=400000, y_0=0
      - Ellipsoid: Clarke 1880 (a=6378249.145, b=6356514.86955)
      - Bursa-Wolf to WGS84: dx=132.938, dy=-128.285, dz=-383.111, rx=0, ry=0, rz=12.7996 arcsec, scale_ppm=0 or 9.9976

    Variant A: scale_ppm = 0 (recommended to try first)
    Variant B: scale_ppm = 9.9976
    """
    variant = str(variant).upper().strip()
    scale_ppm = 0.0 if variant == "A" else 9.9976

    proj4 = (
        "+proj=tmerc +lat_0=0 +lon_0=-62 +k=0.9995 "
        "+x_0=400000 +y_0=0 "
        "+a=6378249.145 +b=6356514.86955 +units=m +no_defs "
        f"+towgs84=132.938,-128.285,-383.111,0,0,12.7996,{scale_ppm}"
    )
    return CRS.from_string(proj4)


def infer_xyz_grid(xyz_path: Path) -> Tuple[np.ndarray, Affine, int, int]:
    """
    Load a whitespace/CSV XYZ with columns X Y Z (in Montserrat grid meters) and infer a regular grid.

    Returns (Z2D, transform, width, height) in row-major order (north-up assumed).
    """
    xs, ys, zs = [], [], []
    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            xs.append(x); ys.append(y); zs.append(z)

    if not xs:
        raise ValueError(f"No numeric XYZ data found in {xyz_path}")

    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")
    zs = np.asarray(zs, dtype="float32")

    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    width = x_unique.size
    height = y_unique.size

    x_unique.sort()
    y_unique.sort()

    dxs = np.diff(x_unique)
    dys = np.diff(y_unique)
    dx = float(np.median(dxs)) if dxs.size else 1.0
    dy = float(np.median(dys)) if dys.size else 1.0

    x_to_ix = {v: i for i, v in enumerate(x_unique)}
    y_to_iy = {v: i for i, v in enumerate(y_unique)}

    Z = np.full((height, width), np.nan, dtype="float32")
    for x, y, z in zip(xs, ys, zs):
        ix = x_to_ix.get(x)
        iy = y_to_iy.get(y)
        if ix is None or iy is None:
            continue
        Z[iy, ix] = z

    minx, maxy = float(x_unique.min()), float(y_unique.max())
    transform = Affine(dx, 0.0, minx, 0.0, -dy, maxy)

    return Z, transform, width, height


def write_source_geotiff(out_path: Path, Z: np.ndarray, transform: Affine, crs: CRS, dtype: str = "float32") -> None:
    profile = {
        "driver": "GTiff",
        "width": Z.shape[1],
        "height": Z.shape[0],
        "count": 1,
        "dtype": dtype,
        "crs": crs.to_wkt(),
        "transform": transform,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(Z.astype(dtype), 1)


def reproject_to_utm20(src_path: Path, dst_path: Path, dst_res: Optional[float] = None, resampling: Resampling = Resampling.bilinear, dtype: str = "float32") -> None:
    dst_crs = CRS.from_epsg(32620)

    with rasterio.open(src_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res
        )
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs.to_wkt(),
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": dtype,
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )


def autodetect_delimiter(header_line: str) -> str:
    if "," in header_line:
        return ","
    if "\t" in header_line:
        return "\t"
    if ";" in header_line:
        return ";"
    return ","


def transform_points_csv(points_in: Path, points_out: Path, mont_crs: CRS) -> None:
    """
    Read points CSV with headers E,N,H,Site (Montserrat grid), transform to:
      - lon/lat WGS84 (EPSG:4326)
      - UTM20N (EPSG:32620)
    and write an output CSV with extra columns.
    """
    wgs84 = CRS.from_epsg(4326)
    utm20 = CRS.from_epsg(32620)

    to_lonlat = Transformer.from_crs(mont_crs, wgs84, always_xy=True)
    to_utm20 = Transformer.from_crs(wgs84, utm20, always_xy=True)

    rows_out = []

    with open(points_in, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise ValueError("Empty points CSV")
        delim = autodetect_delimiter(first)
        f.seek(0)
        reader = csv.reader(f, delimiter=delim)
        headers = next(reader)

        def idx(name_variants):
            for nm in name_variants:
                for j, h in enumerate(headers):
                    if h.strip().lower() == nm:
                        return j
            return None

        iE = idx(["e", "east", "easting", "x"])
        iN = idx(["n", "north", "northing", "y"])
        iH = idx(["h", "elev", "z", "height"])
        iS = idx(["site", "station", "name", "id"])
        if None in (iE, iN, iH, iS):
            raise ValueError(f"Missing one of required columns E,N,H,Site in {points_in} (got headers {headers})")

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            try:
                E = float(row[iE]); N = float(row[iN]); H = float(row[iH])
            except Exception:
                continue
            name = row[iS].strip()

            lon, lat = to_lonlat.transform(E, N)
            ue, un = to_utm20.transform(lon, lat)

            rows_out.append({
                "Site": name,
                "E_mont": E,
                "N_mont": N,
                "H_m": H,
                "lon": lon,
                "lat": lat,
                "UTM20_E": ue,
                "UTM20_N": un,
            })

    out_fields = ["Site", "E_mont", "N_mont", "H_m", "lon", "lat", "UTM20_E", "UTM20_N"]
    with open(points_out, "w", encoding="utf-8", newline="") as w:
        writer = csv.DictWriter(w, fieldnames=out_fields)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Convert Montserrat-grid XYZ DEM to UTM20N GeoTIFF and/or transform Montserrat-grid points CSV to WGS84/UTM20N.")
    ap.add_argument("--xyz", type=str, help="Path to XYZ file (X Y Z columns) in Montserrat local grid.")
    ap.add_argument("--out", type=str, help="Output GeoTIFF path (UTM20N).")
    ap.add_argument("--variant", choices=["A", "B"], default="A", help="Bursa–Wolf variant: A=no scale, B=scale=9.9976 ppm.")
    ap.add_argument("--dtype", default="float32", help="Output dtype for rasters (default float32).")
    ap.add_argument("--dst_res", type=float, default=None, help="Target resolution in meters for UTM20N reprojection (optional).")

    ap.add_argument("--points_in", type=str, help="CSV with E,N,H,Site columns in Montserrat grid.")
    ap.add_argument("--points_out", type=str, help="Output CSV with WGS84/UTM20N columns.")

    args = ap.parse_args()

    mont_crs = build_montserrat_crs(args.variant)

    if args.xyz and args.out:
        xyz_path = Path(args.xyz)
        tmp_src_tif = xyz_path.with_suffix(".mont.tif")
        out_tif = Path(args.out)

        print(f"[DEM] Reading XYZ and inferring grid: {xyz_path}")
        Z, transform, width, height = infer_xyz_grid(xyz_path)
        print(f"[DEM] Grid: {width} x {height}  dx≈{transform.a:.3f}  dy≈{abs(transform.e):.3f} m")

        print(f"[DEM] Writing source CRS GeoTIFF: {tmp_src_tif}")
        write_source_geotiff(tmp_src_tif, Z, transform, mont_crs, dtype=args.dtype)

        print(f"[DEM] Reprojecting to UTM20N: {out_tif}")
        reproject_to_utm20(tmp_src_tif, out_tif, dst_res=args.dst_res, dtype=args.dtype)
        print(f"[DEM] Wrote: {out_tif}")

    if args.points_in and args.points_out:
        points_in = Path(args.points_in)
        points_out = Path(args.points_out)
        print(f"[PTS] Transforming points: {points_in}  →  {points_out}  (variant {args.variant})")
        transform_points_csv(points_in, points_out, mont_crs)
        print(f"[PTS] Wrote: {points_out}")

    if not ((args.xyz and args.out) or (args.points_in and args.points_out)):
        print("Nothing to do. Provide --xyz/--out and/or --points_in/--points_out.")


if __name__ == "__main__":
    main()
