#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Callable, Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from whitebox import WhiteboxTools

import geopandas as gpd
from shapely.geometry import shape as shp_shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# ============================= Optional ASL NodeGrid =============================
_HAVE_ASL = True
try:
    from flovopy.asl.grid import NodeGrid
except Exception:
    _HAVE_ASL = False

# ============================= Small utilities =============================

def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _exists_nonempty(p: Path) -> bool:
    return p.exists() and os.path.getsize(p) > 0

def _print_path(label: str, p: Path | str):
    p = Path(p).resolve()
    print(f"{label}: {p}  exists={p.exists()}  size={os.path.getsize(p) if p.exists() else 'NA'}")

def read_extent(tif: Path) -> tuple[float, float, float, float, rasterio.crs.CRS | None]:
    with rasterio.open(tif) as src:
        b = src.bounds
        return float(b.left), float(b.right), float(b.bottom), float(b.top), src.crs

# ============================= DEM I/O helpers =============================

def normalize_for_wbt(dem_in: Path, dem_out: Path, nodata_val: float = -32768.0) -> Path:
    """
    Make a WBT-friendly GeoTIFF without changing georeferencing:
      - single band float32; replace NaNs with fixed nodata
      - LZW + predictor=1; not tiled
    """
    dem_in = dem_in.resolve(); dem_out = dem_out.resolve()
    with rasterio.open(dem_in) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            arr = np.where(arr == nod, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, nodata_val).astype("float32")

        prof = src.profile.copy()
        prof.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=nodata_val,
            tiled=False,
            compress="LZW",
            predictor=1,
            interleave="pixel",
        )

    dem_out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_out, "w", **prof) as dst:
        dst.write(arr, 1)
    return dem_out

# ============================= GDAL wrappers =============================

def _run(cmd: str) -> None:
    print("[GDAL]", cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"GDAL command failed (rc={rc}): {cmd}")

def gdal_reproject_to_utm20(src_tif: Path, dst_tif: Path, nodata: float = -32768.0,
                            res: float | None = None, overwrite: bool = False) -> Path:
    """
    Reproject to EPSG:32620 using gdalwarp.
    Optional ‘res’ sets target pixel size (meters). If None, gdal chooses.
    """
    src_tif = Path(src_tif); dst_tif = Path(dst_tif)
    if dst_tif.exists() and not overwrite:
        print(f"[GDAL] Exists, skipping: {dst_tif}")
        return dst_tif
    res_part = "" if res is None else f"-tr {res} {res} "
    cmd = (
        f'gdalwarp -t_srs EPSG:32620 {res_part}-dstnodata {nodata} '
        f'-r bilinear -multi -wo NUM_THREADS=ALL_CPUS '
        f'-co COMPRESS=LZW -co PREDICTOR=1 "{src_tif}" "{dst_tif}"'
    )
    _run(cmd)
    return dst_tif

def gdal_cut_to_polygon(src_tif: Path, polygon: Path, dst_tif: Path, nodata: float = -32768.0,
                        overwrite: bool = False) -> Path:
    """
    Cut DEM to island polygon to remove ocean. polygon can be Shapefile/GeoJSON/GPKG.
    """
    src_tif = Path(src_tif); dst_tif = Path(dst_tif)
    if dst_tif.exists() and not overwrite:
        print(f"[GDAL] Exists, skipping: {dst_tif}")
        return dst_tif
    cmd = (
        f'gdalwarp -cutline "{polygon}" -crop_to_cutline -dstnodata {nodata} '
        f'-r bilinear -multi -wo NUM_THREADS=ALL_CPUS '
        f'-co COMPRESS=LZW -co PREDICTOR=1 "{src_tif}" "{dst_tif}"'
    )
    _run(cmd)
    return dst_tif

# ======================= Auto-derive island polygon =======================

def derive_island_polygon_from_dem(
    dem_ll: Path,
    out_gpkg: Path,
    sea_level: float = 0.0,
    min_area_km2: float = 0.1,
    simplify_tol_deg: float = 0.0,
) -> Path:
    """
    Create a coastline/island polygon from DEM:
      - Land mask = valid pixels AND (elev > sea_level) if sea_level is not None,
        else just valid pixels (non-nodata).
      - Extract polygons, keep the largest (area in deg² converted to km² approx).
      - Optionally simplify (in degrees) and save to GPKG layer 'island'.
    Returns path to the GPKG.
    """
    dem_ll = Path(dem_ll).resolve()
    out_gpkg = Path(out_gpkg).resolve()

    with rasterio.open(dem_ll) as src:
        Z = src.read(1, masked=True)
        tr = src.transform
        crs = src.crs

    # Build land mask
    valid = np.isfinite(Z)
    if sea_level is not None:
        land = valid & (Z > sea_level)
    else:
        land = valid

    # Turn mask into polygons
    shp_gen = rio_shapes(land.astype(np.uint8), mask=None, transform=tr)
    polys: list[Polygon] = []
    for geom, val in shp_gen:
        if int(val) == 1:
            g = shp_shape(geom)
            if isinstance(g, (Polygon, MultiPolygon)):
                polys.append(g if isinstance(g, Polygon) else unary_union(g))

    if not polys:
        raise RuntimeError("No land polygon extracted from DEM (check sea_level and nodata).")

    # Merge and keep the largest island
    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        geoms = [merged]
    elif isinstance(merged, MultiPolygon):
        geoms = list(merged.geoms)
    else:
        geoms = [Polygon(merged)]

    # Approx area threshold (in km²) using crude lon/lat scaling (ok for small island)
    def _km2_of_polygon(p: Polygon) -> float:
        # quick approximate conversion: scale lon by cos(lat0)
        lon, lat = p.representative_point().x, p.representative_point().y
        m_per_deg_lat = 111_194.9266
        m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(lat))
        # approximate by converting bounds to meters and area via scale
        minx, miny, maxx, maxy = p.bounds
        sx = m_per_deg_lon
        sy = m_per_deg_lat
        # area scale factor (deg² → m²)
        return float(p.area) * (sx * sy) / 1e6

    geoms = [g for g in geoms if _km2_of_polygon(g) >= min_area_km2]
    if not geoms:
        raise RuntimeError("No polygon passed the minimum area filter.")

    island = max(geoms, key=_km2_of_polygon)

    # Optional simplify
    if simplify_tol_deg and simplify_tol_deg > 0:
        island = island.simplify(simplify_tol_deg, preserve_topology=True) or island

    gdf = gpd.GeoDataFrame({"name": ["island"]}, geometry=[island], crs=crs)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG", layer="island")
    print(f"[CUTLINE] Derived island polygon → {out_gpkg}")
    return out_gpkg

# ======================= Whitebox pipeline ========================

def wbt_run_abs(wbt: WhiteboxTools, tool: str, args_paths: dict, expect: Path | None = None) -> None:
    argv = []
    for k, v in args_paths.items():
        if v is None:
            argv.append(k)
        elif isinstance(v, Path):
            argv.append(f"{k}={str(v.resolve())}")
        else:
            argv.append(f"{k}={v}")
    print(f"WBT: {tool} {' '.join(argv)}")
    for k, v in args_paths.items():
        if isinstance(v, Path):
            _print_path(f"  arg {k}", v)
    ok = wbt.run_tool(tool, argv)
    if not ok and not (expect and expect.exists()):
        raise RuntimeError(f"Whitebox tool failed: {tool}")

def hydrocondition_dem(dem_in: Path, outdir: Path, breach: bool = False) -> Tuple[Path, Path, Path]:
    """Run hydrology **in projected UTM**: returns (filled_dem, d8_flowaccum_cells, d8_pointer)."""
    wbt = WhiteboxTools()
    wbt.set_working_dir(str(outdir.resolve()))
    wbt.verbose = True

    dem_in = dem_in.resolve()
    filled = (outdir / "04_dem_filled.tif").resolve()
    fa     = (outdir / "04_dem_fa.tif").resolve()
    pntr   = (outdir / "04_dem_d8pntr.tif").resolve()

    try:
        if breach:
            wbt_run_abs(wbt, "BreachDepressionsLeastCost",
                        {"--dem": dem_in, "--output": filled}, expect=filled)
        else:
            wbt_run_abs(wbt, "FillDepressions",
                        {"--dem": dem_in, "--output": filled, "--fix_flats": None}, expect=filled)
        if not _exists_nonempty(filled):
            raise RuntimeError("Filled DEM not written")
    except Exception as e:
        print(f"[warn] Primary depression removal failed: {e}")
        try:
            wbt_run_abs(wbt, "BreachDepressionsLeastCost",
                        {"--dem": dem_in, "--output": filled}, expect=filled)
        except Exception as e2:
            print(f"[warn] BreachDepressionsLeastCost also failed: {e2}")
            tmp_fill1 = (outdir / "dem_fill1.tif").resolve()
            wbt_run_abs(wbt, "FillSingleCellPits",
                        {"--dem": dem_in, "--output": tmp_fill1}, expect=tmp_fill1)
            wbt_run_abs(wbt, "BreachDepressionsLeastCost",
                        {"--dem": tmp_fill1, "--output": filled}, expect=filled)

    if not _exists_nonempty(filled):
        raise RuntimeError("Hydroconditioning failed: no filled/breached DEM produced")

    wbt_run_abs(wbt, "D8FlowAccumulation",
                {"--dem": filled, "--output": fa, "--out_type": "cells"}, expect=fa)
    wbt_run_abs(wbt, "D8Pointer",
                {"--dem": filled, "--output": pntr}, expect=pntr)

    return filled, fa, pntr

def choose_threshold_from_fa(fa_tif: Path, percentile: float = 97.5, min_cells: int = 800) -> int:
    with rasterio.open(fa_tif) as src:
        A = src.read(1, masked=True)
        q = float(np.percentile(A.compressed(), percentile))
        fa_min, fa_max = float(A.min()), float(A.max())
    thr = int(max(min_cells, q))
    print(f"FlowAccum stats: min={fa_min:.1f} max={fa_max:.1f} q{percentile:.0f}={q:.1f} → threshold={thr}")
    return thr

def extract_streams(fa_tif: Path, pntr_tif: Path, threshold: int, outdir: Path) -> tuple[Path, Path]:
    wbt = WhiteboxTools()
    wbt.set_working_dir(str(outdir.resolve()))
    wbt.verbose = True

    streams_tif = (outdir / "05_streams.tif").resolve()
    streams_vec = (outdir / "05_streams.shp").resolve()

    wbt_run_abs(wbt, "ExtractStreams",
                {"--flow_accum": fa_tif, "--output": streams_tif, "--threshold": str(threshold)}, expect=streams_tif)
    wbt_run_abs(wbt, "RasterStreamsToVector",
                {"--streams": streams_tif, "--d8_pntr": pntr_tif, "--output": streams_vec}, expect=streams_vec)
    return streams_tif, streams_vec

def fix_streams_crs(streams_vector: Path, outdir: Path, reference_raster: Path, to_epsg: int = 4326) -> tuple[Path, Path]:
    """
    Stamp vector CRS from reference raster (UTM), then also save a WGS84 GeoPackage.
    Returns (native_shp, wgs84_gpkg).
    """
    streams_vector = streams_vector.resolve()
    with rasterio.open(reference_raster) as src:
        src_crs = src.crs

    gdf = gpd.read_file(streams_vector)
    if gdf.crs is None:
        gdf = gdf.set_crs(src_crs, allow_override=True)

    shp_native = (outdir / "06_streams.shp").resolve()
    gdf.to_file(shp_native, driver="ESRI Shapefile")

    gdf_wgs84 = gdf.to_crs(epsg=to_epsg)
    gpkg_wgs84 = (outdir / "06_streams.gpkg").resolve()
    gdf_wgs84.to_file(gpkg_wgs84, driver="GPKG")

    print(f"Wrote Shapefile (native CRS): {shp_native}")
    print(f"Wrote GeoPackage (EPSG:{to_epsg}): {gpkg_wgs84}")
    return shp_native, gpkg_wgs84

# =============================== Plots (optional) =============================

def _geo_aspect(ymin: float, ymax: float) -> float:
    lat_mid = 0.5 * (ymin + ymax)
    c = np.cos(np.deg2rad(lat_mid))
    return 1.0 / max(c, 1e-6)

def quick_raster_png(dem_tif: Path, out_png: Path, cmap="terrain", title="DEM"):
    with rasterio.open(dem_tif) as src:
        z = src.read(1, masked=True)
    xmin, xmax, ymin, ymax, _ = read_extent(dem_tif)

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(np.isfinite(z), z, np.nan), vert_exag=1, dx=1, dy=1)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs, cmap="gray", origin="upper", extent=(xmin, xmax, ymin, ymax), zorder=1)
    ax.imshow(z,  cmap=cmap,  origin="upper", extent=(xmin, xmax, ymin, ymax), alpha=0.35, zorder=2)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(_geo_aspect(ymin, ymax), adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

def plot_streams_only(streams_gpkg: Path, dem_for_extent: Path, out_png: Path):
    xmin, xmax, ymin, ymax, _ = read_extent(dem_for_extent)
    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(ax=ax, linewidth=0.9, color="royalblue")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(_geo_aspect(ymin, ymax), adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Extracted Streams")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

def plot_streams_over_dem_matplotlib(dem_tif: Path, streams_gpkg: Path):
    with rasterio.open(dem_tif) as src:
        dem = src.read(1, masked=True)
    xmin, xmax, ymin, ymax, _ = read_extent(dem_tif)

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(np.isfinite(dem), dem, np.nan), vert_exag=1, dx=1, dy=1)

    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs,  cmap="gray",    origin="upper", extent=(xmin, xmax, ymin, ymax), zorder=1)
    ax.imshow(dem, cmap="terrain", origin="upper", extent=(xmin, xmax, ymin, ymax), alpha=0.35, zorder=2)
    gdf.plot(ax=ax, linewidth=1.0, color="royalblue", zorder=3)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(_geo_aspect(ymin, ymax), adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Streams over DEM (hillshade)")
    fig.tight_layout()
    return fig

def plot_streams_only_metric(streams_gpkg_wgs84: Path, utm_dem: Path, out_png: Path, utm_epsg: int = 32620):
    with rasterio.open(utm_dem) as src:
        west, south = src.transform * (0, src.height)
        east,  north = src.transform * (src.width, 0)
    gdf = gpd.read_file(streams_gpkg_wgs84).to_crs(utm_epsg)

    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(ax=ax, linewidth=0.9, color="royalblue")
    ax.set_xlim(min(west, east), max(west, east))
    ax.set_ylim(min(south, north), max(south, north))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.set_title("Extracted Streams (UTM)")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

def plot_streams_over_dem_metric(utm_dem_filled: Path, streams_gpkg_wgs84: Path, out_png: Path, utm_epsg: int = 32620):
    with rasterio.open(utm_dem_filled) as src:
        dem = src.read(1, masked=True)
        tr  = src.transform
        dx = abs(tr.a); dy = abs(tr.e)
        west, south = tr * (0, src.height)
        east,  north = tr * (src.width, 0)

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(np.isfinite(dem), dem, np.nan), vert_exag=1.0, dx=dx, dy=dy)

    gdf = gpd.read_file(streams_gpkg_wgs84).to_crs(utm_epsg)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs,  cmap="gray",    origin="upper", extent=[west, east, south, north], zorder=1)
    ax.imshow(dem, cmap="terrain", origin="upper", extent=[west, east, south, north], alpha=0.35, zorder=2)
    gdf.plot(ax=ax, linewidth=1.0, color="royalblue", zorder=3)
    ax.set_xlim(min(west, east), max(west, east))
    ax.set_ylim(min(south, north), max(south, north))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.set_title("Streams over DEM (UTM hillshade)")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

# ============================ Elevation at CSV creation ===========================

def _make_lonlat_nearest_sampler(dem_ll_path: Path) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[], None]]:
    ds = rasterio.open(dem_ll_path)
    if ds.crs and ds.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
        print(f"[warn] DEM CRS is {ds.crs}; expected lon/lat (EPSG:4326/CRS84).")
    band1 = ds.read(1)
    nod   = ds.nodata
    tr    = ds.transform
    h, w  = ds.height, ds.width

    def sampler(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        out = np.full(lon.shape, np.nan, float)
        cols, rows = (~tr) * (lon, lat)
        cols = np.round(cols).astype(int)
        rows = np.round(rows).astype(int)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if np.any(valid):
            vals = band1[rows[valid], cols[valid]].astype(float)
            if nod is not None:
                vals = np.where(vals == nod, np.nan, vals)
            out[valid] = vals
        return out

    def close():
        ds.close()

    return sampler, close

def write_top_n_channels_csv_with_elev(
    streams_gpkg: Path,
    dem_ll_path: Path,
    outdir: Path,
    top_n: int = 30,
    min_len_m: float = 200.0,
    lonlat_fmt: str = "%.8f",
):
    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf_m = gdf.to_crs(32620)
    gdf_m["len_m"] = gdf_m.geometry.length
    keep = gdf_m[gdf_m["len_m"] >= min_len_m].sort_values("len_m", ascending=False).head(top_n)
    if keep.empty:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        print("No channels met the min length; wrote 0 CSVs.")
        return

    keep_ll = keep.to_crs(4326)

    sampler, close_dem = _make_lonlat_nearest_sampler(dem_ll_path)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    n_wrote = 0
    try:
        for i, row in keep_ll.reset_index(drop=True).iterrows():
            geom = row.geometry
            if geom is None:
                continue

            if isinstance(geom, LineString):
                parts = [geom]
            elif isinstance(geom, MultiLineString):
                parts = list(geom.geoms)
            else:
                continue

            coords_list = []
            for ls in parts:
                arr = np.asarray(ls.coords, float)
                if arr.shape[0] >= 2:
                    coords_list.append(arr)
            if not coords_list:
                continue

            arr = np.vstack(coords_list)         # [[lon,lat], ...]
            lon, lat = arr[:, 0], arr[:, 1]
            elev = sampler(lon, lat)

            out = np.column_stack([lon, lat, elev])
            np.savetxt(
                outdir / f"channel_{i+1:02d}.csv",
                out,
                fmt=(lonlat_fmt, lonlat_fmt, "%.3f"),
                delimiter=",",
                header="lon,lat,elev_m",
                comments=""
            )
            n_wrote += 1
    finally:
        close_dem()

    print(f"Wrote {n_wrote} channel CSVs with elevation to {outdir}")

# ============================ Diagnostics (optional) ===========================

def print_dem_resolution_stats(dem_ll: Path | None, utm_dem: Path | None):
    if dem_ll and dem_ll.exists():
        with rasterio.open(dem_ll) as src:
            tr = src.transform
            dlon_deg = float(tr.a)
            dlat_deg = float(-tr.e)
            ymin, ymax = src.bounds.bottom, src.bounds.top
            lat_mid = 0.5 * (ymin + ymax)
            mdeg_lat = 111_194.9266
            mdeg_lon = mdeg_lat * np.cos(np.deg2rad(lat_mid))
            dx_m = abs(dlon_deg) * mdeg_lon
            dy_m = abs(dlat_deg) * mdeg_lat
        print(f"[DEM] Lon/Lat DEM pixel ~ {dx_m:.1f} m (x) × {dy_m:.1f} m (y) at lat {lat_mid:.3f}")
    if utm_dem and utm_dem.exists():
        with rasterio.open(utm_dem) as src:
            tr = src.transform
            dx_m = abs(tr.a); dy_m = abs(tr.e)
        print(f"[DEM] UTM DEM pixel   = {dx_m:.2f} m × {dy_m:.2f} m")

# ================================ CLI ==============================

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract drainage network from a DEM; attach elevation to channel CSVs at creation.")
    p.add_argument("--dem", type=Path, required=True, help="Input DEM GeoTIFF (lon/lat; EPSG:4326/CRS84)")
    p.add_argument("--outdir", type=Path, default=Path("wbt_out"))
    p.add_argument("--breach", action="store_true", help="Use breaching instead of FillDepressions")
    p.add_argument("--fa-percentile", type=float, default=97.5, help="Percentile for FA threshold")
    p.add_argument("--min-cells", type=int, default=800, help="Lower bound on flow-accum cells threshold")
    p.add_argument("--prep", action="store_true", help="Normalize DEM to WBT-friendly GTiff (predictor=1)")
    p.add_argument("--cutline", type=Path, help="Optional island polygon (SHP/GPKG/GeoJSON)")
    p.add_argument("--auto-cut", action="store_true", help="Derive island polygon from DEM and use as cutline")
    p.add_argument("--sea-level", type=float, default=0.0, help="Sea level threshold for auto-cut (meters)")
    p.add_argument("--min-area-km2", type=float, default=0.1, help="Min polygon area to keep (auto-cut)")
    p.add_argument("--simplify-deg", type=float, default=0.0, help="Simplify tolerance in degrees (auto-cut)")
    p.add_argument("--top-n", type=int, default=30, help="How many channels to export as CSV")
    p.add_argument("--min-len-m", type=float, default=200.0, help="Min channel length (m) for CSV export")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    p.add_argument("--no-nodegrid", action="store_true", help="Skip NodeGrid export (if flovopy is present)")
    return p.parse_args(argv)

# ================================ Main =============================

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    # Stage 0 — Preview original lon/lat DEM
    work_ll = args.dem.resolve()
    if not args.no_plots:
        quick_raster_png(work_ll, outdir / "00_original_dem.png", title="Original DEM")

    # Stage 1 — Optional normalization
    if args.prep:
        print("Normalizing DEM for WBT…")
        work_ll = normalize_for_wbt(work_ll, outdir / "01_dem_norm.tif")
        _print_path("Normalized DEM", work_ll)
        if not args.no_plots:
            quick_raster_png(work_ll, outdir / "01_dem_norm.png", title="DEM (normalized)")

    # Stage 2 — Optional / Auto island cutline
    cut_ll = None
    if args.cutline and args.cutline.exists():
        cut_ll = outdir / "02_dem_ll_land.tif"
        gdal_cut_to_polygon(work_ll, args.cutline, cut_ll, nodata=-32768.0, overwrite=True)
        work_ll = cut_ll
        _print_path("Cut to island polygon (user)", work_ll)
    elif args.auto_cut:
        gpkg = outdir / "02_island_from_dem.gpkg"
        derive_island_polygon_from_dem(
            work_ll, gpkg,
            sea_level=args.sea_level,
            min_area_km2=args.min_area_km2,
            simplify_tol_deg=args.simplify_deg,
        )
        cut_ll = outdir / "02_dem_ll_land.tif"
        gdal_cut_to_polygon(work_ll, gpkg, cut_ll, nodata=-32768.0, overwrite=True)
        work_ll = cut_ll
        _print_path("Cut to island polygon (auto)", work_ll)

    # Stage 3 — Reproject to UTM (fast)
    utm_dem = gdal_reproject_to_utm20(work_ll, outdir / "03_dem_utm.tif",
                                      nodata=-32768.0, res=10.0, overwrite=True)
    _print_path("DEM (UTM)", utm_dem)

    # Stage 4 — Hydro + FA + Pointer (UTM)
    filled, fa, pntr = hydrocondition_dem(utm_dem, outdir, breach=args.breach)
    _print_path("Filled DEM (UTM)", filled)

    # Stage 5 — Streams (UTM)
    thr = choose_threshold_from_fa(fa, percentile=args.fa_percentile, min_cells=args.min_cells)
    streams_raster, streams_vec = extract_streams(fa, pntr, thr, outdir)

    # Stage 6 — Tag CRS (UTM) and also write WGS84 GPKG
    streams_shp, streams_gpkg = fix_streams_crs(streams_vec, outdir, reference_raster=utm_dem)

    # Stage 7 — Plots (optional)
    if not args.no_plots:
        plot_streams_only(streams_gpkg, work_ll, outdir / "07_streams_only.png")
        fig = plot_streams_over_dem_matplotlib(work_ll, streams_gpkg)
        fig.savefig(outdir / "07_streams_over_dem.png", dpi=300); plt.close(fig)
        plot_streams_only_metric(streams_gpkg, utm_dem, outdir / "07_streams_only_metric.png")
        plot_streams_over_dem_metric(filled, streams_gpkg, outdir / "07_streams_over_dem_metric.png")

    # Stage 8 — CSVs in lon/lat WITH elevation
    write_top_n_channels_csv_with_elev(
        streams_gpkg=streams_gpkg,
        dem_ll_path=work_ll,  # sample in same CRS as CSV coords
        outdir=outdir / "channels_csv",
        top_n=args.top_n,
        min_len_m=args.min_len_m,
    )

    # Optional NodeGrid
    if _HAVE_ASL and not args.no_nodegrid:
        try:
            gdf = gpd.read_file(streams_gpkg).to_crs(4326)
            all_lon, all_lat = [], []
            for geom in gdf.geometry:
                if geom is None:
                    continue
                if isinstance(geom, LineString):
                    parts = [geom]
                elif isinstance(geom, MultiLineString):
                    parts = list(geom.geoms)
                else:
                    continue
                for ls in parts:
                    arr = np.asarray(ls.coords, float)
                    if arr.shape[0] >= 2:
                        all_lon.append(arr[:, 0])
                        all_lat.append(arr[:, 1])
            if all_lon:
                lon = np.concatenate(all_lon); lat = np.concatenate(all_lat)
                sampler, close_dem = _make_lonlat_nearest_sampler(work_ll)
                try:
                    elev = sampler(lon, lat)
                finally:
                    close_dem()
                nodegrid = NodeGrid(lon, lat, node_elev_m=elev, approx_spacing_m=None, dem_tag=os.path.basename(work_ll))
                nodegrid.save(outdir)
                print(f"[NG] NodeGrid ID: {nodegrid.id}  nodes: {lon.size}  elev: yes")
        except Exception as e:
            print(f"[warn] NodeGrid build skipped: {e}")

    print("✅ Drainage extraction finished.")
    print("Check outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()