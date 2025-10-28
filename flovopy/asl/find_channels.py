# flovopy/asl/find_channels
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

# Optional: PyGMT + rioxarray for DEM download/export
_HAVE_PYGMT = True
try:
    import pygmt
    import rioxarray  # noqa: F401 (xarray accessor)
except Exception:
    _HAVE_PYGMT = False

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

def utm_epsg_for(lon: float, lat: float) -> int:
    """
    Compute UTM EPSG for a lon/lat point. Northern hemisphere: 326##, Southern: 327##
    """
    zone = int(np.floor((lon + 180) / 6) + 1)
    if zone < 1: zone = 1
    if zone > 60: zone = 60
    return (32600 if lat >= 0 else 32700) + zone

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

def gdal_reproject_to_utm(src_tif: Path, dst_tif: Path, utm_epsg: int,
                          nodata: float = -32768.0, res: float | None = None,
                          overwrite: bool = False) -> Path:
    """
    Reproject to computed UTM zone using gdalwarp.
    Optional ‘res’ sets target pixel size (meters). If None, gdal chooses.
    """
    src_tif = Path(src_tif); dst_tif = Path(dst_tif)
    if dst_tif.exists() and not overwrite:
        print(f"[GDAL] Exists, skipping: {dst_tif}")
        return dst_tif
    res_part = "" if res is None else f"-tr {res} {res} "
    cmd = (
        f'gdalwarp -t_srs EPSG:{utm_epsg} {res_part}-dstnodata {nodata} '
        f'-r bilinear -multi -wo NUM_THREADS=ALL_CPUS '
        f'-co COMPRESS=LZW -co PREDICTOR=1 "{src_tif}" "{dst_tif}"'
    )
    _run(cmd)
    return dst_tif

def gdal_cut_to_polygon(src_tif: Path, polygon: Path, dst_tif: Path, nodata: float = -32768.0,
                        overwrite: bool = False) -> Path:
    """
    Cut DEM to polygon to remove water/outside region. polygon can be Shapefile/GeoJSON/GPKG.
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

# ======================= Auto-derive land polygon (optional) =======================

def derive_land_polygon_from_dem(
    dem_ll: Path,
    out_gpkg: Path,
    sea_level: float = 0.0,
    min_area_km2: float = 0.1,
    simplify_tol_deg: float = 0.0,
    layer_name: str = "land",
) -> Path:
    """
    Create a land polygon from DEM:
      - Land mask = valid pixels AND (elev > sea_level) if sea_level is not None,
        else just valid pixels (non-nodata).
      - Extract polygons, keep those above area threshold.
      - Optionally simplify and save to GPKG.
    """
    dem_ll = Path(dem_ll).resolve()
    out_gpkg = Path(out_gpkg).resolve()

    with rasterio.open(dem_ll) as src:
        Z = src.read(1, masked=True)
        tr = src.transform
        crs = src.crs

    valid = np.isfinite(Z)
    land = valid if sea_level is None else (valid & (Z > sea_level))

    shp_gen = rio_shapes(land.astype(np.uint8), mask=None, transform=tr)
    polys: list[Polygon] = []
    for geom, val in shp_gen:
        if int(val) == 1:
            g = shp_shape(geom)
            if isinstance(g, (Polygon, MultiPolygon)):
                polys.append(g if isinstance(g, Polygon) else unary_union(g))

    if not polys:
        raise RuntimeError("No land polygon extracted from DEM (check sea_level/nodata).")

    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        geoms = [merged]
    elif isinstance(merged, MultiPolygon):
        geoms = list(merged.geoms)
    else:
        geoms = [Polygon(merged)]

    def _km2_of_polygon(p: Polygon) -> float:
        lon, lat = p.representative_point().x, p.representative_point().y
        m_per_deg_lat = 111_194.9266
        m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(lat))
        return float(p.area) * (m_per_deg_lon * m_per_deg_lat) / 1e6

    geoms = [g for g in geoms if _km2_of_polygon(g) >= min_area_km2]
    if not geoms:
        raise RuntimeError("No polygon passed the minimum area filter.")

    gdf = gpd.GeoDataFrame({"id": list(range(len(geoms)))}, geometry=geoms, crs=crs)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if simplify_tol_deg and simplify_tol_deg > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tol_deg, preserve_topology=True)
    gdf.to_file(out_gpkg, driver="GPKG", layer=layer_name)
    print(f"[CUTLINE] Derived land polygons → {out_gpkg} (layer={layer_name})")
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

def plot_streams_only_metric(streams_gpkg_wgs84: Path, utm_dem: Path, out_png: Path, utm_epsg: int):
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

def plot_streams_over_dem_metric(utm_dem_filled: Path, streams_gpkg_wgs84: Path, out_png: Path, utm_epsg: int):
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
    utm_epsg: int | None = None,
):
    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if utm_epsg is None:
        xmin, xmax, ymin, ymax, _ = read_extent(dem_ll_path)
        utm_epsg = utm_epsg_for((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    gdf_m = gdf.to_crs(utm_epsg)
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

# ============================ PyGMT fetch (optional) ============================

def download_dem_with_pygmt(region: list[float], resolution: str, out_geotiff: Path) -> Path:
    """
    Use PyGMT to fetch Earth Relief (SRTM-derived) and write a GeoTIFF (EPSG:4326).
    region = [lon0, lon1, lat0, lat1]
    resolution examples: '1s' (~30m), '3s' (~90m), '15s', '30s', '1m', '5m', '15m'...
    """
    if not _HAVE_PYGMT:
        raise RuntimeError("PyGMT and rioxarray are required for DEM download (pip install pygmt rioxarray).")
    lon0, lon1, lat0, lat1 = [float(x) for x in region]
    grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=[lon0, lon1, lat0, lat1])
    # Ensure CRS and write GeoTIFF
    grid = grid.rio.write_crs("EPSG:4326", inplace=False)
    out_geotiff.parent.mkdir(parents=True, exist_ok=True)
    grid.rio.to_raster(out_geotiff, compress="LZW")
    print(f"[PyGMT] Saved Earth Relief {resolution} GeoTIFF → {out_geotiff}")
    return out_geotiff

# ============================ Resample lines to points ===========================

def _resample_lines_to_points(gdf_ll: gpd.GeoDataFrame, spacing_m: float = 20.0, utm_epsg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample each LineString/MultiLineString to points every `spacing_m` meters.
    gdf_ll must be in lon/lat; we reproject to UTM (auto if utm_epsg None),
    then bring the resampled points back to lon/lat.
    """
    if gdf_ll.crs is None or gdf_ll.crs.to_epsg() != 4326:
        gdf_ll = gdf_ll.to_crs(4326)
    if utm_epsg is None:
        xmin, xmax, ymin, ymax = *gdf_ll.total_bounds[0:2], *gdf_ll.total_bounds[2:4]  # not pretty but fine
        lonc = (gdf_ll.total_bounds[0] + gdf_ll.total_bounds[2]) / 2.0
        latc = (gdf_ll.total_bounds[1] + gdf_ll.total_bounds[3]) / 2.0
        utm_epsg = utm_epsg_for(lonc, latc)

    gdf_m = gdf_ll.to_crs(utm_epsg)

    lon_all, lat_all = [], []
    for geom in gdf_m.geometry:
        if geom is None:
            continue

        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        else:
            continue

        for ls in parts:
            L = ls.length
            if L <= 0:
                continue
            d = np.arange(0.0, L, spacing_m)
            if d.size == 0 or d[-1] < L:
                d = np.append(d, L)

            pts_m = [ls.interpolate(float(di)) for di in d]
            pts_ll = gpd.GeoSeries(pts_m, crs=utm_epsg).to_crs(4326)

            xy = np.array([(p.x, p.y) for p in pts_ll.geometry], dtype=float)
            if xy.size:
                lon_all.append(xy[:, 0])
                lat_all.append(xy[:, 1])

    if lon_all:
        lon = np.concatenate(lon_all)
        lat = np.concatenate(lat_all)
        keep = ~(np.isnan(lon) | np.isnan(lat))
        lon, lat = lon[keep], lat[keep]
        uniq = np.unique(np.round(np.column_stack([lon, lat]), 8), axis=0)
        return uniq[:, 0], uniq[:, 1]
    else:
        return np.array([]), np.array([])

# ================================ CLI ==============================

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract drainage network from a DEM; can fetch DEM via PyGMT Earth Relief for a given region.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dem", type=Path, help="Input DEM GeoTIFF (lon/lat; EPSG:4326/CRS84)")
    src.add_argument("--region", nargs=4, type=float, metavar=("LON0","LON1","LAT0","LAT1"),
                     help="Lon/Lat region to fetch DEM with PyGMT Earth Relief, e.g. --region -63.3 -62.1 16.4 17.1")
    p.add_argument("--earth-relief", default="30s", help="PyGMT Earth Relief resolution (1s, 3s, 15s, 30s, 1m, 5m, 15m...). Default: 30s")
    p.add_argument("--outdir", type=Path, default=Path("wbt_out"))
    p.add_argument("--breach", action="store_true", help="Use breaching instead of FillDepressions")
    p.add_argument("--fa-percentile", type=float, default=97.5, help="Percentile for FA threshold")
    p.add_argument("--min-cells", type=int, default=800, help="Lower bound on flow-accum cells threshold")
    p.add_argument("--prep", action="store_true", help="Normalize DEM to WBT-friendly GTiff (predictor=1)")
    p.add_argument("--cutline", type=Path, help="Optional polygon (SHP/GPKG/GeoJSON) to crop DEM")
    p.add_argument("--auto-cut", action="store_true", help="Derive land polygon from DEM and use as cutline")
    p.add_argument("--sea-level", type=float, default=0.0, help="Sea level threshold for auto-cut (meters)")
    p.add_argument("--min-area-km2", type=float, default=0.1, help="Min polygon area to keep (auto-cut)")
    p.add_argument("--simplify-deg", type=float, default=0.0, help="Simplify tolerance in degrees (auto-cut)")
    p.add_argument("--top-n", type=int, default=30, help="How many channels to export as CSV")
    p.add_argument("--min-len-m", type=float, default=200.0, help="Min channel length (m) for CSV export")
    p.add_argument("--utm-res-m", type=float, default=10.0, help="Target UTM pixel size (meters) for gdalwarp")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    p.add_argument("--no-nodegrid", action="store_true", help="Skip NodeGrid export (if flovopy is present)")
    return p.parse_args(argv)

# ================================ Main =============================

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    # Stage 0 — Acquire or set DEM (lon/lat, EPSG:4326)
    if args.dem:
        work_ll = args.dem.resolve()
    else:
        # Fetch via PyGMT
        fetch_tif = outdir / "00_earth_relief.tif"
        work_ll = download_dem_with_pygmt(
            region=[args.region[0], args.region[1], args.region[2], args.region[3]],
            resolution=args.earth_relief,
            out_geotiff=fetch_tif,
        )

    if not args.no_plots:
        quick_raster_png(work_ll, outdir / "00_original_dem.png", title="Original DEM")

    # Stage 1 — Optional normalization
    if args.prep:
        print("Normalizing DEM for WBT…")
        work_ll = normalize_for_wbt(work_ll, outdir / "01_dem_norm.tif")
        _print_path("Normalized DEM", work_ll)
        if not args.no_plots:
            quick_raster_png(work_ll, outdir / "01_dem_norm.png", title="DEM (normalized)")

    # Stage 2 — Optional / Auto cutline
    cut_ll = None
    if args.cutline and args.cutline.exists():
        cut_ll = outdir / "02_dem_ll_cut.tif"
        gdal_cut_to_polygon(work_ll, args.cutline, cut_ll, nodata=-32768.0, overwrite=True)
        work_ll = cut_ll
        _print_path("Cut to polygon (user)", work_ll)
    elif args.auto_cut:
        gpkg = outdir / "02_land_from_dem.gpkg"
        derive_land_polygon_from_dem(
            work_ll, gpkg,
            sea_level=args.sea_level,
            min_area_km2=args.min_area_km2,
            simplify_tol_deg=args.simplify_deg,
            layer_name="land",
        )
        cut_ll = outdir / "02_dem_ll_cut.tif"
        gdal_cut_to_polygon(work_ll, gpkg, cut_ll, nodata=-32768.0, overwrite=True)
        work_ll = cut_ll
        _print_path("Cut to polygon (auto)", work_ll)

    # Compute UTM EPSG from DEM center
    xmin, xmax, ymin, ymax, _ = read_extent(work_ll)
    utm_epsg = utm_epsg_for((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    print(f"[CRS] Using UTM EPSG:{utm_epsg}")

    # Stage 3 — Reproject to UTM
    utm_dem = gdal_reproject_to_utm(work_ll, outdir / "03_dem_utm.tif",
                                    utm_epsg=utm_epsg, nodata=-32768.0,
                                    res=args.utm_res_m, overwrite=True)
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
        plot_streams_only_metric(streams_gpkg, utm_dem, outdir / "07_streams_only_metric.png", utm_epsg=utm_epsg)
        plot_streams_over_dem_metric(filled, streams_gpkg, outdir / "07_streams_over_dem_metric.png", utm_epsg=utm_epsg)

    # Stage 8 — CSVs in lon/lat WITH elevation
    write_top_n_channels_csv_with_elev(
        streams_gpkg=streams_gpkg,
        dem_ll_path=work_ll,  # sample elevations on lon/lat DEM
        outdir=outdir / "channels_csv",
        top_n=args.top_n,
        min_len_m=args.min_len_m,
        utm_epsg=utm_epsg,
    )

    # Optional: NodeGrid from stream points
    if _HAVE_ASL and not args.no_nodegrid:
        try:
            gdf_ll = gpd.read_file(streams_gpkg)
            if gdf_ll.crs is None or gdf_ll.crs.to_epsg() != 4326:
                gdf_ll = gdf_ll.to_crs(4326)

            spacing_m = 20.0
            lon, lat = _resample_lines_to_points(gdf_ll, spacing_m=spacing_m, utm_epsg=utm_epsg)

            if lon.size:
                sampler, close_dem = _make_lonlat_nearest_sampler(work_ll)
                try:
                    elev = sampler(lon, lat)
                finally:
                    close_dem()

                nodegrid = NodeGrid(
                    lon,
                    lat,
                    node_elev_m=elev,
                    approx_spacing_m=spacing_m,
                    dem_tag=os.path.basename(work_ll),
                )
                nodegrid.save(outdir)
                print(f"[NG] NodeGrid ID: {nodegrid.id}  nodes={lon.size}  spacing≈{spacing_m} m")
            else:
                print("[NG] No resampled points produced; NodeGrid not written.")
        except Exception as e:
            print(f"[warn] NodeGrid build skipped: {e}")

    print("✅ Drainage extraction finished.")
    print("Check outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()

"""
1) From a local DEM file:

    python find_channels.py --dem dem.tif --outdir out --utm-res-m 20

2) Via PyGMT

    python find_channels.py --region -105.8 -105.2 39.4 39.9 --earth-relief 3s --outdir out
"""