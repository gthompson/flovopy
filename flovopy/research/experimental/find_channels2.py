# flovopy/asl/find_channels
#!/usr/bin/env python3
# The aim of this program is to find channels based on channel width. But it found nothing when i tested it on Fuego
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Callable, Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio import features as rio_features
from rasterio.transform import rowcol
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

# Optional: SciPy for EDT and percentile filter
_HAVE_SCIPY = True
try:
    from scipy.ndimage import distance_transform_edt, percentile_filter
except Exception:
    _HAVE_SCIPY = False

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
    zone = int(np.floor((lon + 180) / 6) + 1)
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone

def _write_geotiff_like(like_tif: Path, out_tif: Path, data: np.ndarray, nodata: float = np.nan, dtype: str = "float32"):
    with rasterio.open(like_tif) as src:
        prof = src.profile.copy()
    prof.update(driver="GTiff", dtype=dtype, count=1, compress="LZW", predictor=1, nodata=nodata)
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(data.astype(dtype), 1)
    print("Saved:", out_tif)
    return out_tif

# ============================= DEM I/O helpers =============================

def normalize_for_wbt(dem_in: Path, dem_out: Path, nodata_val: float = -32768.0) -> Path:
    dem_in = dem_in.resolve(); dem_out = dem_out.resolve()
    with rasterio.open(dem_in) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            arr = np.where(arr == nod, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, nodata_val).astype("float32")
        prof = src.profile.copy()
        prof.update(driver="GTiff", dtype="float32", count=1, nodata=nodata_val,
                    tiled=False, compress="LZW", predictor=1, interleave="pixel")
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

def _safe_unlink(p: Path):
    try:
        Path(p).unlink()
    except FileNotFoundError:
        pass

def gdal_reproject_to_utm(src_tif: Path, dst_tif: Path, nodata: float = -32768.0,
                          res: float | None = None, overwrite: bool = False,
                          utm_epsg: int | None = None) -> Path:
    src_tif = Path(src_tif); dst_tif = Path(dst_tif)
    res_part = "" if res is None else f"-tr {res} {res} "
    ow_part  = "-overwrite " if overwrite else ""
    if dst_tif.exists() and overwrite:
        _safe_unlink(dst_tif)
    if dst_tif.exists() and not overwrite:
        print(f"[GDAL] Exists, skipping: {dst_tif}")
        return dst_tif
    epsg = utm_epsg or 32620
    cmd = (
        f'gdalwarp {ow_part}-t_srs EPSG:{epsg} {res_part}-dstnodata {nodata} '
        f'-r bilinear -multi -wo NUM_THREADS=ALL_CPUS '
        f'-co COMPRESS=LZW -co PREDICTOR=1 "{src_tif}" "{dst_tif}"'
    )
    _run(cmd)
    return dst_tif

def gdal_cut_to_polygon(src_tif: Path, polygon: Path, dst_tif: Path, nodata: float = -32768.0,
                        overwrite: bool = False) -> Path:
    src_tif = Path(src_tif); dst_tif = Path(dst_tif)
    ow_part = "-overwrite " if overwrite else ""
    if dst_tif.exists() and overwrite:
        _safe_unlink(dst_tif)
    if dst_tif.exists() and not overwrite:
        print(f"[GDAL] Exists, skipping: {dst_tif}")
        return dst_tif
    cmd = (
        f'gdalwarp {ow_part}-cutline "{polygon}" -crop_to_cutline -dstnodata {nodata} '
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
        raise RuntimeError("No land polygon extracted from DEM.")
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
    if simplify_tol_deg and simplify_tol_deg > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tol_deg, preserve_topology=True)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG", layer=layer_name)
    print(f"[CUTLINE] Derived land polygons → {out_gpkg} (layer={layer_name})")
    return out_gpkg

# ======================= Whitebox pipeline + metrics ========================

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
    wbt = WhiteboxTools()
    wbt.set_working_dir(str(outdir.resolve()))
    wbt.verbose = True
    dem_in = dem_in.resolve()
    filled = (outdir / "04_dem_filled.tif").resolve()
    fa     = (outdir / "04_dem_fa.tif").resolve()
    pntr   = (outdir / "04_dem_d8pntr.tif").resolve()
    try:
        if breach:
            wbt_run_abs(wbt, "BreachDepressionsLeastCost", {"--dem": dem_in, "--output": filled}, expect=filled)
        else:
            wbt_run_abs(wbt, "FillDepressions", {"--dem": dem_in, "--output": filled, "--fix_flats": None}, expect=filled)
        if not _exists_nonempty(filled):
            raise RuntimeError("Filled DEM not written")
    except Exception as e:
        print(f"[warn] Primary depression removal failed: {e}")
        try:
            wbt_run_abs(wbt, "BreachDepressionsLeastCost", {"--dem": dem_in, "--output": filled}, expect=filled)
        except Exception as e2:
            print(f"[warn] BreachDepressionsLeastCost also failed: {e2}")
            tmp_fill1 = (outdir / "dem_fill1.tif").resolve()
            wbt_run_abs(wbt, "FillSingleCellPits", {"--dem": dem_in, "--output": tmp_fill1}, expect=tmp_fill1)
            wbt_run_abs(wbt, "BreachDepressionsLeastCost", {"--dem": tmp_fill1, "--output": filled}, expect=filled)
    if not _exists_nonempty(filled):
        raise RuntimeError("Hydroconditioning failed: no filled/breached DEM produced")
    wbt_run_abs(wbt, "D8FlowAccumulation", {"--dem": filled, "--output": fa, "--out_type": "cells"}, expect=fa)
    wbt_run_abs(wbt, "D8Pointer", {"--dem": filled, "--output": pntr}, expect=pntr)
    return filled, fa, pntr

def choose_threshold_from_fa(fa_tif: Path, percentile: float = 97.5, min_cells: int = 800) -> int:
    with rasterio.open(fa_tif) as src:
        A = src.read(1, masked=True)
        q = float(np.percentile(A.compressed(), percentile))
        fa_min, fa_max = float(A.min()), float(A.max())
    thr = int(max(min_cells, q))
    print(f"FlowAccum stats: min={fa_min:.1f} max={fa_max:.1f} q{percentile:.0f}={q:.1f} → threshold={thr}")
    return thr

import shutil, os
from pathlib import Path

def _whitebox_exe_path() -> str:
    """
    Return absolute path to the Whitebox CLI.
    Prefer PATH-provided 'whitebox'; fall back to 'whitebox_tools' if present.
    """
    for name in ("whitebox", "whitebox_tools"):
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError(
        "Could not find Whitebox CLI. Install it (e.g., `conda install -c conda-forge whitebox` "
        "or `whitebox-tools`) or add it to PATH."
    )

def _wb_cli(cmd_args: list[str]) -> None:
    exe = _whitebox_exe_path()
    cmd = " ".join([f'"{exe}"'] + cmd_args)
    print("[Whitebox CLI]", cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Whitebox CLI failed (rc={rc})")

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

from shutil import which

def _wb_exe_name() -> str:
    # Prefer the Rust CLI if present; otherwise fall back to the conda 'whitebox' runner
    return "whitebox_tools" if which("whitebox_tools") else "whitebox"

def stream_order_wbt(pntr_tif: Path, streams_raster: Path, outdir: Path) -> Path:
    outdir = Path(outdir).resolve()
    order_r = (outdir / "05_stream_order.tif").resolve()

    # 1) Python wrapper (exact tool name)
    try:
        wbt = WhiteboxTools()
        wbt.set_working_dir(str(outdir))
        wbt.verbose = True
        ok = wbt.run_tool("StrahlerStreamOrder", [
            f'--d8_pntr={pntr_tif}',
            f'--streams={streams_raster}',
            f'--output={order_r}',
        ])
        if ok and order_r.exists():
            return order_r
        else:
            raise RuntimeError("wrapper call returned False or output missing")
    except Exception as e:
        print(f"[warn] wrapper StrahlerStreamOrder failed: {e}")

    # helper: run CLI
    def _wb_cli_run(tool: str, out_path: Path) -> bool:
        cmd = (
            f'whitebox_tools -r={tool} '
            f'--wd="{outdir}" '
            f'--d8_pntr="{pntr_tif}" '
            f'--streams="{streams_raster}" '
            f'--output="{out_path}" -v --compress_rasters=False'
        )
        print("[WhiteboxTools]", cmd)
        rc = os.system(cmd)
        return rc == 0 and out_path.exists() and os.path.getsize(out_path) > 0

    # 2) CLI: Strahler
    if _wb_cli_run("StrahlerStreamOrder", order_r):
        return order_r

    # 3) CLI fallbacks (different definitions of order)
    for alt_tool, alt_name in [
        ("HortonStreamOrder", "horton"),
        ("TopologicalStreamOrder", "topological"),
    ]:
        alt_out = outdir / f"05_stream_order_{alt_name}.tif"
        if _wb_cli_run(alt_tool, alt_out):
            print(f"[info] Used {alt_tool} as fallback.")
            return alt_out

    raise RuntimeError("All stream-order attempts failed (Strahler/Horton/Topological).")

# ======================= Width (EDT) & Depth (HAND / relief) =======================

def _slope_deg(dem_utm_tif: Path) -> np.ndarray:
    with rasterio.open(dem_utm_tif) as src:
        z = src.read(1, masked=True).astype("float64")
        tr = src.transform
        dx, dy = abs(tr.a), abs(tr.e)
    gy, gx = np.gradient(z.filled(np.nan), dy, dx)
    slp = np.rad2deg(np.arctan(np.hypot(gx, gy)))
    return np.ma.array(slp, mask=~np.isfinite(z))

def _rasterize_lines(lines_path: Path, like_tif: Path, burn=1) -> np.ndarray:
    with rasterio.open(like_tif) as src:
        shape, transform = (src.height, src.width), src.transform
        crs = src.crs
    gdf = gpd.read_file(lines_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return rio_features.rasterize(
        ((geom, burn) for geom in gdf.geometry if geom is not None),
        out_shape=shape, transform=transform, fill=0, dtype="uint8"
    )

def width_raster_from_mask(fa_tif: Path, dem_filled_utm: Path,
                           streams_vec: Path, fa_thr_cells: int,
                           slope_thr_deg: float, outdir: Path) -> Path:
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for EDT-based width (pip/conda install scipy).")
    with rasterio.open(fa_tif) as src:
        fa = src.read(1, masked=True)
        tr = src.transform
    slp = _slope_deg(dem_filled_utm)
    ch_mask = (fa.filled(0) >= fa_thr_cells) & (slp.filled(90) <= slope_thr_deg)
    edt = distance_transform_edt(ch_mask, sampling=(abs(tr.e), abs(tr.a)))
    centerline = _rasterize_lines(streams_vec, fa_tif, burn=1)
    width_m = np.where(centerline == 1, 2.0 * edt, np.nan).astype("float32")
    out = _write_geotiff_like(fa_tif, outdir / "06_width_m.tif", width_m, nodata=np.nan, dtype="float32")
    return out

def hand_whitebox(dem_filled_utm: Path, pntr_tif: Path, streams_raster: Path, outdir: Path) -> Path:
    wbt = WhiteboxTools()
    wbt.set_working_dir(str(outdir.resolve())); wbt.verbose = True
    hand_tif = (outdir / "06_hand.tif").resolve()
    try:
        wbt_run_abs(wbt, "ElevationAboveStream", {
            "--dem": dem_filled_utm,
            "--d8_pntr": pntr_tif,
            "--streams": streams_raster,
            "--output": hand_tif
        }, expect=hand_tif)
        return hand_tif
    except Exception as e:
        print(f"[warn] HAND failed: {e}")
        raise

def local_relief_tif(dem_filled_utm: Path, outdir: Path, radius_px: int = 3) -> Path:
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for percentile_filter local-relief.")
    out = (outdir / "06_relief_p90.tif").resolve()
    with rasterio.open(dem_filled_utm) as src:
        z = src.read(1, masked=True).astype("float32")
        prof = src.profile.copy()
    size = radius_px * 2 + 1
    zf = z.filled(np.nan)
    # fill NaNs with global mean to avoid NaN propagation in percentile filter
    zfill = np.where(np.isfinite(zf), zf, np.nanmean(zf))
    p90 = percentile_filter(zfill, percentile=90, size=size, mode='nearest')
    relief = np.where(np.isfinite(zf), p90 - zfill, np.nan).astype("float32")
    prof.update(dtype="float32", nodata=np.nan, compress="LZW", predictor=1)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(relief, 1)
    print("Saved:", out)
    return out

# ======================= Per-segment metrics & filtering =======================

def summarize_stream_metrics(streams_vec: Path, like_tif_utm: Path,
                             width_tif: Path, depth_tif: Path, order_tif: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(streams_vec)
    with rasterio.open(like_tif_utm) as src_like:
        tr = src_like.transform
        res_m = abs(tr.a)  # assume square pixels
    with rasterio.open(width_tif) as wds, \
         rasterio.open(depth_tif) as dds, \
         rasterio.open(order_tif) as ods:
        W_med, D_med, O_max, L_m = [], [], [], []
        for geom in gdf.geometry:
            if geom is None or geom.length <= 0:
                W_med.append(np.nan); D_med.append(np.nan); O_max.append(np.nan); L_m.append(0.0)
                continue
            n = max(2, int(geom.length / res_m))
            xs = np.linspace(0, 1, n)
            pts = [geom.interpolate(float(d), normalized=True) for d in xs]
            coords = [(p.x, p.y) for p in pts]
            # sample rasters
            w = np.array([v[0] for v in wds.sample(coords)])
            d = np.array([v[0] for v in dds.sample(coords)])
            o = np.array([v[0] for v in ods.sample(coords)])
            W_med.append(np.nanmedian(np.where(w > 0, w, np.nan)))
            D_med.append(np.nanmedian(np.where(d >= 0, d, np.nan)))
            O_max.append(np.nanmax(o))
            L_m.append(geom.length)
    gdf["len_m"]    = L_m
    gdf["order"]    = O_max
    gdf["width_m"]  = W_med
    gdf["depth_m"]  = D_med
    gdf["xs_proxy"] = gdf["width_m"] * gdf["depth_m"]
    return gdf

def filter_and_save(gdf: gpd.GeoDataFrame, outdir: Path,
                    min_order=3, min_width_m=90.0, xs_quantile=0.8) -> Path:
    q = float(gdf["xs_proxy"].quantile(xs_quantile)) if np.isfinite(gdf["xs_proxy"]).any() else -np.inf
    sel = gdf[
        (gdf["order"] >= float(min_order)) &
        (gdf["width_m"] >= float(min_width_m)) &
        (gdf["xs_proxy"] >= q)
    ].copy()
    gpkg = outdir / "streams_filtered.gpkg"
    sel.to_file(gpkg, driver="GPKG")
    print(f"Filtered streams: kept {len(sel)}/{len(gdf)} → {gpkg}")
    return gpkg

# =============================== Plots (optional) =============================

def _sorted_bounds(tif: Path) -> tuple[float, float, float, float]:
    with rasterio.open(tif) as src:
        b = src.bounds
        xmin, xmax = (b.left, b.right) if b.left < b.right else (b.right, b.left)
        ymin, ymax = (b.bottom, b.top) if b.bottom < b.top else (b.top, b.bottom)
    return xmin, xmax, ymin, ymax

def _geo_aspect(ymin: float, ymax: float) -> float:
    lat_mid = 0.5 * (ymin + ymax)
    c = np.cos(np.deg2rad(lat_mid))
    return 1.0 / max(c, 1e-6)

def quick_raster_png(dem_tif: Path, out_png: Path, cmap="terrain", title="DEM"):
    with rasterio.open(dem_tif) as src:
        z  = src.read(1, masked=True)
        tr = src.transform
        dx, dy = abs(tr.a), abs(tr.e)
    xmin, xmax, ymin, ymax = _sorted_bounds(dem_tif)
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(np.isfinite(z), z, np.nan), vert_exag=1, dx=dx, dy=dy)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs, cmap="gray", origin="lower", extent=(xmin, xmax, ymin, ymax), zorder=1)
    ax.imshow(z,  cmap=cmap,  origin="lower", extent=(xmin, xmax, ymin, ymax), alpha=0.35, zorder=2)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(_geo_aspect(ymin, ymax), adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)

def plot_streams_only(streams_gpkg: Path, dem_for_extent: Path, out_png: Path):
    xmin, xmax, ymin, ymax = _sorted_bounds(dem_for_extent)
    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(ax=ax, linewidth=0.9, color="royalblue")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(_geo_aspect(ymin, ymax), adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Extracted Streams")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)

def plot_streams_over_dem_matplotlib(dem_tif: Path, streams_gpkg: Path):
    with rasterio.open(dem_tif) as src:
        dem = src.read(1, masked=True)
        tr  = src.transform
        dx, dy = abs(tr.a), abs(tr.e)
    xmin, xmax, ymin, ymax = _sorted_bounds(dem_tif)
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(np.isfinite(dem), dem, np.nan), vert_exag=1, dx=dx, dy=dy)
    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(hs,  cmap="gray",    origin="lower", extent=(xmin, xmax, ymin, ymax), zorder=1)
    ax.imshow(dem, cmap="terrain", origin="lower", extent=(xmin, xmax, ymin, ymax), alpha=0.35, zorder=2)
    gdf.plot(ax=ax, linewidth=1.0, color="royalblue", zorder=3)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect(_geo_aspect(ymin, ymax), adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Streams over DEM (hillshade)")
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
        dem = src.read(1, masked=True); tr = src.transform
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
    band1 = ds.read(1); nod = ds.nodata; tr = ds.transform
    h, w  = ds.height, ds.width
    def sampler(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, float); lat = np.asarray(lat, float)
        out = np.full(lon.shape, np.nan, float)
        cols, rows = (~tr) * (lon, lat)
        cols = np.round(cols).astype(int); rows = np.round(rows).astype(int)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if np.any(valid):
            vals = band1[rows[valid], cols[valid]].astype(float)
            if nod is not None:
                vals = np.where(vals == nod, np.nan, vals)
            out[valid] = vals
        return out
    def close(): ds.close()
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
            arr = np.vstack(coords_list)
            lon, lat = arr[:, 0], arr[:, 1]
            elev = sampler(lon, lat)
            out = np.column_stack([lon, lat, elev])
            np.savetxt(
                outdir / f"channel_{i+1:02d}.csv",
                out, fmt=(lonlat_fmt, lonlat_fmt, "%.3f"),
                delimiter=",", header="lon,lat,elev_m", comments=""
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
            dlon_deg = float(tr.a); dlat_deg = float(-tr.e)
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
    if not _HAVE_PYGMT:
        raise RuntimeError("PyGMT and rioxarray are required for DEM download (pip install pygmt rioxarray).")
    lon0, lon1, lat0, lat1 = [float(x) for x in region]
    grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=[lon0, lon1, lat0, lat1])
    grid = grid.rio.write_crs("EPSG:4326", inplace=False)
    out_geotiff.parent.mkdir(parents=True, exist_ok=True)
    grid.rio.to_raster(out_geotiff, compress="LZW")
    print(f"[PyGMT] Saved Earth Relief {resolution} GeoTIFF → {out_geotiff}")
    return out_geotiff

# ============================ Resample lines to points ===========================

def _resample_lines_to_points(gdf_ll: gpd.GeoDataFrame, spacing_m: float = 20.0, utm_epsg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if gdf_ll.crs is None or gdf_ll.crs.to_epsg() != 4326:
        gdf_ll = gdf_ll.to_crs(4326)
    if utm_epsg is None:
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
                lon_all.append(xy[:, 0]); lat_all.append(xy[:, 1])
    if lon_all:
        lon = np.concatenate(lon_all); lat = np.concatenate(lat_all)
        keep = ~(np.isnan(lon) | np.isnan(lat))
        lon, lat = lon[keep], lat[keep]
        uniq = np.unique(np.round(np.column_stack([lon, lat]), 8), axis=0)
        return uniq[:, 0], uniq[:, 1]
    else:
        return np.array([]), np.array([])
    
from pathlib import Path
import geopandas as gpd
import rasterio as rio

def fix_streams_crs(
    streams_vec_path: Path,
    outdir: Path,
    reference_raster: Path,
    ll_layer_name: str = "streams"
) -> tuple[Path, Path]:
    """
    Ensure streams vector has a CRS (UTM from reference_raster) and produce:
      - UTM shapefile (for metric calculations)
      - WGS84 GPKG (for mapping/exports)

    Returns
    -------
    (streams_shp_utm, streams_gpkg_ll)
    """
    streams_vec_path = Path(streams_vec_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read vector
    gdf = gpd.read_file(streams_vec_path)

    # Get CRS from the reference raster (UTM expected)
    with rio.open(reference_raster) as src:
        utm_crs = src.crs
        if utm_crs is None:
            raise RuntimeError(f"[CRS] Reference raster has no CRS: {reference_raster}")

    # If vector has no CRS, assign UTM from raster; if it has CRS, just ensure it’s UTM
    if gdf.crs is None:
        gdf = gdf.set_crs(utm_crs)
    elif gdf.crs != utm_crs:
        try:
            gdf = gdf.to_crs(utm_crs)
        except Exception as e:
            raise RuntimeError(f"[CRS] Could not reproject streams to UTM: {e}")

    # Write UTM shapefile for metric operations
    streams_shp_utm = outdir / "05_streams_utm.shp"
    gdf.to_file(streams_shp_utm, driver="ESRI Shapefile")

    # Also write a WGS84 copy for mapping/exports
    gdf_ll = gdf.to_crs(4326)
    streams_gpkg_ll = outdir / "05_streams_ll.gpkg"
    gdf_ll.to_file(streams_gpkg_ll, driver="GPKG", layer=ll_layer_name)

    print(f"[CRS] Streams UTM → {streams_shp_utm}")
    print(f"[CRS] Streams WGS84 → {streams_gpkg_ll}")
    return streams_shp_utm, streams_gpkg_ll

# ================================ CLI ==============================

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract drainage network from a DEM; can fetch DEM via PyGMT Earth Relief for a given region. Adds width (EDT), HAND/local-relief depth, and filtering.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dem", type=Path, help="Input DEM GeoTIFF (lon/lat; EPSG:4326/CRS84)")
    src.add_argument("--region", nargs=4, type=float, metavar=("LON0","LON1","LAT0","LAT1"),
                     help="Lon/Lat region for PyGMT Earth Relief, e.g. --region -63.3 -62.1 16.4 17.1")
    p.add_argument("--earth-relief", default="30s", help="PyGMT Earth Relief resolution (1s, 3s, 15s, 30s, 1m, ...). Default: 30s")
    p.add_argument("--outdir", type=Path, default=Path("wbt_out"))
    p.add_argument("--utm-res-m", type=float, default=10.0, help="Target UTM pixel size (meters) for gdalwarp")
    p.add_argument("--prep", action="store_true", help="Normalize DEM to WBT-friendly GTiff")
    p.add_argument("--cutline", type=Path, help="Optional polygon (SHP/GPKG/GeoJSON) to crop DEM")
    p.add_argument("--auto-cut", action="store_true", help="Derive land polygon from DEM and use as cutline")
    p.add_argument("--sea-level", type=float, default=0.0, help="Sea level threshold for auto-cut (meters)")
    p.add_argument("--min-area-km2", type=float, default=0.1, help="Min polygon area to keep (auto-cut)")
    p.add_argument("--simplify-deg", type=float, default=0.0, help="Simplify tolerance in degrees (auto-cut)")
    # Hydrology & extraction
    p.add_argument("--breach", action="store_true", help="Use breaching instead of FillDepressions")
    p.add_argument("--fa-percentile", type=float, default=97.5, help="Percentile for FA threshold")
    p.add_argument("--min-cells", type=int, default=800, help="Lower bound on FA cells threshold")
    # Width/Depth metrics & filtering
    p.add_argument("--width-slope-deg", type=float, default=15.0, help="Max slope (deg) allowed in channel mask for width EDT")
    p.add_argument("--no-hand", action="store_true", help="Skip HAND and use local-relief depth proxy")
    p.add_argument("--relief-radius-px", type=int, default=4, help="Local-relief radius (pixels) if HAND is disabled/fails")
    p.add_argument("--min-order", type=int, default=3, help="Minimum Strahler order to keep")
    p.add_argument("--min-width-m", type=float, default=90.0, help="Minimum median width (m) to keep")
    p.add_argument("--xs-quantile", type=float, default=0.80, help="Retain segments with xs_proxy ≥ this quantile")
    p.add_argument("--keep-unfiltered", action="store_true", help="Also keep unfiltered streams GPKG")
    # CSV/NodeGrid/plots
    p.add_argument("--top-n", type=int, default=30, help="How many channels to export as CSV")
    p.add_argument("--min-len-m", type=float, default=200.0, help="Min channel length (m) for CSV export")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    p.add_argument("--no-nodegrid", action="store_true", help="Skip NodeGrid export (if flovopy is present)")
    return p.parse_args(argv)

# ================================ Main =============================

def main(argv: List[str] | None = None) -> None:
    print('The aim of this program is to find channels based on channel width. But it found nothing when i tested it on Fuego. So still use find_channels.py')
    args = parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    # Stage 0 — Acquire/set DEM (lon/lat EPSG:4326)
    if args.dem:
        work_ll = args.dem.resolve()
    else:
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

    # Stage 2 — Optional cutline
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

    # Stage 5 — Streams (UTM raster + vector)
    thr = choose_threshold_from_fa(fa, percentile=args.fa_percentile, min_cells=args.min_cells)
    streams_raster, streams_vec_raw = extract_streams(fa, pntr, thr, outdir)

    # Stage 6 — Tag CRS (UTM) → produce BOTH UTM .shp and WGS84 .gpkg
    #           (We will use the UTM shapefile for width/depth/metrics.)
    streams_shp, streams_gpkg = fix_streams_crs(streams_vec_raw, outdir, reference_raster=utm_dem)

    # Stage 7 — Stream order (uses raster streams)
    order_r = stream_order_wbt(pntr, streams_raster, outdir)

    # Stage 8 — Width (EDT at centerline) — uses **vector** (UTM shapefile)
    width_tif = width_raster_from_mask(
        fa_tif=fa,
        dem_filled_utm=filled,
        streams_vec=streams_shp,             # ✅ correct: vector shapefile in UTM
        fa_thr_cells=thr,
        slope_thr_deg=args.width_slope_deg,
        outdir=outdir,
    )

    # Stage 9 — Depth proxy: HAND (best) or local relief
    # HAND needs the raster streams + pointer (Whitebox), so we pass the raster.
    if not args.no_hand:
        try:
            depth_tif = hand_whitebox(
                dem_filled_utm=filled,
                pntr_tif=pntr,
                streams_raster=streams_raster,
                outdir=outdir,
            )
        except Exception:
            print("[info] Falling back to local-relief depth proxy.")
            depth_tif = local_relief_tif(filled, outdir, radius_px=args.relief_radius_px)
    else:
        depth_tif = local_relief_tif(filled, outdir, radius_px=args.relief_radius_px)

    # Stage 10 — Metrics & filtering (use UTM shapefile for geometry)
    gdf_metrics = summarize_stream_metrics(
        streams_vec=streams_shp,           # UTM shapefile path
        like_tif_utm=utm_dem,
        width_tif=width_tif,
        depth_tif=depth_tif,
        order_tif=order_r,
    )
    gpkg_metrics = outdir / "06_streams_metrics.gpkg"
    gdf_metrics.to_file(gpkg_metrics, driver="GPKG")
    print("Saved metrics:", gpkg_metrics)

    streams_filtered_gpkg = filter_and_save(
        gdf_metrics, outdir,
        min_order=args.min_order,
        min_width_m=args.min_width_m,
        xs_quantile=args.xs_quantile,
    )

    # Optionally keep the original unfiltered GPKG (already written as streams_gpkg)
    if args.keep_unfiltered:
        _ = streams_gpkg  # no-op, kept on disk

    # Stage 11 — Plots (optional)
    if not args.no_plots:
        # lon/lat plots from original DEM bounds (use filtered WGS84 GPKG)
        plot_streams_only(streams_filtered_gpkg, work_ll, outdir / "07_streams_only.png")
        fig = plot_streams_over_dem_matplotlib(work_ll, streams_filtered_gpkg)
        fig.savefig(outdir / "07_streams_over_dem.png", dpi=300); plt.close(fig)
        # metric (UTM) plots
        plot_streams_only_metric(streams_filtered_gpkg, utm_dem, outdir / "07_streams_only_metric.png", utm_epsg=utm_epsg)
        plot_streams_over_dem_metric(filled, streams_filtered_gpkg, outdir / "07_streams_over_dem_metric.png", utm_epsg=utm_epsg)

    # Stage 12 — CSVs in lon/lat WITH elevation (filtered only)
    write_top_n_channels_csv_with_elev(
        streams_gpkg=streams_filtered_gpkg,
        dem_ll_path=work_ll,
        outdir=outdir / "channels_csv",
        top_n=args.top_n,
        min_len_m=args.min_len_m,
        utm_epsg=utm_epsg,
    )

    # Stage 13 — Optional: NodeGrid from filtered stream points
    if _HAVE_ASL and not args.no_nodegrid:
        try:
            gdf_ll = gpd.read_file(streams_filtered_gpkg)
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
                    lon, lat, node_elev_m=elev, approx_spacing_m=spacing_m,
                    dem_tag=os.path.basename(work_ll),
                )
                nodegrid.save(outdir)
                print(f"[NG] NodeGrid ID: {nodegrid.id}  nodes={lon.size}  spacing≈{spacing_m} m")
            else:
                print("[NG] No resampled points produced; NodeGrid not written.")
        except Exception as e:
            print(f"[warn] NodeGrid build skipped: {e}")

    print("✅ Drainage extraction + metrics + filtering finished.")
    print("Check outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()

"""
Examples
1) Local DEM:
   python find_channels.py --dem dem.tif --outdir out --utm-res-m 20

2) Via PyGMT (Earth Relief 1s ~ 30 m):
   python find_channels.py --region -90.98 -90.78 14.365 14.49 --earth-relief 01s --outdir out

Big-channel filter knobs:
   --min-order 3 --min-width-m 90 --xs-quantile 0.8

Force local relief instead of HAND:
   --no-hand --relief-radius-px 4
"""