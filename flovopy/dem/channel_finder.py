#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling as RioResampling, calculate_default_transform
from whitebox import WhiteboxTools

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# flip with 'labels-only' option (transform_only=True)
from flovopy.dem.flip_geotiff import flip_geotiff


# ============================= Helpers =============================

def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _exists_nonempty(p: Path) -> bool:
    return p.exists() and os.path.getsize(p) > 0

def _print_path(label: str, p: Path | str):
    p = Path(p).resolve()
    print(f"{label}: {p}  exists={p.exists()}  size={os.path.getsize(p) if p.exists() else 'NA'}")

def read_extent(tif: Path) -> tuple[float, float, float, float, rasterio.crs.CRS | None]:
    """Return (west, east, south, north, crs) from the file's bounds."""
    with rasterio.open(tif) as src:
        b = src.bounds
        return float(b.left), float(b.right), float(b.bottom), float(b.top), src.crs

def normalize_for_wbt(dem_in: Path, dem_out: Path, nodata_val: float = -32768.0) -> Path:
    """
    Make a WBT-friendly GeoTIFF without changing georeferencing:
      - single band float32; replace NaNs with fixed nodata
      - LZW + predictor=1 (avoid Predictor=3); not tiled
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

def reproject_to_utm(src_tif: Path, dst_tif: Path, dst_epsg: int = 32620, dst_res: float | None = None) -> Path:
    """
    Reproject lon/lat DEM to a projected CRS (default UTM 20N).
    If dst_res is None, rasterio chooses an appropriate resolution.
    Writes float32, LZW predictor=1, no tiling.
    """
    src_tif = src_tif.resolve(); dst_tif = dst_tif.resolve()
    with rasterio.open(src_tif) as src:
        src_crs = src.crs
        dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg)
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_res
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            nodata=src.nodata if src.nodata is not None else -32768.0,
            count=1,
            driver="GTiff",
            tiled=False,
            compress="LZW",
            predictor=1,
            interleave="pixel",
        )
        data = src.read(1).astype("float32")
        nod = profile["nodata"]
        data = np.where(np.isfinite(data), data, nod).astype("float32")

    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_tif, "w", **profile) as dst:
        out = np.empty((height, width), dtype="float32")
        reproject(
            data, out,
            src_transform=src.transform, src_crs=src_crs,
            dst_transform=transform,     dst_crs=dst_crs,
            resampling=RioResampling.bilinear,
            src_nodata=profile["nodata"], dst_nodata=profile["nodata"],
        )
        dst.write(out, 1)
    return dst_tif


# ======================= Whitebox pipeline ========================

def wbt_run_abs(wbt: WhiteboxTools, tool: str, args_paths: dict, expect: Path | None = None) -> None:
    """Build WBT argv with absolute paths and run."""
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
    filled = (outdir / "dem_filled.tif").resolve()
    fa     = (outdir / "dem_fa.tif").resolve()
    pntr   = (outdir / "dem_d8pntr.tif").resolve()

    # Breach vs fill with fallbacks
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

    # Flow accumulation + pointer
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

    streams_tif = (outdir / "streams.tif").resolve()
    streams_vec = (outdir / "streams.shp").resolve()

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

    shp_native = (outdir / "streams.shp").resolve()
    gdf.to_file(shp_native, driver="ESRI Shapefile")

    gdf_wgs84 = gdf.to_crs(epsg=to_epsg)
    gpkg_wgs84 = (outdir / "streams.gpkg").resolve()
    gdf_wgs84.to_file(gpkg_wgs84, driver="GPKG")

    print(f"Wrote Shapefile (native CRS): {shp_native}")
    print(f"Wrote GeoPackage (EPSG:{to_epsg}): {gpkg_wgs84}")
    return shp_native, gpkg_wgs84


# =============================== Plots =============================

def _geo_aspect(ymin: float, ymax: float) -> float:
    """Matplotlib aspect so that ~1 km in x ≈ 1 km in y on a lon/lat plot."""
    lat_mid = 0.5 * (ymin + ymax)
    c = np.cos(np.deg2rad(lat_mid))
    return 1.0 / max(c, 1e-6)  # y/x data ratio

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

def plot_streams_over_dem_matplotlib(dem_tif: Path, streams_gpkg: Path, out_png: Path):
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
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

def plot_streams_only_metric(streams_gpkg_wgs84: Path, utm_dem: Path, out_png: Path, utm_epsg: int = 32620):
    """Streams-only plot in meters (UTM) → true aspect (1 km == 1 km)."""
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
    """Hillshade + streams in UTM so distances are correct."""
    with rasterio.open(utm_dem_filled) as src:
        dem = src.read(1, masked=True)
        tr  = src.transform
        dx = abs(tr.a); dy = abs(tr.e)  # pixel size meters
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


# ============================ CSV export ===========================

def write_top_n_channels_csv(streams_gpkg: Path, outdir: Path, top_n: int = 30, min_len_m: float = 200.0):
    gdf = gpd.read_file(streams_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    gdf_m = gdf.to_crs(32620)  # Montserrat ~ UTM 20N
    gdf_m["len_m"] = gdf_m.geometry.length
    keep = gdf_m[gdf_m["len_m"] >= min_len_m].sort_values("len_m", ascending=False).head(top_n)

    outdir = Path(outdir); outdir.mkdir(exist_ok=True, parents=True)
    if keep.empty:
        print("No channels met the min length; wrote 0 CSVs.")
        return

    keep_ll = keep.to_crs(4326)
    for i, row in keep_ll.reset_index(drop=True).iterrows():
        coords = list(row.geometry.coords)
        arr = np.asarray(coords, dtype=float)  # [[lon, lat], ...]
        np.savetxt(outdir / f"channel_{i+1:02d}.csv",
                   arr, fmt="%.8f", delimiter=",", header="lon,lat", comments="")
    print(f"Wrote {len(keep_ll)} channel CSVs to {outdir}")


# ================================ CLI ==============================

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract drainage network from a DEM (WBT in UTM; outputs & plots in lon/lat)")
    p.add_argument("--dem", type=Path, required=True, help="Input DEM GeoTIFF (lon/lat)")
    p.add_argument("--outdir", type=Path, default=Path("wbt_out"))
    p.add_argument("--breach", action="store_true", help="Use breaching instead of FillDepressions")
    p.add_argument("--fa-percentile", type=float, default=97.5, help="Percentile for FA threshold")
    p.add_argument("--min-cells", type=int, default=800, help="Lower bound on flow-accum cells threshold")
    p.add_argument("--prep", action="store_true", help="Normalize DEM to WBT-friendly GTiff (predictor=1)")
    p.add_argument("--top-n", type=int, default=30, help="How many main channels to export as CSV")
    p.add_argument("--min-len-m", type=float, default=200.0, help="Min channel length (m) for CSV export")
    p.add_argument("--flip", choices=["horizontal", "vertical", "both"],
                   help="Flip the DEM before processing (fix georeferencing issues)")
    return p.parse_args(argv)


# ================================ Main =============================

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    outdir = ensure_outdir(args.outdir)

    # Stage 0 — Preview original lon/lat DEM
    quick_raster_png(args.dem, outdir / "00_original_dem.png", title="Original DEM")

    # Stage 1 — Optional normalization (safer I/O)
    work_ll = args.dem.resolve()
    if args.prep:
        print("Normalizing DEM for WBT…")
        work_ll = normalize_for_wbt(work_ll, outdir / "dem_norm.tif")
        _print_path("Normalized DEM", work_ll)
        quick_raster_png(work_ll, outdir / "01_dem_norm.png", title="DEM (normalized)")

    # Stage 2 — Optional flip (labels-only if needed to fix E/W)
    if args.flip:
        flipped = outdir / f"dem_flipped_{args.flip}.tif"
        # IMPORTANT: transform_only=True => flip the "labels", not the pixels
        flip_geotiff(work_ll, flipped, mode=args.flip, transform_only=True)
        work_ll = flipped
        _print_path(f"DEM (flipped {args.flip})", work_ll)
        quick_raster_png(work_ll, outdir / "03_flipped_dem.png", title="Flipped DEM")

    # Stage 3 — Reproject to UTM for hydrology
    utm_dem = reproject_to_utm(work_ll, outdir / "dem_utm.tif", dst_epsg=32620, dst_res=None)
    _print_path("DEM (UTM)", utm_dem)

    # Stage 4 — Hydro + FA + Pointer (UTM)
    filled, fa, pntr = hydrocondition_dem(utm_dem, outdir, breach=args.breach)
    _print_path("Filled DEM (UTM)", filled)

    # Stage 5 — Streams (UTM)
    thr = choose_threshold_from_fa(fa, percentile=args.fa_percentile, min_cells=args.min_cells)
    streams_raster, streams_vec = extract_streams(fa, pntr, thr, outdir)

    # Stage 6 — Tag CRS (UTM) and also write WGS84 GPKG
    streams_shp, streams_gpkg = fix_streams_crs(streams_vec, outdir, reference_raster=utm_dem)

    # Stage 7 — Plots (lon/lat + UTM)
    plot_streams_only(streams_gpkg, work_ll, outdir / "08_streams_only.png")
    plot_streams_over_dem_matplotlib(work_ll, streams_gpkg, outdir / "08_streams_over_dem.png")
    plot_streams_only_metric(streams_gpkg, utm_dem, outdir / "08_streams_only_metric.png")
    plot_streams_over_dem_metric(filled, streams_gpkg, outdir / "08_streams_over_dem_metric.png")

    # Stage 8 — CSVs in lon/lat for ASL
    write_top_n_channels_csv(streams_gpkg, outdir / "channels_csv", top_n=args.top_n, min_len_m=args.min_len_m)

    print("✅ Drainage extraction finished.")
    print("Check outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
