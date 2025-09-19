"""
monty_dem_io.py — Lightweight DEM I/O and utilities for Montserrat work
these tools are Python equivalents of code in wadgeDEMs/matlab

Dependencies
------------
- rasterio (GDAL-backed)
- numpy
- matplotlib (optional; for quicklooks)

What you get
------------
- read_surfer_dsaa / write_surfer_dsaa: Surfer ASCII (.grd, DSAA) ⇄ NumPy
- read_sdts_dem: (optional) read SDTS DEMs if your GDAL build supports it
- write_xyz_from_grid: export lon/lat/elev triplets with optional stride
- save_geotiff: write a GeoTIFF from an array + bounds/CRS
- reproject_to_match: reproject/resample a source GeoTIFF onto a reference GeoTIFF grid
- contour_quicklook / raster_quicklook: simple matplotlib previews

All functions are type-hinted and documented. Designed to be imported in your
analysis scripts and notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
except Exception as e:  # pragma: no cover
    rasterio = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore

__all__ = [
    "SurferGrid",
    "read_surfer_dsaa",
    "write_surfer_dsaa",
    "read_sdts_dem",
    "save_geotiff",
    "raster_quicklook",
    "contour_quicklook",
    "write_xyz_from_grid",
    "reproject_to_match",
]


# ------------------------------
# Data classes
# ------------------------------

@dataclass
class SurferGrid:
    """Container for a Surfer ASCII (DSAA) grid and its extents.

    Attributes
    ----------
    Z : np.ndarray
        Array of shape (nrows, ncols) with elevation values (np.nan for NoData).
    x_min, x_max, y_min, y_max : float
        Grid extents in the grid's native coordinates (often metres or degrees).
    """

    Z: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.Z.shape


# ------------------------------
# Surfer ASCII (DSAA) I/O
# ------------------------------

def read_surfer_dsaa(path: Path | str) -> SurferGrid:
    """Read a Surfer ASCII (DSAA) .grd file into a SurferGrid.

    Parameters
    ----------
    path : Path | str
        Path to the DSAA file.

    Returns
    -------
    SurferGrid
        Grid array and extents. Surfer NoData sentinel (1.70141e+38) is mapped to np.nan.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
        if not header.startswith("DSAA"):
            raise ValueError(f"Not a DSAA Surfer file: {p}")
        ncols, nrows = map(int, f.readline().split())
        x_min, x_max = map(float, f.readline().split())
        y_min, y_max = map(float, f.readline().split())
        _zmin, _zmax = map(float, f.readline().split())
        data = np.fromfile(f, sep=" ", dtype=float, count=ncols * nrows)

    Z = data.reshape((nrows, ncols))
    nodata_sentinel = 1.70141e38
    Z[~np.isfinite(Z)] = np.nan
    Z[np.isclose(Z, nodata_sentinel)] = np.nan
    return SurferGrid(Z=Z.astype("float32"), x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def write_surfer_dsaa(path: Path | str, grid: SurferGrid) -> None:
    """Write a SurferGrid to Surfer ASCII (DSAA) format.

    Notes
    -----
    - np.nan values are written as Surfer's NoData sentinel (1.70141e+38).
    """
    p = Path(path)
    Z = grid.Z.astype(float)
    nrows, ncols = Z.shape
    finite = Z[np.isfinite(Z)]
    zmin = float(finite.min()) if finite.size else 0.0
    zmax = float(finite.max()) if finite.size else 0.0
    with p.open("w") as f:
        f.write("DSAA\n")
        f.write(f"{ncols} {nrows}\n")
        f.write(f"{grid.x_min} {grid.x_max}\n")
        f.write(f"{grid.y_min} {grid.y_max}\n")
        f.write(f"{zmin} {zmax}\n")
        nod = 1.70141e+38
        for r in range(nrows):
            row = Z[r, :]
            f.write(
                " ".join(f"{v:.6g}" if np.isfinite(v) else f"{nod:.5e}" for v in row) + "\n"
            )


# ------------------------------
# SDTS DEM reader (optional)
# ------------------------------

def read_sdts_dem(catd_path: Path | str):
    """Read an SDTS DEM given the Catalog Directory file (e.g., *_CATD.DDF).

    Returns
    -------
    Z : np.ndarray
        Elevation array with np.nan for nodata.
    transform : affine.Affine
        Affine geotransform.
    crs : rasterio.crs.CRS
        Coordinate reference system.

    Notes
    -----
    Requires rasterio/GDAL with SDTS driver support. If rasterio is not
    available, raises RuntimeError.
    """
    if rasterio is None:
        raise RuntimeError("rasterio/GDAL not available; cannot read SDTS")

    with rasterio.open(str(catd_path)) as src:
        Z = src.read(1).astype("float32")
        if src.nodata is not None:
            Z = np.where(Z == src.nodata, np.nan, Z)
        return Z, src.transform, src.crs


# ------------------------------
# GeoTIFF writer & previews
# ------------------------------

def save_geotiff(
    path: Path | str,
    Z: np.ndarray,
    *,
    crs: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    transform=None,
    nodata: float = -9999.0,
) -> None:
    """Save an array as a GeoTIFF.

    Provide either `bounds=(west, south, east, north)` **or** a rasterio
    `transform`. CRS string should be something like "EPSG:4326".
    """
    if rasterio is None:
        raise RuntimeError("rasterio/GDAL not available; cannot save GeoTIFF")

    if transform is None:
        if bounds is None:
            raise ValueError("Provide either bounds or transform")
        nrows, ncols = Z.shape
        west, south, east, north = bounds
        dx = (east - west) / ncols
        dy = (north - south) / nrows
        transform = from_origin(west, north, dx, dy)

    profile = {
        "driver": "GTiff",
        "height": Z.shape[0],
        "width": Z.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        data = np.where(np.isfinite(Z), Z, nodata).astype("float32")
        dst.write(data, 1)


def raster_quicklook(path: Path | str, out_png: Path | str, title: str = "") -> None:
    """Save a simple PNG quicklook of a GeoTIFF."
    if rasterio is None:
        raise RuntimeError("rasterio/GDAL not available")
    if plt is None:
        raise RuntimeError("matplotlib not available")
    with rasterio.open(str(path)) as src:
        arr = src.read(1, masked=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(arr)
        if title:
            ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Elevation")
        ax.axis("off")
        fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
        plt.close(fig)


def contour_quicklook(Z: np.ndarray, levels: Optional[Iterable[float]] = None, out_png: Path | str = "contours.png") -> None:
    """Contour plot from a 2D array."
    if plt is None:
        raise RuntimeError("matplotlib not available")
    import numpy as _np

    fig, ax = plt.subplots(figsize=(6, 6))
    lv = levels if levels is not None else _np.arange(_np.nanmin(Z), _np.nanmax(Z), 50)
    cs = ax.contour(Z, levels=lv)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.axis("equal"); ax.axis("off")
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# XYZ exporter
# ------------------------------

def write_xyz_from_grid(
    path: Path | str,
    lonvec: Sequence[float],
    latvec: Sequence[float],
    Z: np.ndarray,
    stride: int = 1,
) -> None:
    """Write an XYZ file (lon, lat, z) from gridded data.

    Parameters
    ----------
    stride : int
        Write every Nth sample (default 1 = full resolution). The ordering is
        row-major with lat varying slowest.
    """
    p = Path(path)
    with p.open("w") as f:
        for i in range(0, len(latvec), stride):
            for j in range(0, len(lonvec), stride):
                z = Z[i, j]
                if np.isfinite(z):
                    f.write(f"{lonvec[j]:.6f} {latvec[i]:.6f} {z:.3f}\n")


# ------------------------------
# Reproject & resample
# ------------------------------

def reproject_to_match(
    src_path: Path | str,
    ref_path: Path | str,
    out_path: Path | str,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Reproject/resample `src_path` so it matches the grid of `ref_path`.

    The output GeoTIFF will have the same CRS, transform, width/height as the
    reference raster.
    """
    if rasterio is None:
        raise RuntimeError("rasterio/GDAL not available")

    with rasterio.open(str(ref_path)) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_height, ref_width = ref.height, ref.width
        ref_profile = ref.profile.copy()

    with rasterio.open(str(src_path)) as src:
        data = np.full((ref_height, ref_width), ref_profile.get("nodata", -9999.0), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
        )

    ref_profile.update(dtype="float32")
    with rasterio.open(str(out_path), "w", **ref_profile) as dst:
        dst.write(data, 1)
