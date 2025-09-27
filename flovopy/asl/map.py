# flovopy/asl/map.py
from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import pygmt
from obspy import Inventory

# correct helper import (do NOT import from this module again)
from flovopy.stationmetadata.utils import inventory2traceid

# optional, used if user passes xarray grids (kept for type compatibility)
import xarray as xr  # noqa: F401

# Prefer Oceania mirror (your existing config)
pygmt.config(GMT_DATA_SERVER="https://oceania.generic-mapping-tools.org")


# ----------------------------
# small helpers
# ----------------------------
def _finite(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except Exception:
        return False


def _finite_arrays(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.array([]), np.array([])
    return x[m], y[m]


def _validate_region(region: List[float]) -> List[float]:
    if region is None:
        return None
    r = list(map(float, region))
    if not all(np.isfinite(r)):
        raise ValueError(f"Region has non-finite values: {region}")
    # Ensure [minlon, maxlon, minlat, maxlat] ordering
    if r[0] > r[1]:
        r[0], r[1] = r[1], r[0]
    if r[2] > r[3]:
        r[2], r[3] = r[3], r[2]
    return r


# ----------------------------
# Basemap
# ----------------------------
def topo_map(
    show: bool = False,
    zoom_level: int = 0,
    inv: Optional[Inventory] = None,
    add_labels: bool = False,
    centerlon: float = -62.177,
    centerlat: float = 16.721,
    contour_interval: float = 100,
    topo_color: bool = True,
    resolution: str = "01s",
    DEM_DIR: Optional[str] = None,
    stations: Optional[List[str]] = None,
    title: Optional[str] = None,
    region: Optional[List[float]] = None,
    levels: Optional[List[float]] = None,
    level_interval: Optional[float] = None,
    cmap: Optional[str] = None,
    projection: Optional[str] = None,
    azimuth: float = 135,
    elevation: float = 30,
    limit: Optional[float] = None,
    figsize: float = 6.0,
    *,
    frame: bool = True,
    return_region: bool = False,
    dem_tif: Optional[str] = None,  # custom GeoTIFF overrides Earth Relief
):
    """
    Create a topographic basemap using PyGMT with optional station overlays.
    Supports PyGMT Earth Relief and custom GeoTIFF (dem_tif).
    """

    # --- Region selection ---
    if region is None:
        if dem_tif:
            try:
                import rasterio
                with rasterio.open(dem_tif) as src:
                    b = src.bounds
                    region = [float(b.left), float(b.right), float(b.bottom), float(b.top)]
                centerlon = 0.5 * (region[0] + region[1])
                centerlat = 0.5 * (region[2] + region[3])
            except Exception as e:
                raise RuntimeError(f"Failed to read GeoTIFF bounds for region: {e}")
        else:
            # Montserrat defaults
            island_center = (-62.195, 16.75)
            volcano_center = (centerlon, centerlat)
            if zoom_level <= 0:
                span_lon_deg = 0.12
                span_lat_deg = 0.18
                c_lon, c_lat = island_center
            else:
                c_lon, c_lat = volcano_center
                base_span_lon = 0.18
                base_span_lat = 0.14
                shrink = 1.7 ** (zoom_level - 1)
                span_lon_deg = base_span_lon / shrink
                span_lat_deg = base_span_lat / shrink
            half_lon = span_lon_deg / 2.0
            half_lat = span_lat_deg / 2.0
            region = [c_lon - half_lon, c_lon + half_lon, c_lat - half_lat, c_lat + half_lat]
    else:
        centerlon = (region[0] + region[1]) / 2.0
        centerlat = (region[2] + region[3]) / 2.0

    region = _validate_region(region)

    # --- Build figure ---
    if projection is None:
        projection = f"M{figsize}i"
    fig = pygmt.Figure()

    # Decide grid source and optional cache (Earth Relief only)
    grid_src = None
    ergrid = None
    shade = None

    if dem_tif:
        # GeoTIFF path: GMT can read directly
        grid_src = dem_tif
        try:
            shade = pygmt.grdgradient(grid=grid_src, radiance=[azimuth, elevation], normalize="t1")
        except Exception:
            shade = None
    else:
        # Earth Relief with cache
        cache_root = os.environ.get("FLOVOPY_CACHE")
        if DEM_DIR is None and cache_root:
            DEM_DIR = os.path.join(cache_root, "dem")
        pklfile = None
        if DEM_DIR:
            os.makedirs(DEM_DIR, exist_ok=True)
            pklfile = os.path.join(
                DEM_DIR,
                f"EarthRelief_{resolution}_{region[0]:.4f}_{region[1]:.4f}_{region[2]:.4f}_{region[3]:.4f}.pkl"
            )

        if pklfile and os.path.exists(pklfile):
            try:
                with open(pklfile, "rb") as f:
                    ergrid = pickle.load(f)
            except Exception:
                ergrid = None

        if ergrid is None:
            ergrid = pygmt.datasets.load_earth_relief(
                resolution=resolution, region=region, registration=None
            )
            if pklfile:
                try:
                    with open(pklfile, "wb") as f:
                        pickle.dump(ergrid, f)
                except Exception:
                    pass

        grid_src = ergrid
        try:
            shade = pygmt.grdgradient(grid=ergrid, radiance=[azimuth, elevation], normalize="t1")
        except Exception:
            shade = None

    # Special 2-color land/ocean mode
    paint_water = (cmap == "gray") or (cmap is None and not topo_color)
    if cmap in ("land", "green"):
        import tempfile
        cptfile = tempfile.NamedTemporaryFile(suffix=".cpt", delete=False).name
        with open(cptfile, "w") as f:
            f.write("# Sea to land CPT\n")
            f.write("-8000  135/206/250   0   135/206/250\n")
            f.write("0       152/251/152   8000 152/251/152\n")
        cmap = cptfile
        shade = None
        paint_water = False

    # Base imagery
    fig.grdimage(
        grid=grid_src,
        region=region,
        projection=projection,
        cmap=cmap,
        shading=shade,
        frame=frame,
    )

    # Coastlines & (optional) water paint
    coast_kwargs = dict(
        region=region,
        projection=projection,
        shorelines="1/0.5p,black",
        frame=(["WSen", "af"] if frame else False),
    )
    if paint_water:
        coast_kwargs["water"] = "lightblue"
    fig.coast(**coast_kwargs)

    # Contours — use `levels=` (not `interval=`) to avoid deprecation
    try:
        if levels is not None:
            fig.grdcontour(grid=grid_src, levels=levels, pen="0.25p,black", limit=limit)
        else:
            step = level_interval if level_interval is not None else contour_interval
            fig.grdcontour(grid=grid_src, levels=step, pen="0.25p,black", limit=limit)
    except Exception:
        # Gracefully skip contours if GMT can't process the raster
        pass

    # Colorbar for explicit "topo" palette
    if cmap and cmap == "topo":
        fig.colorbar(frame='+l"Topography (m)"')

    # Stations
    if inv is not None:
        try:
            seed_ids = inventory2traceid(inv)
            # station list filter (by station code)
            if not stations:
                stations = [sid.split(".")[1] for sid in seed_ids]

            stalon = []
            stalat = []
            for sid in seed_ids:
                try:
                    c = inv.get_coordinates(sid)
                    lo = c.get("longitude", np.nan)
                    la = c.get("latitude", np.nan)
                    if _finite(lo) and _finite(la):
                        stalon.append(float(lo))
                        stalat.append(float(la))
                except Exception:
                    continue

            X, Y = _finite_arrays(np.array(stalon), np.array(stalat))
            if X.size:
                if add_labels:
                    # Plot points and labels
                    fig.plot(x=X, y=Y, style="s0.5c", fill="white", pen="black")
                    # Add station codes near points
                    for sid, x0, y0 in zip(seed_ids, X, Y):
                        sta = sid.split(".")[1]
                        highlight = (sta in stations)
                        fig.text(
                            x=x0, y=y0, text=sta,
                            font=("12p,white" if highlight else "10p,black"),
                            justify="ML", offset="0.2c/0c",
                        )
                else:
                    fig.plot(x=X, y=Y, style="s0.4c", fill="black", pen="white")
        except Exception as e:
            print(f"[MAP:WARN] Station plotting failed: {e}")

    # Title
    if title and frame:
        fig.text(
            text=title,
            x=region[0] + 0.60 * (region[1] - region[0]),
            y=region[2] + 0.90 * (region[3] - region[2]),
            justify="TC",
            font="11p,Helvetica-Bold,black",
        )

    if frame:
        fig.basemap(region=region, frame=True)

    if show:
        fig.show()

    return (fig, region) if return_region else fig


# ----------------------------
# Heatmap
# ----------------------------
def plot_heatmap_montserrat_colored(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    amp_col: str = "amplitude",
    zoom_level: int = 0,
    inventory: Optional[Inventory] = None,
    color_scale: float = 0.4,  # kept for API compatibility
    cmap: str = "turbo",
    log_scale: bool = True,
    node_spacing_m: int = 50,
    outfile: Optional[str] = None,
    region: Optional[List[float]] = None,
    title: Optional[str] = None,
    dem_tif: Optional[str] = None,
):
    """
    Render a colored heatmap of energy (sum amplitude^2) on a topo basemap.
    """
    if df is None or df.empty:
        raise ValueError("plot_heatmap_montserrat_colored: input DataFrame is empty")

    df = df.copy()
    df["energy"] = df[amp_col] ** 2
    grouped = df.groupby([lat_col, lon_col])["energy"].sum().reset_index()

    x = grouped[lon_col].to_numpy(float)
    y = grouped[lat_col].to_numpy(float)
    z = grouped["energy"].to_numpy(float)
    if log_scale:
        z = np.log10(z + 1e-12)

    # Drop non-finite before range calc/plot
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        raise ValueError("plot_heatmap_montserrat_colored: no finite (lon,lat,energy) to plot")

    # color palette
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        raise ValueError("plot_heatmap_montserrat_colored: invalid z-range after filtering")
    pygmt.makecpt(
        cmap=cmap,
        series=[zmin, zmax, (zmax - zmin) / 100.0 if zmax > zmin else 0.01],
        continuous=True,
    )

    fig = topo_map(
        zoom_level=zoom_level,
        inv=inventory,
        topo_color=False,
        region=region,
        title=title,
        dem_tif=dem_tif,
    )

    # approximate symbol size in cm
    symbol_size_cm = node_spacing_m * 0.077 / 50.0
    fig.plot(x=x, y=y, style=f"s{symbol_size_cm}c", fill=z, cmap=True, pen=None)

    fig.colorbar(frame='+l"Log10 Total Energy"' if log_scale else '+l"Total Energy"')

    if region:
        fig.basemap(region=_validate_region(region), frame=True)

    if outfile:
        fig.savefig(outfile)
    else:
        fig.show()
    return fig