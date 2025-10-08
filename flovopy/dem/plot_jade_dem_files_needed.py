# flovopy/asl/map.py
# flovopy/asl/map.py

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple, List

import numpy as np
import pygmt
from obspy import Inventory

# correct helper import
from flovopy.stationmetadata.utils import inventory2traceid
from flovopy.core.mvo import dome_location  # ensure available

# optional, type-compat
import xarray as xr  # noqa: F401

# Prefer Oceania mirror
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
    # Ensure [minlon, maxlon, minlat, maxlat]
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
    inv: Optional["Inventory"] = None,
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
    cmap: Optional[str] = None,           # explicit CPT name or file; if None we choose based on topo_color
    projection: Optional[str] = None,
    hillshade_azimuth: float = 135,
    hillshade_altitude: float = 30,
    add_shading: bool = True,             # NEW: toggle hillshade
    limit: Optional[float] = None,
    figsize: float = 6.0,
    *,
    frame: bool = True,
    return_region: bool = False,
    dem_tif: Optional[str] = None,        # custom GeoTIFF overrides Earth Relief
    dome_location: Optional[dict] = None, # e.g. {"lat": 16.72, "lon": -62.18}
    outfile: Optional[str] = None,
    land_fill: Optional[str] = None,  # NEW: paint land uniformly (e.g. "152/251/152" or "200/200/200")
):
    import os, pickle, numpy as np, pygmt
    from flovopy.stationmetadata.utils import inventory2traceid
    # xarray import kept for type compatibility in callers
    import xarray as xr  # noqa: F401

    def _finite(v) -> bool:
        try:
            return v is not None and np.isfinite(float(v))
        except Exception:
            return False

    def _finite_arrays(x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        return (x[m], y[m]) if np.any(m) else (np.array([]), np.array([]))

    def _validate_region(r):
        if r is None:
            return None
        r = list(map(float, r))
        if not all(np.isfinite(r)):
            raise ValueError(f"Region has non-finite values: {r}")
        if r[0] > r[1]: r[0], r[1] = r[1], r[0]
        if r[2] > r[3]: r[2], r[3] = r[3], r[2]
        return r

    # --- Region selection ---
    # --- Region selection (always honor zoom_level unless region is explicit) ---
    if region is not None:
        # Caller explicitly set region: use it as-is.
        centerlon = (region[0] + region[1]) / 2.0
        centerlat = (region[2] + region[3]) / 2.0
    else:
        dem_bounds = None
        if dem_tif:
            try:
                import rasterio
                with rasterio.open(dem_tif) as _src:
                    b = _src.bounds  # lon/lat if your GeoTIFF is EPSG:4326
                dem_bounds = [float(b.left), float(b.right), float(b.bottom), float(b.top)]
                # If caller didn’t override center, default to DEM center
                if centerlon is None or centerlat is None:
                    centerlon = (dem_bounds[0] + dem_bounds[1]) / 2.0
                    centerlat = (dem_bounds[2] + dem_bounds[3]) / 2.0
            except Exception as e:
                raise RuntimeError(f"Failed to read GeoTIFF bounds for region: {e}")

        # Defaults if no DEM given
        if centerlon is None or centerlat is None:
            # island view by default
            centerlon, centerlat = (-62.195, 16.75)

        # Compute zoomed span (same semantics as before)
        if zoom_level <= 0:
            span_lon_deg, span_lat_deg = 0.12, 0.18
        else:
            base_span_lon, base_span_lat = 0.18, 0.14
            shrink = 1.7 ** (zoom_level - 1)
            span_lon_deg = base_span_lon / shrink
            span_lat_deg = base_span_lat / shrink

        half_lon, half_lat = span_lon_deg / 2.0, span_lat_deg / 2.0
        region = [centerlon - half_lon, centerlon + half_lon,
                centerlat - half_lat, centerlat + half_lat]

        # If a DEM is supplied, clip the zoom window to DEM bounds to avoid empty edges
        if dem_bounds is not None:
            region = [
                max(region[0], dem_bounds[0]),
                min(region[1], dem_bounds[1]),
                max(region[2], dem_bounds[2]),
                min(region[3], dem_bounds[3]),
            ]

    region = _validate_region(region)

    # --- Figure / projection ---
    if projection is None:
        projection = f"M{figsize}i"
    fig = pygmt.Figure()

    # --- Load raster + compute data range (for CPT) ---
    grid_src = None
    zmin = zmax = None
    shade = None

    if dem_tif:
        grid_src = dem_tif
        # Get min/max via rasterio (robust for GeoTIFF)
        try:
            import rasterio
            with rasterio.open(dem_tif) as src:
                # Read a small overview or compute from stats if available; fallback to full read
                stats = src.statistics(1) if hasattr(src, "statistics") else None
                if stats:
                    zmin, zmax = float(stats.min), float(stats.max)
                else:
                    arr = src.read(1, out_dtype="float32", masked=True)
                    if np.ma.isMaskedArray(arr):
                        data = np.asarray(arr.filled(np.nan))
                    else:
                        data = np.asarray(arr, float)
                    zmin = float(np.nanpercentile(data, 1))
                    zmax = float(np.nanpercentile(data, 99))
        except Exception:
            pass
        # Optional hillshade
        if add_shading:
            try:
                shade = pygmt.grdgradient(grid=grid_src,
                                          radiance=[hillshade_azimuth, hillshade_altitude],
                                          normalize="t1")
            except Exception:
                shade = None
    else:
        # Earth Relief + small cache
        cache_root = os.environ.get("FLOVOPY_CACHE")
        if DEM_DIR is None and cache_root:
            DEM_DIR = os.path.join(cache_root, "dem")
        pklfile = None
        if DEM_DIR:
            os.makedirs(DEM_DIR, exist_ok=True)
            pklfile = os.path.join(DEM_DIR, f"EarthRelief_{resolution}_{region[0]:.4f}_{region[1]:.4f}_{region[2]:.4f}_{region[3]:.4f}.pkl")
        ergrid = None
        if pklfile and os.path.exists(pklfile):
            try:
                with open(pklfile, "rb") as f:
                    ergrid = pickle.load(f)
            except Exception:
                ergrid = None
        if ergrid is None:
            ergrid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region, registration=None)
            if pklfile:
                try:
                    with open(pklfile, "wb") as f:
                        pickle.dump(ergrid, f)
                except Exception:
                    pass
        grid_src = ergrid
        # Data range from xarray DataArray
        try:
            zmin = float(np.nanpercentile(ergrid.values, 1))
            zmax = float(np.nanpercentile(ergrid.values, 99))
        except Exception:
            zmin, zmax = None, None
        if add_shading:
            try:
                shade = pygmt.grdgradient(grid=ergrid,
                                          radiance=[hillshade_azimuth, hillshade_altitude],
                                          normalize="t1")
            except Exception:
                shade = None

    '''
    # --- Choose/construct CPT ---
    # If user gives `cmap`, we honor it. Else pick based on topo_color.
    use_palette = cmap
    if use_palette is None:
        use_palette = "geo" if topo_color else "gray"

    # Special 2-color land/sea mode
    two_color_land = (use_palette in ("land", "green"))
    if two_color_land:
        import tempfile
        cptfile = tempfile.NamedTemporaryFile(suffix=".cpt", delete=False).name
        with open(cptfile, "w") as f:
            f.write("# Sea to land CPT\n")
            f.write("-8000  135/206/250   0   135/206/250\n")
            f.write("0       152/251/152   8000 152/251/152\n")
        use_palette = cptfile
        shade = None  # shading looks odd in flat 2-color mode

    # Build a CPT over the raster’s actual range when we can
    try:
        if (zmin is not None) and (zmax is not None) and np.isfinite(zmin) and np.isfinite(zmax) and (zmax > zmin):
            pygmt.makecpt(cmap=use_palette, series=[zmin, zmax], continuous=True)
            use_cmap_flag = True   # tell grdimage to use the current CPT
        else:
            # fallback to named palette (GMT will stretch)
            use_cmap_flag = use_palette
    except Exception:
        use_cmap_flag = use_palette

    # Paint water only in grayscale/no-topo mode (optional aesthetic)
    paint_water = (use_palette == "gray") or (cmap is None and not topo_color)
    '''

    # --- Choose/construct CPT (robust, no zero-thickness slices) ---
    # ---- Choose/construct CPT for the base DEM (unchanged behavior) ----
    use_palette = cmap if cmap is not None else ("geo" if topo_color else "gray")
    use_cmap_flag = None

    # z-range for continuous CPTs
    if (zmin is None) or (zmax is None) or not np.isfinite(zmin) or not np.isfinite(zmax):
        try:
            if dem_tif:
                import rasterio
                with rasterio.open(dem_tif) as _src:
                    arr = _src.read(1, masked=True).filled(np.nan)
                data = np.asarray(arr, float)
            else:
                data = np.asarray(grid_src.values, float)
            zmin = float(np.nanpercentile(data, 1))
            zmax = float(np.nanpercentile(data, 99))
        except Exception:
            zmin, zmax = -8000.0, 8000.0

    # Build a continuous CPT (works for "geo", "gray", or a named CPT)
    try:
        if np.isfinite(zmin) and np.isfinite(zmax) and (zmax > zmin):
            pygmt.makecpt(cmap=use_palette, series=[zmin, zmax], continuous=True)
            use_cmap_flag = True
        else:
            use_cmap_flag = use_palette
    except Exception:
        use_cmap_flag = use_palette

    # In "gray" mode we’ll paint water with fig.coast later
    paint_water = (use_palette == "gray") or (cmap is None and not topo_color)

    # --- Draw raster ---
    fig.grdimage(
        grid=grid_src,
        region=region,
        projection=projection,
        cmap=(True if use_cmap_flag is True else use_cmap_flag),  # True => current CPT; else palette name/file
        shading=shade,
        frame=frame,
    )

    # -------- OPTIONAL UNIFORM LAND OVERLAY (draw after base DEM, before coast) --------
    if land_fill:
        try:
            # Mask the sea to NaN so only land (z >= 0) is drawn
            land_only = pygmt.grdclip(grid=grid_src, below=[0, "NaN"])

            # Build a constant-color CPT for land
            import tempfile
            cpt_const = tempfile.NamedTemporaryFile(suffix=".cpt", delete=False).name
            with open(cpt_const, "w") as _f:
                # Any single slice works; GMT will fill land with this color
                _f.write(f"0 {land_fill} 1 {land_fill}\n")

            # Draw the uniform land on top (no shading)
            fig.grdimage(
                grid=land_only,
                region=region,
                projection=projection,
                cmap=cpt_const,
                shading=False,
                frame=False,  # frame already drawn
            )
        except Exception as _e:
            print(f"[MAP:WARN] Uniform land overlay failed: {_e}")

    # Coastlines (and optional water fill)
    coast_kwargs = dict(
        region=region,
        projection=projection,
        shorelines="1/0.5p,black",
        resolution='f',
        frame=(["WSen", "af"] if frame else False),
    )
    if paint_water:
        coast_kwargs["water"] = "lightblue"
    fig.coast(**coast_kwargs)

    # Contours
    try:
        if levels is not None:
            fig.grdcontour(grid=grid_src, levels=levels, pen="0.25p,black", limit=limit)
        else:
            step = level_interval if level_interval is not None else contour_interval
            fig.grdcontour(grid=grid_src, levels=step, pen="0.25p,black", limit=limit)
    except Exception:
        pass

    # Colorbar only when we’re using a continuous topo palette (not 2-color)
    if isinstance(use_cmap_flag, (str, bool)) and use_cmap_flag is not False:
        # Only add a label if the palette is meant as topography
        if (cmap is None and topo_color) or (cmap in ("geo", "topo")):
            fig.colorbar(frame='+l"Topography (m)"')

    # Stations
    if inv is not None:
        try:
            seed_ids = inventory2traceid(inv)
            if not stations:
                stations = [sid.split(".")[1] for sid in seed_ids]
            xs, ys = [], []
            for sid in seed_ids:
                try:
                    c = inv.get_coordinates(sid)
                    lo, la = c.get("longitude", np.nan), c.get("latitude", np.nan)
                    if _finite(lo) and _finite(la):
                        xs.append(float(lo)); ys.append(float(la))
                except Exception:
                    continue
            X, Y = _finite_arrays(np.array(xs), np.array(ys))
            if X.size:
                if add_labels:
                    fig.plot(x=X, y=Y, style="s0.5c", fill="white", pen="black")
                    for sid, x0, y0 in zip(seed_ids, X, Y):
                        sta = sid.split(".")[1]
                        highlight = (sta in stations)
                        fig.text(x=x0, y=y0, text=sta,
                                 font=("12p,white" if highlight else "10p,black"),
                                 justify="ML", offset="0.2c/0c")
                else:
                    fig.plot(x=X, y=Y, style="s0.4c", fill="black", pen="white")
        except Exception as e:
            print(f"[MAP:WARN] Station plotting failed: {e}")

    # Dome marker
    if dome_location and _finite(dome_location.get("lon")) and _finite(dome_location.get("lat")):
        fig.plot(
            x=float(dome_location["lon"]),
            y=float(dome_location["lat"]),
            style="t1.0c",   # triangle, 1.0 cm size
            fill="yellow",   # solid fill color
            pen="0.5p,black" # thin black outline
        )

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

    if outfile:
        fig.savefig(outfile)
    elif show:
        fig.show()


    return (fig, region) if return_region else fig
'''
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

'''
# ----------------------------
# Heatmap
# ----------------------------
def plot_heatmap_colored(
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