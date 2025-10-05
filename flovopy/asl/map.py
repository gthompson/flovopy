# flovopy/asl/map.py

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple, List, Union
import xarray as xr  
from pathlib import Path
import numpy as np
import pandas as pd
import pygmt
from obspy import Inventory
from flovopy.stationmetadata.utils import inventory2traceid
from flovopy.core.mvo import dome_location  # ensure available

# Prefer Oceania mirror
pygmt.config(GMT_DATA_SERVER="https://oceania.generic-mapping-tools.org")

# Montserrat defaults - move to flovopy.core.mvo ?
island_center = (-62.195, 16.75)
#volcano_center = (centerlon, centerlat)

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
    topo_color: bool = True,
    resolution: str = "01s",
    DEM_DIR: Optional[str] = None,
    stations: Optional[List[str]] = None,
    title: Optional[str] = None,
    region: Optional[List[float]] = None,
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
    dem_tif: Optional[Union[str, Path, "xr.DataArray", np.ndarray, Tuple[np.ndarray, Tuple[float,float,float,float]]]] = None,
    dome_location: Optional[dict] = None, # e.g. {"lat": 16.72, "lon": -62.18}
    outfile: Optional[str] = None,
    land_fill: Optional[str] = None,  # NEW: paint land uniformly (e.g. "152/251/152" or "200/200/200")
    add_coastline: bool = False,
    add_colorbar: bool = False,
    add_topography: bool = True,
    add_scale: bool = True,
):
    """
    Generate a topographic map using PyGMT with optional DEM input, station markers,
    contours, shading, and annotation.

    This function wraps PyGMT to create a flexible cartographic figure centered on a
    specified location or derived from a digital elevation model (DEM). It supports
    multiple sources for the topography:

    - **Path/str**: Local file path to a GeoTIFF DEM on disk.
    - **URL**: Remote HTTP(S) URL pointing to a GeoTIFF DEM (downloaded or streamed).
    - **In-memory DEM**:
        - `xarray.DataArray` with latitude and longitude coordinates.
        - `(numpy.ndarray, (west, east, south, north))` tuple providing raw values
          with geographic bounds.
        - *Not recommended*: bare `numpy.ndarray` without bounds (insufficient for mapping).

    If no custom DEM is provided, the function falls back to PyGMT's built-in
    Earth Relief dataset at the requested resolution.

    Parameters
    ----------
    show : bool, default=False
        Whether to immediately display the figure interactively.
    zoom_level : int, default=0
        Controls zoom span if `region` is not specified. Higher values zoom in further.
    inv : obspy.Inventory or None
        Station inventory object. If provided, station locations are plotted.
    add_labels : bool, default=False
        If True, station codes are plotted alongside station markers.
    centerlon, centerlat : float, optional
        Center longitude and latitude. If not provided, derived from DEM or defaults.
    topo_color : bool, default=True
        Use color topography palette (`geo`) if True, grayscale if False.
    resolution : str, default="01s"
        Earth Relief resolution when no custom DEM is provided.
    DEM_DIR : str or None
        Directory to cache Earth Relief tiles for faster reloads.
    stations : list of str, optional
        Subset of stations (by code) to highlight when plotting inventory.
    title : str, optional
        Map title (plotted at the top of the figure).
    region : list of float, optional
        Geographic bounding box [west, east, south, north]. If given, overrides zoom.
    level_interval : float, optional
        Contour interval in meters for `grdcontour`. If None, no contours.
    cmap : str or None
        Colormap or CPT file. If None, chosen automatically based on `topo_color`.
    projection : str, optional
        GMT projection string. If None, defaults to "M{figsize}i" (Mercator).
    hillshade_azimuth : float, default=135
        Azimuth (degrees) of illumination for hillshading.
    hillshade_altitude : float, default=30
        Altitude (degrees) of illumination for hillshading.
    add_shading : bool, default=True
        Whether to compute and overlay hillshade on the DEM.
    limit : float, optional
        Value limit for contours.
    figsize : float, default=6.0
        Figure size in inches.
    frame : bool, default=True
        Whether to draw a map frame.
    return_region : bool, default=False
        If True, returns both the figure and the computed region bounds.
    dem_tif : str, Path, URL, xarray.DataArray, or (ndarray, bounds), optional
        Custom DEM source. See above for supported types.
    dome_location : dict, optional
        Plot a dome marker. Dictionary with keys {"lat": float, "lon": float}.
    outfile : str or None
        If set, save the map to this file instead of showing it.
    land_fill : str or None
        If set, paints land uniformly with the given RGB triplet string (e.g. "200/200/200").
    add_coastline : bool, default=False
        Overlay coastlines using `fig.coast()`.
    add_colorbar : bool, default=False
        Draw a colorbar for the topography.
    add_topography : bool, default=True
        If False, no DEM is drawn (only coastlines/stations).
    add_scale: bool, default=True
        Draw a scale bar

    Returns
    -------
    fig : pygmt.Figure
        The generated PyGMT figure object.
    region : list of float
        Returned only if `return_region=True`. Geographic region [west, east, south, north].

    Notes
    -----
    - **DEM input flexibility**: The `dem_tif` parameter accepts file paths, URLs,
      or in-memory DEMs. Internally, these are normalized to a file path or
      `xarray.DataArray` for PyGMT compatibility.
    - **Performance**: Large DEMs may be slow to load. Consider cropping or using
      coarser resolutions for interactive use.
    - **Shading**: Hillshading is computed with `pygmt.grdgradient` and may fail
      for certain DEM inputs; in such cases, shading is skipped gracefully.
    - **Station plotting**: If an ObsPy `Inventory` is provided, stations are drawn
      with optional labels. Highlighting is applied if `stations` is specified.

    Examples
    --------
    >>> # From a local DEM file
    >>> topo_map(dem_tif="dems/montserrat.tif", show=True)

    >>> # From a remote DEM URL
    >>> topo_map(dem_tif="https://example.com/montserrat.tif", title="Montserrat")

    >>> # From an in-memory xarray DataArray
    >>> da = xr.DataArray(Z, coords={"lat": lats, "lon": lons}, dims=("lat","lon"))
    >>> topo_map(dem_tif=da, add_colorbar=True)

    >>> # From raw numpy array with bounds
    >>> Z = np.random.randn(200, 300)
    >>> bounds = (-62.3, -62.0, 16.6, 16.85)
    >>> topo_map(dem_tif=(Z, bounds), add_labels=True, inv=my_inventory)
    """
    def _finite(v) -> bool:
        try:
            return v is not None and np.isfinite(float(v))
        except Exception:
            return False

    def _finite_arrays(x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        return (x[m], y[m]) if np.any(m) else (np.array([]), np.array([]))

    # --------------------
    # Region selection
    # --------------------
    if region is not None:
        centerlon = (region[0] + region[1]) / 2.0
        centerlat = (region[2] + region[3]) / 2.0
    else:
        dem_bounds = None
        if dem_tif is not None:
            # Early resolve to pick a reasonable default center from DEM
            grid_src_tmp, dem_bounds, (zmin_tmp, zmax_tmp) = _resolve_dem_to_grid_and_bounds(dem_tif)
            if (centerlon is None) or (centerlat is None):
                centerlon = (dem_bounds[0] + dem_bounds[1]) / 2.0
                centerlat = (dem_bounds[2] + dem_bounds[3]) / 2.0
        # Defaults if still unset
        if centerlon is None or centerlat is None:
            centerlon, centerlat = (-62.195, 16.75)

        # Zoom-derived region
        if zoom_level <= 0:
            span_lon_deg, span_lat_deg = 0.12, 0.18
        else:
            base_span_lon, base_span_lat = 0.18, 0.14
            shrink = 1.7 ** (zoom_level - 1)
            span_lon_deg = base_span_lon / shrink
            span_lat_deg = base_span_lat / shrink

        half_lon, half_lat = span_lon_deg / 2.0, span_lat_deg / 2.0
        region = [
            centerlon - half_lon,
            centerlon + half_lon,
            centerlat - half_lat,
            centerlat + half_lat,
        ]

        # Clip to DEM if we have its bounds
        if dem_tif is not None and dem_bounds is not None and all(np.isfinite(dem_bounds)):
            region = [
                max(region[0], dem_bounds[0]),
                min(region[1], dem_bounds[1]),
                max(region[2], dem_bounds[2]),
                min(region[3], dem_bounds[3]),
            ]

    region = _validate_region(region)

    # --------------------
    # Figure / projection
    # --------------------
    if projection is None:
        projection = f"M{figsize}i"
    fig = pygmt.Figure()

    # --------------------
    # Resolve DEM now (unconditionally) so grid_src is always set when dem_tif is given
    # --------------------
    grid_src = None
    zmin = zmax = None
    shade = None

    if dem_tif is not None:
        grid_src, dem_bounds, (zmin, zmax) = _resolve_dem_to_grid_and_bounds(dem_tif)

        # Compute z-range if missing
        if (zmin is None) or (zmax is None) or not np.isfinite(zmin) or not np.isfinite(zmax):
            try:
                if hasattr(grid_src, "values"):  # xarray.DataArray
                    data = np.asarray(grid_src.values, float)
                else:
                    import rasterio
                    with rasterio.open(str(grid_src)) as _src:
                        data = _src.read(1, out_dtype="float32", masked=True).filled(np.nan)
                zmin = float(np.nanpercentile(data, 1))
                zmax = float(np.nanpercentile(data, 99))
            except Exception:
                # hard fallback
                zmin, zmax = -8000.0, 8000.0

        # Optional hillshade
        if add_shading:
            try:
                shade = pygmt.grdgradient(
                    grid=grid_src,
                    radiance=[hillshade_azimuth, hillshade_altitude],
                    normalize="t1",
                )
            except Exception:
                shade = None

    else:
        # Earth Relief fallback (with optional cache)
        pklfile = None
        if DEM_DIR:
            os.makedirs(DEM_DIR, exist_ok=True)
            pklfile = os.path.join(
                DEM_DIR,
                f"EarthRelief_{resolution}_{region[0]:.4f}_{region[1]:.4f}_{region[2]:.4f}_{region[3]:.4f}.pkl",
            )
        ergrid = None
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
            zmin = float(np.nanpercentile(ergrid.values, 1))
            zmax = float(np.nanpercentile(ergrid.values, 99))
        except Exception:
            zmin = zmax = None

        if add_shading:
            try:
                shade = pygmt.grdgradient(
                    grid=ergrid,
                    radiance=[hillshade_azimuth, hillshade_altitude],
                    normalize="t1",
                )
            except Exception:
                shade = None

    # --------------------
    # CPT (avoid -Z warning by omitting continuous=True without increment)
    # --------------------
    use_palette = cmap if cmap is not None else ("geo" if topo_color else "gray")
    use_cmap_flag = None

    if (zmin is None) or (zmax is None) or not np.isfinite(zmin) or not np.isfinite(zmax):
        # final fallback if range still unknown
        zmin, zmax = -8000.0, 8000.0

    try:
        if np.isfinite(zmin) and np.isfinite(zmax) and (zmax > zmin):
            # Continuous CPT without increment: no -Z emitted, no warning
            pygmt.makecpt(cmap=use_palette, series=[zmin, zmax])
            use_cmap_flag = True  # tells grdimage to use current CPT
        else:
            use_cmap_flag = use_palette
    except Exception:
        use_cmap_flag = use_palette

    paint_water = (use_palette == "gray") or (cmap is None and not topo_color)

    # --------------------
    # Draw raster
    # --------------------
    if add_topography:
        if grid_src is None:
            raise ValueError("grid_src is None; DEM was not resolved. Check `dem_tif` and `region`.")
        fig.grdimage(
            grid=grid_src,
            region=region,
            projection=projection,
            cmap=(True if use_cmap_flag is True else use_cmap_flag),
            shading=shade,
            frame=frame,
        )
    else:
        add_coastline = True

    if add_colorbar and add_topography:
        if isinstance(use_cmap_flag, (str, bool)) and use_cmap_flag is not False:
            if (cmap is None and topo_color) or (cmap in ("geo", "topo")):
                fig.colorbar(frame='+l"Topography (m)"')

    # --------------------
    # Optional uniform land overlay
    # --------------------
    if land_fill:
        try:
            land_only = pygmt.grdclip(grid=grid_src, below=[0, "NaN"])

            import tempfile
            cpt_const = tempfile.NamedTemporaryFile(suffix=".cpt", delete=False).name
            with open(cpt_const, "w") as _f:
                _f.write(f"0 {land_fill} 1 {land_fill}\n")

            fig.grdimage(
                grid=land_only,
                region=region,
                projection=projection,
                cmap=cpt_const,
                shading=False,
                frame=False,
            )
        except Exception as _e:
            print(f"[MAP:WARN] Uniform land overlay failed: {_e}")

    # --------------------
    # Coastlines (and optional water fill)
    # --------------------
    if add_coastline:
        coast_kwargs = dict(
            region=region,
            projection=projection,
            shorelines="1/0.5p,black",
            resolution="f",
            frame=(["WSen", "af"] if frame else False),
        )
        if paint_water:
            coast_kwargs["water"] = "lightblue"
        fig.coast(**coast_kwargs)

    # --------------------
    # Contours
    # --------------------
    try:
        if level_interval:
            fig.grdcontour(grid=grid_src, levels=level_interval, pen="0.1p,black", limit=limit)
            fig.grdcontour(grid=grid_src, levels=level_interval * 5, pen="1.0p,black", limit=limit)
    except Exception:
        pass

    # --------------------
    # Stations
    # --------------------
    if inv is not None:
        try:
            from flovopy.stationmetadata.utils import inventory2traceid
            seed_ids = inventory2traceid(inv)
            if not stations:
                stations = [sid.split(".")[1] for sid in seed_ids]
            xs, ys = [], []
            for sid in seed_ids:
                try:
                    c = inv.get_coordinates(sid)
                    lo, la = c.get("longitude", np.nan), c.get("latitude", np.nan)
                    if _finite(lo) and _finite(la):
                        xs.append(float(lo))
                        ys.append(float(la))
                except Exception:
                    continue
            X, Y = _finite_arrays(np.array(xs), np.array(ys))
            if X.size:
                if add_labels:
                    fig.plot(x=X, y=Y, style="s0.5c", fill="white", pen="black")
                    for sid, x0, y0 in zip(seed_ids, X, Y):
                        sta = sid.split(".")[1]
                        highlight = (sta in stations)
                        fig.text(
                            x=x0,
                            y=y0,
                            text=sta,
                            font=("12p,white" if highlight else "10p,black"),
                            justify="ML",
                            offset="0.2c/0c",
                        )
                else:
                    fig.plot(x=X, y=Y, style="s0.4c", fill="black", pen="white")
        except Exception as e:
            print(f"[MAP:WARN] Station plotting failed: {e}")

    # --------------------
    # Dome marker
    # --------------------
    if dome_location and _finite(dome_location.get("lon")) and _finite(dome_location.get("lat")):
        domelat = float(dome_location["lat"])
        domelon = float(dome_location["lon"])
        if (region[0] < domelon < region[1]) and (region[2] < domelat < region[3]):
            fig.plot(x=domelon, y=domelat, style="t1.0c", fill="red", pen="1p,black")

    # --------------------
    # Title, frame, output
    # --------------------
    if title and frame:
        print(f"[MAP] Adding title {title}")
        draw_multiline_title(fig, region, title)
        '''
        fig.text(
            text=title,
            x=region[0] + 0.50 * (region[1] - region[0]),
            y=region[2] + 0.98 * (region[3] - region[2]),
            justify="TC",
            font="11p,Helvetica-Bold,black",
        )
        '''

    if frame:
        fig.basemap(region=region, frame=True)

    if add_scale:
        fig.basemap(
            region=region,
            projection=projection,
            map_scale="jBL+w4k+o0.5c/0.5c+f+lkm"  # customize as needed
        )

    if outfile:
        print(f"topo_map: Saving to {outfile}")
        fig.savefig(outfile)
    elif show:
        fig.show()

    return (fig, region) if return_region else fig


def draw_multiline_title(
    fig,
    region,                    # [xmin, xmax, ymin, ymax]
    title,                     # str with \n OR list/tuple of lines
    *,
    font="12p,Helvetica-Bold,black",
    pad_rel=0.05,              # gap above top frame as a fraction of map height
    line_rel=0.02,             # line spacing as a fraction of map height
    justify="TC",
):
    # Normalize title → list of lines
    if isinstance(title, (list, tuple)):
        lines = [str(x) for x in title]
    else:
        lines = str(title).splitlines()

    xmin, xmax, ymin, ymax = map(float, region)
    xmid = xmin + 0.5 * (xmax - xmin)
    H    = (ymax - ymin)
    y0   = ymax + pad_rel * H
    dy   = line_rel * H

    for i, line in enumerate(lines):
        fig.text(
            text=line,
            x=xmid,
            y=y0 - i * dy,
            justify=justify,
            font=font,
            no_clip=True,
        )

def _resolve_dem_to_grid_and_bounds(
    dem_src: Union[str, Path, "xr.DataArray", np.ndarray, Tuple[np.ndarray, Tuple[float, float, float, float]]],
) -> Tuple[Union[str, "xr.DataArray"], List[float], Tuple[Optional[float], Optional[float]]]:
    """
    Normalize a DEM source into a GMT-acceptable grid (path or xarray.DataArray),
    plus geographic bounds [W, E, S, N] and (zmin, zmax) if cheaply available.
    """
    zmin = zmax = None

    # Case 1: string or Path (local file or HTTP(S) URL)
    if isinstance(dem_src, (str, Path)):
        src_str = str(dem_src)
        is_url = src_str.lower().startswith(("http://", "https://"))
        try:
            import rasterio
            if is_url:
                # Try opening directly (vsicurl). If GDAL lacks curl, user can pass a local path instead.
                with rasterio.open(src_str) as ds:
                    b = ds.bounds
                    try:
                        stats = ds.stats(1)
                        if stats:
                            zmin, zmax = float(stats.min), float(stats.max)
                    except Exception:
                        pass
                return src_str, [float(b.left), float(b.right), float(b.bottom), float(b.top)], (zmin, zmax)
            else:
                # Local file
                with rasterio.open(src_str) as ds:
                    b = ds.bounds
                    try:
                        stats = ds.stats(1)
                        if stats:
                            zmin, zmax = float(stats.min), float(stats.max)
                    except Exception:
                        pass
                return src_str, [float(b.left), float(b.right), float(b.bottom), float(b.top)], (zmin, zmax)
        except Exception:
            # Fall back: let GMT try to open; bounds unknown
            return src_str, [np.nan, np.nan, np.nan, np.nan], (None, None)

    # Case 2: xarray.DataArray with lon/lat coords
    if xr is not None and isinstance(dem_src, xr.DataArray):
        da = dem_src
        lon_name = next((n for n in da.dims if n.lower() in ("x", "lon", "longitude")), None)
        lat_name = next((n for n in da.dims if n.lower() in ("y", "lat", "latitude")), None)
        if lon_name is None or lat_name is None:
            raise ValueError("xarray.DataArray DEM must have lon/lat (or x/y) dimensions.")
        west, east = float(da[lon_name].min()), float(da[lon_name].max())
        south, north = float(da[lat_name].min()), float(da[lat_name].max())
        try:
            zmin = float(np.nanpercentile(da.values, 1))
            zmax = float(np.nanpercentile(da.values, 99))
        except Exception:
            pass
        return da, [west, east, south, north], (zmin, zmax)

    # Case 3: tuple of (array, bounds)
    if isinstance(dem_src, tuple) and len(dem_src) == 2 and isinstance(dem_src[0], np.ndarray):
        arr, bounds = dem_src
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
            raise ValueError("When passing (array, bounds), bounds must be (west, east, south, north).")
        west, east, south, north = map(float, bounds)
        if xr is None:
            raise ImportError("xarray is required to wrap an in-memory array into a grid for PyGMT.")
        ny, nx = arr.shape
        lons = np.linspace(west, east, nx)
        lats = np.linspace(south, north, ny)
        da = xr.DataArray(arr, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
        try:
            zmin = float(np.nanpercentile(arr, 1))
            zmax = float(np.nanpercentile(arr, 99))
        except Exception:
            pass
        return da, [west, east, south, north], (zmin, zmax)

    # Case 4: bare ndarray without bounds: insufficient info for mapping
    if isinstance(dem_src, np.ndarray):
        raise ValueError(
            "Bare np.ndarray provided without bounds. Pass (array, (west,east,south,north)) "
            "or an xarray.DataArray with lon/lat coords, or set dem_tif to a path/URL."
        )

    raise TypeError(
        "Unsupported dem_tif type. Use str/Path (file or URL), xarray.DataArray, "
        "or np.ndarray plus bounds as (array, (w,e,s,n))."
    )




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