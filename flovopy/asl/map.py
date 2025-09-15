# flovopy/asl/map.py
from __future__ import annotations
import os, pickle
import numpy as np
import pygmt
from obspy import Inventory, UTCDateTime
import pandas as pd
from flovopy.stationmetadata.utils import inventory2traceid
import xarray as xr
pygmt.config(GMT_DATA_SERVER="https://oceania.generic-mapping-tools.org")

def topo_map(
    show=False,
    zoom_level=0,
    inv: Inventory | None = None,
    add_labels: bool = False,
    centerlon=-62.177,
    centerlat=16.721,        # slight north offset to center on volcano dome area
    contour_interval=100,    # legacy spacing
    topo_color=True,         # legacy color toggle
    resolution="03s",
    DEM_DIR=None,
    stations=None,
    title: str | None = None,
    region=None,
    levels=None,
    level_interval=None,
    cmap=None,
    projection=None,
    azimuth=135,
    elevation=30,
    limit=None,
    figsize: float = 6.0,
):
    if region:
        centerlon = (region[0] + region[1]) / 2.0
        centerlat = (region[2] + region[3]) / 2.0

    # default region by zoom
    if not region:
        diffdeglat = 0.1 / (1.5 ** zoom_level)
        diffdeglon = diffdeglat / np.cos(np.deg2rad(centerlat))
        minlon, maxlon = centerlon - diffdeglon, centerlon + diffdeglon
        minlat, maxlat = centerlat - diffdeglat, centerlat + diffdeglat
        region = [minlon, maxlon, minlat, maxlat]

    # cache key for relief
    pklfile = None
    if DEM_DIR:
        os.makedirs(DEM_DIR, exist_ok=True)
        pklfile = os.path.join(DEM_DIR, f"EarthRelief_{centerlon:.4f}_{centerlat:.4f}_{zoom_level}_{resolution}.pkl")

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
            with open(pklfile, "wb") as f:
                pickle.dump(ergrid, f)

    # shading
    shade = pygmt.grdgradient(grid=ergrid, radiance=[azimuth, elevation], normalize="t1")

    # colormap
    if cmap is None:
        cmap = "topo" if topo_color else "gray"

    if projection is None:
        projection = f"M{figsize}i"

    fig = pygmt.Figure()

    fig.grdimage(grid=ergrid, region=region, projection=projection, cmap=cmap, shading=shade, frame=True)

    fig.coast(region=region, projection=projection, shorelines="1/0.5p,black", frame=["WSen", "af"])

    if levels is not None:
        fig.grdcontour(grid=ergrid, levels=levels, pen="0.25p,black", limit=limit)
    else:
        step = level_interval if level_interval is not None else contour_interval
        fig.grdcontour(grid=ergrid, interval=step, pen="0.25p,black", limit=limit)

    if cmap and cmap != "gray":
        fig.colorbar(frame='+l"Topography (m)"')

    # stations
    if inv is not None:
        seed_ids = inventory2traceid(inv)
        if not stations:
            stations = [sid.split(".")[1] for sid in seed_ids]
        stalat = [inv.get_coordinates(sid)["latitude"] for sid in seed_ids]
        stalon = [inv.get_coordinates(sid)["longitude"] for sid in seed_ids]
        if add_labels:
            for lat, lon, sid in zip(stalat, stalon, seed_ids):
                _, sta, _, _ = sid.split(".")
                is_focus = sta in stations
                fig.plot(x=lon, y=lat, style="s0.5c", fill="white" if is_focus else "black", pen="black" if is_focus else "white")
                fig.text(x=lon, y=lat, text=sta, font=("white" if is_focus else "black"), justify="ML", offset="0.2c/0c")
        else:
            fig.plot(x=stalon, y=stalat, style="s0.4c", fill="black", pen="white")

    if title:
        fig.text(
            text=title,
            x=region[0] + 0.60 * (region[1] - region[0]),
            y=region[2] + 0.90 * (region[3] - region[2]),
            justify="TC",
            font="11p,Helvetica-Bold,black",
        )

    fig.basemap(region=region, frame=True)

    if show:
        fig.show()
    return fig


def plot_heatmap_montserrat_colored(
    df, lat_col="latitude", lon_col="longitude", amp_col="amplitude",
    zoom_level=0, inventory=None, color_scale=0.4, cmap="turbo",
    log_scale=True, node_spacing_m=50, outfile=None, region=None, title=None,
):
    """
    Render tessellated color squares for node energy (sum amp^2) on Montserrat topo.
    """
    df = df.copy()
    df["energy"] = df[amp_col] ** 2
    grouped = df.groupby([lat_col, lon_col])["energy"].sum().reset_index()

    x = grouped[lon_col].to_numpy()
    y = grouped[lat_col].to_numpy()
    z = grouped["energy"].to_numpy()
    if log_scale:
        z = np.log10(z + 1e-12)

    # color palette
    pygmt.makecpt(cmap=cmap,
                  series=[np.nanmin(z), np.nanmax(z),
                          (np.nanmax(z) - np.nanmin(z)) / 100.0],
                  continuous=True)

    fig = topo_map(zoom_level=zoom_level, inv=inventory, topo_color=False, region=region, title=title)

    # approximate symbol size in cm
    symbol_size_cm = node_spacing_m * 0.077 / 50.0
    fig.plot(x=x, y=y, style=f"s{symbol_size_cm}c", fill=z, cmap=True, pen=None)

    fig.colorbar(frame='+l"Log10 Total Energy"' if log_scale else '+l"Total Energy"')
    if region:
        fig.basemap(region=region, frame=True)
    if outfile:
        fig.savefig(outfile)
    else:
        fig.show()
    return fig