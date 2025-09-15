# flovopy/asl/map.py
from __future__ import annotations
import os, pickle
import numpy as np
import pygmt
from obspy import Inventory
from flovopy.stationmetadata.utils import inventory2traceid

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