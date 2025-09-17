# grid.py
from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from obspy.core.inventory import Inventory
from obspy import Stream

from flovopy.utils.make_hash import make_hash   
from flovopy.core.mvo import dome_location

# ------------------------------
# Compact grid signature (for IDs)
# ------------------------------
@dataclass(frozen=True)
class GridSignature:
    centerlat: float
    centerlon: float
    nlat: int
    nlon: int
    node_spacing_m: float
    latmin: float
    latmax: float
    lonmin: float
    lonmax: float

    def as_tuple(self) -> Tuple:
        return (
            self.centerlat, self.centerlon,
            self.nlat, self.nlon, self.node_spacing_m,
            self.latmin, self.latmax, self.lonmin, self.lonmax,
        )


class Grid:
    """
    Regular lat/lon grid with ~square spacing (in meters) at the given center latitude.
    RESPONSIBILITY: geometry + plotting only. No station metadata or distances.
    """

    def __init__(self, centerlat: float, centerlon: float, nlat: int, nlon: int, node_spacing_m: float):
        # meters per degree at center latitude
        meters_per_deg_lat = 1000.0 * 111.19492664455873  # degrees2kilometers(1.0) (baked for speed)
        meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(centerlat))
        if meters_per_deg_lon <= 0:
            raise ValueError("meters_per_deg_lon <= 0; check center latitude")

        # degrees per node
        self.node_spacing_lat = node_spacing_m / meters_per_deg_lat
        self.node_spacing_lon = node_spacing_m / meters_per_deg_lon

        # symmetric endpoints so we get exactly n points
        half_lat = (nlat - 1) / 2.0
        half_lon = (nlon - 1) / 2.0
        minlat = centerlat - half_lat * self.node_spacing_lat
        maxlat = centerlat + half_lat * self.node_spacing_lat
        minlon = centerlon - half_lon * self.node_spacing_lon
        maxlon = centerlon + half_lon * self.node_spacing_lon

        # exact counts with inclusive endpoints
        self.latrange = np.linspace(minlat, maxlat, num=nlat, endpoint=True)
        self.lonrange = np.linspace(minlon, maxlon, num=nlon, endpoint=True)
        self.gridlon, self.gridlat = np.meshgrid(self.lonrange, self.latrange, indexing="xy")

        # metadata
        self.centerlat = float(centerlat)
        self.centerlon = float(centerlon)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.node_spacing_m = float(node_spacing_m)

        # identity: pure grid (stations can be added to the hash on demand)
        self.id = self.set_id()  # grid-only ID by default

    # ---------------------------
    # Identity / signatures
    # ---------------------------
    def signature(self) -> GridSignature:
        return GridSignature(
            centerlat=self.centerlat,
            centerlon=self.centerlon,
            nlat=self.nlat,
            nlon=self.nlon,
            node_spacing_m=self.node_spacing_m,
            latmin=float(np.nanmin(self.gridlat)),
            latmax=float(np.nanmax(self.gridlat)),
            lonmin=float(np.nanmin(self.gridlon)),
            lonmax=float(np.nanmax(self.gridlon)),
        )

    def set_id(self, station_ids: Optional[Sequence[str]] = None) -> str:
        """
        Set/return a deterministic ID for this grid. If a list of station IDs is provided,
        they are included in the hash *without being stored on the Grid*.
        """
        sig = self.signature().as_tuple()
        if station_ids:
            sid_tuple = tuple(sorted(station_ids))
            self.id = make_hash(sig, sid_tuple)
        else:
            self.id = make_hash(sig)
        return self.id

    # ---------------------------
    # Plotting
    # ---------------------------
    def plot(
        self,
        fig=None,
        show: bool = True,
        symbol: str = "c",
        scale: float = 1.0,
        fill: Optional[str] = "blue",
        pen: Optional[str] = "0.5p,black",
        topo_map_kwargs: Optional[dict] = None,
        outfile: Optional[str] = None,
    ):
        """
        Plot grid nodes using PyGMT. If a `topo_map(show=False, **kwargs)` factory exists
        in `.map`, it will be used; otherwise a minimal basemap is drawn.
        """
        try:
            import pygmt  # type: ignore
        except Exception as e:
            raise ImportError("PyGMT is required for Grid.plot().") from e

        # Try external topo_map; else make a bare basemap
        if fig is None:
            topo_map_kwargs = topo_map_kwargs or {}
            try:
                from .map import topo_map  # optional dependency
                fig = topo_map(show=False, **topo_map_kwargs)
            except Exception:
                region = [
                    float(np.nanmin(self.gridlon)), float(np.nanmax(self.gridlon)),
                    float(np.nanmin(self.gridlat)), float(np.nanmax(self.gridlat)),
                ]
                fig = pygmt.Figure()
                fig.basemap(region=region, projection="M12c", frame=True)

        size_cm = (self.node_spacing_m / 2000.0) * float(scale)
        stylestr = f"{symbol}{size_cm}c"

        fig.plot(
            x=self.gridlon.reshape(-1),
            y=self.gridlat.reshape(-1),
            style=stylestr,
            pen=pen,
            fill=fill if symbol not in ("x", "+") else None,
        )
        if outfile:
            fig.savefig(outfile, dpi=300)
        if show:
            fig.show()
        return fig

    # ---------------------------
    # Persistence (pure grid)
    # ---------------------------
    def save(self, cache_dir: str, force_overwrite: bool = False) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        pklfile = os.path.join(cache_dir, f"Grid_{self.id}.pkl")
        if os.path.exists(pklfile) and not force_overwrite:
            print(f"[INFO] File exists: {pklfile} (use force_overwrite=True to overwrite).")
            return pklfile
        with open(pklfile, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] Grid saved to {pklfile}")
        return pklfile

    @classmethod
    def load(cls, cache_file: str) -> "Grid":
        if not os.path.isfile(cache_file):
            raise FileNotFoundError(f"No such file: {cache_file}")
        with open(cache_file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle does not contain a {cls.__name__} instance.")
        print(f"[INFO] Grid loaded from {cache_file}")
        return obj


# ---------------------------------
# Convenience helpers (pure, stateless)
# ---------------------------------
def station_ids_from_inventory(inventory: Inventory) -> Tuple[str, ...]:
    """Return sorted seed IDs NET.STA.LOC.CHA from an ObsPy Inventory."""
    ids = [
        f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
        for net in inventory for sta in net for cha in sta
    ]
    return tuple(sorted(ids))

def station_ids_from_stream(st: Stream) -> Tuple[str, ...]:
    """Return sorted, unique seed IDs from a Stream."""
    return tuple(sorted({tr.id for tr in st}))

def initial_source(lat: float = dome_location["lat"], lon: float = dome_location["lon"]) -> dict:
    return {"lat": float(lat), "lon": float(lon)}

def make_grid(
    center_lat: float = dome_location["lat"],
    center_lon: float = dome_location["lon"],
    node_spacing_m: int = 100,
    grid_size_lat_m: int = 10_000,
    grid_size_lon_m: int = 8_000,
) -> Grid:
    """Create a Grid centered at (center_lat, center_lon) with the given physical extent."""
    nlat = int(grid_size_lat_m / node_spacing_m) + 1
    nlon = int(grid_size_lon_m / node_spacing_m) + 1
    return Grid(center_lat, center_lon, nlat, nlon, node_spacing_m)