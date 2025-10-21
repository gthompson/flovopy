# grid.py
from __future__ import annotations

import math
import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Dict, Any
from pathlib import Path
import copy
import pandas as pd
from scipy.spatial import cKDTree

import numpy as np
from obspy.core.inventory import Inventory
from obspy import Stream
import pygmt
from flovopy.utils.make_hash import make_hash
from flovopy.core.mvo import dome_location
from flovopy.asl.map import topo_map

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
    dem_tag: Optional[str] = None  # indicates DEM source/sampler signature
    

    def as_tuple(self) -> Tuple:
        return (
            self.centerlat, self.centerlon,
            self.nlat, self.nlon, self.node_spacing_m,
            self.latmin, self.latmax, self.lonmin, self.lonmax,
            self.dem_tag,
        )



class Grid:
    """
    Regular lat/lon grid with ~square spacing (in meters) at the given center latitude.
    RESPONSIBILITY: geometry (+ optional DEM elevations) + plotting only. No heavy station metadata.

    DEM support (optional):
      - Provide a DEM sampler via `dem=...` to drape nodes on topography.
      - Elevations (meters, + up) are available at `node_elev_m` (shape = grid).
      - DEM sources:
          * ("pygmt",   {"resolution": "01s", "cache_dir": "...", "tag": "01s"})
          * ("geotiff", {"path": "/abs/path/to/dem.tif", "tag": "myDEM"})
          * ("sampler", {"func": callable(lat, lon) -> elevation_m, "tag": "custom"})
          * None (default; flat z=0)

    Distances helper:
      - `distance_to_stations(...)` can compute horizontal (2-D) or 3-D (include station elevation).
    """

    def __init__(
        self,
        centerlat: float,
        centerlon: float,
        nlat: int,
        nlon: int,
        node_spacing_m: float,
        *,
        dem: Optional[Tuple[str, Dict[str, Any]]] = None,
    ):
        # meters per degree at center latitude
        meters_per_deg_lat, meters_per_deg_lon = _meters_per_degree(centerlat)
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

        # DEM / elevations
        self.dem_info: Optional[Tuple[str, Dict[str, Any]]] = dem
        self.node_elev_m: Optional[np.ndarray] = None  # same shape as grid

        # Optional land mask metadata (True = keep node)
        self._node_mask: Optional[np.ndarray] = None       # shape like gridlat
        self._node_mask_idx: Optional[np.ndarray] = None   # flat indices of True

        # identity: pure grid (stations can be added to the hash on demand)
        self.id = self.set_id()

        # sample DEM if requested
        if self.dem_info is not None:
            self._sample_dem_into_node_elevations()

    # ---------------------------
    # Identity / signatures
    # ---------------------------
    def signature(self) -> GridSignature:
        dem_tag = None
        if self.dem_info is not None:
            source, params = self.dem_info
            if source == "pygmt":
                tag = params.get("tag") or params.get("resolution") or "pygmt"
            elif source in ("sampler", "geotiff"):
                tag = params.get("tag") or source
            else:
                tag = str(source)
            dem_tag = f"{source}:{tag}"

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
            dem_tag=dem_tag,
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
    # DEM sampling
    # ---------------------------
    def _sample_dem_into_node_elevations(self):
        """
        Fill `node_elev_m` by sampling the given DEM spec.
        - ("pygmt",   {"resolution": "01s", "cache_dir": "...", "tag": "01s"})
        - ("geotiff", {"path": "/abs/path/to/dem.tif", "tag": "myDEM"})
        - ("sampler", {"func": callable(lat, lon) -> elevation_m})
        """
        source, params = self.dem_info
        if source == "pygmt":
            self.node_elev_m = _dem_sample_pygmt(self.gridlat, self.gridlon, **params)
        elif source == "geotiff":
            path = params.get("path")
            if not path or not os.path.exists(path):
                raise ValueError("dem=('geotiff', {'path': '/abs/path/to/dem.tif'}) requires a valid file path.")
            # fix keyword name (or pass positionally)
            self.node_elev_m = _dem_sample_geotiff(self.gridlat, self.gridlon, tif_path=path)
            # or: self.node_elev_m = _dem_sample_geotiff(self.gridlat, self.gridlon, path)
        elif source == "sampler":
            func = params.get("func")
            if not callable(func):
                raise ValueError("dem=('sampler', {'func': callable}) is required for sampler DEM.")
            self.node_elev_m = _dem_sample_callable(self.gridlat, self.gridlon, func)
        else:
            raise ValueError(f"Unknown DEM source: {source}")
        
    def apply_land_mask_from_dem(self, *, sea_level: float = 0.0) -> np.ndarray:
        """
        Build a boolean mask of 'land' nodes (elevation > sea_level) using
        Grid.node_elev_m if present; otherwise sample the configured DEM.
        Stores:
            - self._node_mask      (bool array, shape = grid)
            - self._node_mask_idx  (flat int indices of True)
        Returns the mask.
        """
        # Ensure we have elevations
        if self.node_elev_m is None:
            if self.dem_info is None:
                raise ValueError("apply_land_mask_from_dem() requires node_elev_m or a DEM in dem_info.")
            # lazily sample now
            self._sample_dem_into_node_elevations()

        elev = np.asarray(self.node_elev_m, float)
        mask = np.isfinite(elev) & (elev > float(sea_level))
        self._node_mask = mask
        self._node_mask_idx = np.flatnonzero(mask.ravel())
        return mask

    def mask_signature(self) -> Optional[str]:
        """
        Short, stable signature of the current node mask, for use in cache keys.
        Returns None if no mask applied.
        """
        idx = self._node_mask_idx
        if idx is None:
            return None
        if idx.size == 0:
            return "mask:0:-1:-1:0"
        h = int(np.bitwise_xor.reduce(idx))  # simple fast checksum
        return f"mask:{idx.size}:{int(idx[0])}:{int(idx[-1])}:{h}"

    def masked_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (lon, lat) flattened arrays honoring the current mask if present.
        If no mask, returns all nodes.
        """
        x = self.gridlon.reshape(-1)
        y = self.gridlat.reshape(-1)
        if self._node_mask_idx is not None:
            sel = self._node_mask_idx
            return x[sel], y[sel]
        return x, y

    # ---------------------------
    # Plotting
    # ---------------------------
    def plot(
        self,
        fig=None,
        show: bool = True,
        symbol: str = "c",
        scale: float = 2.0,
        fill: Optional[str] = "red",
        pen: Optional[str] = "1.0p,red",
        topo_map_kwargs: Optional[dict] = None,
        outfile: Optional[str] = None,
        min_display_spacing: Optional[float] = 50.0,
        force_all_nodes = False, # if True, we plot all nodes regardless of how dense the plot is
    ):
        """
        Plot grid nodes on the canonical topo_map() basemap (no fallbacks).
        """
        # Base kwargs: default to a simple land/sea scheme unless caller overrides
        default_kwargs = {
            "add_labels": True,
            "cmap": "gray",
            "title": "Grid",
            "frame": True,
        #    "topo_color": False,   # disable colorful topography by default
        #    "cmap": "land",        # 2-color sea/land palette defined in topo_map()
        #    "show": False,
        }

        # If this Grid was built from a GeoTIFF DEM, reuse the same raster for the basemap
        if self.dem_info and self.dem_info[0] == "geotiff":
            path = self.dem_info[1].get("path")
            if path:
                default_kwargs["dem_tif"] = path

        # Merge caller overrides
        topo_kw = {**default_kwargs, **(topo_map_kwargs or {})}

        # Always create/obtain a figure via topo_map()
        if fig is None:
            fig = topo_map(**topo_kw)
        # If caller passed (fig, region) because they used return_region=True, keep the fig
        if isinstance(fig, tuple):
            fig = fig[0]

        # Symbol size (cm) roughly scaled by physical spacing
        size_cm = (float(self.node_spacing_m) / 2000.0) * float(scale)
        stylestr = f"{symbol}{size_cm}c"

        # Nodes to plot (honor mask)
        # Nodes to plot (honor mask)
        # Nodes to plot (honor mask)
        x, y = self.masked_lonlat()
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]

        if x.size == 0:
            raise ValueError("[GRID:PLOT] No nodes to plot after masks (check DEM/CRS or mask radius).")

        # Helper: build an outline polygon from scattered lon/lat
        def _outline_polygon(xs, ys, alpha_km=0.5):
            """
            Returns a shapely Polygon (concave hull if possible, else convex hull).
            alpha_km controls concavity (smaller => tighter boundary).
            """
            try:
                from shapely.geometry import MultiPoint
                from shapely.ops import unary_union
                # Try alpha-shape via SciPy Delaunay, fall back to convex hull
                try:
                    from scipy.spatial import Delaunay

                    pts = np.column_stack([xs, ys])
                    if pts.shape[0] < 4:
                        return MultiPoint(pts).convex_hull

                    # crude lon/lat -> meters scaling near island (OK for hull)
                    lat0 = float(np.nanmedian(ys))
                    m_per_deg_lat = 111_132.0
                    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
                    P = np.column_stack([(xs - xs.mean()) * m_per_deg_lon,
                                         (ys - ys.mean()) * m_per_deg_lat])

                    tri = Delaunay(P)
                    # alpha radius in meters
                    alpha = float(alpha_km) * 1000.0

                    # collect triangle edges with circumradius < alpha
                    edges = []
                    for tri_ix in tri.simplices:
                        A, B, C = P[tri_ix]
                        a = np.linalg.norm(B - C)
                        b = np.linalg.norm(C - A)
                        c = np.linalg.norm(A - B)
                        s = 0.5 * (a + b + c)
                        area = max(s * (s - a) * (s - b) * (s - c), 0.0) ** 0.5
                        if area == 0:
                            continue
                        R = (a * b * c) / (4.0 * area)  # circumradius
                        if R <= alpha:
                            i, j, k = tri_ix
                            edges += [(i, j), (j, k), (k, i)]

                    if not edges:
                        return MultiPoint(np.column_stack([xs, ys])).convex_hull

                    # dissolve kept triangle edges into a polygon outline
                    from shapely.geometry import LineString
                    lines = [LineString([(xs[i], ys[i]), (xs[j], ys[j])]) for i, j in edges]
                    merged = unary_union(lines)
                    outline = merged.convex_hull if merged.geom_type != "Polygon" else merged
                    # If MultiPolygon, take largest
                    if outline.geom_type == "MultiPolygon":
                        outline = max(list(outline.geoms), key=lambda g: g.area)
                    return outline

                except Exception:
                    # Fallback: convex hull only
                    return MultiPoint(np.column_stack([xs, ys])).convex_hull
            except Exception:
                return None

        # Decide rendering strategy based on density
        dense = (self.node_spacing_m < (min_display_spacing or 50.0))

        if dense and not force_all_nodes:
            # 1) Try concave/convex hull outline
            poly = _outline_polygon(x, y, alpha_km=0.5)
            if poly is not None and not poly.is_empty:
                try:
                    # Plot unfilled red outline (thick enough to be visible)
                    # Handle Polygon vs MultiPolygon
                    geoms = [poly] if poly.geom_type != "MultiPolygon" else list(poly.geoms)
                    for g in geoms:
                        xs = np.array(g.exterior.coords)[:, 0]
                        ys = np.array(g.exterior.coords)[:, 1]
                        fig.plot(x=xs, y=ys, pen="1.25p,red", transparency=10)
                except Exception as e:
                    print(f"[GRID:PLOT] Hull plot failed ({e}); falling back to bbox.")
                    poly = None

            # 2) If hull failed, plot bounding rectangle
            if (poly is None) or poly.is_empty:
                xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
                bx = [xmin, xmax, xmax, xmin, xmin]
                by = [ymin, ymin, ymax, ymax, ymin]
                fig.plot(x=bx, y=by, pen="1.25p,red", transparency=10)

        else:
            # Sparse enough → plot nodes as before (with optional thinning)
            if self.node_spacing_m < (min_display_spacing or 50.0):
                step = int(np.ceil((min_display_spacing or 50.0) / self.node_spacing_m))
                x = x[::step]; y = y[::step]

            size_cm = (max(self.node_spacing_m, (min_display_spacing or 50.0)) / 2000.0) * float(scale)
            stylestr = f"{symbol}{size_cm}c"

            fig.plot(
                x=x,
                y=y,
                style=stylestr,
                #pen=pen,
                fill=(None if symbol in ("x", "+") else fill),
            )

        # Dome marker
        #try:
        #    fig.plot(x=dome_location["lon"], y=dome_location["lat"], style="a0.3c", fill="red", pen="black")
        #except Exception:
        #    pass

        if outfile:
            print(f'Grid.plot(): saving {outfile}')
            fig.savefig(outfile, dpi=300)
        if show:
            fig.show()
        return fig

    # ---------------------------
    # Distances (2-D or 3-D) utility
    # ---------------------------
    def distance_to_stations(
        self,
        inventory: Inventory,
        *,
        use_elevation_3d: bool = False,
        station_level_fallback: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute distances (km) from each grid node to each station/channel in the inventory.
        - If `use_elevation_3d=False` (default): horizontal distance only.
        - If `use_elevation_3d=True`: include vertical term using station elevation (m)
          and node elevation (m) if available (otherwise node z=0).

        Returns
        -------
        dict: seed_id (NET.STA.LOC.CHA) → distances_km (shape = grid)
        """
        dists: Dict[str, np.ndarray] = {}
        m_per_deg_lat, m_per_deg_lon = _meters_per_degree(self.centerlat)

        # node z (meters)
        if self.node_elev_m is not None and use_elevation_3d:
            node_z = np.asarray(self.node_elev_m, float)
        else:
            node_z = np.zeros_like(self.gridlat, dtype=float)

        # local ENU approximation (appropriate at Montserrat scale)
        for net in inventory:
            for sta in net:
                sta_elev = _safe_to_float(sta.elevation, default=0.0)  # meters
                for cha in sta:
                    seed = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                    slat = _safe_to_float(cha.latitude if cha.latitude is not None else sta.latitude)
                    slon = _safe_to_float(cha.longitude if cha.longitude is not None else sta.longitude)
                    selev = _safe_to_float(cha.elevation if cha.elevation is not None else sta_elev)

                    dlat_m = (self.gridlat - slat) * m_per_deg_lat
                    dlon_m = (self.gridlon - slon) * (m_per_deg_lon * math.cos(math.radians(slat)))
                    if use_elevation_3d:
                        dz_m = node_z - selev
                        d_m = np.sqrt(dlat_m**2 + dlon_m**2 + dz_m**2)
                    else:
                        d_m = np.sqrt(dlat_m**2 + dlon_m**2)

                    dists[seed] = d_m / 1000.0  # km

        return dists

    # ---------------------------
    # Persistence (pure grid+DEM)
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


    def _slice(self, mask: np.ndarray):
        """
        Return a typed sub-grid containing only nodes where mask==True.
        For arbitrary masks, we return a NodeGrid (sparse) to avoid breaking
        Grid invariants (nlat/nlon/rectangularity).
        """
        mask = np.asarray(mask, bool).ravel()
        flat_lat = self.gridlat.ravel()
        flat_lon = self.gridlon.ravel()
        if mask.size != flat_lat.size:
            raise ValueError("mask length != number of grid nodes")

        sel_lat = flat_lat[mask]
        sel_lon = flat_lon[mask]

        # optional elevation carry-over
        if self.node_elev_m is not None:
            sel_elev = np.asarray(self.node_elev_m, float).ravel()[mask]
        else:
            sel_elev = None

        # propagate DEM tag if available
        dem_tag = None
        if self.dem_info is not None:
            src, params = self.dem_info
            dem_tag = params.get("tag") or (src if isinstance(src, str) else None)

        approx = float(self.node_spacing_m) if hasattr(self, "node_spacing_m") else None
        return NodeGrid(
            node_lon=sel_lon,
            node_lat=sel_lat,
            node_elev_m=sel_elev,
            approx_spacing_m=approx,
            dem_tag=dem_tag,
        )
    
    # --- Mask API (typed, fluent) ---
    def apply_mask_indices(self, indices: np.ndarray, *, validate: bool = True) -> "Grid":
        """
        Apply a node mask defined by flat indices into the flattened grid (row-major).
        Stores:
          - self._node_mask_idx (1-D int array)
          - self._node_mask     (2-D bool array, same shape as grid)
        Returns self.
        """
        idx = np.asarray(indices, int).ravel()
        if idx.size == 0:
            self._node_mask_idx = idx
            self._node_mask = np.zeros_like(self.gridlat, dtype=bool)
            return self

        nn = int(self.nlat * self.nlon)
        if validate:
            if np.min(idx) < 0 or np.max(idx) >= nn:
                raise ValueError("apply_mask_indices: index out of bounds for grid size.")
        idx = np.unique(idx)

        mask_flat = np.zeros(nn, dtype=bool)
        mask_flat[idx] = True
        self._node_mask = mask_flat.reshape(self.nlat, self.nlon)
        self._node_mask_idx = idx
        return self

    def apply_mask_boolean(
        self,
        mask_bool: np.ndarray,
        *,
        validate: bool = True,
        mode: str = "replace",   # "replace" (default), "intersect", or "union"
    ) -> "Grid":
        """
        Apply a node mask given as a boolean array.

        Parameters
        ----------
        mask_bool : np.ndarray
            Boolean mask, shape==(nlat,nlon) or flat length==nlat*nlon.
            True = keep node, False = mask out.
        validate : bool
            Check that the shape/size matches grid dimensions.
        mode : {"replace","intersect","union"}
            - "replace": overwrite any existing mask with the new one
            - "intersect": keep only nodes that are True in BOTH
            - "union": keep nodes that are True in EITHER
        """
        b = np.asarray(mask_bool, bool)

        # normalize shape to 2D
        if b.ndim == 1:
            if validate and b.size != self.nlat * self.nlon:
                raise ValueError("apply_mask_boolean: flat mask length != nlat*nlon")
            b = b.reshape(self.nlat, self.nlon)
        else:
            if validate and b.shape != self.gridlat.shape:
                raise ValueError("apply_mask_boolean: 2-D mask shape mismatch.")

        if self._node_mask is None or mode == "replace":
            new_mask = b.copy()
        else:
            if mode == "intersect":
                new_mask = self._node_mask & b
            elif mode == "union":
                new_mask = self._node_mask | b
            else:
                raise ValueError(f"Unknown mode {mode!r}; must be 'replace','intersect','union'")

        self._node_mask = new_mask
        self._node_mask_idx = np.flatnonzero(self._node_mask.ravel())
        return self

    def clear_mask(self) -> "Grid":
        """Remove any active node mask."""
        self._node_mask = None
        self._node_mask_idx = None
        return self

    def get_mask_indices(self) -> np.ndarray | None:
        """Return flat indices of True nodes, or None if no mask is set."""
        return None if self._node_mask_idx is None else np.asarray(self._node_mask_idx, int)

    @property
    def shape(self) -> tuple[int, int] | None:
        """(nlat, nlon) for regular grids; NodeGrid will return None."""
        return (int(self.nlat), int(self.nlon))

    def __str__(self):
        base = f"[GRID] {self.nlat}x{self.nlon} nodes ({self.gridlat.size} total)  spacing={self.node_spacing_m:.1f} m"
        if self._node_mask is None:
            return base + "  [no mask]"
        else:
            kept = int(np.count_nonzero(self._node_mask))
            pct  = kept / self._node_mask.size * 100.0
            return base + f"  [mask: {kept}/{self._node_mask.size} nodes kept ({pct:.1f}%)]"
        
    # inside your Grid class
    def node_lonlat(self, g: int):
        """
        Return (lon, lat, elev_m|nan) for flat node index g.
        Works whether gridlon/gridlat/node_elev_m are 2-D or 1-D.
        """
        import numpy as np
        glon = np.asarray(self.gridlon)
        glat = np.asarray(self.gridlat)
        if glon.shape != glat.shape:
            raise ValueError("gridlon/gridlat shape mismatch")
        # Accept both (nlat, nlon) and (N,)
        nlat, nlon = (glon.shape if glon.ndim == 2 else (1, glon.size))
        if glon.ndim == 2:
            i, j = np.unravel_index(int(g), (nlat, nlon))
            lon = float(glon[i, j]); lat = float(glat[i, j])
            elev = np.asarray(getattr(self, "node_elev_m", None))
            if elev is not None and elev.size == glon.size:
                elev = float(elev.reshape(nlat, nlon)[i, j])
            else:
                elev = float("nan")
        else:
            lon = float(glon.ravel()[int(g)])
            lat = float(glat.ravel()[int(g)])
            elev_arr = np.asarray(getattr(self, "node_elev_m", None))
            elev = float(elev_arr.ravel()[int(g)]) if elev_arr is not None else float("nan")
        return lon, lat, elev

    def nearest_node(self, lon: float, lat: float) -> int:
        """
        Return flat node index of the grid node nearest (lon, lat).
        """
        import numpy as np
        glon = np.asarray(self.gridlon).ravel()
        glat = np.asarray(self.gridlat).ravel()
        k = int(np.argmin((glon - lon) ** 2 + (glat - lat) ** 2))
        return k    

def make_grid(
    center_lat: float = dome_location["lat"],
    center_lon: float = dome_location["lon"],
    node_spacing_m: int = 100,
    grid_size_lat_m: int = 10_000,
    grid_size_lon_m: int = 8_000,
    *,
    dem: Optional[Tuple[str, Dict[str, Any]]] = None,
) -> Grid:
    """
    Create a Grid centered at (center_lat, center_lon) with the given physical extent.

    Examples
    --------
    # Flat grid (old behavior)
    g = make_grid(node_spacing_m=50, grid_size_lat_m=8000, grid_size_lon_m=8000)

    # Drape on PyGMT Earth Relief (01s), cached under ./_dem_cache
    g = make_grid(
        node_spacing_m=50,
        grid_size_lat_m=8000, grid_size_lon_m=8000,
        dem=("pygmt", {"resolution": "01s", "cache_dir": "./_dem_cache", "tag": "01s"})
    )

    # Drape using a local GeoTIFF
    g = make_grid(
        node_spacing_m=50,
        grid_size_lat_m=8000, grid_size_lon_m=8000,
        dem=("geotiff", {"path": "/abs/path/to/dem.tif", "tag": "DEM2020"})
    )

    # Drape using a custom sampler (callable(lat, lon) -> elevation_m)
    g = make_grid(
        dem=("sampler", {"func": my_sampler, "tag": "myDEM"})
    )
    """
    nlat = int(grid_size_lat_m / node_spacing_m) + 1
    nlon = int(grid_size_lon_m / node_spacing_m) + 1
    return Grid(center_lat, center_lon, nlat, nlon, node_spacing_m, dem=dem)


# === Sparse node grid (NodeGrid) ======================
@dataclass(frozen=True)
class NodeGridSignature:
    """
    Signature for a sparse set of nodes. We don't store all points in the
    signature; we use a stable hash of the (lon,lat,optional elev) array.
    """
    n_nodes: int
    approx_spacing_m: float | None
    bbox_lonmin: float
    bbox_lonmax: float
    bbox_latmin: float
    bbox_latmax: float
    has_elevation: bool
    dem_tag: str | None  # e.g., path or short hash of DEM file, if used
    nodes_sha1: str      # sha1 of float64 packed node array for stability

    def as_tuple(self) -> tuple:
        return (self.n_nodes, self.approx_spacing_m, self.bbox_lonmin, self.bbox_lonmax,
                self.bbox_latmin, self.bbox_latmax, self.has_elevation, self.dem_tag, self.nodes_sha1)


class NodeGrid:
    """
    Sparse 'grid' defined by arbitrary nodes. Compatible with ASL plotting and
    caching expectations: provides gridlon, gridlat, optional gridelev, signature(), id, plot(), save(), load().

    Attributes
    ----------
    node_lon : (N,) float array
    node_lat : (N,) float array
    node_elev_m : (N,) float array or None   # elevation above sea level in meters
    approx_spacing_m : float or None         # optional meta for plotting & label
    id : str                                 # stable cache key from signature()
    """
    def __init__(
        self,
        node_lon: np.ndarray,
        node_lat: np.ndarray,
        node_elev_m: np.ndarray | None = None,
        approx_spacing_m: float | None = None,
        dem_tag: str | None = None,
    ):
        node_lon = np.asarray(node_lon, float).reshape(-1)
        node_lat = np.asarray(node_lat, float).reshape(-1)
        if node_lon.size != node_lat.size:
            raise ValueError("node_lon and node_lat must have same length")
        if node_elev_m is not None:
            node_elev_m = np.asarray(node_elev_m, float).reshape(-1)
            if node_elev_m.size != node_lon.size:
                raise ValueError("node_elev_m must match node_lon size")

        self.node_lon = node_lon
        self.node_lat = node_lat
        self.node_elev_m = node_elev_m
        self.approx_spacing_m = None if approx_spacing_m is None else float(approx_spacing_m)
        self.dem_tag = dem_tag

        # For compatibility with existing code that expects 2-D fields:
        self.gridlon = node_lon.copy()
        self.gridlat = node_lat.copy()

        self.id = self.set_id()

    def get_mask_indices(self) -> np.ndarray | None:
        # NodeGrid usually doesn’t use a raster mask; return None to indicate “no mask”.
        return None

    @property
    def shape(self) -> tuple[int, int] | None:
        return None

    def _nodes_sha1(self) -> str:
        """Stable hash of concatenated (lon, lat, elev?) in float64."""
        arrs = [self.node_lon.astype("float64"), self.node_lat.astype("float64")]
        if self.node_elev_m is not None:
            arrs.append(self.node_elev_m.astype("float64"))
        blob = np.ascontiguousarray(np.column_stack(arrs)).view("uint8")
        return hashlib.sha1(blob).hexdigest()

    def signature(self) -> NodeGridSignature:
        lonmin, lonmax = float(np.nanmin(self.node_lon)), float(np.nanmax(self.node_lon))
        latmin, latmax = float(np.nanmin(self.node_lat)), float(np.nanmax(self.node_lat))
        return NodeGridSignature(
            n_nodes=int(self.node_lon.size),
            approx_spacing_m=self.approx_spacing_m,
            bbox_lonmin=lonmin, bbox_lonmax=lonmax,
            bbox_latmin=latmin, bbox_latmax=latmax,
            has_elevation=self.node_elev_m is not None,
            dem_tag=self.dem_tag,
            nodes_sha1=self._nodes_sha1(),
        )

    def set_id(self) -> str:
        self.id = make_hash(self.signature().as_tuple())
        return self.id

    # --- Plot (re-uses topo_map if available) ---
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
        Plot sparse nodes on a simple two-color land/sea basemap by default.
        If this NodeGrid carries a DEM tag from a GeoTIFF build, pass it through.
        """
        try:
            import pygmt  # type: ignore
        except Exception as e:
            raise ImportError("PyGMT is required for NodeGrid.plot().") from e

        default_kwargs = {
            "topo_color": False,
            "cmap": "land",     # no shading two-color scheme in map.py
        }

        # If created from channel_finder with a DEM, we stored basename in dem_tag;
        # when possible, prefer a direct path via topo_map_kwargs['dem_tif'] supplied by caller.
        # (We can't reliably reconstruct a full path from dem_tag alone.)
        topo_map_kwargs = {**default_kwargs, **(topo_map_kwargs or {})}

        if fig is None:
            try:
                from .map import topo_map
                fig = topo_map(show=False, **topo_map_kwargs)
            except Exception:
                region = [
                    float(np.nanmin(self.node_lon)), float(np.nanmax(self.node_lon)),
                    float(np.nanmin(self.node_lat)), float(np.nanmax(self.node_lat)),
                ]
                fig = pygmt.Figure()
                fig.basemap(region=region, projection="M12c", frame=True)

        # Symbol size (cm) – use approx spacing if available
        if self.approx_spacing_m:
            size_cm = (self.approx_spacing_m / 2000.0) * float(scale)
        else:
            size_cm = 0.12 * float(scale)
        stylestr = f"{symbol}{size_cm}c"

        fig.plot(
            x=self.node_lon,
            y=self.node_lat,
            style=stylestr,
            pen=pen,
            fill=fill if symbol not in ("x", "+") else None,
        )

        if outfile:
            fig.savefig(outfile, dpi=300)
        if show:
            fig.show()
        return fig

    # --- Persistence ---
    def save(self, cache_dir: str, force_overwrite: bool = False) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        pklfile = os.path.join(cache_dir, f"NodeGrid_{self.id}.pkl")
        if os.path.exists(pklfile) and not force_overwrite:
            print(f"[INFO] File exists: {pklfile} (use force_overwrite=True to overwrite).")
            return pklfile
        with open(pklfile, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] NodeGrid saved to {pklfile}")
        return pklfile

    @classmethod
    def load(cls, cache_file: str) -> "NodeGrid":
        if not os.path.isfile(cache_file):
            raise FileNotFoundError(f"No such file: {cache_file}")
        with open(cache_file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle does not contain a {cls.__name__} instance.")
        print(f"[INFO] NodeGrid loaded from {cache_file}")
        return obj
    
    def _slice(self, mask: np.ndarray) -> "NodeGrid":
        """
        Return a NodeGrid with nodes where mask==True.
        """
        mask = np.asarray(mask, bool).ravel()
        if mask.size != self.node_lon.size:
            raise ValueError("mask length != number of nodes")
        lon = self.node_lon[mask]
        lat = self.node_lat[mask]
        elev = self.node_elev_m[mask] if self.node_elev_m is not None else None
        return NodeGrid(
            node_lon=lon,
            node_lat=lat,
            node_elev_m=elev,
            approx_spacing_m=self.approx_spacing_m,
            dem_tag=self.dem_tag,
        )
    
    '''
    replace with snap_to_grid
    def nodegrid_to_masked_grid(
        nodegrid: "NodeGrid",
        template: "Grid",
        *,
        tol_cells: float = 0.5,     # max offset allowed between node and snapped cell, in grid-cell units
        set_on_template: bool = True
    ) -> tuple["Grid", np.ndarray]:
        """
        Snap NodeGrid nodes to a rectangular template Grid and create a mask of used cells.

        Parameters
        ----------
        nodegrid : NodeGrid
            Sparse nodes (lon, lat[, elev]).
        template : Grid
            Regular raster to map onto. Must have latrange/lonrange with uniform spacing.
        tol_cells : float
            Maximum allowed angular offset between a node and the snapped grid cell,
            expressed as a fraction of the grid spacing. 0.5 means within half a cell.
        set_on_template : bool
            If True, store the flat masked indices on `template._node_mask_idx` for
            downstream use; otherwise return indices only.

        Returns
        -------
        masked_grid : Grid
            The same template object (for convenience). If `set_on_template=True`,
            it has an attribute `_node_mask_idx` (1-D array of flat indices).
        used_idx : np.ndarray
            Sorted unique flat indices (into template’s flattened grid) that are used.



        # 1) Build your regular grid (with or without DEM sampling/masking)
        grid = make_grid(center_lat=..., center_lon=..., node_spacing_m=..., dem=("geotiff", {"path": dem_tif, "tag": "DEM"}))
        # optionally: grid.apply_land_mask_from_dem(sea_level=0.0)   # if you added this earlier

        # 2) Build/obtain a NodeGrid (e.g., channels/streams)
        nodegrid = nodegrid_from_channel_csvs(channels_dir=..., step_m=100, dem_tif=dem_tif)

        # 3) Snap NodeGrid to template Grid and carry mask indices
        grid, used_idx = nodegrid_to_masked_grid(nodegrid, grid, tol_cells=0.5, set_on_template=True)

        # 4) For any sparse result vector R coming from NodeGrid ops, backfill into a raster:
        R_sparse = np.random.rand(used_idx.size)   # example
        full = np.full(grid.nlat * grid.nlon, np.nan, float)
        full[used_idx] = R_sparse
        R_raster = full.reshape(grid.nlat, grid.nlon)

        # Now you can plot with grdimage / heatmaps against `grid` normally.
        """
        # grid spacing (degrees)
        dlat = float(template.latrange[1] - template.latrange[0]) if template.nlat > 1 else np.nan
        dlon = float(template.lonrange[1] - template.lonrange[0]) if template.nlon > 1 else np.nan
        if not np.isfinite(dlat) or not np.isfinite(dlon) or dlat <= 0 or dlon <= 0:
            raise ValueError("Template grid must have >=2 points per axis with uniform spacing.")

        lat0 = float(template.latrange[0])
        lon0 = float(template.lonrange[0])
        nlat, nlon = int(template.nlat), int(template.nlon)

        # snap each node to nearest grid row/col by index
        lon = np.asarray(nodegrid.node_lon, float).ravel()
        lat = np.asarray(nodegrid.node_lat, float).ravel()

        j = np.rint((lon - lon0) / dlon).astype(int)  # columns
        i = np.rint((lat - lat0) / dlat).astype(int)  # rows

        # reject nodes that fall outside the grid bounds
        inside = (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon)

        # also enforce a tolerance in *angular* offset relative to the snapped centers
        lat_snapped = lat0 + i * dlat
        lon_snapped = lon0 + j * dlon
        d_i = np.abs((lat - lat_snapped) / dlat)   # in cell units
        d_j = np.abs((lon - lon_snapped) / dlon)   # in cell units
        close = (d_i <= tol_cells) & (d_j <= tol_cells)

        ok = inside & close
        if not np.any(ok):
            raise ValueError("No NodeGrid points fall within the template grid (or tol too small).")

        # flat indices into raster
        flat_idx = (i[ok] * nlon + j[ok]).astype(int)
        used_idx = np.unique(flat_idx)  # de-duplicate collisions

        # optionally stash on the grid so raster code can backfill sparse vectors
        if set_on_template:
            setattr(template, "_node_mask_idx", used_idx)

        return template, used_idx
    '''

    def snap_to_grid(
        self,
        template: "Grid",
        *,
        tol_cells: float = 0.5,
        set_on_template: bool = True,
    ) -> np.ndarray:
        """
        Snap this NodeGrid onto a rectangular template Grid and compute the
        flat indices of used raster cells. Optionally stash them on the Grid.

        Returns
        -------
        used_idx : np.ndarray
            Sorted, unique flat indices into template’s flattened grid.
        """
        # --- grid spacing (degrees) ---
        dlat = float(template.latrange[1] - template.latrange[0]) if template.nlat > 1 else np.nan
        dlon = float(template.lonrange[1] - template.lonrange[0]) if template.nlon > 1 else np.nan
        if not np.isfinite(dlat) or not np.isfinite(dlon) or dlat <= 0 or dlon <= 0:
            raise ValueError("Template grid must have >= 2 points per axis with uniform spacing.")

        lat0 = float(template.latrange[0])
        lon0 = float(template.lonrange[0])
        nlat, nlon = int(template.nlat), int(template.nlon)

        # --- snap each node to nearest row/col ---
        lon = np.asarray(self.node_lon, float).ravel()
        lat = np.asarray(self.node_lat, float).ravel()

        j = np.rint((lon - lon0) / dlon).astype(int)  # columns
        i = np.rint((lat - lat0) / dlat).astype(int)  # rows

        # inside bounds?
        inside = (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon)

        # tolerance relative to cell center (in cell units)
        lat_snapped = lat0 + i * dlat
        lon_snapped = lon0 + j * dlon
        d_i = np.abs((lat - lat_snapped) / dlat)
        d_j = np.abs((lon - lon_snapped) / dlon)
        close = (d_i <= tol_cells) & (d_j <= tol_cells)

        ok = inside & close
        if not np.any(ok):
            raise ValueError("No NodeGrid points fall within the template grid (or tol too small).")

        flat_idx = (i[ok] * nlon + j[ok]).astype(int)
        used_idx = np.unique(flat_idx)

        if set_on_template:
            setattr(template, "_node_mask_idx", used_idx)

        return used_idx


    def mask_grid_with_nodes(
        self,
        grid,
        *,
        k: int = 1,
        max_m: float = 20.0,
        flatten_copy: bool = True,
        mask_name: str = "channels_only",   # kept for symmetry; not required by Grid
        return_matches: bool = False,
    ):
        """
        Horizontally match this NodeGrid's nodes to the nearest Grid nodes and keep ONLY those grid nodes.

        Parameters
        ----------
        grid : Grid
            Target grid (regular lat/lon with one node per (lon,lat)); must expose
            grid.gridlon (2-D), grid.gridlat (2-D), nlat, nlon, centerlat.
        k : int
            Query up to k nearest grid nodes per NodeGrid node.
        max_m : float
            Keep neighbors within this horizontal distance (meters).
        flatten_copy : bool
            If True, operate on a deepcopy of the grid and zero-out its node_elev_m.
        mask_name : str
            Unused here (placeholder for naming); masking is applied directly to Grid.
        return_matches : bool
            If True, also return a tidy DataFrame of matches.

        Returns
        -------
        grid_out : Grid
            Masked grid (copy if flatten_copy; otherwise original).
        mask2d : np.ndarray
            2D boolean mask (True=kept node).
        matches : pd.DataFrame   (if return_matches=True)
        """

        # --- 0) Validate NodeGrid coords ---
        lon = np.asarray(getattr(self, "node_lon", None), float)
        lat = np.asarray(getattr(self, "node_lat", None), float)
        if lon is None or lat is None or lon.size == 0 or lon.shape != lat.shape:
            raise ValueError("NodeGrid.node_lon/node_lat must exist and have same non-empty shape.")

        # --- 1) Pull grid geometry directly from Grid() as provided ---
        if not hasattr(grid, "gridlon") or not hasattr(grid, "gridlat"):
            raise RuntimeError("Grid must expose gridlon/gridlat 2-D arrays.")
        glon2d = np.asarray(grid.gridlon, float)
        glat2d = np.asarray(grid.gridlat, float)
        if glon2d.shape != glat2d.shape:
            raise RuntimeError("grid.gridlon and grid.gridlat must have the same 2-D shape.")
        ny, nx = glon2d.shape

        # raveled lon/lat and global flat indices
        glon = glon2d.ravel()
        glat = glat2d.ravel()
        base_flat_idx = np.arange(glon.size, dtype=int)

        # finite only (should be all, but be safe)
        finite = np.isfinite(glon) & np.isfinite(glat)
        glon = glon[finite]; glat = glat[finite]; base_flat_idx = base_flat_idx[finite]
        if glon.size == 0:
            raise ValueError("Grid has no finite nodes to index.")

        # Deduplicate horizontally (in case of any identical lon/lat)
        pairs = np.column_stack([glon, glat])
        uniq_pairs, uniq_idx = np.unique(pairs, axis=0, return_index=True)
        u_glon = uniq_pairs[:, 0]
        u_glat = uniq_pairs[:, 1]
        flat_idx_u = base_flat_idx[uniq_idx]   # KD index -> global flat index (ny*nx)

        # --- 2) Local ENU-like linearization around grid centerlat (or mean) ---
        try:
            lat0 = float(getattr(grid, "centerlat", np.nan))
        except Exception:
            lat0 = np.nan
        if not np.isfinite(lat0):
            lat0 = float(np.nanmean(u_glat))
        # meters-per-degree at reference latitude
        m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat0)

        def _to_xy(lon_deg, lat_deg):
            x = (np.asarray(lon_deg, float) - float(np.nanmean(u_glon))) * m_per_deg_lon
            y = (np.asarray(lat_deg, float) - float(np.nanmean(u_glat))) * m_per_deg_lat
            return x, y

        gx, gy = _to_xy(u_glon, u_glat)
        tree = cKDTree(np.column_stack([gx, gy]))

        # --- 3) Query for NodeGrid nodes ---
        px, py = _to_xy(lon, lat)
        dists, idxs = tree.query(np.column_stack([px, py]), k=k, workers=-1)
        dists = np.atleast_2d(dists)
        idxs  = np.atleast_2d(idxs)

        keep_flat = []
        if return_matches:
            out_rows = []

        for i in range(lon.size):
            for rank in range(dists.shape[1]):
                d = float(dists[i, rank])
                if not np.isfinite(d) or d > max_m:
                    continue
                kd_i   = int(idxs[i, rank])            # index in unique set
                gi_flat = int(flat_idx_u[kd_i])        # global flat index (ny*nx)
                keep_flat.append(gi_flat)
                if return_matches:
                    out_rows.append({
                        "ng_idx": i,
                        "node_lon": float(lon[i]),
                        "node_lat": float(lat[i]),
                        "grid_idx": gi_flat,
                        "grid_lon": float(u_glon[kd_i]),
                        "grid_lat": float(u_glat[kd_i]),
                        "dist_m": d,
                        "rank": rank + 1,
                    })

        keep_flat = np.unique(np.asarray(keep_flat, dtype=int))

        # --- 4) Build keep-only mask2d (True=keep) ---
        mask1d = np.zeros(ny * nx, dtype=bool)
        valid = keep_flat[(keep_flat >= 0) & (keep_flat < mask1d.size)]
        mask1d[valid] = True
        mask2d = mask1d.reshape(ny, nx)

        # --- 5) Optionally flatten a copy; then apply the mask ---
        import copy as _copy
        grid_out = _copy.deepcopy(grid) if flatten_copy else grid
        # flatten elevations
        if hasattr(grid_out, "node_elev_m"):
            try:
                # ensure array exists and has correct shape
                if (getattr(grid_out, "node_elev_m") is None or
                    np.asarray(grid_out.node_elev_m).shape != (ny, nx)):
                    grid_out.node_elev_m = np.zeros((ny, nx), dtype=float)
                else:
                    grid_out.node_elev_m = np.zeros_like(np.asarray(grid_out.node_elev_m, dtype=float))
            except Exception:
                pass  # non-fatal if elevation storage differs

        # Preferred: use Grid.apply_mask_indices if available
        if hasattr(grid_out, "apply_mask_indices"):
            grid_out.apply_mask_indices(keep_flat, validate=False)
        else:
            # fallback: set internal mask fields if present (True=keep)
            if hasattr(grid_out, "_node_mask"):
                grid_out._node_mask = mask2d.copy()
            if hasattr(grid_out, "_node_mask_idx"):
                grid_out._node_mask_idx = keep_flat.copy()

        if return_matches:
            matches = pd.DataFrame(out_rows, columns=[
                "ng_idx","node_lon","node_lat","grid_idx","grid_lon","grid_lat","dist_m","rank"
            ])
            return grid_out, mask2d, matches
        else:
            return grid_out, mask2d






# --- Builders from channel_finder outputs ------------------------------------

def _resample_polyline_lonlat(lon: np.ndarray, lat: np.ndarray, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a lon/lat polyline to ~uniform spacing (meters) along track.
    Uses local meters-per-degree at each segment mid-lat for speed.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    if lon.size < 2:
        return lon, lat

    # cumulative distance (meters) along polyline
    meters = [0.0]
    for i in range(1, lon.size):
        latm = 0.5 * (lat[i] + lat[i-1])
        m_per_deg_lat = 111_194.9266
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(latm))
        dx = (lon[i] - lon[i-1]) * m_per_deg_lon
        dy = (lat[i] - lat[i-1]) * m_per_deg_lat
        meters.append(meters[-1] + math.hypot(dx, dy))
    meters = np.asarray(meters)

    if meters[-1] <= step_m:
        return lon, lat

    # target distances and interpolation
    tgt = np.arange(0.0, meters[-1] + step_m, step_m)
    lon_i = np.interp(tgt, meters, lon)
    lat_i = np.interp(tgt, meters, lat)
    return lon_i, lat_i


def sample_dem_elevations(dem_tif: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Sample DEM elevations (meters) at lon/lat points. Returns NaN for nodata/outside.
    """
    import rasterio
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    z = np.full(lon.shape, np.nan, float)
    with rasterio.open(dem_tif) as src:
        for i, (x, y) in enumerate(zip(lon, lat)):
            try:
                val = list(src.sample([(x, y)], indexes=1))[0][0]
                nod = src.nodata
                z[i] = np.nan if (val == nod or not np.isfinite(val)) else float(val)
            except Exception:
                z[i] = np.nan
    return z


def nodegrid_from_channel_csvs(
    channels_dir: str,
    *,
    step_m: float | None = 100.0,
    dem_tif: str | None = None,
    approx_spacing_m: float | None = None,
    max_points: int | None = None,
) -> NodeGrid:
    """
    Build a NodeGrid from the CSV polylines written by channel_finder.write_top_n_channels_csv().
    - Reads all *.csv under channels_dir (columns: lon,lat)
    - Optionally resamples each polyline to ~step_m spacing
    - Optionally samples a DEM for elevations (meters)

    Returns NodeGrid (sparse grid).
    """
    all_lon, all_lat = [], []
    for name in sorted(os.listdir(channels_dir)):
        if not name.lower().endswith(".csv"):
            continue
        arr = np.loadtxt(os.path.join(channels_dir, name), delimiter=",", skiprows=1)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        lon, lat = arr[:, 0], arr[:, 1]
        if step_m and step_m > 0:
            lon, lat = _resample_polyline_lonlat(lon, lat, step_m)
        all_lon.append(lon); all_lat.append(lat)

    if not all_lon:
        raise ValueError(f"No channel CSVs found in {channels_dir}")

    lon = np.concatenate(all_lon); lat = np.concatenate(all_lat)

    if max_points and lon.size > max_points:
        idx = np.linspace(0, lon.size - 1, num=max_points, dtype=int)
        lon, lat = lon[idx], lat[idx]

    elev = None
    dem_tag = None
    if dem_tif:
        elev = sample_dem_elevations(dem_tif, lon, lat)
        dem_tag = os.path.basename(dem_tif)

    # choose a reasonable spacing meta
    approx = approx_spacing_m if approx_spacing_m is not None else (step_m or None)
    return NodeGrid(lon, lat, node_elev_m=elev, approx_spacing_m=approx, dem_tag=dem_tag)


# ---------------------------------
# DEM samplers
# ---------------------------------
def _dem_sample_pygmt(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    *,
    resolution: str = "01s",
    cache_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> np.ndarray:
    """
    Sample GMT Earth Relief at given (lat2d, lon2d). Returns elevations (m).
    Caches a small pickled grid subset for speed if cache_dir is given.
    """

    latv = np.asarray(lat2d, float); lonv = np.asarray(lon2d, float)
    minlon, maxlon = float(np.nanmin(lonv)), float(np.nanmax(lonv))
    minlat, maxlat = float(np.nanmin(latv)), float(np.nanmax(latv))
    region = [minlon, maxlon, minlat, maxlat]

    # Optional cache of the xarray DataArray
    key = f"EarthRelief_{resolution}_{minlon:.4f}_{maxlon:.4f}_{minlat:.4f}_{maxlat:.4f}_{tag or ''}".replace(" ", "")
    cache_path = os.path.join(cache_dir, f"{key}.pkl") if cache_dir else None

    da = None
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                da = pickle.load(f)
        except Exception:
            da = None

    if da is None:
        da = pygmt.datasets.load_earth_relief(resolution=resolution, region=region, registration=None)
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(da, f)

    # Interpolate to node coordinates
    try:
        import xarray as xr  # noqa: F401
        elev = da.interp(lon=("points", lonv.ravel()), lat=("points", latv.ravel()), method="linear").values
        elev = elev.reshape(latv.shape)
    except Exception:
        lon_coords = np.asarray(da.lon.values)
        lat_coords = np.asarray(da.lat.values)
        j = np.searchsorted(lon_coords, lonv.ravel(), side="left")
        i = np.searchsorted(lat_coords, latv.ravel(), side="left")
        j = np.clip(j, 0, len(lon_coords) - 1)
        i = np.clip(i, 0, len(lat_coords) - 1)
        elev = da.values[i, j].reshape(latv.shape)

    return np.asarray(elev, float)


def _dem_sample_geotiff(lat2d: np.ndarray, lon2d: np.ndarray, tif_path: str) -> np.ndarray:
    """
    Sample elevations (meters) from a GeoTIFF at (lat, lon) points.

    - Accepts DEMs in any CRS; reprojects lon/lat (EPSG:4326) to the DEM's CRS if needed.
    - Returns an array shaped like lat2d with float elevations (NaN for nodata/outside).
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform as rio_transform

    latv = np.asarray(lat2d, float)
    lonv = np.asarray(lon2d, float)
    out = np.full(latv.size, np.nan, float)  # we'll reshape at the end

    # Flatten query points once
    lats = latv.ravel()
    lons = lonv.ravel()

    with rasterio.open(tif_path) as src:
        nod = src.nodata

        # Make sure query coords match the raster CRS
        if src.crs and src.crs.to_epsg() not in (4326,):  # DEM not in EPSG:4326
            xs, ys = rio_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        else:
            xs, ys = lons.tolist(), lats.tolist()

        # rasterio.sample expects iterable of (x, y)
        samples = src.sample(zip(xs, ys), indexes=1)

        # Fill output, honoring nodata
        for i, (val,) in enumerate(samples):
            if val is None or not np.isfinite(val) or (nod is not None and val == nod):
                out[i] = np.nan
            else:
                out[i] = float(val)

    return out.reshape(latv.shape)


def _dem_sample_callable(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray] | Callable[[float, float], float],
) -> np.ndarray:
    """
    Sample elevations using a user-provided function.
    - If `func` is vectorized (accepts array lat, lon), use directly.
    - Else call pointwise.
    """
    try:
        z = func(lat2d, lon2d)  # type: ignore[arg-type]
        z = np.asarray(z, float)
        if z.shape != lat2d.shape:
            raise ValueError("Sampler returned unexpected shape.")
        return z
    except Exception:
        out = np.empty_like(lat2d, dtype=float)
        it = np.nditer(lat2d, flags=["multi_index"])
        while not it.finished:
            i, j = it.multi_index
            out[i, j] = float(func(float(lat2d[i, j]), float(lon2d[i, j])))  # type: ignore[call-arg]
            it.iternext()
        return out


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


def _meters_per_degree(centerlat_deg: float) -> Tuple[float, float]:
    """Return (meters_per_deg_lat, meters_per_deg_lon) at the given latitude."""
    meters_per_deg_lat = 1000.0 * 111.19492664455873  # ~exact degrees2kilometers(1)
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(centerlat_deg))
    return meters_per_deg_lat, meters_per_deg_lon


def _safe_to_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)
    

def _lonlat_to_local_xy(lon, lat, lon0, lat0):
    """
    Convert lon/lat (deg) to local ENU-like meters using spherical scale at lat0.
    Good for ~tens of km. meters_per_degree_fn(lat0) -> (m_per_deg_lat, m_per_deg_lon).
    """
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(lat0)  # already in your module
    x = (np.asarray(lon, float) - lon0) * m_per_deg_lon
    y = (np.asarray(lat, float) - lat0) * m_per_deg_lat
    return x, y


def _build_grid_kdtree_2d(grid):
    """
    KD-tree over UNIQUE (lon,lat) nodes; returns (tree, meta) where meta includes:
      ny, nx, lon0, lat0, glon, glat, flat_idx_u (GLOBAL flat indices ny*nx)
    """
    if hasattr(grid, "lon2d") and hasattr(grid, "lat2d"):
        lon2d = np.asarray(grid.lon2d, float)
        lat2d = np.asarray(grid.lat2d, float)
        ny, nx = lon2d.shape
        glon = lon2d.ravel(); glat = lat2d.ravel()
        base_flat_idx = np.arange(glon.size, dtype=int)  # global flat indices
    else:
        glon, glat = grid.masked_lonlat()  # fallback
        glon = np.asarray(glon, float).ravel()
        glat = np.asarray(glat, float).ravel()
        if hasattr(grid, "lon2d"):
            ny, nx = np.asarray(grid.lon2d).shape
        else:
            raise RuntimeError("Cannot infer grid shape; need lon2d/lat2d on Grid.")
        base_flat_idx = np.arange(glon.size, dtype=int)

    finite = np.isfinite(glon) & np.isfinite(glat)
    glon, glat, base_flat_idx = glon[finite], glat[finite], base_flat_idx[finite]
    if glon.size == 0:
        raise ValueError("Grid has no finite nodes to index.")

    pairs = np.column_stack([glon, glat])
    uniq_pairs, uniq_idx = np.unique(pairs, axis=0, return_index=True)
    u_glon = uniq_pairs[:, 0]
    u_glat = uniq_pairs[:, 1]
    flat_idx_u = base_flat_idx[uniq_idx]

    lon0, lat0 = float(np.nanmean(u_glon)), float(np.nanmean(u_glat))
    gx, gy = _lonlat_to_local_xy(u_glon, u_glat, lon0, lat0)
    tree = cKDTree(np.column_stack([gx, gy]))

    meta = dict(
        ny=int(ny), nx=int(nx),
        lon0=lon0, lat0=lat0,
        glon=u_glon, glat=u_glat,
        flat_idx_u=flat_idx_u,   # maps KD index -> global flat index
    )
    return tree, meta


def _apply_keep_mask_2d(grid, keep_flat, mask_name="channels_only"):
    """Build keep-only 2D mask from flat indices and apply it to grid. Returns (grid_masked, mask2d)."""
    if hasattr(grid, "lon2d"):
        ny, nx = np.asarray(grid.lon2d).shape
    elif hasattr(grid, "shape") and len(grid.shape) >= 2:
        ny, nx = int(grid.shape[0]), int(grid.shape[1])
    else:
        raise RuntimeError("Cannot infer grid 2D shape.")

    keep_flat = np.unique(np.asarray(keep_flat, dtype=int))
    mask1d = np.zeros(ny * nx, dtype=bool)
    valid = keep_flat[(keep_flat >= 0) & (keep_flat < mask1d.size)]
    mask1d[valid] = True
    mask2d = mask1d.reshape(ny, nx)

    grid_masked = grid
    applied = False
    if hasattr(grid, "add_mask"):
        try:
            grid.add_mask(mask2d, name=mask_name, mode="keep")
        except TypeError:
            grid.add_mask(mask2d, name=mask_name)
        applied = True
    if not applied and hasattr(grid, "set_mask"):
        try:
            grid.set_mask(mask2d, keep=True)
        except TypeError:
            grid.set_mask(mask2d)
        applied = True
    if not applied and hasattr(grid, "with_mask"):
        grid_masked = grid.with_mask(mask2d)
        applied = True
    if not applied and hasattr(grid, "mask"):
        grid.mask = mask2d
        applied = True
    if not applied:
        raise RuntimeError("Could not apply mask: unknown Grid mask API.")
    return grid_masked, mask2d


def apply_channel_land_circle_mask(
    grid: "Grid",
    nodegrid: "NodeGrid",
    *,
    k: int = 4,
    max_m: float = 20.0,
    dome_location: dict | None = None,
    radius_km: float = 1.0,
) -> "Grid":
    """
    Keep nodes: within radius_km of dome  OR  (on land AND along channel).

    - Uses grid.apply_land_mask_from_dem() for land
    - Uses nodegrid.mask_grid_with_nodes(..., flatten_copy=False) to get channel mask2d
    - Builds the circle mask in meters
    - Applies final mask with grid.apply_mask_boolean(...)

    Returns the same grid (mutated) for convenience.
    """
    import numpy as np

    # 1) Land mask (True=keep)
    land_mask = grid.apply_land_mask_from_dem(sea_level=0.0)  # shape (nlat, nlon)

    # 2) Channel mask via NodeGrid → ask for mask2d but DO NOT flatten or write into grid
    _, channel_mask2d = nodegrid.mask_grid_with_nodes(
        grid,
        k=k,
        max_m=max_m,
        flatten_copy=False,      # don't zero elevations, don't write a new copy
        return_matches=False,    # we only need the mask
    )

    # 3) Circle mask around dome (True inside circle)
    centerlon = getattr(grid, "centerlon", float(np.nan))
    centerlat = getattr(grid, "centerlat", float(np.nan))
    if dome_location and "lon" in dome_location and "lat" in dome_location:
        centerlon = float(dome_location["lon"])
        centerlat = float(dome_location["lat"])

    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(grid.centerlat)
    dlat_m = (grid.gridlat - centerlat) * m_per_deg_lat
    dlon_m = (grid.gridlon - centerlon) * m_per_deg_lon
    dist_m = np.hypot(dlat_m, dlon_m)
    circle_mask = dist_m <= (float(radius_km) * 1000.0)

    # (optional) some quick stats
    n_all = grid.gridlat.size
    kept_circle = int(circle_mask.sum())
    kept_land   = int(land_mask.sum())
    kept_chan   = int(channel_mask2d.sum())
    print(f"[CIRCLE] keep {kept_circle}/{n_all} ({100*kept_circle/n_all:.1f}%)")
    print(f"[LAND  ] keep {kept_land}/{n_all} ({100*kept_land/n_all:.1f}%)")
    print(f"[CHAN  ] keep {kept_chan}/{n_all} ({100*kept_chan/n_all:.1f}%)")

    # 4) Boolean recipe: within_circle OR (on_land AND on_channel)
    final_mask = circle_mask | (land_mask & channel_mask2d)

    # 5) Apply once (True = keep)
    grid.apply_mask_boolean(final_mask, validate=True, mode="replace")

    # Optional: show final mask stats
    kept_final = int(final_mask.sum())
    print(f"[FINAL ] keep {kept_final}/{n_all} ({100*kept_final/n_all:.1f}%)")

    return grid

import numpy as np
import pandas as pd
from typing import Dict, Optional

def summarize_station_node_distances(
    node_distances_km: Dict[str, np.ndarray],
    *,
    reduce_to_station: bool = False,   # True → aggregate NET.STA across channels
    include_median: bool = True,       # add p50
    sort_by: str = "min_km",           # "min_km" | "max_km" | "station"
    to_csv: Optional[str] = None       # path to save the table (optional)
) -> pd.DataFrame:
    """
    Build a summary table of min/median/max station→node distances (km).

    Parameters
    ----------
    node_distances_km : dict[seed_id -> np.ndarray]
        Each array is length n_nodes (dense), with masked nodes as NaN.
    reduce_to_station : bool
        If True, collapse multiple channels to a single NET.STA row by
        aggregating over all channels of that station.
    include_median : bool
        If True, include p50 column.
    sort_by : str
        Column to sort by.
    to_csv : str | None
        If provided, save the resulting table to this path.

    Returns
    -------
    pd.DataFrame with columns:
        - station  (seed_id or NET.STA if reduced)
        - n_total  (total nodes)
        - n_finite (unmasked nodes per station)
        - pct_masked
        - min_km, [median_km], max_km
    """
    def _seed_to_station(seed: str) -> str:
        # seed like NET.STA.LOC.CHA -> NET.STA
        parts = seed.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else seed

    rows = []
    n_total = None

    for sid, d in node_distances_km.items():
        d = np.asarray(d, dtype=float).ravel()
        if n_total is None:
            n_total = d.size
        finite = np.isfinite(d)
        n_finite = int(finite.sum())

        if n_finite == 0:
            min_km = np.nan
            med_km = np.nan
            max_km = np.nan
        else:
            vals = d[finite]
            min_km = float(np.nanmin(vals))
            max_km = float(np.nanmax(vals))
            med_km = float(np.nanmedian(vals)) if include_median else np.nan

        rows.append({
            "station": sid,
            "n_total": n_total,
            "n_finite": n_finite,
            "pct_masked": 0.0 if n_total == 0 else float(100.0 * (n_total - n_finite) / n_total),
            "min_km": min_km,
            **({"median_km": med_km} if include_median else {}),
            "max_km": max_km,
        })

    df = pd.DataFrame(rows)

    if reduce_to_station:
        # Group by NET.STA, aggregate across channels
        key = df["station"].map(_seed_to_station)
        agg = {
            "n_total": "max",                     # same for all rows
            "n_finite": "sum",                    # sum of finite across channels
            "pct_masked": "mean",                 # not strictly meaningful; keep for rough sense
            "min_km": "min",
            "max_km": "max",
        }
        if include_median:
            # median across channels of their per-station medians (robust enough)
            agg["median_km"] = "median"
        df["station"] = key
        df = df.groupby("station", as_index=False).agg(agg)

    # Sorting
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
    else:
        df = df.sort_values("station").reset_index(drop=True)

    if to_csv:
        df.to_csv(to_csv, index=False)

    return df