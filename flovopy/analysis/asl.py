# Standard Library
import os
import math
import glob
import pickle
from pprint import pprint

# Scientific Stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyGMT for mapping and relief data
import pygmt
pygmt.config(GMT_DATA_SERVER="https://oceania.generic-mapping-tools.org")

# ObsPy core and event tools
import obspy
from obspy import UTCDateTime, read_events, Inventory
from obspy.core.event import Event, Catalog, ResourceIdentifier, Origin, Amplitude, QuantityError, OriginQuality, Comment
from obspy.geodetics import locations2degrees, degrees2kilometers, gps2dist_azimuth
# Your internal or local modules (assumed to exist)
# For example:
from flovopy.stationmetadata.utils import inventory2traceid
from flovopy.processing.sam import VSAM, DSAM  # VSAM class for corrections and simulation
from flovopy.core.mvo import dome_location
from scipy.ndimage import uniform_filter1d


def topo_map(
    show=False,
    zoom_level=0,
    inv=None,
    add_labels=False,
    centerlon=-62.177,
    centerlat=16.711+0.01,  # slight north offset to center plot onto Montserrat, rather than volcano dome
    contour_interval=100,      # legacy: still supported
    topo_color=True,           # legacy: True=color topo, False=grayscale
    resolution="03s",
    DEM_DIR=None,    
    # Not in original version:
    stations=[], 
    title=None, 
    region=None,
    # ---- modern/extended options ----
    levels=None,               # explicit contour levels (list/array or CSV string)
    level_interval=None,       # if set, overrides contour_interval for spacing
    cmap=None,                 # explicit colormap name; if None, uses topo_color to choose
    projection=None,
    azimuth=135,               # hillshade light azimuth (deg)
    elevation=30,              # hillshade light elevation (deg)
    limit=None,                 # e.g., "-1300/1300" to clamp contour display range
    figsize: float = 6.0,       # NEW: map width in inches):
):

    """
    Plot a topographic map of Montserrat using modern PyGMT with backward-compatible options.

    This function wraps PyGMT’s `grdimage` and `grdcontour` to provide a shaded-relief map 
    with optional seismic station overlays. It preserves legacy arguments (`contour_interval`, 
    `topo_color`) for compatibility with older code, while also supporting modern options 
    like `levels` and `cmap`.

    Parameters
    ----------
    show : bool, default=False
        If True, immediately display the map in an interactive viewer.
    zoom_level : int, default=0
        Zoom level scaling for the map extent. Higher values zoom in closer.
    inv : obspy.Inventory or None
        Station inventory. If provided, station locations are plotted as symbols.
    add_labels : bool, default=False
        If True, annotate stations with their codes.
    centerlon, centerlat : float, defaults=(-62.177, 16.711)
        Map center in decimal degrees (Montserrat coordinates).
    contour_interval : int, default=100
        **Legacy option.** Contour spacing in meters when `levels` is not specified.
    topo_color : bool, default=True
        **Legacy option.** If True, use a hypsometric color map ("topo"). 
        If False, use grayscale shaded relief.
    resolution : str, default="03s"
        Resolution of the topography grid. Options include "30s", "15s", "03s", etc.
    DEM_DIR : str or None
        Directory for caching Earth relief grids as pickled files. If None, caching is skipped.

    levels : list, array, str, or None, default=None
        **Modern option.** Explicit contour levels (e.g., [-1200,-800,-400,0,400,800,1200]) 
        or comma-separated string ("0,100,200,..."). Overrides `contour_interval`.
    level_interval : int or None, default=None
        **Modern option.** Contour spacing. Used if `levels` is None. Overrides `contour_interval` 
        if both are provided.
    cmap : str or None, default=None
        Colormap for topography. If None, chosen automatically from `topo_color` 
        ("topo" if True, "gray" if False).
    projection : str, default="M4i"
        GMT projection string, e.g., "M4i" for Mercator with 4-inch width.
    azimuth : float, default=135
        Azimuth (direction of illumination) for shaded relief in degrees.
    elevation : float, default=30
        Elevation angle of illumination in degrees.
    limit : str or None, default=None
        Depth/elevation range to display contours, e.g., "-1300/1300".
    figsize : float, default=6.0
        Width of the map in inches. The height is scaled automatically to match the region aspect ratio.
    

    Returns
    -------
    fig : pygmt.Figure
        The PyGMT figure object containing the rendered map.

    Notes
    -----
    - Hillshading is applied using `pygmt.grdgradient` with `radiance=[azimuth, elevation]`.
    - Contours are drawn with either `levels` (modern) or `contour_interval`/`level_interval` (legacy).
    - Colorbar is shown only if a non-grayscale colormap is used.
    - This function is backward-compatible with older calls that used 
      `contour_interval` and `topo_color`.

    Examples
    --------
    Basic (legacy-style): use hypsometric colors and 100 m contour spacing
    >>> fig = topo_map(
    ...     show=False,
    ...     contour_interval=100,   # legacy spacing
    ...     topo_color=True,        # legacy color map toggle → uses "topo"
    ...     resolution="03s",
    ... )
    >>> fig.show()

    Grayscale shaded relief (legacy-style toggle)
    >>> fig = topo_map(
    ...     show=False,
    ...     contour_interval=100,
    ...     topo_color=False,       # grayscale instead of color
    ... )
    >>> fig.show()

    Modern: explicit contour levels and a custom colormap
    >>> fig = topo_map(
    ...     show=False,
    ...     levels=[-1200, -800, -400, 0, 400, 800, 1200],  # explicit levels
    ...     cmap="geo",               # override color map
    ...     limit="-1300/1300",       # clip displayed contours
    ... )
    >>> fig.show()

    Modern: provide levels as a CSV-like string and annotate every 200 m
    >>> fig = topo_map(
    ...     show=False,
    ...     levels="0,200,400,600,800,1000,1200",
    ...     cmap="oleron",
    ...     level_interval=None,      # ignored when `levels` is provided
    ... )
    # If you want labels on contours, set annotation inside the function call to grdcontour.

    Station overlay from an ObsPy Inventory (with labels)
    >>> from obspy import read_inventory
    >>> inv = read_inventory("MVO_inventory.xml")  # example path
    >>> fig = topo_map(
    ...     show=False,
    ...     inv=inv,
    ...     add_labels=True,
    ...     topo_color=True,
    ... )
    >>> fig.show()

    Zoom and illumination control
    >>> fig = topo_map(
    ...     show=False,
    ...     zoom_level=2,         # zoom in further
    ...     azimuth=135,          # light from SE
    ...     elevation=35,         # higher sun angle
    ...     topo_color=True,
    ... )
    >>> fig.show()

    Use a local cache for Earth relief grids (speeds up repeats)
    >>> fig = topo_map(
    ...     show=False,
    ...     DEM_DIR="/path/to/cache_dir",  # pickled grid cache
    ...     contour_interval=100,
    ...     topo_color=True,
    ... )
    >>> fig.show()

    Pick a different projection (e.g., 5-inch wide Mercator)
    >>> fig = topo_map(
    ...     show=False,
    ...     projection="M5i",
    ...     topo_color=True,
    ... )
    >>> fig.show()

    """
    if region:
        centerlon = (region[0]+region[1])/2
        centerlat = ((region[2]+region[3])/2)

    # --- optional cache file ---
    pklfile = None
    if DEM_DIR:
        os.makedirs(DEM_DIR, exist_ok=True)
        pklfile = os.path.join(
            DEM_DIR,
            f"EarthReliefData{centerlon}.{centerlat}.{zoom_level}.{resolution}.pkl"
        )

    #f stations is None:
    #    stations = []

    if not region:
        # define plot geographical range
        diffdeglat = 0.1/(1.5**zoom_level)
        diffdeglon = diffdeglat/np.cos(np.deg2rad(centerlat))
        minlon, maxlon = centerlon-diffdeglon, centerlon+diffdeglon  #-62.25, -62.13
        minlat, maxlat = centerlat-diffdeglat, centerlat+diffdeglat  # 16.66, 16.83
        region=[minlon, maxlon, minlat, maxlat]
        print(f'topo_map: region={region}')

    # --- load relief grid ---
    ergrid = None
    if pklfile and os.path.exists(pklfile):
        try:
            with open(pklfile, "rb") as f:
                ergrid = pickle.load(f)
        except Exception:
            ergrid = None

    if ergrid is None:
        try:
            ergrid = pygmt.datasets.load_earth_relief(
                resolution=resolution, region=region, registration=None
            )
            if pklfile:
                with open(pklfile, "wb") as f:
                    pickle.dump(ergrid, f)
        except Exception:
            print("Cannot load any topo data")
            return None

    # --- hillshade (modern approach) ---
    shade = pygmt.grdgradient(
        grid=ergrid,
        radiance=[azimuth, elevation],
        normalize="t1",
    )

    # --- choose colormap (back-compat + override) ---
    if cmap is None:
        cmap = "topo" if topo_color else "gray"

    # --- projection / figure size ---
    if projection is None:
        projection = f"M{figsize}i"   # width in inches    
    
    # Visualization
    fig = pygmt.Figure()
    
    '''
    if topo_color:
        # make color pallets
        print('Making color pallet')
        pygmt.makecpt(
            cmap='topo',
            series='-1300/1300/%d' % contour_interval,
            continuous=True
        )
        print('Calling grdimage')
        # plot high res topography
        fig.grdimage(
            grid=ergrid,
            region=region,
            projection='M4i',
            shading=True,
            frame=True
            )
    '''
    
    # --- raster with hillshade ---
    fig.grdimage(
        grid=ergrid,
        region=region,
        projection=projection,
        cmap=cmap,
        shading=shade,
        frame=True,
    )

    # Coastlines for crisp edges
    fig.coast(
        region=region,
        projection=projection,
        shorelines="1/0.5p,black",
        frame=["WSen", "af"],
    )
    
    # --- contours: support both new `levels` and old `contour_interval` ---
    if levels is not None:
        # Explicit levels provided
        fig.grdcontour(
            grid=ergrid,
            levels=levels,             # explicit
            pen="0.25p,black",
            limit=limit,               # e.g., "-1300/1300"
        )
    else:
        # Fall back to spacing (prefer level_interval if given; else contour_interval)
        step = level_interval if level_interval is not None else contour_interval
        fig.grdcontour(
            grid=ergrid,
            levels=contour_interval,             # legacy-compatible
            annotation=None,           # add e.g., annotation=200 if you want labels
            pen="0.25p,black",
            limit=limit,
        )
    
    # Colorbar only if we're using a meaningful color map
    if cmap and cmap != "gray":
        fig.colorbar(frame='+l"Topography (m)"')

    if inv:
        seed_ids = inventory2traceid(inv)#, force_location_code='')
        if not stations:
            stations = [id.split('.')[1] for id in seed_ids]
        stalat = [inv.get_coordinates(seed_id)['latitude'] for seed_id in seed_ids]
        stalon = [inv.get_coordinates(seed_id)['longitude'] for seed_id in seed_ids]
        
        if add_labels:
            #print('Adding station labels')
            for thislat, thislon, this_id in zip(stalat, stalon, seed_ids):
                net, sta, loc, chan = this_id.split('.')
                if sta in stations:
                    fig.plot(x=thislon, y=thislat, style="s0.5c", fill="white", pen='black') 
                    fig.text(x=thislon, y=thislat, text=sta, textfiles=None, \
                            font="white",
                            justify="ML",
                            offset="0.2c/0c",)
                else:
                    fig.plot(x=thislon, y=thislat, style="s0.4c", fill="black", pen='white') 
                    fig.text(x=thislon, y=thislat, text=sta, textfiles=None, \
                            font="black",
                            justify="ML",
                            offset="0.2c/0c",)
        else:
            fig.plot(x=stalon, y=stalat, style="s0.4c", fill="black", pen='white') 

    if title:
        fig.text(
            text=title,
            x=region[0] + (region[1]-region[0]) * 0.60,
            y=region[2] + (region[3]-region[2]) * 0.90,
            justify="TC",
            font="11p,Helvetica-Bold,black"
        )
    
    fig.basemap(region=region, frame=True)

    if show:
        fig.show();
    
    return fig

class Grid:
    def __init__(self, centerlat, centerlon, nlat, nlon, node_spacing_m):
        """
        Build a regular lat/lon grid with square spacing (node_spacing_m),
        centered at (centerlat, centerlon), with nlat x nlon nodes.
        Keeps attribute names compatible with your existing code:
        - self.gridlat, self.gridlon
        - self.node_spacing_lat, self.node_spacing_lon (in degrees)
        - self.latrange, self.lonrange (1D arrays)
        """
        # meters per degree at the center latitude
        meters_per_deg_lat = degrees2kilometers(1.0) * 1000.0  # ~111,320 m
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(centerlat))

        # degree spacing that corresponds to node_spacing_m
        node_spacing_lat = node_spacing_m / meters_per_deg_lat
        node_spacing_lon = node_spacing_m / meters_per_deg_lon

        # symmetric endpoints so we get exactly n points
        half_lat = (nlat - 1) / 2.0
        half_lon = (nlon - 1) / 2.0
        minlat = centerlat - half_lat * node_spacing_lat
        maxlat = centerlat + half_lat * node_spacing_lat
        minlon = centerlon - half_lon * node_spacing_lon
        maxlon = centerlon + half_lon * node_spacing_lon

        # exact counts with inclusive endpoints
        latrange = np.linspace(minlat, maxlat, num=nlat, endpoint=True)
        lonrange = np.linspace(minlon, maxlon, num=nlon, endpoint=True)

        gridlon, gridlat = np.meshgrid(lonrange, latrange, indexing="xy")

        # store (keep your original attribute names)
        self.gridlon = gridlon
        self.gridlat = gridlat
        self.node_spacing_lat = node_spacing_lat
        self.node_spacing_lon = node_spacing_lon
        self.lonrange = lonrange
        self.latrange = latrange

        # also store convenient metadata
        self.centerlat = centerlat
        self.centerlon = centerlon
        self.nlat = nlat
        self.nlon = nlon
        self.node_spacing_m = node_spacing_m

    def plot(self, fig=None, show=True,
            symbol="c", scale=1.0, fill="blue", pen="0.5p,black",
            topo_map_kwargs=None):
        """
        Plot grid nodes on a PyGMT topo map.

        Parameters
        ----------
        DEM_DIR : path or None
        fig : pygmt.Figure or None
            If None, creates a new topo map. Otherwise overlays on given figure.
        show : bool
        symbol : str
            GMT symbol code, e.g. 'c' (circle), 's' (square), 't' (triangle), 'x' (x-mark).
            (Note: '+' may not be supported; use 'x' for a cross.)
        scale : float
            Multiplier on default size (default size ~ node_spacing_m / 2000 cm).
        fill : str or None
            Fill color for filled symbols (ignored for line-only symbols like 'x').
        pen : str or None
            Outline/line pen (e.g. '0.5p,black').
        topo_map_kwargs : dict or None
            Extra kwargs forwarded to topo_map when creating a new figure.
        """
        if fig is None:
            topo_map_kwargs = topo_map_kwargs or {}
            fig = topo_map(show=False, **topo_map_kwargs)

        size_cm = (self.node_spacing_m / 2000.0) * float(scale)
        stylestr = f"{symbol}{size_cm}c"

        fig.plot(
            x=self.gridlon.reshape(-1),
            y=self.gridlat.reshape(-1),
            style=stylestr,
            pen=pen,
            fill=fill if symbol not in ("x", "+") else None,  # no fill for line-only marks
        )

        if show:
            fig.show()
        return fig

def initial_source(lat=dome_location['lat'], lon=dome_location['lon']):
    return {'lat':lat, 'lon':lon}

def make_grid(center_lat=dome_location['lat'], center_lon=dome_location['lon'], node_spacing_m = 100, grid_size_lat_m = 10000, grid_size_lon_m = 8000):
    nlat = int(grid_size_lat_m/node_spacing_m) + 1
    nlon = int(grid_size_lon_m/node_spacing_m) + 1
    return Grid(center_lat, center_lon, nlat, nlon, node_spacing_m)  

def synthetic_source_from_grid(
    grid: Grid,
    sampling_interval: float = 60.0,
    DR_cm2: float = 100.0,
    t0: obspy.UTCDateTime | None = None,
    order: str = "C",
):
    """
    Build a synthetic_source dict (lat, lon, DR, t) from a Grid.
    """
    if t0 is None:
        t0 = obspy.UTCDateTime(0)

    lat_flat = grid.gridlat.ravel(order=order).astype(float)
    lon_flat = grid.gridlon.ravel(order=order).astype(float)
    npts = lat_flat.size

    return {
        "lat": lat_flat,
        "lon": lon_flat,
        "DR": np.full(npts, float(DR_cm2)),
        "t": [t0 + i * sampling_interval for i in range(npts)],
    }


def simulate_SAM(inv, source, units='m/s', surfaceWaves=False, wavespeed_kms=1.5, peakf=8.0, Q=None, noise_level_percent=0.0, verbose=False):
    if units == 'm/s':
        sam_class = VSAM
    elif units == 'm':
        sam_class = DSAM
    npts = len(source['DR'])
    if isinstance(inv, Inventory):
        seed_ids = inventory2traceid(inv)#, force_location_code='')
    else:
        seed_ids = []
        return None
    
    dataframes = {}
    for id in seed_ids:
        net, sta, loc, chan = id.split('.') 
        coordinates = inv.get_coordinates(id)
        stalat = coordinates['latitude']
        stalon = coordinates['longitude']
        distance_km = degrees2kilometers(locations2degrees(stalat, stalon, source['lat'], source['lon']))

        gsc = sam_class.compute_geometrical_spreading_correction(distance_km, chan, surfaceWaves=surfaceWaves, wavespeed_kms=wavespeed_kms, peakf=peakf)
        isc = sam_class.compute_inelastic_attenuation_correction(distance_km, peakf, wavespeed_kms, Q)

        # Instead of assigning arrays as single elements, build the DataFrame properly:
        times = [UTCDateTime().timestamp + i for i in range(npts)]  # or use source['t'] if available
        amplitude = source['DR'] / (gsc * isc) * 1e-7
        if noise_level_percent > 0.0:
            amplitude += np.multiply(amplitude, np.random.uniform(0, 1, size=npts))

        dataframes[id] = pd.DataFrame({
            'time': times,
            'mean': amplitude
        })

    return sam_class(dataframes=dataframes, sampling_interval=1.0, verbose=verbose)

def plot_SAM(
    samobj,
    gridobj,
    K: int = 3,                         # number of random source nodes
    metric: str = "mean",
    DEM_DIR=None,
    inv=None,
    colors=None,                         # list like ["yellow","red","magenta"]
    seed: int | None = None,             # set for reproducibility
    show_map: bool = True,
    figsize: float = 6.0,               # map width in inches
):
    """
    Plot SAM values for K randomly-chosen source nodes.

    - PyGMT map: each chosen source node is plotted as a colored circle.
    - Matplotlib: grouped bars per station in the same colors (width = 0.9/K).

    Returns
    -------
    fig_map : pygmt.Figure
        The PyGMT map figure (or None if show_map=False).
    chosen_nodes : np.ndarray
        The flattened-node indices that were selected.
    """

    # ----- choose K random nodes -----
    rng = np.random.default_rng(seed)
    total_nodes = gridobj.gridlat.size
    K = max(1, min(K, total_nodes))  # clamp
    chosen_nodes = rng.choice(total_nodes, size=K, replace=False)

    # ----- prepare DSAM values -----
    # x tick labels = station codes (from dsamobj.dataframes keys)
    stations = [sid.split(".")[1] for sid in samobj.dataframes]
    st = samobj.to_stream(metric=metric)

    # for each chosen node, collect one y-value per station
    # y_matrix shape: (K, nstations)
    y_matrix = np.empty((K, len(stations)), dtype=float)
    for i, node in enumerate(chosen_nodes):
        y_matrix[i, :] = [tr.data[node] for tr in st]

    # ----- Matplotlib grouped bar chart -----
    x = np.arange(len(stations))
    if colors is None:
        # default palette (expand if K > 3)
        base = ["yellow", "red", "magenta", "dodgerblue", "limegreen", "orange", "purple"]
        colors = (base * ((K + len(base) - 1) // len(base)))[:K]
    barw = 0.9 / K
    # center offsets so the group is centered on each station tick
    offsets = (np.arange(K) - (K - 1) / 2.0) * barw

    plt.figure()
    bars = []
    for i in range(K):
        bars.append(
            plt.bar(
                x + offsets[i],
                y_matrix[i, :],
                width=barw,
                color=colors[i],
                edgecolor="black",
                linewidth=0.5,
                label=f"Node {chosen_nodes[i]}",
            )
        )

    plt.xticks(x, stations, rotation=45, fontsize=8)
    plt.xlabel("Station")
    plt.ylabel(metric)
    plt.title(f"{metric} by station for {K} source node(s)")
    plt.legend(ncols=min(K, 4), fontsize=8, frameon=False)

    # ----- PyGMT topo map with colored source nodes -----
    fig_map = None
    if show_map:
        fig_map = topo_map(
            show=False,
            zoom_level=0,
            inv=inv,
            add_labels=True,
            centerlon=-62.177,
            centerlat=16.711,
            contour_interval=100,
            topo_color=False,        # keeps your old behavior
            resolution="03s",
            DEM_DIR=DEM_DIR,
            figsize=figsize,
        )

        # plot each chosen node in matching color
        lon_flat = gridobj.gridlon.ravel()
        lat_flat = gridobj.gridlat.ravel()
        for i, node in enumerate(chosen_nodes):
            fig_map.plot(
                x=[lon_flat[node]],
                y=[lat_flat[node]],
                style="c0.28c",
                pen=f"1p,{colors[i]}",
                fill=colors[i],
            )

        fig_map.show()

    return fig_map, chosen_nodes

# pretty sure that i had a different version here that worked. this one is crashing because trying to plot nodenum 100 of a 100-length tr.data
# what I really should be plotting is the corrections at node 100

class ASL:
    def __init__(self, samobject, metric, inventory, gridobj, window_seconds):
        ''' 
        ASL: Simple amplitude-based source location for volcano-seismic data 
        This program takes a VSAM object as input
        Then armed with an inventory that provides station coordinates, it attempts
        to find a location for each sample by reducing amplitudes based on the grid
        node to station distance. Grid nodes are contained in a Grid object.

        ASL can use any one of the mean, max, median, VLP, LP, or VT metrics from a VSAM object

        '''

        if isinstance(samobject, VSAM):
            pass
        else:
            print('invalid type passed as samobject. Aborting')
            return

        self.samobject = samobject
        self.metric = metric
        self.inventory = inventory
        self.gridobj = gridobj
        self.node_distances_km = {}
        self.station_coordinates = {}
        self.amplitude_corrections = {}
        self.window_seconds = window_seconds
        self.surfaceWaves = False
        self.wavespeed_kms = None
        self.wavelength_km = None
        self.Q = None
        self.peakf = None       
        self.located = False
        self.source = None
        self.event = None
        
    def setup(self, surfaceWaves=False):  
        self.compute_grid_distances()
        self.compute_amplitude_corrections(surfaceWaves = surfaceWaves)

    def compute_grid_distances(self, use_stream=False):
        """
        Computes and stores distances from each grid node to each station/channel.

        If `use_stream=True`, it will compute only for trace IDs in self.samobject.to_stream().
        Otherwise, it uses all available channels in self.inventory.

        Results are stored in:
            - self.node_distances_km: dict of {seed_id: np.array of distances in km}
            - self.station_coordinates: dict of {seed_id: {latitude, longitude, elevation}}
        """


        node_distances_km = {}
        station_coordinates = {}

        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)
        nodelatlon = list(zip(gridlat, gridlon))

        if use_stream:
            st = self.samobject.to_stream()
            trace_ids = list({tr.id for tr in st})
            print(f"[INFO] Computing distances using {len(trace_ids)} trace IDs from stream.")
        else:
            trace_ids = []
            for net in self.inventory:
                for sta in net:
                    for cha in sta:
                        trace_ids.append(f"{net.code}.{sta.code}..{cha.code}")
            print(f"[INFO] Computing distances using {len(trace_ids)} channels from inventory.")

        for seed_id in trace_ids:
            try:
                coords = self.inventory.get_coordinates(seed_id)
                stalat = coords['latitude']
                stalon = coords['longitude']
                station_coordinates[seed_id] = coords
            except Exception as e:
                print(f"[WARN] Skipping {seed_id}: {e}")
                continue

            try:
                distances_deg = [
                    locations2degrees(nlat, nlon, stalat, stalon)
                    for nlat, nlon in nodelatlon
                ]
                distances_km = [degrees2kilometers(d) for d in distances_deg]
                node_distances_km[seed_id] = np.array(distances_km)
            except Exception as e:
                print(f"[ERROR] Distance calc failed for {seed_id}: {e}")
                continue

        self.node_distances_km = node_distances_km
        self.station_coordinates = station_coordinates
        print(f"[DONE] Grid distances computed for {len(node_distances_km)} trace IDs.")

            
    @staticmethod
    def set_peakf(metric, df):
        if metric in ['mean', 'median', 'max', 'rms']:
            ratio = df['VT'].sum()/df['LP'].sum()
            peakf = np.sqrt(ratio) * 4
        elif metric == 'VLP':
            peakf = 0.1
        elif metric == 'LP':
            peakf = 2.0
        elif metric == 'VT':
            peakf = 8.0
        return peakf


    def compute_amplitude_corrections(
        self,
        surfaceWaves=False,
        wavespeed_kms=None,
        Q=None,
        fix_peakf=None,
        cache_dir="asl_cache",
        force_recompute=False,
    ):
        """
        Compute amplitude corrections for all channels in the inventory using geometric spreading
        and inelastic attenuation. Results are cached for reuse across events.

        Parameters
        ----------
        surfaceWaves : bool
            Whether to use surface wave assumptions.
        wavespeed_kms : float or None
            Wave speed in km/s. Default is 1.5 for surface, 3.0 otherwise.
        Q : float or None
            Attenuation factor. Required if inelastic corrections are used.
        fix_peakf : float or None
            Fixed peak frequency to use in calculations. If None, defaults to 1.0.
        cache_dir : str
            Directory for saving and loading cached corrections.
        """
        if not wavespeed_kms:
            wavespeed_kms = 1.5 if surfaceWaves else 3.0
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        peakf = fix_peakf or 2.0
        cache_key = f"ampcorr_Q{int(round(Q or 99))}_V{int(round(wavespeed_kms))}_f{int(round(peakf))}_{'surf' if surfaceWaves else 'body'}.pkl"
        cache_path = os.path.join(cache_dir, cache_key)

        # Try loading cached corrections
        if os.path.exists(cache_path) and not force_recompute:
            try:
                with open(cache_path, "rb") as f:
                    self.amplitude_corrections = pickle.load(f)
                print(f"[CACHE HIT] Loaded amplitude corrections from {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to load cache: {e}")
                self.amplitude_corrections = {}
        else:
            print(f"[INFO] Creating amplitude corrections for all inventory channels (Q={Q}, f={peakf})")
            corrections = {}
            for net in self.inventory:
                for sta in net:
                    for cha in sta:
                        seed_id = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                        try:
                            dist_km = self.node_distances_km[seed_id]
                        except KeyError:
                            print(f"[WARN] No node distances for {seed_id}, skipping")
                            continue

                        gsc = VSAM.compute_geometrical_spreading_correction(
                            dist_km, cha.code[-3:], surfaceWaves=surfaceWaves,
                            wavespeed_kms=wavespeed_kms, peakf=peakf
                        )
                        isc = VSAM.compute_inelastic_attenuation_correction(
                            dist_km, peakf, wavespeed_kms, Q
                        )
                        corrections[seed_id] = gsc * isc

            # Save new cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(corrections, f)
                print(f"[CACHE SAVE] Amplitude corrections saved to {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to save cache: {e}")
            self.amplitude_corrections = corrections

        # Assign class attributes
        self.surfaceWaves = surfaceWaves
        self.wavespeed_kms = wavespeed_kms
        self.Q = Q
        self.peakf = peakf
        self.wavelength_km = peakf * wavespeed_kms



    '''
    def metric2stream(self):
        st = self.samobject.to_stream(metric=self.metric)
        if st[0].stats.sampling_rate != self.window_seconds:
            window = np.ones(self.window_seconds) / self.window_seconds
            for tr in st:
                tr.data = np.convolve(tr.data, window, mode='same')
        return st       
    '''



    def metric2stream(self):
        """
        Return the SAM stream (already produced by self.samobject).
        If window_seconds > samobject.sampling_interval, apply a centered moving
        average of length round(window_seconds / sampling_interval) samples.

        Assumes self.samobject.to_stream(metric=...) returns a Stream whose traces
        are sampled at ~1 / sampling_interval Hz (typically 1 Hz).
        """
        st = self.samobject.to_stream(metric=self.metric)

        # Get the DSAM/VSAM sampling interval (seconds per sample)
        dt = float(getattr(self.samobject, "sampling_interval", 1.0))
        if not np.isfinite(dt) or dt <= 0:
            # Fallback to trace metadata if needed
            fs = float(getattr(st[0].stats, "sampling_rate", 1.0) or 1.0)
            dt = 1.0 / fs

        win_sec = float(getattr(self, "window_seconds", 0.0) or 0.0)

        # Only smooth if the window is strictly longer than one sample
        if win_sec > dt:
            win_samples = max(2, int(round(win_sec / dt)))  # at least 2 samples for an actual window
            for tr in st:
                # If someone hands you a non-1/dt stream, scale the window accordingly
                fs_tr = float(getattr(tr.stats, "sampling_rate", 1.0) or 1.0)
                dt_tr = 1.0 / fs_tr
                w_tr = max(2, int(round(win_sec / dt_tr)))

                x = tr.data.astype(float)
                # NaN-safe centered moving average
                m = np.isfinite(x).astype(float)
                xf = np.where(np.isfinite(x), x, 0.0)

                num = uniform_filter1d(xf, size=w_tr, mode="nearest")
                den = uniform_filter1d(m,  size=w_tr, mode="nearest")
                tr.data = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

                # Keep metadata consistent
                tr.stats.sampling_rate = fs_tr
        # else: window ≤ sample spacing → no smoothing

        return st
    
    def locate(self):
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        lendata = len(st[0].data)
        t = st[0].times('utcdatetime')

        source_DR = np.empty(lendata, dtype=float)
        source_lat = np.empty(lendata, dtype=float)
        source_lon = np.empty(lendata, dtype=float)
        source_misfit = np.empty(lendata, dtype=float)
        source_azgap = np.empty(lendata, dtype=float)
        source_nsta = np.empty(lendata, dtype=int)

        for i in range(lendata):
            y = [tr.data[i] for tr in st]
            best_misfit = 1e15
            best_j = -1
            for j in range(len(gridlat)):
                c = [self.amplitude_corrections[id][j] for id in seed_ids]
                reduced_y = np.multiply(y, c)
                this_misfit = np.nanstd(reduced_y) / np.nanmedian(reduced_y)
                if this_misfit < best_misfit:
                    best_misfit = this_misfit
                    best_j = j

            DR_values = [y[tracenum] * self.amplitude_corrections[id][best_j] for tracenum, id in enumerate(seed_ids)]
            source_DR[i] = np.nanmedian(DR_values)
            source_lat[i] = gridlat[best_j]
            source_lon[i] = gridlon[best_j]
            source_misfit[i] = best_misfit

            station_coords = []
            for seed_id in seed_ids:
                coords = self.station_coordinates.get(seed_id)
                if coords:
                    station_coords.append((coords['latitude'], coords['longitude']))
            azgap, nsta = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
            source_azgap[i] = azgap
            source_nsta[i] = nsta

        self.source = {
            't': t,
            'lat': source_lat,
            'lon': source_lon,
            'DR': source_DR * 1e7,
            'misfit': source_misfit,
            'azgap': source_azgap,
            'nsta': source_nsta
        }

        self.source_to_obspyevent()
        self.located = True
        return self.source

        # Here is where i would add loop over shrinking grid

    def locate(self, *, min_stations=3, eps=1e-9):
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)

        # Stream and station order (frozen)
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        nsta = len(seed_ids)
        lendata = len(st[0].data)
        t = st[0].times('utcdatetime')

        # Build data matrix once: (nsta, lendata)
        Y = np.vstack([tr.data.astype(float) for tr in st])  # rows match seed_ids order

        # Build corrections matrix once: C[sta, node]
        # Assert node length & ordering match gridlat/gridlon
        first_corr = self.amplitude_corrections[seed_ids[0]]
        nnodes = len(first_corr)
        assert nnodes == gridlat.size == gridlon.size, \
            f"Node count mismatch: corr={nnodes}, grid={gridlat.size}"

        C = np.empty((nsta, nnodes), dtype=float)
        for k, sid in enumerate(seed_ids):
            ck = np.asarray(self.amplitude_corrections[sid], dtype=float)
            if ck.size != nnodes:
                raise ValueError(f"Corrections length mismatch for {sid}: {ck.size} != {nnodes}")
            C[k, :] = ck

        # Precompute station coordinates (for azgap) once
        station_coords = []
        for sid in seed_ids:
            coords = self.station_coordinates.get(sid)
            if coords:
                station_coords.append((coords['latitude'], coords['longitude']))

        # Outputs
        source_DR     = np.empty(lendata, dtype=float)
        source_lat    = np.empty(lendata, dtype=float)
        source_lon    = np.empty(lendata, dtype=float)
        source_misfit = np.empty(lendata, dtype=float)
        source_azgap  = np.empty(lendata, dtype=float)
        source_nsta   = np.empty(lendata, dtype=int)

        # Loop over time samples
        for i in range(lendata):
            y = Y[:, i]  # shape (nsta,)

            best_j = -1
            best_m = np.inf

            # Iterate nodes; use vector ops on the station dimension
            for j in range(nnodes):
                reduced = y * C[:, j]                    # (nsta,)
                finite  = np.isfinite(reduced)
                nfin    = int(finite.sum())
                if nfin < min_stations:
                    continue
                r = reduced[finite]
                med = np.nanmedian(r)
                if not np.isfinite(med):
                    continue
                # robust scatter: MAD instead of std (optional)
                # mad = 1.4826 * np.nanmedian(np.abs(r - med))
                # m = mad / (abs(med) + eps)
                m = float(np.nanstd(r) / (abs(med) + eps))  # your original metric with eps
                if m < best_m:
                    best_m = m
                    best_j = j

            # If nothing passed the min_stations gate, fall back (all stations)
            if best_j < 0:
                # Choose node with minimal m even without min_stations
                for j in range(nnodes):
                    reduced = y * C[:, j]
                    r = reduced[np.isfinite(reduced)]
                    if r.size == 0:
                        continue
                    med = np.nanmedian(r)
                    if not np.isfinite(med):
                        continue
                    m = float(np.nanstd(r) / (abs(med) + eps))
                    if m < best_m:
                        best_m = m
                        best_j = j

            # DR from the chosen node: median across stations (finite only)
            reduced_best = y * C[:, best_j]
            rbest = reduced_best[np.isfinite(reduced_best)]
            source_DR[i] = np.nanmedian(rbest) if rbest.size else np.nan

            source_lat[i] = gridlat[best_j]
            source_lon[i] = gridlon[best_j]
            source_misfit[i] = best_m

            # Azimuthal gap (station set is fixed per call)
            azgap, nsta_eff = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
            source_azgap[i] = azgap
            source_nsta[i]  = nsta_eff

        self.source = {
            't': t,
            'lat': source_lat,
            'lon': source_lon,
            'DR': source_DR * 1e7,
            'misfit': source_misfit,
            'azgap': source_azgap,
            'nsta': source_nsta
        }
        self.source_to_obspyevent()
        self.located = True
        return 

    def fast_locate(self):
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)
        st = self.metric2stream()

        seed_ids = [tr.id for tr in st]
        t = st[0].times('utcdatetime')
        n = len(t)

        source_DR = np.empty(n, dtype=float)
        source_lat = np.empty(n, dtype=float)
        source_lon = np.empty(n, dtype=float)
        source_misfit = np.empty(n, dtype=float)
        source_azgap = np.empty(n, dtype=float)
        source_nsta = np.empty(n, dtype=int)

        for i in range(n):

            DR_stations_nodes = np.full((len(st), len(gridlat)), np.nan)

            for j, tr in enumerate(st):
                seed_id = tr.id
                station = tr.stats.station

                # Try using seed_id for correction, fallback to station
                correction = self.amplitude_corrections.get(seed_id)
                if correction is None:
                    correction = self.amplitude_corrections.get(station)
                    if correction is None:
                        print(f"[WARN] No correction for {seed_id} or {station}, skipping")
                        continue

                DR_stations_nodes[j] = correction * tr.data[i]

            DR_mean_nodes = np.nanmean(DR_stations_nodes, axis=0)
            DR_std_nodes = np.nanstd(DR_stations_nodes, axis=0)
            misfit = np.divide(DR_std_nodes, DR_mean_nodes)
            lowest_misfit_index = np.nanargmin(misfit)

            source_DR[i] = DR_mean_nodes[lowest_misfit_index]
            source_lat[i] = gridlat[lowest_misfit_index]
            source_lon[i] = gridlon[lowest_misfit_index]
            source_misfit[i] = misfit[lowest_misfit_index]

            # Compute azgap & nsta at this location
            station_coords = []
            for tr in st:
                coords = self.station_coordinates.get(tr.id) or self.station_coordinates.get(tr.stats.station)
                if coords:
                    station_coords.append((coords['latitude'], coords['longitude']))

            azgap, nsta = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
            source_azgap[i] = azgap
            source_nsta[i] = nsta

        self.source = {
            't': t,
            'lat': source_lat,
            'lon': source_lon,
            'DR': source_DR * 1e7,
            'misfit': source_misfit,
            'azgap': source_azgap,
            'nsta': source_nsta
        }

        self.source_to_obspyevent()
        self.located = True
        return

        # TODO: Refactor module to use station-based amplitude corrections consistently, rather than full SEED IDs.



    def plot(self, zoom_level=1, threshold_DR=0, scale=1, join=False, number=0, \
             add_labels=False, equal_size=False, outfile=None, 
             stations=None, title=None, region=None, normalize=True):
        source = self.source

        if source:
                
            # timeseries of DR vs threshold_amp
            t_dt = [this_t.datetime for this_t in source['t']]
            plt.figure()
            plt.plot(t_dt, source['DR'])
            plt.plot(t_dt, np.ones(source['DR'].size) * threshold_DR)
            plt.xlabel('Date/Time (UTC)')
            plt.ylabel('Reduced Displacement (${cm}^2$)')
              
            if threshold_DR>0:
                
                indices = source['DR']<threshold_DR
                source['DR'][indices]=0.0
                source['lat'][indices]=None
                source['lon'][indices]=None
            

            # Trajectory map
            x = source['lon']
            y = source['lat']
            DR = source['DR']
            if equal_size:
                symsize = scale * np.ones(len(DR))
            elif normalize:
                symsize = np.divide(DR, np.nanmax(DR))*scale
            else:
                symsize = scale * np.sqrt(DR)
            #print('symbol size = ',symsize)
    
                
            maxi = np.argmax(DR)

            fig = topo_map(zoom_level=zoom_level, inv=self.inventory, \
                                        centerlat=y[maxi], centerlon=x[maxi], add_labels=add_labels, 
                                        topo_color=False, stations=stations, title=title, region=region)


            if number:
                if number<len(x):
                    ind = np.argpartition(DR, -number)[-number:]
                    x = x[ind]
                    y = y[ind]
                    DR = DR[ind]
                    mascxi = np.argmax(DR)
                    symsize = symsize[ind]
            pygmt.makecpt(cmap="viridis", series=[0, len(x)])
            timecolor = [i for i in range(len(x))]

            fig.plot(x=x, y=y, size=symsize, style="cc", pen=None, fill=timecolor, cmap=True)

            fig.colorbar(
                frame='+l"Sequence"',
                #     position="x11.5c/6.6c+w6c+jTC+v" #for vertical colorbar
                )

            if region:
                fig.basemap(region=region, frame=True)
            if outfile:
                fig.savefig(outfile)
            else:
                fig.show();  

            if join:
                fig.plot(x=x, y=y, style="r-", pen="1p,red")
    

            
        else: # no location data      
            fig = topo_map(zoom_level=zoom_level, inv=self.inventory, show=True, add_labels=add_labels)
            #fig._cleanup()


    def plot_reduced_displacement(self, threshold_DR=0, outfile=None):
        source = self.source
        if source:
            # timeseries of DR vs threshold_amp
            t_dt = [this_t.datetime for this_t in source['t']]
            plt.figure()
            plt.plot(t_dt, source['DR'])
            plt.plot(t_dt, np.ones(source['DR'].size) * threshold_DR)
            plt.xlabel('Date/Time (UTC)')
            plt.ylabel('Reduced Displacement (${cm}^2$)')
            if outfile:
                plt.savefig(outfile)
            else:   
                plt.show()

    def plot_misfit(self, outfile=None):
        source = self.source
        if source:
            t_dt = [this_t.datetime for this_t in source['t']]
            # repeat but for misfit rather than DR
            plt.figure()
            plt.plot(t_dt, source['misfit'])
            plt.xlabel('Date/Time (UTC)')
            plt.ylabel('Misfit (std/median)')
            if outfile:
                plt.savefig(outfile)
            else:   
                plt.show()
    
    def print_source(self):
        print(pd.DataFrame(self.source))

    def source_to_csv(self, csvfile):
        source = self.source
        if source:
            df = pd.DataFrame(source)
            df.to_csv(csvfile, index=False)
            print(f"CSV file created: {csvfile}")
        
    def source_to_obspyevent(self, event_id=None):
        """
        Converts self.source (dict) into an ObsPy Event with Origins and Amplitudes,
        storing azimuthal gap, station count, misfit, and distance metrics in each Origin.
        """
        source = self.source
        if not event_id:
            event_id = source['t'][0].strftime("%Y%m%d%H%M%S")

        event = Event()
        event.resource_id = ResourceIdentifier(f"smi:example.org/event/{event_id}")
        event.event_type = "landslide"

        # Optional comment about units
        comment_text = "Note: Origin.quality distance fields are in kilometers, not degrees."
        event.comments.append(Comment(text=comment_text))

        azgap = source.get('azgap', [None] * len(source['t']))
        nsta = source.get('nsta', [None] * len(source['t']))
        misfits = source.get('misfit', [None] * len(source['t']))
        coords_dict = self.station_coordinates  # {seed_id: {'latitude': x, 'longitude': y, ...}}

        for i, (t, lat, lon, DR, misfit_val) in enumerate(zip(
            source['t'], source['lat'], source['lon'], source['DR'], misfits
        )):
            origin = Origin()
            origin.resource_id = ResourceIdentifier(f"smi:example.org/origin/{event_id}_{i:03d}")
            origin.time = UTCDateTime(t) if not isinstance(t, UTCDateTime) else t
            origin.latitude = lat
            origin.longitude = lon
            origin.depth = 0

            # --- Origin Quality ---
            oq = OriginQuality()
            oq.standard_error = float(misfit_val) if misfit_val is not None else None
            oq.azimuthal_gap = float(azgap[i]) if azgap[i] is not None else None
            oq.used_station_count = int(nsta[i]) if nsta[i] is not None else None

            # Distance metrics in km (stored in fields meant for degrees — documented above)
            distances_km = []
            for coords in coords_dict.values():
                dist_m, _, _ = gps2dist_azimuth(lat, lon, coords['latitude'], coords['longitude'])
                distances_km.append(dist_m / 1000.0)

            if distances_km:
                oq.minimum_distance = min(distances_km)
                oq.maximum_distance = max(distances_km)
                oq.median_distance = float(np.median(distances_km))

            origin.quality = oq

            # Optionally, store misfit as time uncertainty too
            origin.time_errors = QuantityError(uncertainty=misfit_val)

            event.origins.append(origin)

            # --- Amplitude object ---
            amplitude = Amplitude()
            amplitude.resource_id = ResourceIdentifier(f"smi:example.org/amplitude/{event_id}_{i:03d}")
            amplitude.generic_amplitude = DR
            amplitude.unit = "other"  # cm² doesn't fit Enum; document externally
            amplitude.pick_id = origin.resource_id  # Soft link
            amplitude.time_window = None

            event.amplitudes.append(amplitude)

        self.event = event



    def save_event(self, outfile=None):
        """            outfile (str): Output QuakeML filename.  """
        if self.located and outfile:
            # Create a Catalog and write to QuakeML
            catalog = Catalog(events=[self.event])
            catalog.write(outfile, format="QUAKEML")

            print(f"QuakeML file created: {outfile}")

    def print_event(self):
        if self.located:
            pprint(self.event)

import xarray as xr
def plot_heatmap_montserrat_colored(df, lat_col='latitude', lon_col='longitude', amp_col='amplitude',
                                     zoom_level=0, inventory=None, color_scale=0.4,
                                     cmap='turbo', log_scale=True, contour=False,
                                     node_spacing_m=50, outfile=None, region=None, title=None):
    """
    Plot ASL heatmap on Montserrat topography using tessellated color-filled squares.

    Parameters:
    - df: pandas DataFrame with lat/lon and amplitude columns
    - lat_col, lon_col, amp_col: column names
    - zoom_level: zoom level for topo_map
    - inventory: ObsPy Inventory for station overlay
    - color_scale: multiplier for colormap normalization
    - cmap: GMT colormap name (e.g., 'turbo', 'hot', 'viridis')
    - log_scale: apply log10 scaling to energy
    - contour: overlay contour lines on interpolated energy field
    - node_spacing_m: expected ASL grid spacing in meters (default = 50)
    - outfile: if given, save the figure; otherwise show it
    """
    df = df.copy()
    df['energy'] = df[amp_col] ** 2

    # Aggregate total energy per node
    grouped = df.groupby([lat_col, lon_col])['energy'].sum().reset_index()
    x = grouped[lon_col].to_numpy()
    y = grouped[lat_col].to_numpy()
    z = grouped['energy'].to_numpy()

    # Log scale if requested
    if log_scale:
        z = np.log10(z + 1e-10)

    # Compute symbol size in degrees (longitude spacing)
    meters_per_deg_lon = 111320 * np.cos(np.radians(16.7))  # Montserrat lat ~16.7°
    #node_size_deg = node_spacing_m / meters_per_deg_lon
    symbol_size_m = node_spacing_m * 0.077/50

    # Build color palette
    pygmt.makecpt(cmap=cmap,
                  series=[np.nanmin(z), np.nanmax(z), (np.nanmax(z) - np.nanmin(z)) / 100],
                  continuous=True)

    # Create base map
    fig = topo_map(zoom_level=zoom_level, inv=inventory, topo_color=False, region=region, title=title)

    # Plot colored square tiles

    fig.plot(x=x, y=y, style=f"s{symbol_size_m}c", fill=z, cmap=True, pen=None)

    """
    # Create a DataFrame for pygmt
    points = pd.DataFrame({
        "longitude": x,
        "latitude": y,
        "energy": z,
    })

    # Create NumPy array directly
    points = np.column_stack((x, y, z))

    fig.plot(
        data=points,
        style=f"s{symbol_size_m}c",  # Squares
        cmap=True,
        pen="none"  # You *can* say "none" here now
    )
    
    # Create DataFrame for PyGMT
    points = pd.DataFrame({
        "longitude": x,
        "latitude": y,
        "size": symbol_size_m  # Size in cm
    })

    # Plot with constant fill color, no border lines
    fig.plot(
        data=points,
        style="c",                         # Circle, size from 'size' column
        fill="red",                        # All symbols filled red
        pen="none",                        # No borders
    )
    """



    # Optional contours
    """
    contour = False
    if contour:
        from scipy.interpolate import griddata
        xi = np.linspace(min(x), max(x), 200)
        yi = np.linspace(min(y), max(y), 200)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), z, (Xi, Yi), method='linear')

        grid = xr.DataArray(
            Zi,
            coords={"lat": yi, "lon": xi},
            dims=["lat", "lon"],
        )

        fig.grdcontour(grid=grid, region=fig.region, interval=0.5, pen="0.75p,gray")
    """

    # Add colorbar
    fig.colorbar(frame='+l"Log10 Total Energy"' if log_scale else '+l"Total Energy"')
    if region:
        fig.basemap(region=region, frame=True)
        
    # Save or show
    if outfile:
        fig.savefig(outfile)
    else:
        fig.show()

    return fig

def compute_azimuthal_gap(origin_lat, origin_lon, station_coords):
    """
    Computes the azimuthal gap and station count.
    
    Parameters
    ----------
    origin_lat : float
        Latitude of source
    origin_lon : float
        Longitude of source
    station_coords : list of (lat, lon)
        Coordinates of stations used

    Returns
    -------
    az_gap : float
        Maximum azimuthal gap in degrees
    n_stations : int
        Number of stations used
    """
    azimuths = []
    for stalat, stalon in station_coords:
        _, az, _ = gps2dist_azimuth(origin_lat, origin_lon, stalat, stalon)
        azimuths.append(az)

    if len(azimuths) < 2:
        return 360.0, len(azimuths)  # Maximum gap with only 1 station

    azimuths = sorted(azimuths)
    azimuths.append(azimuths[0] + 360.0)  # Wrap around

    gaps = [azimuths[i+1] - azimuths[i] for i in range(len(azimuths)-1)]
    return max(gaps), len(azimuths) - 1



def extract_asl_diagnostics(topdir, outputcsv, timestamp=True):
    """
    Extracts ASL diagnostics from QuakeML files and associated event directories.

    Parameters:
        topdir (str): Base directory containing ASL event folders
        output_csv (str): Optional path to save results as a CSV file

    Returns:
        pd.DataFrame: DataFrame of all ASL-origin diagnostic info
    """
    all_dirs = sorted(glob.glob(os.path.join(topdir, "*MVO*")))
    lod = []

    for thisdir in all_dirs:
        # Check if ASL map exists
        mapfile = glob.glob(os.path.join(thisdir, 'map_Q100*.png'))
        if not (len(mapfile) == 1 and os.path.isfile(mapfile[0])):
            continue

        qmlfile = glob.glob(os.path.join(thisdir, 'event*Q*.qml'))
        if not (len(qmlfile) == 1 and os.path.isfile(qmlfile[0])):
            continue

        try:
            cat = read_events(qmlfile[0])
            ev = cat.events[0]
            for i, origin in enumerate(ev.origins):
                r = {}
                r['qml_path'] = qmlfile[0]
                r['time'] = origin.time.isoformat() if origin.time else None
                r['latitude'] = origin.latitude
                r['longitude'] = origin.longitude
                r['depth_km'] = origin.depth / 1000 if origin.depth else None
                r['amplitude'] = ev.amplitudes[i].generic_amplitude if i < len(ev.amplitudes) else None

                # OriginQuality
                oq = origin.quality
                if oq:
                    r['azimuthal_gap'] = oq.azimuthal_gap
                    r['station_count'] = oq.used_station_count
                    r['misfit'] = oq.standard_error
                    r['min_dist_km'] = oq.minimum_distance
                    r['max_dist_km'] = oq.maximum_distance
                    r['median_dist_km'] = oq.median_distance

                lod.append(r)

        except Exception as e:
            print(f"[WARN] Could not parse {qmlfile[0]}: {e}")

    df = pd.DataFrame(lod)

    if timestamp:
        outputcsv.replace('.csv', f'{UTCDateTime().timestamp}.csv')

    df.to_csv(outputcsv, index=False)
    print(f"[✓] Saved ASL diagnostics to: {outputcsv}")
    return df

def compare_asl_sources(asl1, asl2, atol=1e-8, rtol=1e-5):
    """
    Compare the 'source' output dictionaries from two ASL objects.
    
    Returns True if all values match; otherwise, returns a dict of diffs.
    """
    source1 = asl1.source
    source2 = asl2.source
    
    diffs = {}
    keys1 = set(source1.keys())
    keys2 = set(source2.keys())
    
    if keys1 != keys2:
        diffs['key_mismatch'] = {'only_in_1': keys1 - keys2, 'only_in_2': keys2 - keys1}
        return diffs
    
    for key in keys1:
        v1 = source1[key]
        v2 = source2[key]

        # Convert ObsPy UTCDateTime to float timestamps for comparison
        if hasattr(v1, '__getitem__') and hasattr(v1[0], 'timestamp'):
            v1 = np.array([x.timestamp for x in v1])
            v2 = np.array([x.timestamp for x in v2])

        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if v1.shape != v2.shape:
                diffs[key] = f"Shape mismatch: {v1.shape} vs {v2.shape}"
            elif np.issubdtype(v1.dtype, np.number):
                if not np.allclose(v1, v2, atol=atol, rtol=rtol, equal_nan=True):
                    diffs[key] = f"Numeric mismatch in {key} (not allclose)"
            else:
                if not np.array_equal(v1, v2):
                    diffs[key] = f"Non-numeric mismatch in {key} (not array_equal)"
        elif isinstance(v1, (list, tuple)):
            if v1 != v2:
                diffs[key] = f"Mismatch in list/tuple {key}: {v1} != {v2}"
        else:
            if v1 != v2:
                diffs[key] = f"Mismatch in {key}: {v1} != {v2}"

    result = True if not diffs else diffs

    if result is True:
        print("✅ ASL sources are identical.")
    else:
        print("❌ Differences found:")
        for k, v in result.items():
            print(f"{k}: {v}")
    
    

if __name__ == "__main__":
    pass
