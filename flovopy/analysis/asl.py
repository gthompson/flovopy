# Standard Library
import os
import math
import pickle
from pprint import pprint

# Scientific Stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyGMT for mapping and relief data
import pygmt

# ObsPy core and event tools
import obspy
from obspy import Stream, Trace, UTCDateTime, read_events
from obspy.core.event import Event, Catalog, ResourceIdentifier, Origin, Amplitude, QuantityError, OriginQuality, Comment
from obspy.geodetics import locations2degrees, degrees2kilometers, gps2dist_azimuth
# Your internal or local modules (assumed to exist)
# For example:
from flovopy.core.inventory import inventory2traceid
from flovopy.processing.sam import VSAM  # VSAM class for corrections and simulation
from flovopy.core.mvo import dome_location
import os
import glob


def montserrat_topo_map(show=False, zoom_level=0, inv=None, add_labels=False, centerlon=-62.177, centerlat=16.711, contour_interval=100, \
                        topo_color=True, resolution='03s', DEM_DIR=None, stations=[], title=None, region=None):

    #define etopo data file
    # ergrid = 'path_to_local_data_file'
    #ergrid = '@earth_relief_30s' #30 arc second global relief (SRTM15+V2.1 @ 1.0 km)
    #ergrid = '@earth_relief_15s' #15 arc second global relief (SRTM15+V2.1)
    #ergrid = '@earth_relief_03s' #3 arc second global relief (SRTM3S)
    if region:
        centerlon = (region[0]+region[1])/2
        centerlat = ((region[2]+region[3])/2)
    if DEM_DIR:
        pklfile = os.path.join(DEM_DIR, f'EarthReliefData{centerlon}.{centerlat}.{zoom_level}.{resolution}.pkl')
    else:
        pklfile = None
    #f stations is None:
    #    stations = []

    if not region:
        # define plot geographical range
        diffdeglat = 0.08/(2**zoom_level)
        diffdeglon = diffdeglat/np.cos(np.deg2rad(centerlat))
        minlon, maxlon = centerlon-diffdeglon, centerlon+diffdeglon  #-62.25, -62.13
        minlat, maxlat = centerlat-diffdeglat, centerlat+diffdeglat  # 16.66, 16.83
        region=[minlon, maxlon, minlat, maxlat]
        print(f'montserrat_topo_map: region={region}')

    if pklfile:
        if os.path.exists(pklfile):
            print(f'Loading {pklfile}')
            with open(pklfile, 'rb') as fileptr:
                ergrid = pickle.load(fileptr)    
    else:        
        try:
            print('Reading topo (earth relief) data from GMT website')
            ergrid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region, registration=None)
            print("ergrid downloaded")
            if pklfile:
                with open(pklfile, 'wb') as fileptr: 
                    print(f'Writing {pklfile}')
                    # A new file will be created 
                    pickle.dump(ergrid, fileptr)
        except:
            print("Cannot load any topo data")
            return None

    #print(ergrid)
    
    # Visualization
    fig = pygmt.Figure()
    
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
    
    # plot continents, shorelines, rivers, and borders
    fig.coast(
        region=region,
        projection='M4i',
        shorelines=True,
        frame=True
        )
    
    # plot the topographic contour lines
    fig.grdcontour(
        grid=ergrid,
        #interval=contour_interval,
        levels=contour_interval,
        annotation="%d+f6p" % contour_interval,
        limit="-1300/1300", #to only display it below 
        pen="a0.15p"
        )
    
    if topo_color:
        fig.colorbar(
            frame='+l"Topography"',
        #     position="x11.5c/6.6c+w6c+jTC+v" #for vertical colorbar
            )

    if inv:
        seed_ids = inventory2traceid(inv, force_location_code='')
        if not stations:
            stations = [id.split('.')[1] for id in seed_ids]
            #invstations
        stalat = [inv.get_coordinates(seed_id)['latitude'] for seed_id in seed_ids]
        stalon = [inv.get_coordinates(seed_id)['longitude'] for seed_id in seed_ids]
        
        if add_labels:
            #print('Adding station labels')
            for thislat, thislon, this_id in zip(stalat, stalon, seed_ids):
                net, sta, loc, chan = this_id.split('.')
                if sta in stations:
                    fig.plot(x=thislon, y=thislat, style="s0.3c", fill="green", pen='darkgreen') 
                    fig.text(x=thislon, y=thislat, text=sta, textfiles=None, \
                            font="darkgreen",
                            justify="ML",
                            offset="0.2c/0c",)
                else:
                    fig.plot(x=thislon, y=thislat, style="s0.3c", fill="red", pen='darkred') 
                    fig.text(x=thislon, y=thislat, text=sta, textfiles=None, \
                            font="darkred",
                            justify="ML",
                            offset="0.2c/0c",)
        else:
            fig.plot(x=stalon, y=stalat, style="s0.3c", fill="dodgerblue4", pen='2p,blue') 

    if title:
        fig.text(
            text=title,
            x=region[0] + (region[1]-region[0]) * 0.60,
            y=region[2] + (region[3]-region[2]) * 0.90,
            justify="TC",
            font="11p,Helvetica-Bold,black"
        )
    print('GOT HERE 1')
    fig.basemap(region=region, frame=True)

    if show:
        fig.show();
    print('GOT HERE 2')
    return fig

class Grid:
    def __init__(self, centerlat, centerlon, nlat, nlon, node_spacing_m):
        deg2m = degrees2kilometers(1.0) * 1000.0
        node_spacing_lat = node_spacing_m / deg2m
        minlat = centerlat - (nlat-1)/2 * node_spacing_lat
        maxlat = centerlat + (nlat-1)/2 * node_spacing_lat
        latrange = np.array([lat for lat in np.arange(minlat, maxlat, node_spacing_lat)])
        node_spacing_lon = node_spacing_lat / np.cos(centerlat / (2 * math.pi))
        minlon = centerlon - (nlon-1)/2 * node_spacing_lon
        maxlon = centerlon + (nlon+1)/2 * node_spacing_lon
        lonrange = np.array([lon for lon in np.arange(minlon, maxlon, node_spacing_lon)])
        gridlon, gridlat = np.meshgrid(lonrange, latrange)
        self.gridlon = gridlon
        self.gridlat = gridlat
        self.node_spacing_lat = node_spacing_lat
        self.node_spacing_lon = node_spacing_lon
        self.lonrange = lonrange
        self.latrange = latrange

    def plot(self, node_spacing_m, DEM_DIR=None):
        fig = montserrat_topo_map(DEM_DIR=DEM_DIR)
        #plt.plot(self.gridlon, self.gridlat, marker='+', color='k', linestyle='none')
        #plt.show()
        symsize = node_spacing_m/2000
        stylestr = f'+{symsize}c'
        
        fig.plot(x=self.gridlon.reshape(-1), y=self.gridlat.reshape(-1), style=stylestr, pen='black')
        fig.basemap(region=[minlon, maxlon, minlat, maxlat], frame=True)
        fig.show()
        #fig._cleanup()

def initial_source(lat=dome_location['lat'], lon=dome_location['lon']):
    return {'lat':lat, 'lon':lon}

def make_grid(center_lat=dome_location['lat'], center_lon=dome_location['lon'], node_spacing_m = 100, grid_size_lat_m = 10000, grid_size_lon_m = 8000):
    nlat = int(grid_size_lat_m/node_spacing_m) + 1
    nlon = int(grid_size_lon_m/node_spacing_m) + 1
    return Grid(center_lat, center_lon, nlat, nlon, node_spacing_m)  


def simulate_VSAM(inv, source, units='m/s', surfaceWaves=False, wavespeed_kms=1.5, peakf=8.0, Q=None, noise_level_percent=0.0):
    npts = len(source['DR'])
    seed_ids = inventory2seedids(inv, force_location_code='')
    st = obspy.Stream()
    
    for id in seed_ids:
        coordinates = inv.get_coordinates(id)
        stalat = coordinates['latitude']
        stalon = coordinates['longitude']
        distance_km = degrees2kilometers(locations2degrees(stalat, stalon, source['lat'], source['lon']))
        tr = obspy.Trace()
        tr.id = id
        tr.stats.starttime = source['t'][0]
        tr.stats.delta = source['t'][1] - source['t'][0]
        gsc = VSAM.compute_geometrical_spreading_correction(distance_km, tr.stats.channel, surfaceWaves=surfaceWaves, wavespeed_kms=wavespeed_kms, peakf=peakf)
        isc = VSAM.compute_inelastic_attenuation_correction(distance_km, peakf, wavespeed_kms, Q)
        tr.data = source['DR'] / (gsc * isc) * 1e-7
        if noise_level_percent > 0.0:
            tr.data += np.multiply(np.nanmax(tr.data), np.random.uniform(0, 1, size=npts) )
            pass # do something here
        tr.stats['units'] = units
        st.append(tr)
    return VSAM(stream=st, sampling_interval=tr.stats.delta)

def plot_VSAM(dsamobj, gridobj, nodenum, metric='mean', DEM_DIR=None):
    x = [id for id in dsamobj.dataframes]
    st = dsamobj.to_stream(metric=metric)
    y = [tr.data[nodenum] for tr in st]
    plt.figure()
    plt.bar(x, y, width=1.0)
    fig = montserrat_topo_map(show=False, zoom_level=0, inv=None, add_labels=False, centerlon=-62.177, centerlat=16.711, contour_interval=100, topo_color=True, resolution='03s', DEM_DIR=DEM_DIR)
    print('is figure?', isinstance(fig, pygmt.Figure))
    #ax = fig.axes()
    #ax[0].plot(gridobj.gridlon[nodenum], gribobj.gridlat[nodenum], 'o')
    fig.plot(gridobj.gridlon[nodenum], gridobj.gridlat[nodenum], 'o')
    #fig._cleanup()

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
        if os.path.exists(cache_path):
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



    def metric2stream(self):
        st = self.samobject.to_stream(metric=self.metric)
        if st[0].stats.sampling_rate != self.window_seconds:
            window = np.ones(self.window_seconds) / self.window_seconds
            for tr in st:
                tr.data = np.convolve(tr.data, window, mode='same')
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
        return self.source

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
            
            if join:
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
                print('Got here 0')
                fig = montserrat_topo_map(zoom_level=zoom_level, inv=self.inventory, \
                                          centerlat=y[maxi], centerlon=x[maxi], add_labels=add_labels, 
                                          topo_color=False, stations=stations, title=title, region=region)

                print('Got here 3')
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
                print('Got here 4')
                fig.plot(x=x, y=y, size=symsize, style="cc", pen=None, fill=timecolor, cmap=True)
                print('Got here 4.1')
                fig.colorbar(
                    frame='+l"Sequence"',
                    #     position="x11.5c/6.6c+w6c+jTC+v" #for vertical colorbar
                    )

                '''
                    fig.plot(x=x, y=y, size=symsize, style="cc", fill='black', pen='1p,black')
                    k = 1
                    for i in range(len(x)):
                        if DR[i] > threshold_DR:
                            fig.text(x=x[i], y=y[i], text=f"{k}", textfiles=None, \
                                #font="Courier-Bold",
                                font="red",
                                justify="ML",
                                offset="0.2c/0c",)
                            k += 1
                         
                else:
                    fig.plot(x=x, y=y, size=symsize, style="cc", fill='black', pen='1p,black')
                    fig.plot(x=x, y=y, style="f1c/0.05c+c", fill='black', pen='0.5p,black')
                    fig.plot(x=x[maxi], y=y[maxi], size=symsize[maxi], style="cc", fill='red', pen='1p,red')
                '''
                print('Got here 5')
                if region:
                    fig.basemap(region=region, frame=True)
                if outfile:
                    fig.savefig(outfile)
                else:
                    fig.show();  
                print('\n')
                #fig._cleanup()           
                
            else:    
                # Heatmap
                df = pd.DataFrame()
                df['time'] = source['t']
                df['lon'] = source['lon']
                df['lat'] = source['lat']
                df['DR'] = source['DR']
                df['energy'] = np.multiply(source['DR'], source['DR'])
                unique_locationsDF = df.groupby(['lat', 'lon'])['energy'].sum().reset_index()
                fig = montserrat_topo_map(zoom_level=zoom_level, inv=self.inventory)
                x=unique_locationsDF['lon'].to_numpy()
                y=unique_locationsDF['lat'].to_numpy()
                symsize = np.sqrt(unique_locationsDF['energy'].to_numpy())
                symsize = np.divide(symsize, np.nanmax(symsize))*scale
                fig.plot(x=x, y=y, size=symsize, style='cc', fill='black', pen='2p,black')
                if outfile:
                    fig.savefig(outfile)
                else:
                    fig.show();   
                #fig._cleanup()
            
            

            '''
            # time-longitude plot
            plt.figure()
            plt.scatter(t_dt, lon, s=source['DR']*cross_scale, marker='x')  

            # time-latitude plot
            plt.figure()
            plt.scatter(t_dt, lat, s=source['DR']*cross_scale, marker='x')
            '''
            
        else: # no location data      
            fig = montserrat_topo_map(zoom_level=zoom_level, inv=self.inventory, show=True, add_labels=add_labels)
            #fig._cleanup()




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


def plot_heatmap_montserrat_colored(df, lat_col='latitude', lon_col='longitude', amp_col='amplitude',
                                     zoom_level=0, inventory=None, color_scale=0.4,
                                     cmap='turbo', log_scale=True, contour=False,
                                     node_spacing_m=50, outfile=None):
    """
    Plot ASL heatmap on Montserrat topography using tessellated color-filled squares.

    Parameters:
    - df: pandas DataFrame with lat/lon and amplitude columns
    - lat_col, lon_col, amp_col: column names
    - zoom_level: zoom level for montserrat_topo_map
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
    fig = montserrat_topo_map(zoom_level=zoom_level, inv=inventory, topo_color=False)

    # Plot colored square tiles

    fig.plot(x=x, y=y, style=f"s{symbol_size_m}c", fill=z, cmap=True, pen=None)

    # Optional contours
    if contour:
        from scipy.interpolate import griddata
        xi = np.linspace(min(x), max(x), 200)
        yi = np.linspace(min(y), max(y), 200)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        fig.grdcontour(grid=Zi, region=fig.region, interval=0.5, pen="0.75p,gray")

    # Add colorbar
    fig.colorbar(frame='+l"Log10 Total Energy"' if log_scale else '+l"Total Energy"')

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



def extract_asl_diagnostics(topdir, output_csv=None):
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

    if output_csv:
        output_path = output_csv
    else:
        timestamp = int(UTCDateTime().timestamp)
        output_path = f'/home/thompsong/Dropbox/ASL_results_{timestamp}.csv'

    df.to_csv(output_path, index=False)
    print(f"[✓] Saved ASL diagnostics to: {output_path}")
    return df

 
