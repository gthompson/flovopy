import numpy as np

from obspy import read, Stream, UTCDateTime
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.event import Catalog, Event, Origin, Magnitude, Comment
from obspy.clients.nrl import NRL
from obspy.core.util import AttribDict
from obspy.geodetics import locations2degrees, degrees2kilometers


def inventory_fix_ids(inv, netcode='MV'):
    for network in inv.networks:
        if not network.code:
            network.code = netcode
        for station in network.stations:
            for channel in station.channels:
                if channel.code[0] in 'FCES':
                    shortperiod = True
                if channel.code[0] in 'GDBH':
                    shortperiod = False
                Fs = channel.sample_rate
                nslc = network.code + '.' + station.code + '..' + channel.code
                if netcode=='MV':
                    nslc = correct_nslc_mvo(nslc, Fs, shortperiod=shortperiod)
                net, sta, loc, chan = nslc.split('.')
                channel.code = chan
            station.code = sta

######################################################################
##  Additional tools for ObsPy Inventory class                      ##
######################################################################

def inventory2traceid(inv, chancode='', force_location_code='*'):
    trace_ids = list()

    for networkObject in inv:
        if chancode:
            networkObject = networkObject.select(channel=chancode)
        stationObjects = networkObject.stations

        for stationObject in stationObjects:
            channelObjects = stationObject.channels
            for channelObject in channelObjects:
                this_trace_id = networkObject.code + '.' + stationObject.code + f'.{force_location_code}.' + channelObject.code
                trace_ids.append(this_trace_id)
    
    return trace_ids


def attach_station_coordinates_from_inventory(inventory, st):
    """ attach_station_coordinates_from_inventory """
    from obspy.core.util import AttribDict
    for tr in st:
        for netw in inventory.networks:
            for sta in netw.stations:
                if tr.stats.station == sta.code and netw.code == tr.stats.network:
                    for cha in sta.channels:
                        if tr.stats.location == cha.location_code:
                            tr.stats.coordinates = AttribDict({
                                'latitude':cha.latitude,
                                'longitude':cha.longitude,
                                'elevation':cha.elevation})
                            #tr.stats.latitude = cha.latitude
                            #tr.stats.longitude = cha.longitude  
                            
                                                      
def attach_distance_to_stream(st, olat, olon):
    import obspy.geodetics
    for tr in st:
        try:
            alat = tr.stats['coordinates']['latitude']
            alon = tr.stats['coordinates']['longitude']
            print(alat, alon, olat, olon)
            distdeg = obspy.geodetics.locations2degrees(olat, olon, alat, alon)
            distkm = obspy.geodetics.degrees2kilometers(distdeg)
            tr.stats['distance'] =  distkm * 1000
        except Exception as e:
            print(e)
            print('cannot compute distance for %s' % tr.id)

def create_trace_inventory(tr, netname='', sitename='', net_ondate=None, \
                           sta_ondate=None, lat=0.0, lon=0.0, elev=0.0, depth=0.0, azimuth=0.0, dip=-90.0, stationXml=None):
    from obspy.core.inventory import Inventory, Network, Station, Channel, Site
    from obspy.clients.nrl import NRL    
    inv = Inventory(networks=[], source='Glenn Thompson')
    if not sta_ondate:
        net_ondate = tr.stats.starttime
    if not sta_ondate:
        sta_ondate = tr.stats.starttime        
    net = Network(
        # This is the network code according to the SEED standard.
        code=tr.stats.network,
        # A list of stations. We'll add one later.
        stations=[],
        description=netname,
        # Start-and end dates are optional.
        start_date=net_ondate)

    sta = Station(
        # This is the station code according to the SEED standard.
        code=tr.stats.station,
        latitude=lat,
        longitude=lon,
        elevation=elev,
        creation_date=sta_ondate, 
        site=Site(name=sitename))

    cha = Channel(
        # This is the channel code according to the SEED standard.
        code=tr.stats.channel,
        # This is the location code according to the SEED standard.
        location_code=tr.stats.location,
        # Note that these coordinates can differ from the station coordinates.
        latitude=lat,
        longitude=lon,
        elevation=elev,
        depth=depth,
        azimuth=azimuth,
        dip=dip,
        sample_rate=tr.stats.sampling_rate)

    # By default this accesses the NRL online. Offline copies of the NRL can
    # also be used instead
    nrl = NRL()
    # The contents of the NRL can be explored interactively in a Python prompt,
    # see API documentation of NRL submodule:
    # http://docs.obspy.org/packages/obspy.clients.nrl.html
    # Here we assume that the end point of data logger and sensor are already
    # known:
    response = nrl.get_response( # doctest: +SKIP
        sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
        datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])


    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)
    
    # And finally write it to a StationXML file. We also force a validation against
    # the StationXML schema to ensure it produces a valid StationXML file.
    #
    # Note that it is also possible to serialize to any of the other inventory
    # output formats ObsPy supports.
    if stationXml:
        print('Writing inventory to %s' % stationXml)
        inv.write(stationXml, format="stationxml", validate=True)    
    return inv


def _merge_channels(chan1, chan, channel_codes):
    index2 = channel_codes.index(chan.code)
    print('Want to merge channel:')
    print(chan)
    print('into:')
    print(chan1[index2])
    print(chan1[index2].startDate)

    
def _add_channel(chan1, chan):
    # add as new channel
    print('Adding new channel %s' % chan.code)
    chan1.append(chan)  
    
def _merge_stations(sta1, sta, station_codes):
    index = station_codes.index(sta.code)
    print('Merging station')  
    for chan in sta.channels:
        channel_codes = [chan.code for chan in sta1[index].channels]
        if chan.code in channel_codes: 
            #merge_channels(sta1[index].channels, chan, channel_codes)
            _add_channel(sta1[index].channels, chan)
        else: # add as new channel
            _add_channel(sta1[index].channels, chan)           
            
def _add_station(sta1, sta):
    # add as new station
    print('Adding new station %s' % sta.code)
    sta1.append(sta)            
            
def merge_inventories(inv1, inv2): # obsolete - can just add inventories
    netcodes1 = [this_net.code for this_net in inv1.networks]
    for net2 in inv2.networks:
        if net2.code in netcodes1:
            netpos = netcodes1.index(net2.code)
            for sta in net2.stations:
                if inv1.networks[netpos].stations:
                    station_codes = [sta.code for sta in inv1.networks[netpos].stations]
                    if sta.code in station_codes: 
                        _merge_stations(inv1.networks[netpos].stations, sta, station_codes)
                    else:
                        _add_station(inv1.networks[netpos].stations, sta)
                else:
                    _add_station(inv1.networks[netpos].stations, sta)            
        else: # this network code from inv2 does not exist in inv1
            inv1.networks.append(net2)
            netpos = -1
            
def modify_inventory(inv, thisid, lat=None, lon=None, elev=None):
    #netcodes = [this_net.code for this_net in inv.networks]
    thisnet, thissta, thisloc, thischan = thisid.split('.')
    for netindex, net in enumerate(inv.networks):
        if net.code==thisnet:
            for sta in net.stations:
                station_codes = [sta.code for sta in net.stations]
                if sta.code==thissta: 
                    if lat:
                        sta.latitude = lat
                    if lon:
                        sta.longitude = lon
                    if elev:
                        sta.elevation = elev
                    for chan in sta.channels:
                        if chan.code==thischan and chan.location_code==thisloc:
                            print('match found')
                            chan.depth=0.0
                            code0 = chan.code[0]
                            if lat:
                                chan.latitude = lat
                            if lon:
                                chan.longitude = lon
                            if elev:
                                chan.elevation = elev
                            if chan.sample_rate < 80.0 and chan.sample_rate > 20.0:
                                if chan.code[0] == 'E': 
                                    code0 = 'S'
                                elif chan.code[0] == 'H': 
                                    code0 = 'B'
                            elif chan.sample_rate > 80.0 and chan.sample_rate < 250.0:
                                if chan.code[0] == 'B': 
                                    code0 = 'H'
                                elif chan.code[0] == 'S': 
                                    code0 = 'E'
                            chan.code = code0 + chan.code[1:]
                                

def show_response(inv):
    # Loop over the stations and channels to access the sensitivity
    for network in inv:
        print(f'Network: {network}')
        for station in network:
            print(f'Station: {station}')
            for channel in station:
                print(f'Channel: {channel}')
                if channel.response is not None:
                    # The response object holds the overall sensitivity
                    #sensitivity = channel.response.sensitivity
                    print(f'Response: {channel.response}')
    print('*****************************\n\n\n')



def has_response(inventory, trace):
    """
    Check if a response exists for a given Trace in an ObsPy Inventory.

    :param inventory: ObsPy Inventory object
    :param trace: ObsPy Trace object (from a Stream)
    :return: True if response exists, False otherwise
    """
    try:
        # Get response for the trace's network, station, location, channel, and start time
        resp = inventory.get_response(trace.id, trace.stats.starttime)
        return resp is not None  # Response exists
    except Exception:
        return False  # No response found



def calculate_sensitivity_all(inventory):
    # Iterate through all networks and stations in the inventory
    for network in inventory:
        for station in network:
            for channel in station:
                if channel.response:  # Check if the station has a response
                    # Recalculate the sensitivity for the station's response
                    #sensitivity = calculate_sensitivity(channel.response)
                    channel.response.recalculate_overall_sensitivity()
                    
                    # Now you can do something with the recalculated sensitivity, such as:
                    # - Print it
                    #print(f"{network.code}.{station.code}{channel.code} has recalculated sensitivity: {sensitivity}")
                    
                    # - Store it in the station's metadata (if needed)
                    # station.sensitivity = sensitivity  # You can add an attribute for sensitivity if required

    # Optionally, save the modified inventory if you want to store the changes
    # inventory.write("modified_inventory.xml", format="STATIONXML")   

def subset_inv(inv, st, st_subset):
    # subset an inventory based on a stream object which is a subset of another
    try:
        inv_new = inv.copy() # got an error here once that Inventory has no copy(), but it does
        for tr in st:
            if len(st_subset.select(id=tr.id))==0:
                inv_new = inv_new.remove(network=tr.stats.network, station=tr.stats.station, location=tr.stats.location, channel=tr.stats.channel)
        return inv_new
    except:
        print('Failed to subset inventory. Returning unchanged')
        return inv



def plot_inv(inv):
    inv.plot(water_fill_color=[0.0, 0.5, 0.8], continent_fill_color=[0.1, 0.6, 0.1], size=30);
    return