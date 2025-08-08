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

def plot_inv(inv):
    inv.plot(water_fill_color=[0.0, 0.5, 0.8], continent_fill_color=[0.1, 0.6, 0.1], size=30);
    return