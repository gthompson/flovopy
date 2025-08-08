import os
import pandas as pd
from obspy import UTCDateTime
from obspy.core.inventory.response import Response, InstrumentSensitivity
from obspy.clients.nrl import NRL
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from flovopy.stationmetadata.utils import (
    responses2inventory,
    expand_channel_code,
    build_dataframe_from_table
)
from collections import defaultdict


def NRL2inventory(
    nrl_path, net, sta, loc, chans,
    datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0,
    lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='',
    ondate=UTCDateTime(1970, 1, 1), offdate=UTCDateTime(2025, 12, 31),
    sensitivity=None, units=None
) -> Inventory:
    """
    Constructs an ObsPy Inventory using datalogger/sensor response from the IRIS NRL or local NRL path.

    If sensor/datalogger keys are not found, provides fallback PAZ responses for some supported sensors.

    Parameters:
    -----------
    nrl_path : str
        Path to local NRL directory or None to use remote IRIS NRL.
    net : str
        Network code (e.g. 'XX').
    sta : str
        Station code (e.g. 'ABCD').
    loc : str
        Location code (e.g. '00').
    chans : list of str
        List of channel codes (e.g. ['HDF', 'HDF']).
    datalogger : str
        Datalogger type ('Centaur', 'RT130', etc.).
    sensor : str
        Sensor model name or keyword (e.g. 'TCP', 'L-22', 'infrabsu').
    Vpp : float
        Peak-to-peak voltage (1 or 40 for Centaur).
    fsamp : float
        Sampling rate in Hz.
    lat, lon, elev, depth : float
        Station coordinates.
    sitename : str
        Optional site name (currently unused).
    ondate, offdate : UTCDateTime
        Start and end date for metadata validity.
    sensitivity : float, optional
        Fallback sensitivity value for custom sensor.
    units : str, optional
        Input units for fallback response (e.g. 'm/s', 'Pa').

    Returns:
    --------
    inventory : obspy.core.inventory.Inventory
        Inventory containing one station with the specified response.
    """
    if os.path.isdir(nrl_path):
        nrl = NRL(nrl_path)
    else:
        nrl = NRL("http://ds.iris.edu/NRL/")

    # Datalogger NRL keys
    if datalogger == 'Centaur':
        if Vpp == 40:
            datalogger_keys = ['Nanometrics', 'Centaur', '40 Vpp (1)', 'Off', 'Linear phase', f"{int(fsamp)}"]
        elif Vpp == 1:
            datalogger_keys = ['Nanometrics', 'Centaur', '1 Vpp (40)', 'Off', 'Linear phase', f"{int(fsamp)}"]
        else:
            raise ValueError(f"Unsupported Vpp: {Vpp}")
    elif datalogger == 'RT130':
        datalogger_keys = ['REF TEK', 'RT 130 & 130-SMA', '1', f"{int(fsamp)}"]
    else:
        raise ValueError(f"Unsupported datalogger: {datalogger}")

    # Sensor NRL keys
    if sensor == 'TCP':
        sensor_keys = ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']
    elif sensor == 'L-22':
        sensor_keys = ['Sercel/Mark Products', 'L-22D', '2200 Ohms', '10854 Ohms']
    elif sensor.lower().startswith('chap'):
        sensor_keys = ['Chaparral Physics', '25', 'Low: 0.4 V/Pa']
    elif sensor.lower() == 'infrabsu':
        sensor_keys = ['JeffreyBJohnson', 'infraBSU', '0.000046 V/Pa']
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

    try:
        thisresponse = nrl.get_response(sensor_keys=sensor_keys, datalogger_keys=datalogger_keys)
    except Exception as e:
        print(f"[WARNING] NRL lookup failed for {net}.{sta}.{loc} ({sensor}): {e}")

        # Fallback PAZ response for selected sensors
        if sensor.lower() == 'infrabsu':
            poles = [-0.301593 + 0j]
            zeros = [0j]
            sensitivity = sensitivity or 18400.0
        elif sensor.lower().startswith('chap'):
            poles = [-1.0 + 0j, -3.03 + 0j, -3.03 + 0j]
            zeros = [0j, 0j, 0j]
            sensitivity = sensitivity or 160000.0
        else:
            raise ValueError(f"No fallback response available for sensor: {sensor}")

        input_units = units or 'm/s'
        output_units = 'Counts'

        try:
            thisresponse = Response.from_paz(
                zeros=zeros, poles=poles, sensitivity=sensitivity,
                input_units=input_units, output_units=output_units
            )
        except Exception as err:
            print(f"[ERROR] Failed to build fallback PAZ response: {err}")
            thisresponse = Response(instrument_sensitivity=InstrumentSensitivity(
                value=sensitivity, frequency=1.0,
                input_units=input_units, output_units=output_units
            ))

    responses = {chan: thisresponse for chan in chans}
    inventory = responses2inventory(
        net, sta, loc, fsamp, responses,
        lat=lat, lon=lon, elev=elev, depth=depth,
        start_date=ondate, end_date=offdate
    )
    return inventory

'''
def build_inventory_from_csv(csv_path: str, nrl_path: str = None) -> Inventory:
    """
    Builds a full ObsPy Inventory by reading a CSV file of station metadata.

    Each row in the CSV defines one station's configuration and is converted into a
    response using the IRIS NRL or local NRL. All stations are merged into one Inventory.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file containing network, station, location, channel, and response info.
    nrl_path : str, optional
        Path to local NRL directory, or None to use remote IRIS NRL.

    Returns:
    --------
    master_inventory : obspy.core.inventory.Inventory
        Full Inventory containing all stations described in the CSV.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["ondate"] = df["ondate"].apply(UTCDateTime)
    df["offdate"] = df["offdate"].apply(UTCDateTime)

    master_inventory = Inventory(networks=[], source="build_inventory_from_csv")

    for _, row in df.iterrows():
        net = str(row["network"]).strip()
        sta = str(row["station"]).strip()
        loc = str(row["location"]).strip()
        chan_list = expand_channel_code(row["channel"])
        lat = float(row["lat"] or 0.0)
        lon = float(row["lon"] or 0.0)
        elev = float(row["elev"] or 0.0)
        depth = float(row["depth"] or 0.0)
        fsamp = float(row["fsamp"] or 100.0)
        vpp = float(row["vpp"] or 40.0)
        datalogger = row['datalogger']
        sensor = row['sensor']

        try:
            inv = NRL2inventory(
                nrl_path, net, sta, loc, chan_list,
                datalogger=datalogger, sensor=sensor, Vpp=vpp, fsamp=fsamp,
                lat=lat, lon=lon, elev=elev, depth=depth,
                ondate=row["ondate"], offdate=row["offdate"]
            )
            master_inventory += inv
        except Exception as e:
            print(f"[ERROR] Skipped {net}.{sta}.{loc}: {e}")

    return master_inventory
'''
def sensor_type_dispatch(
    row, 
    nrl_path=None, 
    infrabsu_xml=None,
    verbose=False
) -> Inventory:
    """
    Handle special sensor types (Raspberry Shake, infraBSU, Chaparral) and return an Inventory.
    """

    # Extract row values safely
    net = str(row["network"]).strip()
    sta = str(row["station"]).strip()
    loc = str(row["location"]).strip()
    chans = expand_channel_code(row["channel"])
    lat = float(row.get("lat", 0.0))
    lon = float(row.get("lon", 0.0))
    elev = float(row.get("elev", 0.0))
    depth = float(row.get("depth", 0.0))
    fsamp = float(row.get("fsamp", 100.0))
    vpp = float(row.get("vpp", 40.0))
    datalogger = str(row.get('datalogger', '')).strip()
    sensor = str(row.get('sensor', '')).strip()
    ondate = row.get("ondate", UTCDateTime("2000-01-01"))
    offdate = row.get("offdate", UTCDateTime("2100-01-01"))

    inv = Inventory(networks=[], source="sensor_type_dispatch")

    # --- Special handling ---
    if datalogger.upper().startswith("RS"):
        if datalogger.upper() == "RSB":
            inv = get_rsb(sta, loc)
        elif datalogger.upper() == "RBOOM":
            inv = get_rboom(sta, loc)
        elif datalogger.upper() == "RS1D":
            inv = get_rs1d_v4(sta, loc, fsamp=fsamp)
        elif datalogger.upper() == "RS3D":
            inv = get_rs3d_v5(sta, loc)
        else:
            raise ValueError(f"Unknown Raspberry Shake model: {datalogger}")
        for netobj in inv:
            for staobj in netobj:
                staobj.latitude = lat
                staobj.longitude = lon
                staobj.elevation = elev
                staobj.depth = depth
                for ch in staobj:
                    ch.latitude = lat
                    ch.longitude = lon
                    ch.elevation = elev
                    ch.depth = depth
        return inv

    elif sensor.lower().startswith("infrabsu") and infrabsu_xml:
        for ch in chans:
            inv += build_combined_infrabsu_centaur_stationxml(
                fsamp=fsamp,
                vpp=vpp,
                stationxml_path=infrabsu_xml,
                network=net,
                station=sta,
                location=loc,
                channel=ch.strip(),
                latitude=lat,
                longitude=lon,
                elevation=elev,
                depth=depth,
                start_date=ondate,
                end_date=offdate
            )
        return inv

    elif sensor.upper().startswith("CHAP"):
        inv = NRL2inventory(
            nrl_path,
            net,
            sta,
            loc,
            chans,
            fsamp=fsamp,
            Vpp=vpp,
            datalogger=datalogger,
            sensor=sensor,
            lat=lat,
            lon=lon,
            elev=elev,
            depth=depth,
            ondate=ondate,
            offdate=offdate
        )
        return inv

    else:
        # Default: try NRL
        inv = NRL2inventory(
            nrl_path,
            net,
            sta,
            loc,
            chans,
            fsamp=fsamp,
            Vpp=vpp,
            datalogger=datalogger,
            sensor=sensor,
            lat=lat,
            lon=lon,
            elev=elev,
            depth=depth,
            ondate=ondate,
            offdate=offdate
        )
        return inv
    
def build_inventory_from_dataframe(df, nrl_path=None, infrabsu_xml=None, verbose=True):
    """
    Converts a cleaned metadata DataFrame into an ObsPy Inventory.

    This function dispatches each row to an appropriate sensor handling function
    based on sensor type (e.g., Raspberry Shake, infraBSU, Chaparral, or NRL-based).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned metadata table with one row per NSLC entry. Must include columns:
        'network', 'station', 'location', 'channel', 'sensor', 'datalogger',
        'lat', 'lon', 'elev', 'depth', 'fsamp', 'vpp', 'ondate', and 'offdate'.
    nrl_path : str, optional
        Path to local NRL directory. If None, remote IRIS NRL is used.
    infrabsu_xml : str, optional
        Path to cached base StationXML file for infraBSU response reuse.
    verbose : bool, default=True
        If True, logs progress and warnings.

    Returns
    -------
    obspy.core.inventory.Inventory
        Combined ObsPy Inventory object for all valid rows.
    """
    inventory = Inventory(networks=[], source="build_inventory_from_dataframe")

    for _, row in df.iterrows():
        try:
            inv_piece = sensor_type_dispatch(
                row,
                nrl_path=nrl_path,
                infrabsu_xml=infrabsu_xml,
                verbose=verbose
            )
            inventory += inv_piece
        except Exception as e:
            if verbose:
                net = row.get("network", "?")
                sta = row.get("station", "?")
                loc = row.get("location", "")
                chan = row.get("channel", "")
                print(f"[WARN] Failed to parse {net}.{sta}.{loc}.{chan}: {e}")

    return inventory

def build_inventory_from_table(path, sheet_name='ksc_stations_master', nrl_path=None, infrabsu_xml=None, verbose=True):
    """
    Builds an ObsPy Inventory from an Excel or CSV metadata table.

    This function reads a metadata file (CSV or Excel), processes it into a clean
    DataFrame, and converts the DataFrame into an ObsPy Inventory. It supports
    response reuse for infraBSU sensors if a prebuilt StationXML is provided.

    Parameters
    ----------
    path : str
        Path to the Excel (.xlsx) or CSV (.csv) metadata file.
    sheet_name : str, optional
        Sheet name to read (used only for Excel files). Default is 'ksc_stations_master'.
    nrl_path : str, optional
        Path to a local copy of the NRL. If None, remote NRL is used.
    infrabsu_xml : str, optional
        Path to a prebuilt StationXML file for infraBSU response reuse.
    verbose : bool, default=True
        If True, print progress and warnings.

    Returns
    -------
    obspy.core.inventory.Inventory
        ObsPy Inventory object representing all stations and responses.
    """
    df = build_dataframe_from_table(path, sheet_name=sheet_name)
    return build_inventory_from_dataframe(df, nrl_path=nrl_path, infrabsu_xml=infrabsu_xml, verbose=verbose)


def create_trace_inventory(tr, netname='', sitename='', net_ondate=None, \
                           sta_ondate=None, lat=0.0, lon=0.0, elev=0.0, depth=0.0, azimuth=0.0, dip=-90.0, stationXml=None):
    """
    Create a minimal ObsPy Inventory for a single trace, using NRL responses.

    This function constructs a minimal station inventory for one trace using metadata from
    the trace header and assumed/default metadata values. It fetches the instrument response
    from the online NRL (Nominal Response Library) and optionally writes a StationXML file.

    Parameters:
    tr (Trace): ObsPy Trace object.
    netname (str): Description of the network.
    sitename (str): Description of the station site.
    net_ondate (UTCDateTime or None): Start date for the network. Defaults to trace starttime.
    sta_ondate (UTCDateTime or None): Start date for the station. Defaults to trace starttime.
    lat (float): Station latitude.
    lon (float): Station longitude.
    elev (float): Station elevation in meters.
    depth (float): Channel sensor depth in meters.
    azimuth (float): Sensor azimuth in degrees.
    dip (float): Sensor dip in degrees (e.g. -90 for vertical).
    stationXml (str or None): Optional output path for StationXML file.

    Returns:
    Inventory: ObsPy Inventory containing the trace's metadata and instrument response.
    """
 
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


           
'''            
def merge_inventories(inv1, inv2):
    """
    Merge two ObsPy Inventory objects with code-aware deduplication and hierarchical merging.

    This function merges `inv2` into `inv1` in place, taking care to avoid duplication
    at the network, station, and channel levels. If a network, station, or channel in
    `inv2` already exists in `inv1`, it is not blindly appended — instead, the function
    recursively inspects and merges child elements.

    ⚠️ This differs significantly from `inv1 += inv2` or `inv1 + inv2`, which simply
    concatenates all elements and does not perform deduplication. Using `Inventory.__add__()`
    may result in invalid StationXML if multiple networks, stations, or channels share
    the same codes but have differing metadata.

    This method is particularly useful when combining inventories from overlapping sources,
    or when preserving clean NSLC structures is critical (e.g., for visualization, export,
    or further metadata refinement).

    Parameters:
    ----------
    inv1 : obspy.core.inventory.inventory.Inventory
        The primary inventory object to be modified in place.

    inv2 : obspy.core.inventory.inventory.Inventory
        The secondary inventory to merge into `inv1`.

    Returns:
    -------
    None
        Modifies `inv1` in place.
    """


    
    def _add_channel(chan1, chan):
        """
        Add a new channel to an existing list of channels.

        Parameters:
        chan1 (list[Channel]): List of existing channels in a station.
        chan (Channel): New channel to add.
        """
        # add as new channel
        print('Adding new channel %s' % chan.code)
        chan1.append(chan)  
        

    def _merge_stations(sta1, sta, station_codes):
        """
        Merge a station into an existing list of stations by merging or adding channels.

        Parameters:
        sta1 (list[Station]): List of existing stations in a network.
        sta (Station): New station to merge.
        station_codes (list[str]): List of station codes in the network.
        """
        index = station_codes.index(sta.code)
        print('Merging station')  
        for chan in sta.channels:
            channel_codes = [chan.code for chan in sta1[index].channels]
            if chan.code in channel_codes:
                print(f"Channel {chan.code} already exists — skipping.")
                continue
            else:
                _add_channel(sta1[index].channels, chan)       
                
    def _add_station(sta1, sta):
        """
        Add a new station to a list of stations.

        Parameters:
        sta1 (list[Station]): List of existing stations in a network.
        sta (Station): Station object to add.
        """
        # add as new station
        print('Adding new station %s' % sta.code)
        sta1.append(sta) 

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
'''




def merge_inventories(*inventories):
    """
    Merge and patch one or more ObsPy Inventory objects into a single, deduplicated Inventory.

    This function:
    - Combines all provided inventories into a new Inventory.
    - Deduplicates networks and stations.
    - Merges stations with the same code, even across overlapping or adjacent time windows.
    - Merges channels by (code, location_code) and allows multiple versions if:
        - Time windows do not overlap, or
        - Metadata differs (with a warning).
    - Fills missing site name and description where possible.

    Parameters:
    -----------
    *inventories : obspy.core.inventory.Inventory
        One or more Inventory objects to be merged and cleaned.

    Returns:
    --------
    Inventory
        A new, merged, deduplicated, and patched Inventory object.
    """
    def channels_overlap(c1, c2):
        """Check if two channels overlap in time."""
        s1 = c1.start_date or UTCDateTime(0)
        e1 = c1.end_date or UTCDateTime(2100, 1, 1)
        s2 = c2.start_date or UTCDateTime(0)
        e2 = c2.end_date or UTCDateTime(2100, 1, 1)
        return s1 <= e2 and s2 <= e1

    def channel_metadata_equivalent(c1, c2):
        """Compare essential metadata fields between two channels."""
        keys = ["code", "location_code", "sample_rate", "azimuth", "dip"]
        for key in keys:
            if getattr(c1, key, None) != getattr(c2, key, None):
                return False
        # Compare responses (hash or text representation)
        try:
            return c1.response.__str__() == c2.response.__str__()
        except Exception:
            return False

    merged_inv = Inventory(networks=[], source="Merged and patched by merge_inventories()")

    all_networks = defaultdict(list)
    for inv in inventories:
        for net in inv:
            all_networks[net.code].append(net)

    for net_code, nets in all_networks.items():
        merged_net = Network(code=net_code, description=nets[0].description or "")
        stations_by_code = defaultdict(list)

        for net in nets:
            for sta in net:
                stations_by_code[sta.code].append(sta)

        for sta_code, sta_list in stations_by_code.items():
            sta_list.sort(key=lambda s: s.start_date or UTCDateTime(0))
            merged_stations = []

            while sta_list:
                current = sta_list.pop(0).copy()
                current_start = current.start_date or UTCDateTime(0)
                current_end = current.end_date or UTCDateTime(2100, 1, 1)

                i = 0
                while i < len(sta_list):
                    other = sta_list[i]
                    other_start = other.start_date or UTCDateTime(0)
                    other_end = other.end_date or UTCDateTime(2100, 1, 1)

                    if other_start <= current_end and other_end >= current_start:
                        sta_list.pop(i)

                        # Patch missing site info
                        if not current.site.name and other.site.name:
                            current.site.name = other.site.name
                        if not current.site.description and other.site.description:
                            current.site.description = other.site.description

                        # Merge channels with smart logic
                        for new_chan in other.channels:
                            match_found = False
                            for existing_chan in current.channels:
                                if (new_chan.code == existing_chan.code and
                                    new_chan.location_code == existing_chan.location_code):
                                    if channels_overlap(new_chan, existing_chan):
                                        if channel_metadata_equivalent(new_chan, existing_chan):
                                            match_found = True  # exact duplicate
                                        else:
                                            print(f"[WARN] Channel metadata mismatch for {sta_code}.{new_chan.location_code}.{new_chan.code} during overlapping period")
                                    # Else no time overlap — both can exist
                            if not match_found:
                                current.channels.append(new_chan)

                        # Extend time window
                        current_start = min(current_start, other_start)
                        current_end = max(current_end, other_end)
                        current.start_date = current_start
                        current.end_date = current_end
                    else:
                        i += 1

                if not current.site.name:
                    current.site.name = f"AUTO_NAME_{sta_code}"
                if not current.site.description:
                    current.site.description = f"Autogenerated site description for {sta_code}"

                merged_stations.append(current)

            merged_net.stations.extend(merged_stations)

        merged_inv.networks.append(merged_net)

    return merged_inv