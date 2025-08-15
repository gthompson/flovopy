# flovopy/stationmetadata/utils.py
from obspy import UTCDateTime, Trace
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.io.xseed import Parser
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.core.util import AttribDict
import pandas as pd
from collections import defaultdict
from pathlib import Path

def get_templates_dir() -> Path:
    """
    Return the absolute path to the stationxml_templates directory.
    Works regardless of OS, working directory, or installation type.
    """
    return Path(__file__).resolve().parent / "stationxml_templates"

def inventory2traceid(inv):
    """
    Convert an ObsPy Inventory to a list of trace IDs in NET.STA.LOC.CHA format.

    Parameters
    ----------
    inv : obspy.core.inventory.inventory.Inventory
        ObsPy Inventory object.

    Returns
    -------
    trace_ids : list of str
        List of trace IDs in 'NET.STA.LOC.CHA' format for each channel in the inventory.
    """
    trace_ids = []
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                trace_id = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                trace_ids.append(trace_id)
    return trace_ids

def has_response(inventory, trace):
    """
    Check whether a response exists in the inventory for a given trace.

    Parameters:
    inventory (Inventory): ObsPy inventory object.
    trace (Trace): ObsPy Trace object.

    Returns:
    bool: True if response exists, False otherwise.
    """
    try:
        resp = inventory.get_response(trace.id, trace.stats.starttime)
        return resp is not None
    except Exception:
        return False


def calculate_sensitivity_all(inventory):
    """
    Recalculate the overall sensitivity for all channels in the inventory.

    Parameters:
    inventory (Inventory): ObsPy inventory object.
    """
    for network in inventory:
        for station in network:
            for channel in station:
                if channel.response:
                    channel.response.recalculate_overall_sensitivity()


def subset_inv(inv, st, st_subset):
    """
    Return a subset of the inventory corresponding to a given stream subset.

    Parameters:
    inv (Inventory): Full ObsPy inventory.
    st (Stream): Original full stream.
    st_subset (Stream): Subset of the stream.

    Returns:
    Inventory: Subsetted inventory.
    """
    try:
        inv_new = inv.copy()
        for tr in st:
            if len(st_subset.select(id=tr.id)) == 0:
                inv_new = inv_new.remove(
                    network=tr.stats.network,
                    station=tr.stats.station,
                    location=tr.stats.location,
                    channel=tr.stats.channel)
        return inv_new
    except Exception:
        print('Failed to subset inventory. Returning unchanged')
        return inv


def expand_channel_code(base: str):
    """
    Expand a channel code like 'EHZ' or 'EHZNEZ' into a list of channel codes.

    Parameters:
    base (str): Base channel code or sequence.

    Returns:
    list[str]: List of expanded channel codes.
    """
    base = str(base).strip()
    if len(base) <= 3:
        return [base]
    prefix = base[:2]
    suffix = base[2:]
    return [prefix + ch for ch in suffix]

def build_dataframe_from_table(path, sheet_name='ksc_stations_master'):
    """
    Load and clean a metadata table from CSV or Excel.

    Parameters:
    -----------
    path : str
        Path to a CSV or Excel file.
    sheet_name : str or int
        Sheet name or index if reading Excel.

    Returns:
    --------
    pd.DataFrame
        Cleaned and expanded metadata table with one row per NSLC entry.
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path, dtype={"location": str})
    elif path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path, sheet_name=sheet_name, dtype={"location": str})
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls")

    # Clean and normalize
    df.columns = df.columns.str.strip().str.lower()
   
    #df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    for col in ['network', 'station', 'location', 'channel']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()

    df['location'] = df['location'].fillna('').astype(str)

    # Date conversion
    for col in ['ondate', 'offdate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].apply(lambda x: UTCDateTime(x) if pd.notnull(x) else None)

    # Optional numeric defaults
    def safe_float(series, default=0.0):
        return pd.to_numeric(series, errors='coerce').fillna(default).astype(float)

    for col, default in {
        'lat': 0.0, 'lon': 0.0, 'elev': 0.0, 'depth': 0.0,
        'fsamp': 100.0, 'vpp': 40.0
    }.items():
        if col in df.columns:
            df[col] = safe_float(df[col], default)

    for col in ['sensor', 'datalogger']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()

    # Expand channels
    expanded_rows = []
    for _, row in df.iterrows():
        channels = expand_channel_code(row.get("channel", ""))
        for ch in channels:
            new_row = row.copy()
            new_row["channel"] = ch
            expanded_rows.append(new_row)

    df = pd.DataFrame(expanded_rows)
    return df

def apply_coordinates_from_csv(inventory, csv_path):
    """
    Apply station/channel coordinates from a CSV file to the inventory.

    Parameters:
    inventory (Inventory): ObsPy inventory to update.
    csv_path (str): Path to CSV with fields: Network, Station, Latitude, Longitude, Elevation, Depth.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Network', 'Station'])
    df['Depth'] = df['Depth'].fillna(0.0)
    df.set_index(df['Network'].str.strip() + '.' + df['Station'].str.strip(), inplace=True)
    for net in inventory:
        for sta in net:
            key = f"{net.code}.{sta.code}"
            if key in df.index:
                row = df.loc[key]
                lat, lon, ele, dep = row['Latitude'], row['Longitude'], row['Elevation'], row['Depth']
                sta.latitude = lat
                sta.longitude = lon
                sta.elevation = ele
                sta.depth = dep
                for ch in sta:
                    ch.latitude = lat
                    ch.longitude = lon
                    ch.elevation = ele
                    ch.depth = dep
                print(f"[OK] Updated coordinates for {key}")
            else:
                print(f"[WARN] No coordinates found for {key}")


def attach_station_coordinates_from_inventory(inventory, st):
    """
    Attach coordinates from inventory to each trace in a Stream.

    Parameters:
    inventory (Inventory): ObsPy inventory with station/channel metadata.
    st (Stream): ObsPy Stream to modify in-place.
    """
    for tr in st:
        for netw in inventory.networks:
            for sta in netw.stations:
                if tr.stats.station == sta.code and netw.code == tr.stats.network:
                    for cha in sta.channels:
                        if tr.stats.location == cha.location_code:
                            tr.stats.coordinates = AttribDict({
                                'latitude': cha.latitude,
                                'longitude': cha.longitude,
                                'elevation': cha.elevation
                            })


def attach_distance_to_stream(st, olat, olon):
    """
    Attach source distance (in meters) to each trace based on coordinates.

    Parameters:
    st (Stream): ObsPy Stream object.
    olat (float): Origin latitude.
    olon (float): Origin longitude.
    """
    for tr in st:
        try:
            if not hasattr(tr.stats, 'coordinates'):
                continue
            alat = tr.stats['coordinates']['latitude']
            alon = tr.stats['coordinates']['longitude']
            distdeg = locations2degrees(olat, olon, alat, alon)
            distkm = degrees2kilometers(distdeg)
            tr.stats['distance'] = distkm * 1000
        except Exception as e:
            print(e)
            print(f'Cannot compute distance for {tr.id}')


def merge_duplicate_stations_and_patch_site(inventory):
    """
    Merge stations with the same code in a network and unify their channels.

    Parameters:
    inventory (Inventory): ObsPy inventory with possibly duplicated stations.

    Returns:
    Inventory: Cleaned inventory with merged stations.
    """
    merged_networks = []
    for net in inventory:
        station_map = defaultdict(list)
        for sta in net:
            station_map[sta.code].append(sta)
        new_stations = []
        for code, stations in station_map.items():
            all_channels = []
            for sta in stations:
                all_channels.extend(sta.channels)
            merged_station = Station(
                code=code,
                latitude=stations[0].latitude,
                longitude=stations[0].longitude,
                elevation=stations[0].elevation,
                channels=all_channels,
                site=stations[0].site or Site(name=f"{code}_SITE", description=f"Autogenerated site description for {code}"),
                creation_date=stations[0].creation_date,
                start_date=stations[0].start_date,
                end_date=stations[0].end_date
            )
            if not merged_station.site.name:
                merged_station.site.name = f"{code}_SITE"
            if not merged_station.site.description:
                merged_station.site.description = f"Autogenerated site description for {code}"
            new_stations.append(merged_station)
        merged_networks.append(Network(code=net.code, stations=new_stations))
    return Inventory(networks=merged_networks, source=inventory.source)


def write_inventory_as_resp(inventory, seed_tempfile, resp_outdir):
    """
    Write RESP files from an ObsPy inventory using an intermediate SEED file.

    Parameters:
    inventory (Inventory): ObsPy inventory object.
    seed_tempfile (str): Path to temporary SEED file.
    resp_outdir (str): Output directory for RESP files.
    """
    inventory.write(seed_tempfile, format='SEED')
    sp = Parser(seed_tempfile)
    sp.write_resp(folder=resp_outdir, zipped=False)
    print(f"[OK] RESP files written to {resp_outdir}")


def modify_inventory(inv, thisid, lat=None, lon=None, elev=None):
    """
    Modify channel and station coordinates in-place, and update channel codes based on sample rate.

    Parameters:
    inv (Inventory): ObsPy inventory object.
    thisid (str): Full channel ID in NET.STA.LOC.CHA format.
    lat (float): New latitude.
    lon (float): New longitude.
    elev (float): New elevation.
    """
    thisnet, thissta, thisloc, thischan = thisid.split('.')
    for net in inv.networks:
        if net.code == thisnet:
            for sta in net.stations:
                if sta.code == thissta:
                    if lat:
                        sta.latitude = lat
                    if lon:
                        sta.longitude = lon
                    if elev:
                        sta.elevation = elev
                    for chan in sta.channels:
                        if chan.code == thischan and chan.location_code == thisloc:
                            chan.depth = 0.0
                            code0 = chan.code[0]
                            if lat:
                                chan.latitude = lat
                            if lon:
                                chan.longitude = lon
                            if elev:
                                chan.elevation = elev
                            if 20.0 < chan.sample_rate < 80.0:
                                if code0 == 'E':
                                    code0 = 'S'
                                elif code0 == 'H':
                                    code0 = 'B'
                            elif 80.0 < chan.sample_rate < 250.0:
                                if code0 == 'B':
                                    code0 = 'H'
                                elif code0 == 'S':
                                    code0 = 'E'
                            chan.code = code0 + chan.code[1:]


def responses2inventory(net, sta, loc, fsamp, responses, lat=None, lon=None,
                        elev=None, depth=None, start_date=UTCDateTime(1900, 1, 1), end_date=UTCDateTime(2100, 1, 1)):
    """
    Build a minimal ObsPy inventory object from manually defined response objects.

    Parameters:
    net (str): Network code.
    sta (str): Station code.
    loc (str): Location code.
    fsamp (float): Sampling rate.
    responses (dict): Dictionary mapping channel codes to ObsPy Response objects.
    lat (float): Latitude.
    lon (float): Longitude.
    elev (float): Elevation.
    depth (float): Sensor depth.
    start_date (UTCDateTime): Start of deployment.
    end_date (UTCDateTime): End of deployment.

    Returns:
    Inventory: ObsPy Inventory with one station.
    """
    channels = []
    for chan, responseObj in responses.items():
        channel = Channel(
            code=chan,
            location_code=loc,
            latitude=lat,
            longitude=lon,
            elevation=elev,
            depth=depth,
            sample_rate=fsamp,
            start_date=start_date,
            end_date=end_date,
            response=responseObj,
        )
        channels.append(channel)
    station = Station(
        code=sta,
        latitude=lat,
        longitude=lon,
        elevation=elev,
        creation_date=UTCDateTime(),
        channels=channels,
        start_date=start_date,
        end_date=end_date,
    )
    network = Network(code=net, stations=[station])
    return Inventory(networks=[network], source="USF_instrument_responses.py")


def inventory_fix_ids(inv, netcode='MV'):
    """
    Fix missing network and station codes in an ObsPy inventory and update channel codes based on sample rate.

    Parameters:
    inv (Inventory): ObsPy Inventory object to modify in-place.
    netcode (str): Default network code to assign if missing.

    Notes:
    - Updates each channel code using `correct_nslc_mvo()` if network code is 'MV'.
    - Uses sampling rate and initial channel code letter to determine if the sensor is short-period.
    """
    from flovopy.core.trace_utils import correct_nslc_mvo
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


def show_response(inv):
    """
    Print response information for each channel in an ObsPy Inventory.

    Parameters:
    inv (Inventory): ObsPy Inventory object.

    Notes:
    - Prints network, station, channel, and response details.
    - Useful for debugging or verifying that responses are loaded correctly.
    """
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
    """
    Plot the spatial distribution of stations in an ObsPy Inventory using default styling.

    Parameters:
    inv (Inventory): ObsPy Inventory object.

    Returns:
    matplotlib.figure.Figure: The figure object of the plot.
    """
    inv.plot(water_fill_color=[0.0, 0.5, 0.8], continent_fill_color=[0.1, 0.6, 0.1], size=30);
    return


def validate_inventory(inv: Inventory, verbose=True) -> list:
    """
    Validate an ObsPy Inventory for structural and temporal issues.

    Checks include:
    - Missing site name or description
    - Missing channels or response info
    - Incomplete channel metadata (code, sample rate)
    - Invalid or missing start/end dates
    - Overlapping or gapped channel time windows (per code/location)

    Parameters:
    -----------
    inv : obspy.core.inventory.Inventory
        The inventory object to validate.

    verbose : bool
        Whether to print a summary of issues.

    Returns:
    --------
    issues : list of str
        List of validation warnings. Empty list means the inventory passed all checks.
    """
    issues = []

    for net in inv:
        for sta in net:
            sid = f"{net.code}.{sta.code}"

            # Site info
            if not sta.site or not sta.site.name:
                issues.append(f"Missing site name for station {sid}")
            if not sta.site or not sta.site.description:
                issues.append(f"Missing site description for station {sid}")

            # Station time window checks
            if not sta.start_date:
                issues.append(f"Missing start_date for station {sid}")
            if not sta.end_date:
                issues.append(f"Missing end_date for station {sid}")
            elif sta.start_date and sta.end_date and sta.start_date > sta.end_date:
                issues.append(f"Station {sid} has start_date after end_date")

            if not sta.channels:
                issues.append(f"No channels found for station {sid}")

            # Group channels for temporal checks
            chan_groups = defaultdict(list)

            for cha in sta:
                cid = f"{sid}.{cha.location_code}.{cha.code}"

                if not cha.code:
                    issues.append(f"Missing channel code for {cid}")
                if not cha.sample_rate:
                    issues.append(f"Missing sample rate for {cid}")
                if not cha.response:
                    issues.append(f"No response info for {cid}")
                if not cha.start_date:
                    issues.append(f"Missing start_date for {cid}")
                if not cha.end_date:
                    issues.append(f"Missing end_date for {cid}")
                elif cha.start_date and cha.end_date and cha.start_date > cha.end_date:
                    issues.append(f"Channel {cid} has start_date after end_date")

                # Group by code and location for overlap checks
                key = (cha.code, cha.location_code)
                chan_groups[key].append(cha)

            # Check for overlaps/gaps within grouped channels
            for (code, loc), chans in chan_groups.items():
                chans.sort(key=lambda c: c.start_date or UTCDateTime(0))
                for i in range(1, len(chans)):
                    prev = chans[i-1]
                    curr = chans[i]

                    prev_end = prev.end_date or UTCDateTime(2100, 1, 1)
                    curr_start = curr.start_date or UTCDateTime(0)

                    if curr_start < prev_end:
                        issues.append(
                            f"Channel overlap: {sid}.{loc}.{code} "
                            f"({prev.start_date} to {prev.end_date}) overlaps with "
                            f"({curr.start_date} to {curr.end_date})"
                        )
                    elif curr_start > prev_end:
                        issues.append(
                            f"Channel gap: {sid}.{loc}.{code} "
                            f"gap between {prev.end_date} and {curr.start_date}"
                        )

    if verbose:
        if issues:
            print("[WARNING] The following issues were found in the Inventory:")
            for issue in issues:
                print(" -", issue)
        else:
            print("[OK] No structural or temporal issues detected.")

    return issues


def get_calib(tr: Trace, inv) -> float:
    calib_value = 1.0
    try:
        for station in inv.networks[0].stations:
            if station.code == tr.stats.station:
                for channel in station.channels:
                    if channel.code == tr.stats.channel:
                        _, calib_value = channel.response._get_overall_sensitivity_and_gain()
    except Exception:
        pass
    return float(calib_value)