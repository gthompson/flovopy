# flovopy/stationmetadata/sensors.py
"""
flovopy.stationmetadata.sensors
--------------------------------
Provides response modeling, gain calculation, and StationXML construction utilities
for common seismic and infrasound sensors and dataloggers, including Raspberry Shake,
infraBSU, and Nanometrics Centaur-based configurations.
"""
import os
import requests
from obspy import UTCDateTime
from obspy.core.inventory.response import Response
from obspy.io.xseed import Parser
from obspy.core.inventory import read_inventory, Inventory, Site, InstrumentSensitivity
from flovopy.stationmetadata.utils import responses2inventory
import numpy as np
from obspy.clients.nrl import NRL

# Sensor modeling and response functions

def centaur(inputVoltageRange: float) -> float:
    """
    Returns counts per volt for a Nanometrics Centaur digitizer.

    Parameters:
        inputVoltageRange (float): Input voltage range in volts (e.g., 1.0 or 40.0).

    Returns:
        float: Conversion factor in counts/volt.
    """
    countsPerVolt = 0.4e6 * 40 / inputVoltageRange
    return countsPerVolt

def trillium() -> float:
    """
    Returns sensitivity for a Trillium Compact 120 seismometer.

    Returns:
        float: Sensitivity in volts per m/s.
    """
    voltsPerMS = 754  # V / (m/s)
    return voltsPerMS

def infraBSU(HgInThisSensor: float = 0.5) -> float:
    """
    Returns sensitivity for an infraBSU microphone (LP21 STair model).

    Parameters:
        HgInThisSensor (float): Mercury equivalent in inches. Default is 0.5.

    Returns:
        float: Sensitivity in volts per pascal (V/Pa).
    """
    oneInchHg2Pa = 3386.4
    voltsPerPa = 46e-6  # from infraBSU quick start guide
    return voltsPerPa

def ChaparralM25() -> float:
    """
    Returns sensitivity for a Chaparral M25 microphone in low-gain mode.

    Returns:
        float: Sensitivity in volts per pascal (V/Pa).
    """
    voltsPerPaLowGain = 0.4  # recommended for 24-bit digitizers
    return voltsPerPaLowGain


def get_rboom(sta, loc, fsamp=100.0, lat=0.0, lon=0.0, elev=0.0, depth=0.0,
              start_date=UTCDateTime(1900, 1, 1), end_date=UTCDateTime(2100, 1, 1)):
    """
    Build an ObsPy Inventory for a Raspberry Boom (HDF only) by reusing get_rsb() 
    and removing the EHZ seismic channel.

    Parameters
    ----------
    sta : str
        Station code to assign.
    loc : str
        Location code to assign.
    fsamp : float, optional
        Default sample rate for HDF channel (Hz).
    lat, lon, elev, depth : float, optional
        Station coordinates and depth (meters).
    start_date, end_date : obspy.UTCDateTime, optional
        Start and end of channel validity.

    Returns
    -------
    obspy.core.inventory.Inventory
        Inventory containing only the HDF channel.
    """
    # First, build the full RSB inventory (HDF + EHZ)
    inv = get_rsb(
        sta=sta, loc=loc, fsamp=fsamp,
        lat=lat, lon=lon, elev=elev, depth=depth,
        start_date=start_date, end_date=end_date
    )

    if inv is None:
        return None

    try:
        station = inv[0][0]
        # Keep only channels with code 'HDF'
        station.channels = [ch for ch in station.channels if ch.code == "HDF"]
        if not station.channels:
            print(f"[WARN] No HDF channel found for {sta} — inventory will be empty.")
    except Exception as e:
        print(f"[ERROR] Failed to strip EHZ channel: {e}")
        return None

    return inv


def get_rsb(sta, loc, fsamp=100.0,
            lat=0.0, lon=0.0, elev=0.0, depth=0.0,
            start_date=UTCDateTime(1900, 1, 1),
            end_date=UTCDateTime(2100, 1, 1)):
    """
    Raspberry Shake & Boom (EHZ seismic + HDF infrasound).

    Tries to fetch a published RSB dataless file. 
    If unavailable, falls back to hardcoded PAZ definitions.
    """
    net = "AM"

    # 1. Try matching pattern from other get_rs* functions
    rsb_url = 'https://manual.raspberryshake.org/_downloads/57ab6152abedf7bb15f86fdefa71978c/RSnBV3-20s.dataless.xml-reformatted.xml'  
    rsb_dataless = "rsb_v3.dataless"
    rsb_xml = "rsb_v3.xml"

    try:
        inv = _fetch_and_parse_response(sta, loc, rsb_url, rsb_dataless, rsb_xml)
        if inv:
            # Optionally adjust sample rates - but would they have to be propagated through stages?
            '''
            for chan in inv[0][0].channels:
                if chan.code.startswith("EHZ"):
                    chan.sample_rate = fsamp
                elif chan.code.startswith("HDF"):
                    chan.sample_rate = fsamp
            '''
            return inv
    except Exception as e:
        print(f"[WARN] Could not fetch RSB dataless, falling back to PAZ: {e}")

    # 2. Fallback — hardcoded PAZ definitions
    hdf_resp = Response.from_paz(
        zeros=[0j, 0j],
        poles=[-0.312 + 0j, -0.312 + 0j],
        stage_gain=56000,   # TODO: verify
        input_units="Pa",   # infrasound pressure
        output_units="Counts",
    )

    ehz_resp = Response.from_paz(
        zeros=[0j, 0j, 0j],
        poles=[-1.0 + 0j, -3.03 + 0j, -3.03 + 0j],
        stage_gain=399650000,  # TODO: verify
        input_units="m/s",     # ground velocity
        output_units="Counts",
    )

    responses = {
        "HDF": (hdf_resp, fsamp_hdf),
        "EHZ": (ehz_resp, fsamp_ehz),
    }

    return responses2inventory(
        net=net, sta=sta, loc=loc,
        fsamp=None,  # let responses dict define per-channel fsamp
        responses=responses,
        lat=lat, lon=lon, elev=elev, depth=depth,
        start_date=start_date, end_date=end_date,
    )
'''
def get_rs1d_v4(sta, loc, fsamp=100.0):
    url = 'https://manual.raspberryshake.org/_downloads/e324d5afda5534b3266cd8abdd349199/out4.response.restored-plus-decimation.dataless'
    responsefile = 'rs1d_v4.dataless'
    xmlfile = 'rs1d_v4.xml'

    if not os.path.isfile(responsefile):
        response = requests.get(url)
        if response.status_code == 200:
            with open(responsefile, "wb") as f:
                f.write(response.content)
            print("[OK] File downloaded: rs1d_v4.dataless")
        else:
            print(f"[ERROR] Failed to download file. HTTP {response.status_code}")
            return None

    sp = Parser(responsefile)
    sp.write_seed(xmlfile)
    inventory = read_inventory(xmlfile)

    if fsamp == 50.0:
        channel = inventory[0][0][0]
        channel.sample_rate = 50.0
        for stage in channel.response.response_stages:
            if hasattr(stage, "decimation_input_sample_rate"):
                stage.decimation_input_sample_rate = 50.0
            if hasattr(stage, "decimation_factor") and stage.decimation_factor != 1:
                print(f"[WARN] Stage {stage.stage_sequence_number} uses decimation factor {stage.decimation_factor}")

    replace_sta_loc(inventory, sta, loc)
    return inventory
'''

def get_rs1d_v4(sta, loc, fsamp=100.0):
    """
    Return Inventory for Raspberry Shake 1D (v4 response). Most devices are 100 Hz,
    but early units were 50 Hz; pass fsamp=50.0 to adjust metadata coherently.

    Parameters
    ----------
    sta : str
        Station code to assign.
    loc : str
        Location code to assign.
    fsamp : float, default 100.0
        Desired native sample rate for the channel(s).

    Returns
    -------
    obspy.core.inventory.Inventory or None
    """
    inv = _fetch_and_parse_response(
        sta, loc,
        url='https://manual.raspberryshake.org/_downloads/e324d5afda5534b3266cd8abdd349199/out4.response.restored-plus-decimation.dataless',
        responsefile='rs1d_v4.dataless',
        xmlfile='rs1d_v4.xml'
    )
    if inv is None:
        return None

    # If caller asked for 50 Hz (or any non-default), adjust channels & response stages
    if fsamp and isinstance(fsamp, (int, float)):
        for net in inv:
            for sta_obj in net:
                for ch in sta_obj.channels:
                    # Set top-level channel sample rate
                    ch.sample_rate = float(fsamp)

                    # Propagate through response stages if present
                    stages = getattr(ch.response, "response_stages", None)
                    if not stages:
                        continue

                    current_rate = float(fsamp)
                    for stage in stages:
                        # Set input rate into this stage
                        if hasattr(stage, "decimation_input_sample_rate"):
                            stage.decimation_input_sample_rate = current_rate

                        # If stage has decimation, compute output rate and carry forward
                        if hasattr(stage, "decimation_factor") and stage.decimation_factor:
                            factor = float(stage.decimation_factor)
                            if factor != 1.0:
                                print(f"[WARN] Stage {stage.stage_sequence_number} uses decimation factor {factor}")
                            out_rate = current_rate / factor
                            # If stage exposes output rate, set it too
                            if hasattr(stage, "decimation_output_sample_rate"):
                                stage.decimation_output_sample_rate = out_rate
                            current_rate = out_rate
                        else:
                            # No decimation at this stage; carry rate forward unchanged
                            pass
                    # Optionally: you could verify `current_rate` == ch.sample_rate or leave as-is.
                    # Here we’re modeling the *input chain* starting at ch.sample_rate; some files
                    # model the chain the other way around. We leave instrument sensitivity frequency unchanged.

    # Overwrite station & location codes for consistency (same as other helpers)
    replace_sta_loc(inv, sta, loc)
    return inv

def get_rs3d_v5(sta, loc):
    return _fetch_and_parse_response(
        sta, loc,
        url='https://manual.raspberryshake.org/_downloads/858bf482b1cd1b06780e9722c1d0b2db/out4.response.restored-EHZ-plus-decimation.dataless-new',
        responsefile='rs3d_v5.dataless',
        xmlfile='rs3d_v5.xml')

def get_rs3d_v3(sta, loc):
    return _fetch_and_parse_response(
        sta, loc,
        url='https://manual.raspberryshake.org/_downloads/85ad0fd97072077ea8a09e42ab15db4b/out4.response.restored-plus-decimation.dataless',
        responsefile='rs3d_v3.dataless',
        xmlfile='rs3d_v3.xml')

def _fetch_and_parse_response(sta: str, loc: str, url: str, responsefile: str, xmlfile: str) -> Inventory:
    """
    Downloads and parses a dataless SEED file, then updates station/location codes.

    Parameters:
        sta (str): Desired station code.
        loc (str): Desired location code.
        url (str): URL to download the response file.
        responsefile (str): Local filename to save the dataless SEED.
        xmlfile (str): Filename to save the converted StationXML.

    Returns:
        Inventory: Modified ObsPy Inventory object.
    """
    if not os.path.isfile(responsefile):
        response = requests.get(url)
        if response.status_code == 200:
            with open(responsefile, "wb") as f:
                f.write(response.content)
            print(f"[OK] File downloaded: {responsefile}")
        else:
            print(f"[ERROR] Failed to download file. HTTP {response.status_code}")
            return None
    sp = Parser(responsefile)
    sp.write_seed(xmlfile)
    inventory = read_inventory(xmlfile)
    replace_sta_loc(inventory, sta, loc)
    return inventory

def replace_sta_loc(inventory, sta='DUMM', loc=''):
    """
    Replace the station code and location code in a given inventory object.
    
    Assumes a single network and a single station.
    """
    try:
        network = inventory[0]
        station = network[0]
        station.code = sta
        for channel in station.channels:
            channel.location_code = loc
    except Exception as e:
        print(f"[ERROR] Failed to update station/location: {e}")

def download_infraBSU_stationxml(save_path: str):
    """
    Downloads the StationXML file for the 21s infraBSU sensor from IRIS and saves it locally.

    Parameters:
        save_path (str): Full path to save the StationXML file.
    """
    url = (
        "https://service.iris.edu/irisws/nrl/1/combine?"
        "instconfig=sensor_JeffreyBJohnson_infraBSU_LP21_SG0.000046_STairPressure"
        "&format=stationxml"
    )
    print(f"[INFO] Downloading infraBSU StationXML from:\n{url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"[ERROR] Failed to download StationXML: HTTP {response.status_code}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"[INFO] Saved infraBSU StationXML to: {save_path}")

def build_combined_infrabsu_centaur_stationxml(
    fsamp: float = 100.0,
    vpp: int = 40,
    stationxml_path: str = None,
    network: str = 'XX',
    station: str = 'DUMM',
    location: str = '10',
    channel: str = 'HDF',
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1970, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
    sitename: str = None
) -> Inventory:
    """
    Combines infraBSU sensor StationXML with a Centaur datalogger stage.

    Returns:
        Inventory: Updated metadata and full response.
    """
    print(f"[INFO] Reading infraBSU StationXML from: {stationxml_path}")
    sensor_inv = read_inventory(stationxml_path)
    sensor_resp = sensor_inv[0][0][0].response  # Assume 1 net, 1 sta, 1 chan

    # Fetch Centaur response stage
    nrl = NRL("http://ds.iris.edu/NRL/")
    if vpp == 40:
        dl_keys = ['Nanometrics', 'Centaur', '40 Vpp (1)', 'Off', 'Linear phase', f"{int(fsamp)}"]
    elif vpp == 1:
        dl_keys = ['Nanometrics', 'Centaur', '1 Vpp (40)', 'Off', 'Linear phase', f"{int(fsamp)}"]
    else:
        raise ValueError(f"Unsupported Vpp: {vpp}")
    
    # Dummy sensor stage to get only the datalogger
    dummy_sensor_keys = ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']
    centaur_resp = nrl.get_response(sensor_keys=dummy_sensor_keys, datalogger_keys=dl_keys)

    # Merge response stages (skip stage 0 from dummy sensor)
    combined_resp = sensor_resp
    combined_resp.response_stages += centaur_resp.response_stages[1:]

    # Recompute overall sensitivity
    gains = [stage.stage_gain for stage in combined_resp.response_stages if hasattr(stage, 'stage_gain')]
    overall_gain = np.prod(gains)

    combined_resp.instrument_sensitivity = InstrumentSensitivity(
        value=overall_gain,
        frequency=1.0,
        input_units=combined_resp.response_stages[0].input_units,
        output_units=combined_resp.response_stages[-1].output_units
    )

    # Patch metadata
    chan_obj = sensor_inv[0][0][0]
    chan_obj.code = channel
    chan_obj.location_code = location
    chan_obj.latitude = latitude
    chan_obj.longitude = longitude
    chan_obj.elevation = elevation
    chan_obj.depth = depth
    chan_obj.sample_rate = fsamp
    chan_obj.start_date = start_date
    chan_obj.end_date = end_date
    chan_obj.response = combined_resp

    station_obj = sensor_inv[0][0]
    station_obj.code = station
    station_obj.latitude = latitude
    station_obj.longitude = longitude
    station_obj.elevation = elevation
    station_obj.start_date = start_date
    station_obj.end_date = end_date
    station_obj.creation_date = UTCDateTime()
    station_obj.site = Site(
        name=sitename or f"{station}_SITE",
        description=f"InfraBSU component {channel} at {station}"
    )

    sensor_inv[0].code = network
    return sensor_inv

# Precomputed sensitivity constants
countsPerMS = centaur(40.0) * trillium()
countsPerPa40 = centaur(40.0) * infraBSU(0.5)
countsPerPa1 = centaur(1.0) * infraBSU(0.5)
countsPerPaChap40 = centaur(40.0) * ChaparralM25()
countsPerPaChap1 = centaur(1.0) * ChaparralM25()

# Hardcoded response values from instrument documentation or files
rs1d = 469087000  # counts/m/s
rboom = 56000     # counts/Pa
rsb = 399650000   # counts/m/s for shake & boom EHZ
rs3d = 360000000