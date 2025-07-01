import os
import requests
from obspy import UTCDateTime, Stream
from obspy.core.inventory import Inventory, Network, Station, Channel, InstrumentSensitivity, Response
from obspy.io.xseed import Parser
#from obspy.core.inventory.util import NRL
from obspy.clients.nrl import NRL
from obspy.core.inventory.inventory import read_inventory

""" the idea of this is to give ways to just use the overall sensitivyt easily, or the full
 instrument response for different types of datalogger/sensors USF has used """

def get_rboom(sta, loc):
    return get_rsb(sta, loc)

def get_rsb(sta, loc, fsamp=100.0, lat=0.0, lon=0.0, \
            elev=0.0, depth=0.0, start_date=UTCDateTime(1900,1,1), end_date=UTCDateTime(2100,1,1) ):
    net = 'AM'
    # Step 1: Download the StationXML file from the provided URL
    url = 'https://manual.raspberryshake.org/_downloads/57ab6152abedf7bb15f86fdefa71978c/RSnBV3-20s.dataless.xml-reformatted.xml'
    xmlfile = 'rsb_v3.xml'
    if not os.path.isfile(xmlfile):
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to a local file
            with open(xmlfile, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")

        else:
            print(f"Failed to download file. HTTP status code: {response.status_code}")
            return None

    # Step 2: Read the downloaded StationXML file into an ObsPy Inventory object
    '''try:
        inventory = read_inventory(xmlfile)
    except:'''
    responses = {}
    poles = [-0.312 + 0.0j, -0.312 + 0.0j]
    zeros = [0.0 + 0j, 0.0 + 0j]
    sensitivity = 56000
    # note: cannot use Pa in response, so use m/s but remember it is really Pa
    responses['HDF'] = Response.from_paz(zeros, poles, sensitivity, input_units='m/s', \
                                                        output_units='Counts' )
    poles = [-1.0, -3.03, -3.03 ]
    zeros = [0.0 + 0j, 0.0 + 0j, 0.0 + 0j]
    sensitivity = 399650000
    responses['EHZ'] = Response.from_paz(zeros, poles, sensitivity, input_units='m/s', \
                                                        output_units='Counts' )
    
    inventory = responses2inventory(net, sta, loc, fsamp, responses, lat=lat, lon=lon, \
                    elev=elev, depth=depth, start_date=start_date, end_date=end_date)

    return inventory

def get_rs1d_v4(sta, loc):
    # Step 1: Download the StationXML file from the provided URL
    url = 'https://manual.raspberryshake.org/_downloads/e324d5afda5534b3266cd8abdd349199/out4.response.restored-plus-decimation.dataless'
    
    responsefile = 'rs1d_v4.dataless'
    xmlfile = 'rs1d_v4.xml'
    if not os.path.isfile(responsefile):
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to a local file
            with open(responsefile, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")

        else:
            print(f"Failed to download file. HTTP status code: {response.status_code}")
            return None
    sp = Parser(responsefile)
    sp.write_seed(xmlfile)

    # Step 2: Read the downloaded StationXML file into an ObsPy Inventory object
    inventory = read_inventory(xmlfile)
    replace_sta_loc(inventory, sta, loc)
    return inventory

def get_rs3d_v5(sta, loc):
    # Step 1: Download the StationXML file from the provided URL
    url = 'https://manual.raspberryshake.org/_downloads/858bf482b1cd1b06780e9722c1d0b2db/out4.response.restored-EHZ-plus-decimation.dataless-new'
    
    responsefile = 'rs3d_v5.dataless'
    xmlfile = 'rs3d_v5.xml'
    if not os.path.isfile(responsefile):
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to a local file
            with open(responsefile, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")

        else:
            print(f"Failed to download file. HTTP status code: {response.status_code}")
            return None
    sp = Parser(responsefile)
    sp.write_seed(xmlfile)

    # Step 2: Read the downloaded StationXML file into an ObsPy Inventory object
    inventory = read_inventory(xmlfile)
    replace_sta_loc(inventory, sta, loc)
    return inventory

def get_rs3d_v3(sta, loc):
    # Step 1: Download the StationXML file from the provided URL
    url = 'https://manual.raspberryshake.org/_downloads/85ad0fd97072077ea8a09e42ab15db4b/out4.response.restored-plus-decimation.dataless'
    
    responsefile = 'rs3d_v3.dataless'
    xmlfile = 'rs3d_v3.xml'
    if not os.path.isfile(responsefile):
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the content to a local file
            with open(responsefile, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")

        else:
            print(f"Failed to download file. HTTP status code: {response.status_code}")
            return None
    sp = Parser(responsefile)
    sp.write_seed(xmlfile)

    # Step 2: Read the downloaded StationXML file into an ObsPy Inventory object
    inventory = read_inventory(xmlfile)
    replace_sta_loc(inventory, sta, loc)
    return inventory

def replace_sta_loc(inv, sta='DUMM', loc=''):
    # assumes only 1 network with 1 station

    # Find the station (you can loop over networks and stations if needed)

    network = inv[0]  # Replace with the network code
    station = network[0] #.get_station("STA_CODE")  # Replace with the station code

    # Change the station name (code) and location code
    station.code = sta  # Change to the new station name
    station.location_code = loc     

def centaur(inputVoltageRange):
    countsPerVolt = 0.4e6 * 40/inputVoltageRange;
    return countsPerVolt

def trillium():
    voltsPerMS = 754; # V / (m/s)
    return voltsPerMS

def infraBSU(HgInThisSensor=0.5):
    # model 0.5" is default
    # 0.1 - 40 Hz flat
    oneInchHg2Pa = 3386.4;
    linearRangeInPa = oneInchHg2Pa * HgInThisSensor;
    selfNoisePa = 5.47e-3;
    voltsPerPa = 46e-6; # from infraBSU quick start guide
    return voltsPerPa

import os
import tempfile
import requests
from obspy import UTCDateTime
from obspy.clients.nrl import NRL
from obspy.core.inventory.response import Response
from obspy.io.xseed import Parser

'''
import tempfile
import os
import requests
from obspy.clients.nrl import NRL
from obspy.io.xseed import Parser
from obspy.core.inventory.response import Response

def build_combined_infrabsu_centaur_response(fsamp: float = 100.0,
                                              vpp: int = 40,
                                              fallback_resp_url: str = "https://ds.iris.edu/NRL/sensors/johnson/RESP.XX.IS020..BDF.INFRABSU.0_048.0_000046"
                                              ) -> Response:
    """
    Build a fallback Response object for an infraBSU sensor connected to a Centaur datalogger.
    Falls back to using a known-good RESP file if NRL lookup fails.

    Parameters:
        fsamp (float): Sampling rate in Hz.
        vpp (int): Peak-to-peak input voltage (usually 1 or 40).
        fallback_resp_url (str): URL to a known-good RESP file for infraBSU sensor.

    Returns:
        Response: A combined Response object.
    """
    print(f'[INFO] Building fallback response for infraBSU from RESP + Centaur\nParams: fsamp={fsamp}, vpp={vpp}, url={fallback_resp_url}')

    # Download the fallback infraBSU RESP file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".resp") as tf:
        resp_path = tf.name
        r = requests.get(fallback_resp_url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download RESP file: {fallback_resp_url}")
        tf.write(r.content)

    try:
        parser = Parser(resp_path)

        # --- Patch Blockette 50 (station-level metadata) ---
        for blk50 in parser.blockettes.get(50, []):
            if not hasattr(blk50, 'network_identifier_code') or not str(blk50.network_identifier_code).strip():
                blk50.network_identifier_code = "XX"

            if not hasattr(blk50, 'station_call_letters') or not str(blk50.station_call_letters).strip():
                blk50.station_call_letters = "DUMM"

            if not hasattr(blk50, 'station_epoch_year'):
                blk50.station_epoch_year = 2000
            else:
                try:
                    blk50.station_epoch_year = int(blk50.station_epoch_year)
                except (TypeError, ValueError):
                    blk50.station_epoch_year = 2000

            if not hasattr(blk50, 'channel_count'):
                blk50.channel_count = 1
            else:
                try:
                    blk50.channel_count = int(blk50.channel_count)
                except (ValueError, TypeError):
                    blk50.channel_count = 1


            if not hasattr(blk50, 'latitude') or blk50.latitude in ("", None):
                blk50.latitude = 0.0
            if not hasattr(blk50, 'longitude') or blk50.longitude in ("", None):
                blk50.longitude = 0.0
            if not hasattr(blk50, 'elevation') or blk50.elevation in ("", None):
                blk50.elevation = 0.0

        # --- Patch Blockette 52 (channel-level metadata) ---
        for blk52 in parser.blockettes.get(52, []):
            if not hasattr(blk52, 'location_identifier') or not str(blk52.location_identifier).strip():
                blk52.location_identifier = ""

            if not hasattr(blk52, 'channel_identifier') or not str(blk52.channel_identifier).strip():
                blk52.channel_identifier = "HDF"

            if not hasattr(blk52, 'channel_number'):
                blk52.channel_number = 1
            else:
                try:
                    blk52.channel_number = int(blk52.channel_number)
                except (TypeError, ValueError):
                    blk52.channel_number = 1

            if not hasattr(blk52, 'instrument_depth') or blk52.instrument_depth in ("", None):
                blk52.instrument_depth = 0.0
            else:
                try:
                    blk52.instrument_depth = float(blk52.instrument_depth)
                except (TypeError, ValueError):
                    blk52.instrument_depth = 0.0

            if not hasattr(blk52, 'azimuth') or blk52.azimuth in ("", None):
                blk52.azimuth = 0.0
            else:
                try:
                    blk52.azimuth = float(blk52.azimuth)
                except (TypeError, ValueError):
                    blk52.azimuth = 0.0

            if not hasattr(blk52, 'dip') or blk52.dip in ("", None):
                blk52.dip = 0.0
            else:
                try:
                    blk52.dip = float(blk52.dip)
                except (TypeError, ValueError):
                    blk52.dip = 0.0




        test_response = parser.get_inventory()
        print(test_response)
        sensor_response = parser.get_inventory()[0][0][0].response

    except Exception as e:
        print(f"[ERROR] Could not parse or patch RESP file: {e}")
        raise
    finally:
        os.remove(resp_path)

    # Fetch Centaur datalogger stage from NRL
    nrl = NRL("http://ds.iris.edu/NRL/")
    if vpp == 40:
        datalogger_keys = ['Nanometrics', 'Centaur', '40 Vpp (1)', 'Off', 'Linear phase', f"{int(fsamp)}"]
    elif vpp == 1:
        datalogger_keys = ['Nanometrics', 'Centaur', '1 Vpp (40)', 'Off', 'Linear phase', f"{int(fsamp)}"]
    else:
        raise ValueError(f"Unsupported Vpp: {vpp}")

    dummy_sensor_keys = ['Generic', 'Broadband', '1.0 V/count']

    try:
        centaur_response = nrl.get_response(sensor_keys=dummy_sensor_keys, datalogger_keys=datalogger_keys)
    except Exception as e:
        raise RuntimeError(f"Failed to get Centaur response from NRL: {e}")

    # Combine sensor and datalogger stages
    combined_response = Response(
        instrument_sensitivity=sensor_response.instrument_sensitivity,
        response_stages=sensor_response.response_stages + centaur_response.response_stages
    )

    return combined_response
'''

import os
import requests

def download_infraBSU_stationxml(save_path='/data/station_metadata/infraBSU_21s_0.5inch.xml'):
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

from obspy.clients.nrl import NRL
from obspy.core.inventory import Inventory, read_inventory
from obspy import UTCDateTime
#from obspy.core.inventory.util import read_inventory

def build_combined_infrabsu_centaur_stationxml(
    fsamp=100.0,
    vpp=40,
    stationxml_path='/data/station_metadata/infraBSU_21s_0.5inch.xml',
    network='XX', station='DUMM', location='10', channel='HDF',
    latitude=0.0, longitude=0.0, elevation=0.0, depth=0.0,
    start_date=UTCDateTime(1970, 1, 1), end_date=UTCDateTime(2100, 1, 1), sitename=None,
) -> Inventory:
    """
    Combines infraBSU sensor StationXML with a Centaur datalogger stage.

    Returns:
        Inventory with updated metadata and full response.
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


    # Merge response stages
    combined_resp = sensor_resp
    combined_resp.response_stages += centaur_resp.response_stages

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
    station_obj.creation_date=UTCDateTime()
    station_obj.site=Site(name=sitename or f"{station}_SITE",
                description=f"Autogenerated site description for {station}")
        

    network_obj = sensor_inv[0]
    network_obj.code = network


    return sensor_inv


def ChaparralM25():
    # 36 V p2p
    # 0.1 - 200 Hz flat
    selfNoisePa = 3e-3;
    voltsPerPaHighGain = 2.0; # 18 Pa full scale. 
    voltsPerPaLowGain = 0.4; # 90 Pa full scale. recommended for 24-bit digitizers. 
    voltsPerPaVMod = 0.4 * 90/720; # 720 Pa full scale.
    # Volcano mod reduces sensitivity further.
    return voltsPerPaLowGain

countsPerMS = centaur(40.0) * trillium()
countsPerPa40 = centaur(40.0) * infraBSU(0.5)
countsPerPa1 = centaur(1.0) * infraBSU(0.5)
countsPerPaChap40 = centaur(40.0) * ChaparralM25()
countsPerPaChap1 = centaur(1.0) * ChaparralM25()
rs1d    = 469087000 # counts/m/s
rboom   = 56000 # counts/Pa
rsb     = 399650000 # counts/m/s for a shake&boom EHZ
rs3d    = 360000000

def correctUSFstations(st, apply_calib=False, attach=True, return_inventories=False):
    chap25 = {}
    datalogger = 'Centaur'
    sensor = 'TCP'
    Vpp = 40
    inventories = Inventory()
    for tr in st:
        if tr.stats.network=='AM':
            continue
        if tr.stats.sampling_rate>=50:
            #tr.detrend()
            if tr.stats.channel[1]=='H': # a seismic velocity high-gain channel. L for low gain, N for accelerometer
                # trace is in counts. there are 0.3 counts/ (nm/s).
                #tr.data = tr.data / 0.3 * 1e-9 # now in m/s
                calib = countsPerMS
                units = 'm/s'
                                
            if tr.stats.channel[1]=='D': # infraBSU channel?
                #trtr.data = tr.data / countsPerPa40
                # Assumes 0.5" infraBSU sensors at 40V p2p FS
                # But could be 1" or 5", could be 1V p2p FS or could be Chaparral M-25
                units = 'Pa'
                sensor = 'infraBSU'
                Vpp = 1
                calib = countsPerPa1 # default is to assume a 0.5" infraBSU and 1V p2p
                # First let's sort out all the SEED ids that correspond to the Chaparral M-25
                if tr.stats.station=='BCHH1' and tr.stats.channel[2]=='1':
                    calib = countsPerPaChap40
                    chap25 = {'id':tr.id, 'p2p':40}
                    sensor = 'Chaparral'
                    Vpp = 40
                elif tr.id == 'FL.BCHH3.10.HDF':
                    if tr.stats.starttime < UTCDateTime(2022,5,26): # Chaparral M25. I had it set to 1 V FS. Should have used 40 V FS. 
                        calib = countsPerPaChap1
                        chap25 = {'id':tr.id, 'p2p':1}
                        sensor = 'Chaparral'
                        Vpp = 1
                    else:
                        calib = countsPerPaChap40
                        chap25 = {'id':tr.id, 'p2p':40}
                        sensor = 'Chaparral'
                        Vpp = 40
                elif tr.id=='FL.BCHH4.00.HDF':
                    calib=countsPerPaChap40    
                    sensor = 'Chaparral'
                    Vpp = 40
                # anything left is infraBSU and we assume we used a 0.5" sensor, but might sometimes have used the 1" or even the 5"
                # assume we used a 1V peak2peak, except for the following cases
                elif tr.stats.station=='BCHH' or tr.stats.station=='BCHH1' or tr.stats.station[0:3]=='SAK' or tr.stats.network=='NU':
                    calib = countsPerPa40
                    Vpp = 40
                elif chap25:
                        net,sta,loc,chan = chap25['id'].split('.')
                        if tr.stats.network == net and tr.stats.station == sta and tr.stats.location == loc:
                            if chap25['p2p']==40:
                                calib = countsPerPa40
                                Vpp = 40
                            else:
                                Vpp = 1
            if return_inventories or attach:
                inv = NRL2inventory(tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, \
                            datalogger=datalogger, sensor=sensor, Vpp=Vpp, fsamp=tr.stats.sampling_rate, \
                                sensitivty=calib, units=units)
                inventories = inventories + inv
            if apply_calib:
                tr.data = tr.data / calib
                tr.stats['sensitivity'] = calib
                tr.stats['units'] = units
                add_to_trace_history(tr, 'calibration_applied')   
            tr.stats['datalogger'] = datalogger
            tr.stats['sensor'] = sensor           
    if attach:
        st.attach_response(inventories)
    if return_inventories:
        return inventories    
     
''' Old version. ChatGPT suggested update below
def NRL2inventory(net, sta, loc, chans, datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, \
             lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='', \
                ondate=UTCDateTime(1970,1,1), offdate=UTCDateTime(2025,12,31), sensitivity=None, units=None):
    nrl = NRL('http://ds.iris.edu/NRL/')
    if datalogger == 'Centaur':
        if Vpp==40:
            datalogger_keys = ['Nanometrics', 'Centaur', '40 Vpp (1)', 'Off', 'Linear phase', "%d" % fsamp]
        elif Vpp==1:
            datalogger_keys = ['Nanometrics', 'Centaur', '1 Vpp (40)', 'Off', 'Linear phase', "%d" % fsamp]
    elif datalogger == 'RT130':
        datalogger_keys = ['REF TEK', 'RT 130 & 130-SMA', '1', "%d" % fsamp]
    else:
        print(datalogger, ' not recognized')
        print(nrl.dataloggers[datalogger])
    print(datalogger_keys)

    if sensor == 'TCP':
        sensor_keys = ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']
    elif sensor == 'L-22':
        sensor_keys = ['Sercel/Mark Products','L-22D','2200 Ohms','10854 Ohms']
    elif sensor[:4] == 'Chap':
        sensor_keys = ['Chaparral Physics', '25', 'Low: 0.4 V/Pa']
    elif sensor.lower() == 'infrabsu':
        #sensor_keys = ['JeffreyBJohnson', 'sensor_JeffreyBJohnson_infraBSU_LP21_SG0.000046_STairPressure', '0.000046 V/Pa']
        sensor_keys = ['JeffreyBJohnson', 'infraBSU', '0.000046 V/Pdef NRL2inventory(net, sta, loc, chans,
                  datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0,
                  lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='',
                  ondate=UTCDateTime(1970, 1, 1), offdate=UTCDateTime(2025, 12, 31),
                  sensitivity=None, units=None):

    nrl = NRL("http://ds.iris.edu/NRL/")

    # Define datalogger keys
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

    # Define sensor keys
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

        # Fallback infrasound and Chaparral, if they fail
        if sensor.lower() == 'infrabsu':

            try:
                print('Trying to build a response object for InfraBSU from online RESP file')
                thisresponse = build_combined_infrabsu_centaur_response(fsamp=fsamp, vpp=Vpp)
                print(thisresponse)
            except Exception as e: # try PAZ
                print(e)
                print('Building from NRL, and then from RESP both failed for infraBSU. Trying to build from poles & zeros')
                poles = [-0.301593 + 0j]
                zeros = [0j]
                if sensitivity is None:
                    sensitivity = 18400.0  # default

        elif sensor.lower().startswith('chap'):
            print('Trying to build from poles and zeros for Chaparral M25')
            poles = [-1.0 + 0j, -3.03 + 0j, -3.03 + 0j]
            zeros = [0j, 0j, 0j]
            if sensitivity is None:
                sensitivity = 160000.0
        else:
            raise ValueError(f"No fallback response available for sensor: {sensor}")

        input_units = 'Pa'
        output_units = 'V'
        try:
            thisresponse = Response.from_paz(
                zeros, poles, sensitivity,
                input_units=input_units,
                output_units=output_units
            )
        except Exception as err:
            print(f"[ERROR] Failed to build fallback PAZ response for {net}.{sta}.{loc}: {err}")
            sens_obj = InstrumentSensitivity(
                value=sensitivity,
                frequency=1.0,
                input_units=input_units,
                output_units=output_units
            )
            thisresponse = Response(instrument_sensitivity=sens_obj)

    # Warn if evalresp won't work
    if not thisresponse.response_stages:
        print(f"[WARNING] No response stages defined for {net}.{sta}.{loc} — evalresp/remove_response() will fail.")a']
    else:
        print(sensor, ' not recognized')
        print(nrl.sensors[sensor])
    print(sensor_keys)

    responses = {}
    thisresponse = None
    try:
        thisresponse = nrl.get_response(sensor_keys=sensor_keys, datalogger_keys=datalogger_keys)
    except:
        if units=='Pa':
            units='m/s'
            print(f'Warning: units changed from {units} to m/s because Obspy cannot work with Pa')
        # use just a sensitivity instead
        if sensor.lower()=='infrabsu':
            poles = [-0.301593 + 0.0j]
            zeros = [0.0 + 0j]
        if poles or zeros:
            # although input units are Pa, obspy can only compute overall sensitivity and remove response if units are seismic.
            try:
                thisresponse = Response.from_paz(zeros, poles, sensitivity, input_units='m/s', output_units='Counts' )
            except:
                print('poles/zeros failed when recomputing overall sensitivity')
                sensitivityObj = InstrumentSensitivity(sensitivity, 1.0, units, 'Counts')
                thisresponse = Response(instrument_sensitivity=sensitivityObj)
        else:
            sensitivityObj = InstrumentSensitivity(sensitivity, 1.0, units, 'Counts')
            thisresponse = Response(instrument_sensitivity=sensitivityObj)

    for thischan in chans:
        responses[thischan]  = thisresponse  

    inventory = responses2inventory(net, sta, loc, fsamp, responses, lat=lat, lon=lon, \
                        elev=elev, depth=depth, start_date=ondate, end_date=offdate)

    return inventory
'''


from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.inventory.response import Response, InstrumentSensitivity
from obspy.clients.nrl import NRL
from obspy import UTCDateTime

def NRL2inventory(net, sta, loc, chans,
                  datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0,
                  lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='',
                  ondate=UTCDateTime(1970, 1, 1), offdate=UTCDateTime(2025, 12, 31),
                  sensitivity=None, units=None):

    nrl = NRL("http://ds.iris.edu/NRL/")

    # Define datalogger keys
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

    # Define sensor keys
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

    # Try NRL service
    try:
        thisresponse = nrl.get_response(sensor_keys=sensor_keys, datalogger_keys=datalogger_keys)
    except Exception as e:
        print(f"[WARNING] NRL lookup failed for {net}.{sta}.{loc} ({sensor}): {e}")

        # Fallback infrasound and Chaparral, if they fail
        if sensor.lower() == 'infrabsu':

            try:
                print('Trying to build a response object for InfraBSU from online RESP file')
                thisresponse = build_combined_infrabsu_centaur_response(fsamp=fsamp, vpp=Vpp)
                #print(thisresponse)
            except Exception as e: # try PAZ
                print(e)
                print('Building from NRL, and then from RESP both failed for infraBSU. Trying to build from poles & zeros')
                poles = [-0.301593 + 0j]
                zeros = [0j]
                if sensitivity is None:
                    sensitivity = 18400.0  # default

        elif sensor.lower().startswith('chap'):
            print('Trying to build from poles and zeros for Chaparral M25')
            poles = [-1.0 + 0j, -3.03 + 0j, -3.03 + 0j]
            zeros = [0j, 0j, 0j]
            if sensitivity is None:
                sensitivity = 160000.0
        else:
            raise ValueError(f"No fallback response available for sensor: {sensor}")

        input_units = 'Pa'
        output_units = 'V'
        try:
            thisresponse = Response.from_paz(
                zeros, poles, sensitivity,
                input_units=input_units,
                output_units=output_units
            )
        except Exception as err:
            print(f"[ERROR] Failed to build fallback PAZ response for {net}.{sta}.{loc}: {err}")
            sens_obj = InstrumentSensitivity(
                value=sensitivity,
                frequency=1.0,
                input_units=input_units,
                output_units=output_units
            )
            thisresponse = Response(instrument_sensitivity=sens_obj)

    # Warn if evalresp won't work
    if not thisresponse.response_stages:
        print(f"[WARNING] No response stages defined for {net}.{sta}.{loc} — evalresp/remove_response() will fail.")

    # Build Channel objects
    channels = []
    for chan in chans:
        ch = Channel(
            code=chan,
            location_code=loc,
            latitude=lat,
            longitude=lon,
            elevation=elev,
            depth=depth,
            sample_rate=fsamp,
            start_date=ondate,
            end_date=offdate,
            response=thisresponse
        )
        channels.append(ch)

    # Build Station and Network
    site = Site(name=sitename or f"{sta}_SITE",
                description=f"Autogenerated site description for {sta}")
    station = Station(
        code=sta,
        latitude=lat,
        longitude=lon,
        elevation=elev,
        creation_date=UTCDateTime(),
        channels=channels,
        site=site,
        start_date=ondate,
        end_date=offdate
    )

    network = Network(code=net, stations=[station])
    inventory = Inventory(networks=[network], source="NRL2inventory")

    return inventory

import csv
def apply_coordinates_from_csv(inventory, csv_path):
    """
    Updates station coordinates in an ObsPy Inventory from a CSV file with header:
    Network,Station,Latitude,Longitude,Elevation,Depth
    """
    coords = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Network'].strip()}.{row['Station'].strip()}"
            coords[key] = {
                'latitude': float(row['Latitude']),
                'longitude': float(row['Longitude']),
                'elevation': float(row['Elevation']),
                'depth': float(row.get('Depth', 0.0))
            }

    for net in inventory:
        for sta in net:
            key = f"{net.code}.{sta.code}"
            if key in coords:
                c = coords[key]
                sta.latitude = c['latitude']
                sta.longitude = c['longitude']
                sta.elevation = c['elevation']
                sta.depth = c['depth']
                for ch in sta:
                    ch.latitude = c['latitude']
                    ch.longitude = c['longitude']
                    ch.elevation = c['elevation']
                    ch.depth = c['depth']
                print(f"[OK] Updated coordinates for {key}")
            else:
                print(f"[WARN] No coordinates found for {key}")

from obspy.core.inventory import Inventory, Network, Station, Site
from collections import defaultdict

def merge_duplicate_stations_and_patch_site(inventory):
    """
    Merges duplicate stations with same network and code, and ensures every statiimport csv
                    inv = NRL2inventory(
                        str(row["network"]).strip(),
                        str(row["station"]).strip(),
                        str(row["location"]).strip(),
                        chan_list,          
                        fsamp=float(row["fsamp"]),
                        Vpp=int(row["vpp"]),
                        datalogger=row['datalogger'].strip(),
                        sensor=row['datalogger'].strip(),
                        latitude=float(row["lat"]),
                        longitude=float(row["lon"]),
                        elevation=float(row["elev"]),
                        depth=float(row["depth"]),
                        start_date=UTCDateTime(row["ondate"]),
                        end_date=UTCDateTime(row["offdate"])
                    )
                elif ch[1]=='D':on has a Site.
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

from obspy.core.inventory import Inventory
from obspy.io.xseed import Parser

def write_inventory_as_resp(inventory, seed_tempfile, resp_outdir):
    """
    Writes RESP files for all channels in an ObsPy Inventory.
    Requires writing to Dataless SEED first.
    """
    # Write to temporary dataless SEED
    inventory.write(seed_tempfile, format='SEED')

    # Read with Parser and export
    sp = Parser(seed_tempfile)
    sp.write_resp(folder=resp_outdir, zipped=False)
    print(f"[OK] RESP files written to {resp_outdir}")




def responses2inventory(net, sta, loc, fsamp, responses, lat=None, lon=None, \
                        elev=None, depth=None, start_date=UTCDateTime(1900,1,1), end_date=UTCDateTime(2100,1,1)):
    channels=[]
    for chan, responseObj in responses.items():
        channel = Channel(code=chan,
                      location_code=loc,
                      latitude=lat,
                      longitude=lon,
                      elevation=elev,
                      depth=depth,
                      sample_rate=fsamp,
                      start_date=start_date,
                      end_date=end_date,
                      response = responseObj,
                    )
        channels.append(channel)
    station = Station(code=sta,
                      latitude=lat,
                      longitude=lon,
                      elevation=elev,
                      creation_date=UTCDateTime(),
                      channels=channels,
                      start_date=start_date,
                      end_date=end_date,
                      )

    network = Network(code=net,
                     stations=[station])
    inventory = Inventory(networks=[network], source="USF_instrument_responses.py")
    return inventory

def change_stage_gain(thisresponse, stagenum, newgain):
    # might need to change stage 3 of Chaparral gain from 400,000 to 160,000 so it gets the overall sensitivity right
    # Loop through the stages of the response and modify the gain for a specific stage
    # You can identify stages by checking their type, name, or other properties.
    stage = thisresponse[stagenum-1]
    # Print the details of each stage to identify which one you want to modify
    print(f"Stage Type: {stage.stage_type}, Gain: {stage.gain}")
    
    # Modify the gain for a specific stage (you can add more conditions to target a specific stage)
    if stage.stage_type == "GAIN":  # Change "GAIN" to whatever stage you want
        stage.gain *= newgain  # Set the new gain value


# Function to calculate the overall sensitivity for a station's response
def calculate_sensitivity(response):
    sensitivity = 1.0  # Default sensitivity (neutral multiplier)

    for stage in response:
        if stage.stage_type == "GAIN":
            sensitivity *= stage.gain  # Multiply the sensitivity by the stage gain
        elif stage.stage_type == "POLY1":
            # For example, if POLY1 is used for the response, apply the polynomial scaling (if needed)
            # This depends on your instrument response and how it's encoded.
            # You could access the parameters of the POLY1 stage to modify the calculation.
            sensitivity *= stage.coefficients[0]  # If this applies to the specific polynomial stage.

    return sensitivity


if __name__ == '__main__':

    print('calibrations:')
    print('trillium+centaur40 = %f' % countsPerMS)        
    print('infraBSU+centaur40 = %f' % countsPerPa40)        
    print('infraBSU+centaur1 = %f' % countsPerPa1)        
    print('chaparralM25+centaur40 = %f' % countsPerPaChap40)        
    print('chaparralM25+centaur1 = %f' % (countsPerPaChap1))   
    print('************************************\n\n')




from obspy import UTCDateTime, read_inventory
#from flovopy.core.usf import NRL2inventory, apply_coordinates_from_csv, merge_duplicate_stations_and_patch_site, write_inventory_as_resp
import os

import pandas as pd
from obspy import UTCDateTime
from flovopy.core.usf import build_combined_infrabsu_centaur_stationxml
from obspy.core.inventory import Inventory

from obspy import UTCDateTime

def expand_channel_code(base: str):
    """
    Expands compact channel codes like 'EHZNE' → ['EHZ', 'EHN', 'EHE']
    or 'HD123456' → ['HD1', 'HD2', ..., 'HD6'].
    """
    base = str(base).strip()
    if len(base) <= 3:
        return [base]  # Nothing to expand

    prefix = base[:2]
    suffix = base[2:]

    return [prefix + ch for ch in suffix]

def build_inventory_from_csv(csv_path: str,
                              stationxml_path: str = "/data/station_metadata/infraBSU_21s_0.5inch.xml") -> Inventory:
    """
    Reads a CSV of infraBSU sensor entries and builds an ObsPy Inventory by calling
    build_combined_infrabsu_centaur_stationxml() for each row.

    Parameters:
        csv_path (str): Path to the CSV file.
        stationxml_path (str): Path to cached infraBSU StationXML file.

    Returns:
        Inventory: Combined ObsPy Inventory object.
    """
    df = pd.read_csv(csv_path)


    # Strip column names and values
    df.columns = df.columns.str.strip()
    df['location'] = df['location'].astype(str).str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Strip leading/trailing whitespace from string columns only
    #for col in df.select_dtypes(include=['object']).columns:
    #    df[col] = df[col].str.strip()

    # Convert dates to UTCDateTime
    df["ondate"] = df["ondate"].apply(UTCDateTime)
    df["offdate"] = df["offdate"].apply(UTCDateTime)
    #print(df.dtypes)
    master_inventory = Inventory(networks=[], source="build_inventory_from_csv")

    # Updated loop
    for _, row in df.iterrows():
        #try:
            chan_list = expand_channel_code(row["channel"])
            #print(row['channel'], '->', chan_list)
            if row['vpp'].strip()=='':
                row['vpp']=32 # REFTECK RT130 PASSCAL gain

            if row['channel'][1]=='H':
                inv = NRL2inventory(
                    str(row["network"]).strip(),
                    str(row["station"]).strip(),
                    str(row["location"]).strip(),
                    chan_list,          
                    fsamp=float(row["fsamp"]),
                    Vpp=int(row["vpp"]),
                    datalogger=row['datalogger'].strip(),
                    sensor=row['sensor'].strip(),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    elev=float(row["elev"]),
                    depth=float(row["depth"]),
                    ondate=UTCDateTime(row["ondate"]),
                    offdate=UTCDateTime(row["offdate"])
                )
            elif row['channel'][1]=='D':
                for ch in chan_list:
                     inv = build_combined_infrabsu_centaur_stationxml(
                        fsamp=float(row["fsamp"]),
                        vpp=int(row["vpp"]),
                        stationxml_path=stationxml_path,
                        network=str(row["network"]).strip(),
                        station=str(row["station"]).strip(),
                        location=str(row["location"]).strip(),
                        channel=ch.strip(),
                        latitude=float(row["lat"]),
                        longitude=float(row["lon"]),
                        elevation=float(row["elev"]),
                        depth=float(row["depth"]),
                        start_date=UTCDateTime(row["ondate"]),
                        end_date=UTCDateTime(row["offdate"])
                    )                   
            master_inventory += inv

        #except Exception as e:
        #    print(f"[ERROR] Failed for {row['network']}.{row['station']}.{row['location']}.{row['channel']}: {e}")
    
    return master_inventory

def get_stationXML_inventory(
    xmlfile='/data/station_metadata/KSC.xml',
    seedfile='/data/station_metadata/KSC.dataless',
    respdir='/data/station_metadata/RESP/',
    metadata_csv='/data/station_metadata/ksc.csv',
    coord_csv='/data/station_metadata/ksc_coordinates_only.csv',
    overwrite=False
):
    from obspy import Inventory

    if os.path.isfile(xmlfile) and not overwrite:
        inv = read_inventory(xmlfile)
        print(f"[INFO] Loaded existing StationXML: {xmlfile}")
    else:
        print("[INFO] Creating new inventory from USF definitions...")

        inv = build_inventory_from_csv(csv_path = metadata_csv, 
                              stationxml_path = "/data/station_metadata/infraBSU_21s_0.5inch.xml")

        # Apply coordinates from CSV
        if os.path.isfile(coord_csv):
            apply_coordinates_from_csv(inv, coord_csv)

        # Merge + site patch
        inv = merge_duplicate_stations_and_patch_site(inv)

        # Save StationXML
        inv.write(xmlfile, format='STATIONXML', validate=True)
        print(f"[OK] Wrote StationXML to {xmlfile}")

    return inv


import os
from obspy import UTCDateTime
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.io.xseed import Parser
import shutil

def inventory2dataless_and_resp(inv, output_dir="/data/station_metadata/resp",
                                stationxml_seed_converter_jar="/home/thompsong/stationxml-seed-converter.jar"):
    """
    Splits an ObsPy Inventory into one per channel, saves as StationXML,
    and attempts to convert each to Dataless SEED and RESP format.
    """
    os.makedirs(output_dir, exist_ok=True)

    for net in inv:
        for sta in net:
            for cha in sta:
                try:
                    # Build a minimal Inventory
                    new_channel = Channel(
                        code=cha.code,
                        location_code=cha.location_code,
                        latitude=cha.latitude,
                        longitude=cha.longitude,
                        elevation=cha.elevation,
                        depth=cha.depth,
                        azimuth=cha.azimuth,
                        dip=cha.dip,
                        sample_rate=cha.sample_rate,
                        start_date=cha.start_date,
                        end_date=cha.end_date,
                        response=cha.response
                    )

                    new_station = Station(
                        code=sta.code,
                        latitude=sta.latitude,
                        longitude=sta.longitude,
                        elevation=sta.elevation,
                        site=Site(name=sta.site.name if sta.site.name else ""),
                        channels=[new_channel],
                        start_date=sta.start_date,
                        end_date=sta.end_date
                    )

                    new_network = Network(
                        code=net.code,
                        stations=[new_station]
                    )

                    mini_inv = Inventory(networks=[new_network], source=inv.source)

                    # Create safe filenames
                    sdt = (cha.start_date or UTCDateTime(0)).format_iris_web_service().replace(":", "-")
                    edt = (cha.end_date or UTCDateTime(2100, 1, 1)).format_iris_web_service().replace(":", "-")
                    basename = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}_{sdt}_{edt}"
                    xmlfile = os.path.join(output_dir, f"{basename}.xml")

                    # 1. Write StationXML
                    mini_inv.write(xmlfile, format="stationxml")
                    print(f"[OK] Wrote StationXML: {xmlfile}")

                    # 2. Convert to Dataless SEED via external JAR
                    dataless_file = os.path.join(output_dir, f"{basename}.dseed")
                    java_cmd = f"java -jar {stationxml_seed_converter_jar} -s {xmlfile} -o {dataless_file}"
                    ret = os.system(java_cmd)
                    if ret != 0 or not os.path.exists(dataless_file):
                        raise RuntimeError(f"[WARN] Dataless SEED conversion failed for {xmlfile}")
                    print(f"[OK] Wrote Dataless SEED: {dataless_file}")

                    # 3. Convert to RESP
                    resp_dir = os.path.join(output_dir, f"{basename}_resp")
                    os.makedirs(resp_dir, exist_ok=True)
                    sp = Parser(dataless_file)
                    sp.write_resp(folder=resp_dir, zipped=False)
                    print(f"[OK] Wrote RESP files to: {resp_dir}")

                    # Optionally clean up dataless SEED file
                    # os.remove(dataless_file)

                except Exception as e:
                    print(f"[ERROR] Failed for {net.code}.{sta.code}.{cha.location_code}.{cha.code}: {e}")



if __name__ == '__main__':

    xmlfile = '/data/station_metadata/KSC.xml'
    #dataless = '/data/station_metadata/KSC.dataless'
    respdir = '/data/station_metadata/RESP'

    print('### calibrations only ###:')
    print('trillium+centaur40 = %f' % countsPerMS)        
    print('infraBSU+centaur40 = %f' % countsPerPa40)        
    print('infraBSU+centaur1 = %f' % countsPerPa1)        
    print('chaparralM25+centaur40 = %f' % countsPerPaChap40)        
    print('chaparralM25+centaur1 = %f' % (countsPerPaChap1))   
    print('************************************\n\n')


    print('### Building full inventories with responses ###')
    print('First try to get combined response for infraBSU and Centaur:')
    #download_infraBSU_stationxml()
    inv = get_stationXML_inventory(xmlfile=xmlfile, overwrite=True)
    print(inv)

    inventory2dataless_and_resp(inv, output_dir=respdir,
                                stationxml_seed_converter_jar="/home/thompsong/stationxml-seed-converter.jar")