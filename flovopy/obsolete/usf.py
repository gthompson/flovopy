import os
import requests
import pandas as pd
import numpy as np
from collections import defaultdict

from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.inventory.response import Response, InstrumentSensitivity
from obspy.clients.nrl import NRL
from obspy.io.xseed import Parser


""" the idea of this is to give ways to just use the overall sensitivyt easily, or the full
 instrument response for different types of datalogger/sensors USF has used """
'''
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

def get_rs1d_v4(sta, loc, fsamp=100.0):
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

    if fsamp==50.0:
        # Access the channel object (assuming 1 net, 1 station, 1 channel)
        channel = inv[0][0][0]

        # Set new sampling rate
        channel.sample_rate = 50.0

        # Adjust decimation stages (if needed)
        for stage in channel.response.response_stages:
            if hasattr(stage, "decimation_input_sample_rate"):
                stage.decimation_input_sample_rate = 50.0
            if hasattr(stage, "decimation_factor") and stage.decimation_factor != 1:
                print(f"Warning: Stage {stage.stage_sequence_number} uses decimation factor {stage.decimation_factor} which may need recalculating.")

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

def replace_sta_loc(inv, sta='DUMM', loc=''):
    # assumes only 1 network with 1 station

    # Find the station (you can loop over networks and stations if needed)

    network = inv[0]  # Replace with the network code
    station = network[0] #.get_station("STA_CODE")  # Replace with the station code

    # Change the station name (code) and location code
    station.code = sta  # Change to the new station name
    station.location_code = loc     



def download_infraBSU_stationxml(save_path=None):
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
    fsamp=100.0,
    vpp=40,
    stationxml_path=None,
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
    combined_resp.response_stages += centaur_resp.response_stages[1:] # the first stage is bogus - it is for a Trillium

    # Recompute overall sensitivity


    # Get only valid gain values from all stages
    gains = [stage.stage_gain for stage in combined_resp.response_stages if hasattr(stage, 'stage_gain')]

    # Multiply all the stage gains to get the overall system gain
    overall_gain = np.prod(gains)

    # Set the instrument sensitivity
    combined_resp.instrument_sensitivity = InstrumentSensitivity(
        value=overall_gain,
        frequency=1.0,  # Set to the defined frequency (e.g., 1.0 Hz or your sensor's nominal freq)
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
    station_obj.creation_date=UTCDateTime()
    station_obj.site=Site(name=sitename or f"{station}_SITE",
                description=f"InfraBSU component {channel} at {station}")
        

    network_obj = sensor_inv[0]
    network_obj.code = network


    return sensor_inv

'''
  

'''
def NRL2inventory(nrl_path, net, sta, loc, chans,
                  datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0,
                  lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='',
                  ondate=UTCDateTime(1970, 1, 1), offdate=UTCDateTime(2025, 12, 31),
                  sensitivity=None, units=None):
    # Suppress only DeprecationWarnings
    #warnings.filterwarnings("ignore", category=ObsPyDeprecationWarning)
    if os.path.isdir(nrl_path):
        nrl = NRL(nrl_path)
    else:
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

def apply_coordinates_from_csv(inventory, csv_path):
    """
    Updates station coordinates in an ObsPy Inventory from a CSV file with header:
    Network,Station,Latitude,Longitude,Elevation,Depth
    """
    # Load CSV into a DataFrame and drop rows with missing Network or Station
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Network', 'Station'])

    # Fill missing depths with 0.0
    df['Depth'] = df['Depth'].fillna(0.0)

    # Set index for fast lookup
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
'''



'''
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

def build_inventory_from_csv(csv_path: str, stationxml_path: str = None, nrl_path: str = None) -> Inventory:
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
            datalogger = row['datalogger'].strip()
            network=str(row["network"]).strip()
            station=str(row["station"]).strip()
            location=str(row["location"]).strip()
            lat=float(row["lat"].strip() or 0.0)
            lon=float(row["lon"].strip() or 0.0)
            elev=float(row["elev"].strip() or 0.0)
            depth=float(row["depth"].strip() or 0.0)    
            fsamp=float(row["fsamp"].strip() or 100.0)
            ondate=UTCDateTime(row["ondate"].strip() or '2000-01-01')
            offdate=UTCDateTime(row["offdate"].strip() or '2100-01-01')            
            sensor=row['sensor'].strip()

            if datalogger.upper()[0:2] == 'RS' or datalogger.upper() == 'RBOOM':
                # a type of Raspberry Shake
                if datalogger.upper() == 'RSB':
                    inv = get_rsb(station, location)
                elif datalogger.upper() == 'RBOOM':
                    inv = get_rboom(station, location)
                elif datalogger.upper() == 'RS1D':
                    inv = get_rs1d_v4(station, location, fsamp=fsamp)
                elif datalogger.upper() == 'RS3D':
                    inv = get_rs3d_v5(station, location)  # or version 3?    

                for net in inv:
                    for sta in net:
                        sta.latitude = lat
                        sta.longitude = lon
                        sta.elevation = elev
                        sta.depth = depth
                        for ch in sta:
                            ch.latitude = lat
                            ch.longitude = lon
                            ch.elevation = elev
                            ch.depth = depth    
                master_inventory += inv                            
            else:
                chan_list = expand_channel_code(row["channel"])
                vpp = float(row['vpp'] or 32.0) # 32 is default REFTEK RT130 PASSCAL gain, rather than peak-to-peak input voltage

                if row['channel'][1]=='H' or sensor.upper[0:4]=='CHAP': # special case for Chaparral M-25
                    inv = NRL2inventory(
                        nrl_path,
                        network,
                        station,
                        location,
                        chan_list,          
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
                    master_inventory += inv
                elif 'infraBSU' in sensor:
                        for ch in chan_list:
                            inv = build_combined_infrabsu_centaur_stationxml(
                                fsamp=fsamp,
                                vpp=vpp,
                                stationxml_path=stationxml_path,
                                network=network,
                                station=station,
                                location=location,
                                channel=ch.strip(),
                                latitude=lat,
                                longitude=lon,
                                elevation=elev,
                                depth=depth,
                                start_date=ondate,
                                end_date=offdate
                            )   
                            master_inventory += inv
                else:
                    print('CSV line not parsed successfully:\n', row)                
 

        #except Exception as e:
        #    print(f"[ERROR] Failed for {row['network']}.{row['station']}.{row['location']}.{row['channel']}: {e}")
    
    return master_inventory
'''

def get_stationXML_inventory(
    xmlfile=None,
    seedfile=None,
    respdir=None,
    metadata_csv=None,
    coord_csv=None,
    infraBSUstationxml=None,
    nrl_path=None,
    overwrite=False
):


    if os.path.isfile(xmlfile) and not overwrite:
        inv = read_inventory(xmlfile)
        print(f"[INFO] Loaded existing StationXML: {xmlfile}")
    else:
        print("[INFO] Creating new inventory from USF definitions...")

        inv = build_inventory_from_csv(csv_path = metadata_csv, stationxml_path = infraBSUstationxml, nrl_path=nrl_path)

        # Apply coordinates from CSV
        if os.path.isfile(coord_csv):
            apply_coordinates_from_csv(inv, coord_csv)

        # Merge + site patch
        inv = merge_duplicate_stations_and_patch_site(inv)

        # Save StationXML
        inv.write(xmlfile, format='STATIONXML', validate=True)
        print(f"[OK] Wrote StationXML to {xmlfile}")

    return inv




def inventory2dataless_and_resp(inv, output_dir=None, stationxml_seed_converter_jar=None):
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
   


if __name__ == '__main__':

    import platform

    # Get user's home directory
    home = os.path.expanduser("~")

    # Adjust paths based on OS
    if platform.system() == 'Darwin':  # macOS
        metadata_dir = os.path.join(home, 'Dropbox', 'DATA', 'station_metadata')
    else:
        metadata_dir = '/data/station_metadata'
    os.makedirs(metadata_dir, exist_ok=True)

    xmlfile = os.path.join(metadata_dir, 'KSC.xml')
    # dataless = os.path.join(metadata_dir, 'KSC.dataless')
    respdir = os.path.join(metadata_dir, 'RESP')
    os.makedirs(respdir, exist_ok=True)

    metadata_csv = os.path.join(metadata_dir, 'ksc.csv')
    coord_csv = os.path.join(metadata_dir, 'ksc_coordinates_only.csv')

    NRLpath = os.path.join(metadata_dir, 'NRL')
    infraBSUstationXML = os.path.join(metadata_dir, 'infraBSU_21s_0.5inch.xml')

    # Always use $HOME/bin for converter JAR
    stationxml_seed_converter_jar = os.path.join(home, 'bin', 'stationxml-seed-converter.jar')

    print('### calibrations only ###:')
    print('trillium+centaur40 = %f' % countsPerMS)        
    print('infraBSU+centaur40 = %f' % countsPerPa40)        
    print('infraBSU+centaur1 = %f' % countsPerPa1)        
    print('chaparralM25+centaur40 = %f' % countsPerPaChap40)        
    print('chaparralM25+centaur1 = %f' % (countsPerPaChap1))   
    print('************************************\n\n')


    print('### Building full inventories with responses ###')
    print('First try to get combined response for infraBSU and Centaur:')

    if not os.path.isfile(infraBSUstationXML):
        download_infraBSU_stationxml(save_path=infraBSUstationXML)
    inv = get_stationXML_inventory(xmlfile=xmlfile, overwrite=True, infraBSUstationxml=infraBSUstationXML, metadata_csv=metadata_csv, coord_csv=coord_csv, nrl_path=NRLpath)
    print(inv)

    inventory2dataless_and_resp(inv, output_dir=respdir, stationxml_seed_converter_jar=stationxml_seed_converter_jar)
