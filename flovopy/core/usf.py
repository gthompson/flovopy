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
        sensor_keys = ['JeffreyBJohnson', 'infraBSU', '0.000046 V/Pa']
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


    print('Raspberry Boom')
    inv = get_rboom('R1234', '00')
    print(inv)
    show_response(inv)
    masterinv = inv

    print('Raspberry Shake 3D')
    inv = get_rs3d_v3('R2345', '00')
    print(inv)
    show_response(inv)
    masterinv = masterinv + inv

    print('Raspberry Shake 1D')
    inv = get_rs1d_v4('R3456', '00')
    print(inv)
    show_response(inv)
    masterinv = masterinv + inv

    print('Centaur-Trillium')
    inv = NRL2inventory('FL', 'BCHH', '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100, \
                        sitename='Beach House original', ondate=UTCDateTime(2016,2,24) )
    show_response(inv)
    #inv.write("BCHH_original_seismic.sml", format="stationxml", validate=True)
    masterinv = masterinv + inv

    print('Chaparral 40V p2p')
    inv = NRL2inventory('FL', 'BCHH', '10', ['HDF'], datalogger='Centaur', sensor='Chap', Vpp=40, fsamp=100, \
                        sitename='Beach House Sonic', ondate=UTCDateTime(2017,8,1) )
    #inv.write("BCHH_sonic_Chap.sml", format="stationxml", validate=True) # Replace PA with Pa and 400,000 in InstrumentSensivitity Value with 160,000
    show_response(inv)
    masterinv = masterinv + inv

    print('Chaparral 1V p2p')
    inv = NRL2inventory('FL', 'BCHH2', '10', ['HDF'], datalogger='Centaur', sensor='Chap', Vpp=1, fsamp=100, \
                        sitename='Beach House Sonic', ondate=UTCDateTime(2017,8,1) )
    #inv.write("BCHH_sonic_Chap.sml", format="stationxml", validate=True) # Replace PA with Pa and 400,000 in InstrumentSensivitity Value with 160,000
    show_response(inv)
    masterinv = masterinv + inv   
    # note need to apply an extra calib of 0.4 in Trace.stats.calib because this is ignored otherwise
    # or need to change units in response stages from Pa to m/s

    print('infraBSU 40V p2p')
    inv = NRL2inventory('FL', 'BCHH2', '10', ['HD4', 'HD5', 'HD6'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100, \
                    lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='', \
                        ondate=UTCDateTime(1970,1,1), offdate=UTCDateTime(2025,12,31), \
                            sensitivity=countsPerPa40, units='m/s')
    show_response(inv)
    masterinv = masterinv + inv   
    
    print('infraBSU 1V p2p')
    inv = NRL2inventory('FL', 'BCHH3', '20', ['HD7', 'HD8', 'HD9'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100, \
                    lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='', \
                        ondate=UTCDateTime(1970,1,1), offdate=UTCDateTime(2025,12,31), \
                            sensitivity=countsPerPa40, units='m/s')    

    print(inv)
    show_response(inv)
    masterinv = masterinv + inv      


    print('****************************\n\nMaster inventory')
    print(masterinv)