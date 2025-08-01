import os
import struct

from obspy import read, Stream, Trace
from obspy.core.inventory import read_inventory

from flovopy.core.preprocessing import fix_trace_id,  _get_band_code, _adjust_band_code_for_sensor_type
from flovopy.core.trace_utils import remove_empty_traces

dome_location = {'lat':16.71111, 'lon':-62.17722}

##########################################################################
####                    Montserrat Trace tools                        ####
##########################################################################
def fix_trace_mvo_wrapper(trace):
    legacy = True
    sta = trace.stats.station
    if (sta[0:2] == 'MB' and sta!='MBET') or sta[0:3]=='MTB' or sta[0:3]=='BLV':
        legacy = False
    fix_trace_mvo(trace, legacy=legacy, netcode='MV')

def fix_trace_mvo(trace, legacy=False, netcode='MV'):
    fix_y2k_times_mvo(trace)
    fix_sample_rate(trace)
    original_id = trace.id
    trace.id = correct_nslc_mvo(trace.id, trace.stats.sampling_rate)
    mvo_id = trace.id
    fix_trace_id(trace, legacy=legacy, netcode=netcode)
    fixed_id = trace.id
    #print(f'{original_id} -> {mvo_id} -> {fixed_id}')

def load_mvo_master_inventory(XMLDIR):
    master_station_xml = os.path.join(XMLDIR, 'MontserratDigitalSeismicNetwork.xml')
    if os.path.exists(master_station_xml):
        print('Loading ',master_station_xml)
        return read_inventory(master_station_xml)
    else:
        print('Could not find ',master_station_xml)        
        return None

def fix_sample_rate(st, Fs=75.0):
    if isinstance(st, Stream):
        for tr in st:
            fix_sample_rate(tr, Fs=Fs)  # Recursive call for each Trace
    elif isinstance(st, Trace):
        tr = st
        if tr.stats.sampling_rate > Fs * 0.99 and tr.stats.sampling_rate < Fs * 1.01:
            tr.stats.sampling_rate = Fs 
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace object.")    

def fix_y2k_times_mvo(st):
    if isinstance(st, Stream):
        for tr in st:
            fix_y2k_times_mvo(tr, Fs=75.0)  # Recursive call for each Trace
    elif isinstance(st, Trace):
        tr = st
        yyyy = tr.stats.starttime.year
        if yyyy == 1991 or yyyy == 1992 or yyyy == 1993: # digitize
            # OS9/Seislog for a while subtracted 8 years to avoid Y2K problem with OS9
            # should be already fixed but you never know
            tr.stats.starttime._set_year(yyyy+8)
        if yyyy < 1908:
            # getting some files exactly 100 years off
            tr.stats.starttime._set_year(yyyy+100) 
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace object.")
        
# bool_ASN: set True only if data are from MVO analog seismic network
def read_mvo_waveform_file(wavpath, bool_ASN=False, verbose=False, \
                                            seismic_only=False, vertical_only=False):
    if os.path.exists(wavpath):
        st = read(wavpath)
    else:
        print('ERROR. %s not found.' % wavpath)
        return Stream()
    
    if vertical_only:
        st = st.select(component='Z')
    elif seismic_only:
        for tr in st:
            if not tr.stats.channel[1] in 'HL':
                st.remove(tr)
    remove_empty_traces(st)
    for tr in st:
        fix_trace_mvo(tr)

    return st

def correct_nslc_mvo(traceID, Fs, shortperiod=None):
    # Montserrat trace IDs are often bad. return correct trace ID
    # also see fix_nslc_montserrat in /home/thompsong/Developer/SoufriereHillsVolcano/AnalogSeismicNetworkPaper/LIB/fix_mvoe_traceid.ipynb
    # special case - based on waveform analysis, this trace is either noise or a copy of MV.MBLG..SHZ
    if traceID == '.MBLG.M.DUM':
        traceID= 'MV.MBLG.10.SHZ'
    traceID = traceID.replace("?", "x")

    oldnet, oldsta, oldloc, oldcha = traceID.split('.')

    net = 'MV'    
    sta = oldsta.strip()
    loc = oldloc.strip()
    chan = oldcha.strip()
    if not chan:
       chan = 'SHZ'

    if 'J' in loc or 'J' in chan:
        Fs = 75.0
    # Deal with AEF files which have channel 'S JZ' or channel 'SBJZ'
    if chan[0:3] == 'S J':
        chan = 'SH' + chan[3:]
    elif chan[0:3] == 'SBJ':
        chan = 'BH' + chan[3:]


    # Deal with the weird microbarometer ids
    # from the old DSN
    if chan == 'A N' and loc == 'J':
        chan = 'SDO' # barometer, or maybe acoustic pressure sensor, at 75 Hz
        loc = ""
    elif chan == 'PRS': 
        if loc:
            chan ='SD' + loc[0]
            loc = loc[1:]
        else:
            chan = 'SDO'
    # from the new DSN
    elif loc == 'S' and chan == 'AP': 
        chan = 'EDO'
        loc = ""
    elif loc in '0123456789' and chan == 'PR':
        chan = 'EDO'
        loc = loc.zfill(2)
    # final catch
    elif 'AP' in chan or 'PR' in chan or 'PH' in chan or chan=='S A':
        instrumentcode = 'D'
        orientationcode = 'O'
        if chan[-1].isnumeric():
            loc = chan[-1].zfill(2)
        elif loc.isnumeric():
            loc = loc.zfill(2)
        else:
            loc = ''

    else: # Now deal with seismic channels
        # deal with picks in Seisan S-files
        instrumentcode = 'H' 
        orientationcode = 'x'
        if len(chan)==2:
            if chan[1] in 'ZNE':
                orientationcode = chan[1]
            elif chan[1] == 'H' and loc in 'ZNE':
                chan = chan + loc
                loc = ''

        if loc == '--' or loc == 'J' or loc=='I':
            loc = ''    

        if not shortperiod:
            if chan:
                if 'SB' in chan or chan[0] in 'BH':
                    shortperiod = False
                    chan = 'BH' + chan[2:]
                else:
                    shortperiod = True

        #print(f'shortperiod: {shortperiod}')

        # Determine the correct band code
        expected_band_code = _get_band_code(Fs) # this assumes broadband sensor
        #print(f'band_code 1: {expected_band_code}')

        # adjust band_code if short-period sensor
        expected_band_code = _adjust_band_code_for_sensor_type(chan[0], expected_band_code, short_period=shortperiod)
        #print(f'band_code 2: {expected_band_code}')
        chan = expected_band_code + chan[1:]    

        
        if sta[0:2] != 'MB' and (shortperiod and 'L' in chan or sta[-1]=='L'): # trying to account for a low gain sensor, but this should be dealt with by processing legacy IDs
            print(f'Warning: {traceID} might be a legacy ID for a low gain sensor from an old analog network')
        
        if 'Z' in loc or 'Z' in chan:
            orientationcode = 'Z'
        elif 'N' in loc or 'N' in chan:
            orientationcode = 'N'
        elif 'E' in loc or 'E' in chan:
            orientationcode = 'E'    
        elif len(chan)>1:
            if chan[1].strip():
                instrumentcode = chan[1] 
            if len(chan)>2:
                orientationcode = chan[2]
                if orientationcode=='H':
                    orientationcode='x'

        
        # Montserrat BB network 1996-2004 had weirdness like
        # BB stations having channels 'SB[Z,N,E]' and
        # SP stations having channels 'S [Z,N,E]'
        # location code was usually 'J' for seismic, 'E' for pressure
        # channel was 'PRS' for pressure
        # there were also 'A N' channels co-located with single-component Integra LA100s, so perhaps those were some other
        # type of seismometer, oriented Northt?
        # let's handle these directly here
        if len(chan)==2:

            # could be a 2006 era waveform trace ID where given as .STAT.[ZNE].[BS]H
            if len(loc)==1:
                #chan=chan+loc # now length 3
                if loc in 'ZNE':
                    orientationcode = loc
                    loc = ''
                #if not loc.isnumeric():
                #    loc='' 
            elif len(loc)==0:
                # could be arrival row from an Sfile, where the "H" is omitted
                # or an AEF line where trace dead and orientation missing
                #instrumentcode = 'H'
                if chan[1] in 'ZNE':
                    orientationcode = chan[1]
                #else:
                #    orientationcode = '' # sometimes get two-character chans from AEF lines which omit component when trace is dead, e.g. 01-0954-24L.S200601, 


        
        elif len(chan)==3:
            
            if chan[0:2]=='SB':
                # just because we know it is BB sensor
                instrumentcode = 'H' # alternative is L, which only applies for low-gain short-period station

        chan = expected_band_code + instrumentcode + orientationcode

    newID = net + "." + sta + "." + loc + "." + chan
    #print(traceID,'->',newID)
    return newID

def change_last_sample(tr):
    # For some SEISAN files from Montserrat - possibly from SEISLOG conversion,
    # the last value in the time series was always some absurdly large value
    # So remove the last sample
    tr.data = tr.data[0:-2]

def swap32(i):
    # Change the endianess
    return struct.unpack("<i", struct.pack(">i", i))[0]




