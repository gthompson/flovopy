import os
import struct

from obspy import read, Stream, Trace
from obspy.core.inventory import read_inventory

from flovopy.core.trace_utils import remove_empty_traces, fix_trace_mvo

dome_location = {'lat':16.71111, 'lon':-62.17722}

##########################################################################
####                    Montserrat Trace tools                        ####
##########################################################################


def load_mvo_master_inventory(XMLDIR):
    master_station_xml = os.path.join(XMLDIR, 'MontserratDigitalSeismicNetwork.xml')
    if os.path.exists(master_station_xml):
        print('Loading ',master_station_xml)
        return read_inventory(master_station_xml)
    else:
        print('Could not find ',master_station_xml)        
        return None

        
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



def change_last_sample(tr):
    # For some SEISAN files from Montserrat - possibly from SEISLOG conversion,
    # the last value in the time series was always some absurdly large value
    # So remove the last sample
    tr.data = tr.data[0:-2]

def swap32(i):
    # Change the endianess
    return struct.unpack("<i", struct.pack(">i", i))[0]




