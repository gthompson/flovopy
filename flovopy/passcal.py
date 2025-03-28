import os
import glob
from obspy import read, Stream
from flovopy.core.waveform_archive import Stream_to_24H  # Adjust path if Stream_to_24H is defined elsewhere


#######################################################################
##                                                  RT130 tools                                                                ##
#######################################################################

def rt130YYYYJJJdirectory_to_StreamDay(TOPDIR, YYYYJJJ, NETWORK, chans, digitizerStationPairs):
    """ 
    Explore a YYYYJJJ directory from one or many RT130 digitizers. 
    Read the Reftek waveform data files into an ObsPy Stream object that is 24-hours long.
    
    A YYYYJJJ directory has a directory structure like:
    
        2020346
        ├── 91F8
        │   ├── 0
        │   ├── 1
        │   └── 9
        ├── 9338
        │   ├── 0
        │   └── 1
        ├── 9D7C
        │   ├── 0
        │   └── 1
        └── AB13
            ├── 0
            ├── 1
            └── 9
    
    where 91F8, 9338, 9D7C and AB13 are the names of Reftek RT130 digitizers, 
    and the 1 directories contain Reftek waveform data in (usually) 1-hour chunks.
    
    Inputs:
        TOPDIR                - the directory beneath which the YYYYJJJ directories are stored
        YYYYJJJ               - the YYYYJJJ directory to process
        NETWORK               - the 2-character seismic network ID
        chans                 - a list of channel names for each station, usually EHZ, EH1 and EH2
        digitizerStationPairs - some Reftek waveform files do not contain station metadata in the header
                                Missing pairings can be described in a dictionary

    Example:
        TOPDIR = './RAW'
        YYYYJJJ = '2020346'
        NETWORK = '1R'
        chans = ['EHZ','EH1','EH2']
        digitizerStationPairs = {'9338':'BHP4'}    
        all_traces = rt130YYYYJJJdirectory_to_StreamDay(TOPDIR, YYYYJJJ, NETWORK, chans, digitizerStationPairs)


    Created to convert ROCKETSEIS data.    
    """

    all_traces = Stream()
    digitizerDirs = glob.glob(os.path.join(TOPDIR, YYYYJJJ, '????'))

    # Convert the 1 directories
    for thisDigitizerDir in digitizerDirs:
        oneDir = os.path.join(thisDigitizerDir, '1')
        if os.path.exists(oneDir):
            rt130Files = sorted(glob.glob(os.path.join(oneDir, '?????????_????????')))
            for rt130File in rt130Files:
                try:
                    st = read(rt130File)  
                    for c in range(3):
                        st[c].stats.network = NETWORK
                        st[c].stats.channel = chans[c]
                except:
                    print(rt130File, ' is bad\n')                
                else:

                    for tr in st:
                        if not tr.stats.station:
                            print('No station metadata in %s' % rt130File)
                            thisDigitizer = os.path.basename(thisDigitizerDir)
                            if digitizerStationPairs[thisDigitizer]:
                                tr.stats.station = digitizerStationPairs[thisDigitizer]

                        if tr.stats.station:
                            try:
                                all_traces.append(tr)
                            except:
                                print("Could not add trace")
                                
    all_traces = Stream_to_24H(all_traces)
    
    return all_traces
    
    
    
    
def chooseCorrectParFile(yyyyjjj):
    
    correctparfile = ""
    parfilelist = sorted(glob.glob('%s/network20?????.par' % CONFIGDIR))
    for thisparfile in parfilelist:
        parfileyyyyjjj = thisparfile[-11:-4]
        #print(parfileyyyyjjj, yyyyjjj)
        if parfileyyyyjjj <= yyyyjjj:
            correctparfile = thisparfile
    return correctparfile