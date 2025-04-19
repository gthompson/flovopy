import os
from glob import glob
#import datetime as dt
from obspy import UTCDateTime

# need to leave this here to prevent circular imports between Wavfile and Sfile
def filetime2spath(filetime, mainclass='L', db=None, seisan_top=None, fullpath=True):
    """Create full path to a SEISAN S-file from a datetime object."""
    spath = '%02d-%02d%02d-%02d%s.S%4d%02d' % (
        filetime.day, filetime.hour, filetime.minute, filetime.second,
        mainclass, filetime.year, filetime.month
    )
    if fullpath:
        spath = os.path.join(f"{filetime.year}", f"{filetime.month:02d}", spath)
        if db:
            spath = os.path.join("REA", db, spath)
            if seisan_top:
                spath = os.path.join(seisan_top, spath)
    return spath

def spath2datetime(spath):
    """Extract datetime from SEISAN S-file path."""
    basename = os.path.basename(spath)
    if '.S' in spath: # 
        parts = basename.split('.S')
        yyyy = int(parts[1][0:4])
        mm = int(parts[1][4:6])
        parts = parts[0].split('-')
        dd = int(parts[0])
        HH = int(parts[1][0:2])
        MM = int(parts[1][2:4])
        SS = float(parts[2][0:2])
        return UTCDateTime(yyyy, mm, dd, HH, MM, SS)
    else:
        return None    

def legacy_or_not(fname):
    fnamelower = os.path.basename(fname).lower()
    """ This part fails because of 1997 files like /data/SEISAN_DB/REA/MVOE_/1997/04/04-0231-34L.S199704 
    if 'mvo' in fnamelower or 'asne' in fnamelower or 'dsne' in fnamelower or 'spn' in fnamelower:
        print(f'Processing {fname}')
    else:
        print(f'Not processing {fname}')          
        return None, None
    """
    legacy = False
    network = 'DSN'
    if 'asne' in fnamelower or 'spn' in fnamelower:
        legacy = True # WAV file from the analog network
        network = 'ASN'
    return legacy, network


def parse_string(line, pos0, pos1, astype='float', stripstr=True, default=None):
    """Safely extract a substring from a line and convert to int/float/str."""
    _s = line[pos0:pos1]
    if stripstr:
        _s = _s.strip()
    if not _s:
        return default
    try:
        if astype == 'float':
            return float(_s)
        elif astype == 'int':
            return int(_s)
        return _s
    except ValueError:
        return default

'''
def correct_nslc(traceID, sampling_rate=100.0, shortperiod=False):
    """
    Fix NSLC code for a waveform based on sampling rate and whether it's shortperiod.
    Returns a waveform_id dict suitable for ObsPy Pick.
    """
    try:
        parts = traceID.strip().split('.')
        if len(parts) == 4:
            network, station, location, channel = parts
        else:
            station = parts[1] if len(parts) > 1 else ""
            channel = parts[-1]
            network, location = "", ""

        if not channel:
            channel = "???"  # Fallback

        band_code = 'H' if sampling_rate >= 80 else 'S' if shortperiod else 'B'
        comp_code = channel[-1] if len(channel) > 0 else '?'
        chan_code = band_code + 'H' + comp_code  # e.g., 'HHZ', 'SHZ', 'BHZ'

        return {
            "network_code": network,
            "station_code": station,
            "location_code": location,
            "channel_code": chan_code
        }
    except Exception as e:
        print(f"[correct_nslc] Failed to parse traceID '{traceID}': {e}")
        return {
            "network_code": "",
            "station_code": "",
            "location_code": "",
            "channel_code": "???"
        }
'''

def filetime2wavpath(filetime, sfilepath, y2kfix=False, numchans=0, dbstring='MVO__'):
    """Create full path to a WAV file from datetime object."""

    if len(dbstring) < 5:
        dbstring += '_' * (5 - len(dbstring))

    wavdir = os.path.dirname(sfilepath).replace('REA', 'WAV')


    if not y2kfix and filetime.year < 2000:
        basename = '%2d%02d-%02d-%02d%02d-%02dS.%s_%03d' % (
            filetime.year - 1900, filetime.month, filetime.day,
            filetime.hour, filetime.minute, filetime.second,
            dbstring, numchans
        )
    else:
        basename = '%4d-%02d-%02d-%02d%02d-%02dS.%s_%03d' % (
            filetime.year, filetime.month, filetime.day,
            filetime.hour, filetime.minute, filetime.second,
            dbstring, numchans
        )

    return os.path.join(wavdir, basename)

def find_matching_wavfiles(filetime, sfilepath, y2kfix=False):
    # Should match any wavfile (even SPN) within 1-s of filetime
    matching_wavfiles = []
    for timeshift in [0, -1, 1]:
        wavpattern = filetime2wavpath(filetime + timeshift, sfilepath, y2kfix=y2kfix)
        potentialwavfiles = glob(wavpattern.split('.')[0]+".*")
        matching_wavfiles.extend(potentialwavfiles)
    return matching_wavfiles

