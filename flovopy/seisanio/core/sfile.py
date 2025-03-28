import os
import datetime as dt
from obspy.core.event import Pick, QuantityError
from obspy.io.nordic.core import readheader, readwavename, _is_sfile
from obspy import read

from core.wavfile import Wavfile
from core.aeffile import AEFfile
from utils.helpers import spath2datetime, parse_string, correct_nslc


class Sfile:
    def __init__(self, path, use_mvo_parser=False, fast_mode=False):
        self.path = path.strip()
        self.filetime = spath2datetime(self.path)
        self.otime = None
        self.mainclass = None
        self.subclass = None
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.z_indicator = None
        self.agency = None
        self.no_sta = None
        self.rms = None
        self.magnitude = []
        self.magnitude_type = []
        self.magnitude_agency = []
        self.last_action = None
        self.action_time = None
        self.analyst = None
        self.id = None
        self.url = None
        self.gap = None
        self.maximum_intensity = None
        self.arrivals = []
        self.wavfiles = []
        self.aeffiles = []
        self.aefrows = []
        self.events = None

        self.error = {
            'origintime': None,
            'latitude': None,
            'longitude': None,
            'depth': None,
            'covxy': None,
            'covxz': None,
            'covyz': None
        }

        self.focmec = {
            'strike': None,
            'dip': None,
            'rake': None,
            'agency': None,
            'source': None,
            'quality': None
        }

        if not os.path.exists(self.path):
            return

        if _is_sfile(self.path):
            if fast_mode:
                self._parse_sfile_fast()
            elif use_mvo_parser:
                self._parse_sfile()
            else:
                self.events = readheader(self.path)
                wavnames = readwavename(self.path)
                wavpath = os.path.dirname(self.path).replace("REA", "WAV")
                for wavfile in wavnames:
                    self.wavfiles.append(Wavfile(os.path.join(wavpath, wavfile)))

    def _parse_sfile_fast(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if len(line) < 80:
                continue
            if line[79] == '1':
                self.mainclass = line[21:23].strip()
            elif line[79] == '6':
                wavnames = line[1:79].split()
                for wavname in wavnames:
                    wavpath = os.path.dirname(self.path).replace("REA", "WAV")
                    self.wavfiles.append(Wavfile(os.path.join(wavpath, wavname)))
            elif "VOLC MAIN" in line:
                self.subclass = line.split("VOLC MAIN")[-1].strip()

    def maximum_magnitude(self):
        mag, mtype, agency = None, None, None
        for i, m in enumerate(self.magnitude):
            if mag is None or (m > mag and 'MVO' in self.magnitude_agency[i]):
                mag = m
                mtype = self.magnitude_type[i]
                agency = self.magnitude_agency[i]
        return mag, mtype, agency

    def to_dict(self):
        sdict = {
            'path': self.path,
            'filetime': self.filetime,
            'mainclass': self.mainclass,
            'subclass': self.subclass,
            'wavfile1': self.wavfiles[0].path if self.wavfiles else None,
            'wavfile2': self.wavfiles[1].path if len(self.wavfiles) > 1 else None,
            'num_magnitudes': len(self.magnitude),
            'magnitude': self.maximum_magnitude()[0],
            'magnitude_type': self.maximum_magnitude()[1],
            'num_wavfiles': len(self.wavfiles),
            'num_aeffiles': len(self.aeffiles),
            'located': self.longitude is not None,
            'num_arrivals': len(self.arrivals),
            'error_exists': self.error['latitude'] is not None,
            'focmec_exists': self.focmec['strike'] is not None
        }
        return sdict

    def __str__(self):
        return f"<Sfile: {self.path}, time={self.filetime}, subclass={self.subclass}>"
    
def get_sfile_list(SEISAN_DATA, DB, startdate, enddate): 
    """
    make a list of Sfiles between 2 dates
    """

    event_list=[]
    reapath = os.path.join(SEISAN_DATA, 'REA', DB)
    years=list(range(startdate.year,enddate.year+1))
    for year in years:
        if year==enddate.year and year==startdate.year:
            months=list(range(startdate.month,enddate.month+1))
        elif year==startdate.year:
            months=list(range(startdate.month,13))
        elif year==enddate.year:
            months=list(range(1,enddate.month+1))
        else:
            months=list(range(1,13))
        for month in months:
            #print month
            yearmonthdir=os.path.join(reapath, "%04d" % year, "%02d" % month)
            flist=sorted(glob(os.path.join(yearmonthdir,"*L.S*")))
            for f in flist:
                #fdt = sfilename2datetime(f)
                fdt = spath2datetime(f)
                #print(f, fdt)
                if fdt>=startdate and fdt<enddate:
                    event_list.append(f)
    return event_list 
