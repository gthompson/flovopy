import os
from glob import glob
import datetime as dt
from obspy.core.event import Pick, QuantityError
from obspy.io.nordic.core import readheader, readwavename, _is_sfile
from obspy import read

from flovopy.seisanio.core.wavfile import Wavfile
from flovopy.seisanio.core.aeffile import AEFfile
from flovopy.seisanio.utils.helpers import spath2datetime
from flovopy.core.mvo import correct_nslc_mvo

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
                self.parse_sfile()
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
                self.subclass = line.split("VOLC MAIN")[-1].strip()[0]
                print(self.subclass)

    def parse_sfile(self, verbose=False):
        if verbose:
            print('Parsing ', self.path)
        fptr = open(self.path, 'r')
        row = fptr.readlines()
        fptr.close()
        _aeffiles = list()

        for line in row:
            result = line.find("VOLC")
            if line[-1] == '3' or result > -1:
                if line[1:7] == 'ExtMag':
                    self.magnitude.append(float(line[8:12]))
                    self.magnitude_type.append(line[12])
                    self.magnitude_agency.append(line[13:16])
                elif line[1:4] == 'URL':
                    self.url = line[5:78].rstrip()
                elif result:
                    if line[result:result + 9] == "VOLC MAIN":
                        self.subclass = line[result + 10:result + 20].strip()
                    else:
                        if not self.path in _aeffiles:
                            _aeffiles.append(self.path)
                            continue

            if line.find('VOLC MBLYTBH') > -1:
                _aeffiles.append(self.path)
                continue

            if len(line) < 80:
                if len(line.strip()) > 0 and verbose:
                    print("Processing %s: ignoring this line: %s" % (self.path, line))
                continue

            if line[79] == '1':
                # Parse origin time and event details
                try:
                    oyear, omonth, oday = int(line[1:5]), int(line[6:8]), int(line[8:10])
                    ohour, ominute = int(line[11:13]), int(line[13:15])
                    dummy = line[15:20].strip()
                    osecond = float(dummy) if dummy else 0.0
                    if int(osecond) == 60:
                        ominute += 1
                        osecond -= 60.0
                    if osecond > 60:
                        osecond /= 100
                    self.otime = dt.datetime(oyear, omonth, oday, ohour, ominute, int(osecond), int((osecond % 1) * 1e6))
                except:
                    if verbose:
                        print('Failed 1-21:\n01233456789' * 8 + f"\n{line}")

                self.mainclass = line[21:23].strip()
                self.latitude = parse_string(line, 23, 30, 'float')
                self.longitude = parse_string(line, 30, 38, 'float')
                self.depth = parse_string(line, 38, 43, 'float')
                self.z_indicator = line[43].strip()
                self.agency = line[45:48].strip()
                self.no_sta = parse_string(line, 49, 51, 'int', default=0)
                self.rms = parse_string(line, 51, 55, 'float')

                for i in [(55, 59, 59, 60, 63), (63, 67, 67, 68, 71), (71, 75, 75, 76, 79)]:
                    try:
                        mag = float(line[i[0]:i[1]])
                        mag_type = line[i[2]]
                        mag_agency = line[i[3]:i[4]].strip()
                        self.magnitude.append(mag)
                        self.magnitude_type.append(mag_type)
                        self.magnitude_agency.append(mag_agency)
                    except:
                        if verbose:
                            print(f'Failed {i[0]}-{i[4]}:\n01233456789' * 8 + f"\n{line}")

            if line[79] == '2':
                self.maximum_intensity = int(line[27:29])

            if line[79] == '6':
                wavfiles = line[1:79].split()
                for wavfile in wavfiles:
                    wavfullpath = os.path.join(os.path.dirname(self.path).replace('REA', 'WAV'), wavfile)
                    self.wavfiles.append(Wavfile(wavfullpath))
                    aeffullpath = wavfullpath.replace('WAV', 'AEF')
                    if os.path.exists(aeffullpath) and aeffullpath not in _aeffiles:
                        _aeffiles.append(aeffullpath)

            if line[79] == 'E':
                self.gap = parse_string(line, 5, 8, 'int', default=-1)
                self.error['origintime'] = parse_string(line, 14, 20, 'float')
                self.error['latitude'] = parse_string(line, 24, 30, 'float')
                self.error['longitude'] = parse_string(line, 32, 38, 'float')
                self.error['depth'] = parse_string(line, 38, 43, 'float')
                self.error['covxy'] = parse_string(line, 43, 55, 'float')
                self.error['covxz'] = parse_string(line, 55, 67, 'float')
                self.error['covyz'] = parse_string(line, 67, 79, 'float')

            if line[79] == 'F' and 'dip' not in self.focmec:
                self.focmec['strike'] = float(line[0:10])
                self.focmec['dip'] = float(line[10:20])
                self.focmec['rake'] = float(line[20:30])
                self.focmec['agency'] = line[66:69]
                self.focmec['source'] = line[70:77]
                self.focmec['quality'] = line[77]

            if line[79] == 'H':
                _osec = float(line[16:22])
                _yyyy = int(line[1:5])
                _mm = int(line[6:8])
                _dd = int(line[8:10])
                _hh = int(line[11:13])
                _mi = int(line[13:15])
                _ss = int(_osec)
                _ms = int((_osec - _ss) * 1.e6)
                self.otime = dt.datetime(_yyyy, _mm, _dd, _hh, _mi, _ss, _ms)
                self.latitude = float(line[23:32].strip())
                self.longitude = float(line[33:43].strip())
                self.depth = float(line[44:52].strip())
                self.rms = float(line[53:59].strip())

            if line[79] == 'I':
                self.last_action = line[8:11]
                self.action_time = line[12:26]
                self.analyst = line[30:33]
                self.id = int(line[60:74])

            if line[79] == ' ' and line[1] == 'M':
                asta = line[1:5].strip()
                achan = line[5:8].strip()
                aphase = line[8:16].strip()
                ahour = int(line[18:20].strip())
                aminute = int(line[20:22].strip())
                asecond = line[22:28].strip()
                amillisecond = int(asecond.split('.')[1]) if '.' in asecond else 0
                asecond = int(float(asecond))
                if asecond >= 60:
                    aminute += 1
                    asecond -= 60
                    if aminute >= 60:
                        aminute -= 60
                        ahour += 1
                if ahour > 23:
                    ahour -= 24
                atime = self.otime or self.filetime
                try:
                    atime = dt.datetime(atime.year, atime.month, atime.day, ahour, aminute, asecond, 1000 * amillisecond)
                except:
                    if verbose:
                        print('Failed 18-29:\n01233456789' * 8 + f"\n{line}")

                thisarrival = {
                    'sta': asta,
                    'chan': achan,
                    'phase': aphase,
                    'time': atime
                }
                if line[64:79].strip():
                    thisarrival['time_residual'] = parse_string(line, 64, 68, 'float')
                    thisarrival['weight'] = parse_string(line, 68, 70, 'int')
                    thisarrival['distance_km'] = parse_string(line, 72, 75, 'float')
                    thisarrival['azimuth'] = parse_string(line, 77, 79, 'int')

                onset = 'questionable'
                if aphase.startswith('I'):
                    onset = 'impulsive'
                    aphase = aphase[1:]
                elif aphase.startswith('E'):
                    onset = 'emergent'
                    aphase = aphase[1:]

                if achan:
                    traceID = f".{asta}..{achan}"
                    Fs = 75 if asta[:2] == 'MB' and self.filetime.year < 2005 else 100
                    fixedID = correct_nslc_mvo(traceID, Fs, shortperiod=False)
                    if 'time_residual' in thisarrival:
                        tres = QuantityError(uncertainty=thisarrival['time_residual'])
                        p = Pick(time=atime, waveform_id=fixedID, onset=onset, phase_hint=aphase,
                                time_errors=tres, backazimuth=(180 + thisarrival['azimuth']) % 360)
                    else:
                        p = Pick(time=atime, waveform_id=fixedID, onset=onset, phase_hint=aphase)
                    self.arrivals.append(p)

        for _aeffile in _aeffiles:
            self.aeffiles.append(AEFfile(_aeffile))


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
        """ summary string of an Sfile object, for print command """
        str = "S-file path: " + self.path
        str += "\n\tFile time: " + self.filetime.strftime("%Y-%m-%d %H:%M:%S.%f")
        if self.otime:
            str += "\n\tOrigin time: " + self.otime.strftime("%Y-%m-%d %H:%M:%S.%f")
        if self.mainclass:
            str += "\n\tMainclass: " + self.mainclass
        if self.subclass:
            str += "\n\tSubclass: " + self.subclass
        if self.longitude:
            str += "\n\tLongitude: %f " %  self.longitude
        if self.latitude:
            str += "\n\tLatitude: %f" % self.latitude
        if self.depth:
            str += "\n\tDepth: %f" % self.depth
        if self.z_indicator:
            str += "\n\tZ_indicator: " + self.z_indicator
        if self.agency:
            str += "\n\tAgency: " + self.agency
        if self.id:
            str += "\n\tId: %d " % self.id
        if self.rms:
            str += "\n\trms: %f" % self.rms
        for count in range(len(self.magnitude)):
            str += "\n\tMagnitude: %.1f, %s, %s" % (self.magnitude[count], self.magnitude_type[count], self.magnitude_agency[count])
        if self.last_action:
            str += "\n\tLast action: " + self.last_action
        if self.action_time:
            str += "\n\tLast action time: " + self.action_time
        if self.analyst:
            str += "\n\tAnalyst: " + self.analyst
        if self.url:
            str += "\n\tURL: " + self.url
        if self.gap:
            str += "\n\tGap: %d" % self.gap
        if self.error:
            str += "\n\tError: " + pprint.pformat(self.error)
        if self.focmec:
            str += "\n\tFocmec: " + pprint.pformat(self.focmec)
        str += "\n\tNumber of arrivals: %d" % len(self.arrivals)
        if len(self.arrivals)>0:
            str += "\n\tArrivals:" 
            for pick in self.arrivals:
                str += pick.__str__()
        if len(self.wavfiles)>0:
            str += "\n\tWAV files:" 
            for wavfile in self.wavfiles:
                str += "\n\t\t" + wavfile.__str__()
        if len(self.aeffiles)>0:
            str += "\n\tAEF files:" 
            for aeffile in self.aeffiles:
                str += "\n\t\t" + aeffile.__str__()
        if self.events:
            str += "\n\tEvents:"
            for event in self.events:
                str += "\n\t\t" + event.__str__()

        return str
    
    def to_css(self):
        """ Write CSS3.0 database event from an Sfile
        This is potentially less lossy than via an ObsPy Catalog object first """
        pass

    def to_csv(self, csvfile):
        sdict = self.to_dict()
        sdict.to_csv(csvfile)
    
    def to_obspyevent(self):
        thisevent = self.events
        return thisevent
    
    def to_pickle(self):
        import pickle
        picklefile = self.path.replace('REA','PICKLE') + '.pickle'
        with open(picklefile, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def cat(self):
        """ Show contents of an Sfile """
        fptr = open(self.path,'r')
        contents = fptr.read()
        fptr.close()
        print(contents)
                 
    def printEvents(self):
        evobj = sfileobj.events
        print(evobj)
        print(evobj.event_descriptions)
        for i, originobj in enumerate(evobj.origins):
            print(i, originobj)


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


def parse_string(line, pos0, pos1, astype='float', stripstr=True, default=None):
    _s = line[pos0:pos1]
    if stripstr:
        _s = _s.strip()
    if not _s:
        return default # check if this makes sense
    if astype=='float':           
        return float(_s)
    elif astype=='int':           
        return int(_s)        
    else:
        return _s
    
def read_pickle(picklefile):
    import pickle
    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        self = pickle.load(f)
    return self