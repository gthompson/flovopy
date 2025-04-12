import os
from glob import glob
from obspy.io.nordic.core import readheader, readwavename, _is_sfile, _read_picks, _read_picks_nordic_old, _read_picks_nordic_new
from obspy.io.nordic.utils import _get_line_tags
from obspy import UTCDateTime, read
from obspy.core.event import Event, Origin, Magnitude, WaveformStreamID #, Catalog, Pick, QuantityError
from flovopy.seisanio.core.wavfile import Wavfile
from flovopy.seisanio.core.aeffile import AEFfile
from flovopy.seisanio.utils.helpers import filetime2wavpath
from flovopy.core.mvo import correct_nslc_mvo
from flovopy.core.enhanced import EnhancedEvent
import json

class Sfile:
    def __init__(self, path, use_mvo_parser=False, verbose=False, parse_aef=True, try_external_aeffile=False):
        self.path = path.strip()
        self.filetime = spath2datetime(self.path)
        self.mainclass = None
        self.subclass = None
        self.agency = None
        self.last_action = None
        self.action_time = None
        self.analyst = None
        self.analyst_delay = None
        self.dsnwavfileobj = None
        self.asnwavfileobj = None
        self.aeffileobj = None
        self.eventobj = None

        if not os.path.exists(self.path):
            return

        if _is_sfile(self.path):
            self.eventobj = readheader(self.path)
            self.mainclass = self.eventobj.event_descriptions[-1].get('text').strip()
            self.agency=self.eventobj.creation_info.get("agency_id")
            wavnames = readwavename(self.path)
            wavpath = os.path.dirname(self.path).replace("REA", "WAV")
            if not wavnames:
                wavpattern = filetime2wavpath(self.filetime, self.path, y2kfix=False)
                potentialwavfiles = glob(wavpattern.split('.')[0]+".*")
                if len(potentialwavfiles)==1:
                    wavnames = [os.path.basename(potentialwavfiles[0])]
                
            for wavname in wavnames:
                wavpath = os.path.join(os.path.dirname(self.path).replace("REA", "WAV"), wavname)
                wnu = wavname.upper()
                if 'MVO' in wnu:
                    self.dsnwavfileobj = Wavfile(wavpath)
                elif 'ASN' in wnu or 'SPN' in wnu:
                    self.asnwavfileobj = Wavfile(wavpath)
            # pick up extra information        
            if use_mvo_parser:
                self.parse_sfile(verbose=verbose, parse_aef=parse_aef, try_external_aeffile=try_external_aeffile)

    def parse_sfile(self, verbose=False, parse_aef=True, try_external_aeffile=False):
        if verbose:
            print('Parsing ', self.path)
        fptr = open(self.path, 'r')
        lines = fptr.readlines()
        linetags = _get_line_tags(fptr, report=True)
        #for k,v in linetags.items():
        #    print(k, v)
        fptr.close()
        picklines = []
        picklineheader = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO SNR AR TRES W  DIS CAZ'

        for line in lines:
            if len(line.strip())==0:
                continue
            if 'STAT SP IPHASEW' in line:
                picklineheader = line
            elif line.strip()[0]=='M':
                picklines.append(line)                
            line = line.strip()    
            if not line:
                continue
            if line == 'STAT CMP   MAX AVG  TOTAL ENG             FREQUENCY BINS (Hz)       MAX  3':
                # This means there is AEF data in the S-file
                if parse_aef:
                    self.aeffileobj = AEFfile(self.path)
                    try_external_aeffile = False # since we already got AEF data from the S-file
            elif line[-1] == 'I':
                # e.g. 
                # ACTION:REG 07-03-01 09:23 OP:vab  STATUS:               ID:20070301042934     I
                # 0123456789012345678901234567890123456789012345678901234567890123456789012345678
                # 0         1         2         3         4         5         6         7
                yy = line[11:13]
                if yy > '90':
                    yyyy = int('19' + yy)
                else:
                    yyyy = int('20' + yy)
                self.action_time = UTCDateTime(yyyy, int(line[14:16]), int(line[17:19]), int(line[20:22]), int(line[23:25]) )
                self.filetime = UTCDateTime(line[59:73])
                self.last_action = line[7:10].strip()
                self.analyst = line[29:33].strip()
                if self.last_action:
                    analyst_delay = self.action_time - self.filetime
                    if analyst_delay < 7 * 86400:
                        self.analyst_delay = analyst_delay
            elif line[-1] == '3':
                if line.startswith('VOLC MAIN'):
                    self.subclass = line[10]
                
        if self.dsnwavfileobj and try_external_aeffile:
            aeffullpath = self.dsnwavfileobj.path.replace('WAV', 'AEF')
            if os.path.exists(aeffullpath):
                self.aeffileobj = AEFfile(aeffullpath)    

        if picklines:
            #_read_picks(linetags, self.event, nordic_format='UKN')
            _read_picks_nordic_old(picklines, self.eventobj, picklineheader, self.eventobj.origins[0].time)
            #_read_picks_nordic_new(picklines, self.event, picklineheader, self.event.origins[0].time)
            # For 1997 data, all picks are marked as "S[ZNE]" even for broadband stations. Do we want to check against the WAV file?
            if self.dsnwavfileobj:
                if os.path.isfile(self.dsnwavfileobj.path):
                    st = read(self.dsnwavfileobj.path)
                    for tr in st:
                        correct_nslc_mvo(tr.id, tr.stats.sampling_rate)
                    ids = [tr.id for tr in st]
                    
                    for p in self.eventobj.picks:
                        sta = p.waveform_id.station_code
                        cha = p.waveform_id.channel_code
                        if not cha:
                            cha = 'SHZ'
                        for id in ids:
                            tnet, tsta, tloc, tchan = id.split('.')
                            if sta == tsta:
                                if cha[-1] == tchan[-1]:
                                    p.waveform_id = WaveformStreamID(tnet, tsta, tloc, tchan)


        # correct short Trace IDs given in phase pick lines
        self.correct_obspy_event_ids()
            

    def correct_obspy_event_ids(self):
        # Correct NSLC codes for picks and arrivals
        if self.eventobj and hasattr(self.eventobj, "picks"):
            for pick in self.eventobj.picks:
                original_id = pick.waveform_id.get_seed_string()
                if pick.time < UTCDateTime(2005,1,1):
                    Fs = 75.0
                else:
                    Fs = 100.0
                corrected_id = correct_nslc_mvo(original_id, Fs)
                pick.waveform_id.station_code = corrected_id.split('.')[1]
                pick.waveform_id.channel_code = corrected_id.split('.')[3]
                pick.waveform_id.network_code = corrected_id.split('.')[0]
                pick.waveform_id.location_code = corrected_id.split('.')[2]

        if self.eventobj and hasattr(self.eventobj, "origins"):
            for origin in self.eventobj.origins:
                if hasattr(origin, "arrivals"):
                    for arrival in origin.arrivals:
                        if hasattr(arrival, "pick_id"):
                            try:
                                # Locate the corresponding pick
                                pick = next(p for p in self.eventobj.picks if p.resource_id == arrival.pick_id)
                                if pick.time < UTCDateTime(2005,1,1):
                                    Fs = 75.0
                                else:
                                    Fs = 100.0
                                corrected_id = correct_nslc_mvo(pick.waveform_id.get_seed_string(), Fs)
                                pick.waveform_id.station_code = corrected_id.split('.')[1]
                                pick.waveform_id.channel_code = corrected_id.split('.')[3]
                                pick.waveform_id.network_code = corrected_id.split('.')[0]
                                pick.waveform_id.location_code = corrected_id.split('.')[2]
                            except StopIteration:
                                print(f"[WARN] Arrival pick_id not found: {arrival.pick_id}")


    def to_dict(self):
        """
        Return a dictionary of all public attributes of the Sfile object.
        Nested objects like WAV and AEF files are handled appropriately.
        """
        sdict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # skip private attributes
            else:
                try:
                    json.dumps(value)  # check if it's JSON-serializable
                    sdict[key] = value
                except (TypeError, ValueError):
                    sdict[key] = str(value)  # fallback to string representation
        return sdict

    
    def __str__(self):
        """
        Return a pretty-printed string summary of the Sfile object, using to_dict().
        Automatically includes all public attributes.
        """
        from pprint import pformat

        try:
            summary = f"Sfile object: {self.path}\n"
        except AttributeError:
            summary = "Sfile object\n"

        try:
            summary += pformat(self.to_dict(), indent=4)
        except Exception as e:
            summary += f"[ERROR] Could not generate summary: {e}"

        return summary

        
    def to_css(self):
        """ Write CSS3.0 database event from an Sfile
        This is potentially less lossy than via an ObsPy Catalog object first """
        print('not implemented')
        pass

    def to_csv(self, csvfile):
        sdict = self.to_dict()
        sdict.to_csv(csvfile)
    
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
        evobj = self.eventobj
        print(evobj)
        print(evobj.event_descriptions)
        for i, originobj in enumerate(evobj.origins):
            print(i, originobj)

    def to_enhancedevent(self, stream=None):
        """
        Convert Sfile object to an EnhancedEvent object, ready to go into an EnhancedCatalog.
        The metrics dict can contain any attributes of an Sfile object that do not belong in
        an Obspy Event 

        Parameters
        ----------
        stream : obspy.Stream or EnhancedStream, optional
            Associated waveform data.

        Returns
        -------
        EnhancedEvent
        """
        if self.eventobj:
            eventobj = self.eventobj
        else:
            raise ValueError("No ObsPy Event object found in Sfile.")

        wav_paths = [w.path for w in (self.dsnwavfileobj, self.asnwavfileobj) if hasattr(w, "path")]
        aef_path = self.aeffileobj.path if self.aeffileobj else None
        trigger_window = getattr(self.aeffileobj, "trigger_window", None) if self.aeffileobj else None
        average_window = getattr(self.aeffileobj, "average_window", None) if self.aeffileobj else None

        # Basic feature/metric dictionary from known Sfile fields
        metrics = {
            "filetime": self.filetime,
            "mainclass": self.mainclass,
            "subclass": self.subclass,
            "analyst": self.analyst,
            "analyst_delay": self.analyst_delay,
            "aefrows": getattr(self.aeffileobj, "aefrows", None) if self.aeffileobj else None
        }
        return EnhancedEvent(
            obspy_event=eventobj,
            metrics=metrics,
            sfile_path=self.path,
            wav_paths=wav_paths,
            aef_path=aef_path,
            trigger_window=trigger_window,
            average_window=average_window,
            stream=stream
        )

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


"""
Deprecated code

Attributes:
        self.otime = None

        self.latitude = None
        self.longitude = None
        self.depth = None
        self.z_indicator = None

        self.no_sta = None
        self.rms = None
        self.magnitude = []
        self.magnitude_type = []
        self.magnitude_agency = []

        self.id = None
        self.url = None
        self.gap = None
        self.maximum_intensity = None
        self.picks = []

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



def read_pickle(picklefile):
    import pickle
    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        self = pickle.load(f)
    return self

    def _deprecated_code(self):       
        for line in lines:                  
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

            if line[-1] == '1':
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

                print(line)
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

            if line[-1] == '2':
                self.maximum_intensity = int(line[27:29])

            if line[-1] == '6':
                wavfiles = line[1:79].split()
                for wavfile in wavfiles:
                    wavfullpath = os.path.join(os.path.dirname(self.path).replace('REA', 'WAV'), wavfile)
                    self.wavfiles.append(Wavfile(wavfullpath))
                    aeffullpath = wavfullpath.replace('WAV', 'AEF')
                    if os.path.exists(aeffullpath) and aeffullpath not in _aeffiles:
                        _aeffiles.append(aeffullpath)

            if line[-1] == 'E':
                self.gap = parse_string(line, 5, 8, 'int', default=-1)
                self.error['origintime'] = parse_string(line, 14, 20, 'float')
                self.error['latitude'] = parse_string(line, 24, 30, 'float')
                self.error['longitude'] = parse_string(line, 32, 38, 'float')
                self.error['depth'] = parse_string(line, 38, 43, 'float')
                self.error['covxy'] = parse_string(line, 43, 55, 'float')
                self.error['covxz'] = parse_string(line, 55, 67, 'float')
                self.error['covyz'] = parse_string(line, 67, 79, 'float')

            if line[-1] == 'F' and 'dip' not in self.focmec:
                self.focmec['strike'] = float(line[0:10])
                self.focmec['dip'] = float(line[10:20])
                self.focmec['rake'] = float(line[20:30])
                self.focmec['agency'] = line[66:69]
                self.focmec['source'] = line[70:77]
                self.focmec['quality'] = line[77]

            if line[-1] == 'H':
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

            if line[-1] == 'I':

                self.last_action = line[8:11]
                self.action_time = line[12:26]
                self.analyst = line[30:33]
                self.id = int(line[60:74])

            if line[-1] == ' ' and line[1] == 'M':
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

                thispick = {
                    'sta': asta,
                    'chan': achan,
                    'phase': aphase,
                    'time': atime
                }
                if line[64:79].strip():
                    thispick['time_residual'] = parse_string(line, 64, 68, 'float')
                    thispick['weight'] = parse_string(line, 68, 70, 'int')
                    thispick['distance_km'] = parse_string(line, 72, 75, 'float')
                    thispick['azimuth'] = parse_string(line, 77, 79, 'int')

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
                    if 'time_residual' in thispick:
                        tres = QuantityError(uncertainty=thispick['time_residual'])
                        p = Pick(time=atime, waveform_id=fixedID, onset=onset, phase_hint=aphase,
                                time_errors=tres, backazimuth=(180 + thispick['azimuth']) % 360)
                    else:
                        p = Pick(time=atime, waveform_id=fixedID, onset=onset, phase_hint=aphase)
                    self.picks.append(p)
                
            
        if _aeffile in _aeffiles:
            if parse_aef:
                self.aeffiles.append(AEFfile(_aeffile))
            

    def maximum_magnitude(self):
        mag, mtype, agency = None, None, None
        for i, m in enumerate(self.magnitude):
            if mag is None or (m > mag and 'MVO' in self.magnitude_agency[i]):
                mag = m
                mtype = self.magnitude_type[i]
                agency = self.magnitude_agency[i]
        return mag, mtype, agency
    
    def to_obspyevent(self):
        if self.otime and self.latitude and self.longitude and self.depth is not None:
            origin = Origin(time=self.otime, latitude=self.latitude, longitude=self.longitude, depth=self.depth)
        else:
            origin = Origin(time=self.otime)

        if self.magnitude:
            magnitude = Magnitude(mag=self.magnitude[0], magnitude_type=self.magnitude_type[0] if self.magnitude_type else None)
        else:
            magnitude = None

        self.parsed_event = Event(origins=[origin])
        if magnitude:
            self.parsed_event.magnitudes.append(magnitude)
        if self.picks:
            self.parsed_event.picks = self.picks

"""