"""
AEFfile: Parser for SEISAN amplitude-energy-frequency (AEF) files.

AEF files contain summary metrics computed by the legacy `ampengfft` C program, which was
part of the standard analyst workflow at the Montserrat Volcano Observatory (MVO). Each 
AEF file corresponds to a SEISAN-format event waveform file and includes processed results
such as:

- Maximum average amplitude (in m/s)
- Total signal energy (in J/kg)
- Dominant frequency (in Hz)
- Percentage energy in predefined spectral bands ("SSAM")

The ampengfft tool computed these metrics using a preconfigured FFT and energy slicing algorithm.
Results were either:

1. Embedded directly in the corresponding event `.S` file (as lines starting with `VOLC`)
2. Or saved to a separate `.AEF` file with a filename derived from the event.

This parser reads both standalone AEF files and embedded `VOLC` lines from `.S` files,
returning a structured list of dictionaries, one per station-channel. Each entry includes:

- `station`, `channel`: Network/station/channel info
- `amplitude`: Maximum average amplitude (float)
- `energy`: Total energy (float)
- `maxf`: Frequency of peak spectral amplitude (Hz)
- `ssam`: Dictionary with:
    - `bands`: List of N+1 frequency edges (Hz)
    - `fractions`: List of N values, each representing the fraction of total energy in that band
    - `energies`: Optional per-band energy values, if present
"""


import os
import re
from obspy import read
from flovopy.core.trace_utils import correct_nslc_mvo
from flovopy.seisanio.core.wavfile import wavpath2datetime
from flovopy.seisanio.utils.helpers import legacy_or_not, find_matching_wavfiles, spath2datetime

class AEFfile:
    def __init__(self, path=None, filetime=None, verbose=False):
        self.aefrows = []  # List of trace-level rows (dicts)
        self.trigger_window = None
        self.average_window = None
        self.path = path.strip() if path else None
        if filetime:
            self.filetime = filetime
        self.legacy, self.network = legacy_or_not(self.path)
        if 'AEF' in self.path: # we are processing AEF data from a WAV file
            self.filetime = wavpath2datetime(self.path)
            self.wavpath = self.path.replace('AEF', 'WAV').replace('aef','')
        elif 'REA' in self.path: # we are processing AEF data from an S-File
            self.filetime = spath2datetime(self.path)
            matching_wavfiles = find_matching_wavfiles(self.filetime, self.path, y2kfix=False)
            if len(matching_wavfiles)>0:
                self.wavpath = matching_wavfiles[0]
        self.wav_ids = []
        self.aef_ids = []
        
        if os.path.isfile(self.wavpath):
            st = read(self.wavpath)
            for tr in st:
                tr.id = correct_nslc_mvo(tr.id, tr.stats.sampling_rate)
            self.wav_ids = [tr.id for tr in st]            
            del st

        if not os.path.exists(self.path):
            print(f"[WARN] {self.path} does not exist")
            return

        try:
            with open(self.path, 'r') as f:
                lines = f.readlines()            
        except IOError as e:
            print(f"[ERROR] Could not read {self.path}: {e}")
            return

        for line in lines:
            if verbose:
                print(line)
            if len(line) < 80: 
                continue
            line = line.strip()
            if not line:
                continue
            if line[-1]!='3':
                continue
            if 'trigger window' in line.lower():
                if verbose:
                    print('trigger window found')
                self.trigger_window = self._extract_window(line, 'trigger window')
                continue

            if 'average window' in line.lower():
                if verbose:
                    print('average window found')
                self.average_window = self._extract_window(line, 'average window')
                continue

            # Skip header lines
            line = line.lstrip()
            if "VOLC MAIN" in line:
                if verbose:
                    print('header line found')
                continue
            if not line.startswith("VOLC"):
                if verbose:
                    print('other line found')
                continue

            if verbose:
                print('Trying to parse AEF line')

            aefrow = self.parse_aefline(line)
            if aefrow:
                if verbose:
                    print(aefrow)
                self.aef_ids.append(aefrow['fixed_id'])
                self.aefrows.append(aefrow)


    def _extract_window(self, line, keyword):
        i_start = line.lower().find(keyword)
        if i_start > -1:
            substring = line[i_start:i_start + 30]
            eq_index = substring.find('=')
            s_index = substring.find('s')
            if eq_index != -1 and s_index != -1:
                valstr = substring[eq_index + 1:s_index].strip()
                try:
                    return float(valstr)
                except ValueError:
                    print(f"[WARN] Could not parse value for {keyword} in line: {line.strip()}")
        return None

    def to_dict(self, include_ssam=False):
        """
        Return a dictionary of all public attributes of the AEFfile object.
        """
        import json
        aefdict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # skip private attributes
            elif key == "ssam" and not include_ssam:
                continue
            else:
                try:
                    json.dumps(value)  # check if it's JSON-serializable
                    aefdict[key] = value
                except (TypeError, ValueError):
                    aefdict[key] = str(value)  # fallback to string representation
        return aefdict

    
    def __str__(self):
        """
        Return a pretty-printed string summary of the AEFfile object, using to_dict().
        Automatically includes all public attributes.
        """
        from pprint import pformat

        try:
            summary = f"AEFfile object: {self.path}\n"
        except AttributeError:
            summary = "AEFfile object\n"

        try:
            summary += pformat(self.to_dict(), indent=4)
        except Exception as e:
            summary += f"[ERROR] Could not generate summary: {e}"

        return summary

    def parse_aefline(self, line):
        line = line.split('VOLC ')[-1]
        try:
            station = line[0:4].strip()
            channel = line[5:9].strip()
            line = line[9:] # just in case A, E, or F are in the Station or Channel name

            a_idx = line.find("A")
            e_idx = line.find("E")
            f_idx = line.find("F")

            amplitude = float(line[a_idx + 1:e_idx].strip())
            energy = float(line[e_idx + 1:f_idx].strip())

            trace_id = f"MV.{station}..{channel}"
            # Determine analog/digital network based on station prefix
            if self.legacy:
                fixed_id = correct_nslc_mvo(trace_id, 100.0, shortperiod=True)
            else:
                sr = 75.0 if self.filetime and self.filetime.year <= 2004 else 100.0
                fixed_id = correct_nslc_mvo(trace_id, sr, shortperiod=None)

            if self.wav_ids: # we use ids from stream to check we have correct sampling rate and id
                fnet, fsta, floc, fchan = fixed_id.split('.')
                for id in self.wav_ids:
                    tnet, tsta, tloc, tchan = id.split('.')
                    if fsta == tsta:
                        if fchan[-1] == tchan[-1]:
                            fixed_id = id

            # Parse SSAM bins
            ssam = self.parse_F(line, energy, f_idx + 1)

            # Parse max frequency from the end of the line
            maxf = None
            match = re.search(r"(\d+\.\d+)\s*3?$", line)
            if match:
                maxf = float(match.group(1))

            return {
                'station': station,
                'channel': channel,
                'id': trace_id,
                'fixed_id': fixed_id,
                'amplitude': amplitude,
                'energy': energy,
                'ssam': ssam,
                'maxf': maxf
            }

        except Exception as e:
            print(f"[ERROR] Failed to parse AEF line:\n  {line.strip()}\n  {e}")
            return None

    def parse_F(self, line, energy, startindex):
        """
        Extracts SSAM frequency bin percentages and converts to energy.
        """
        F = {
            "frequency_bands": [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 30.0],
            "percentages": [],
            "energies": []
        }

        N = len(F['frequency_bands'])-1 # frequency bands defines bin edges. So N actual measurements.
        while startindex < len(line) and len(F["percentages"]) < N:
            valstr = line[startindex:startindex + 3].strip()
            if "." not in valstr:
                try:
                    val = int(valstr)
                    F["percentages"].append(val)
                    F["energies"].append(val / 100.0 * energy)
                except ValueError as e:
                    #raise e
                    break
            startindex += 3
        return F

    def to_dict(self):
        """
        Return a structured dictionary for export or database insertion.
        """
        return {
            "path": self.path,
            "trigger_window": self.trigger_window,
            "average_window": self.average_window,
            "trace_metrics": self.aefrows
        }
