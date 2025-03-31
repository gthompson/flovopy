import os
import datetime as dt
from obspy import read
from flovopy.seisanio.utils.helpers import filetime2spath, correct_nslc


class Wavfile:
    def __init__(self, path=''):
        self.path = path.strip()
        self.filetime = None
        self.st = None

        try:
            wavbase = os.path.basename(self.path)
            parts = wavbase.split('.')[0].split('-')

            if len(parts) == 5:
                yyyy, mm = parts[0], parts[1]
            elif len(parts) == 4:
                yy = parts[0][0:2]
                yyyy = '19' + yy if yy.startswith('9') else '20' + yy
                mm = parts[0][2:4]

            dd, HHMI, SS = parts[-3], parts[-2], parts[-1][0:2]
            HH, MI = HHMI[0:2], HHMI[2:4]

            self.filetime = dt.datetime(
                int(yyyy), int(mm), int(dd),
                hour=int(HH), minute=int(MI), second=int(SS)
            )
        except Exception as e:
            print(f"[Wavfile] Failed to parse filetime from: {path} -- {e}")

    def find_sfile(self, mainclass='L'):
        """
        Try to find the corresponding S-file for this WAV file.
        """
        db = os.path.dirname(self.path).split('/')[-3]
        seisan_data = self.path.split('/WAV')[0]
        sfile = filetime2spath(self.filetime, mainclass=mainclass, db=db,
                               seisan_data=seisan_data, fullpath=True)

        return sfile, os.path.exists(sfile)

    def register(self, evtype, userid, overwrite=False, evtime=None):
        from obspy.io.nordic.core import blanksfile  # Delayed import
        if not evtime:
            evtime = self.filetime
        return blanksfile(self.path, evtype, userid, overwrite=overwrite, evtime=evtime)

    def read(self):
        if os.path.exists(self.path):
            try:
                self.st = read(self.path)
                return True
            except Exception as e:
                print(f"[Wavfile.read] Failed to read {self.path}: {e}")
                self.st = None
        else:
            print(f"[Wavfile.read] File does not exist: {self.path}")
        return False

    def plot(self, equal_scale=False):
        if self.st:
            self.st.plot(equal_scale=equal_scale)
            return True
        return False

    def __str__(self):
        out = f"wavfile: {self.path}"
        if self.st:
            out += "\n\t" + str(self.st)
        return out

