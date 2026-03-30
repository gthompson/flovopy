import os
from obspy import read, UTCDateTime

from flovopy.seisanio.utils.helpers import filetime2spath, legacy_or_not
from flovopy.core.trace_utils import fix_trace_id
from flovopy.research.mvo.mvo_ids import fix_trace_mvo


class Wavfile:
    def __init__(self, path=""):
        self.path = path.strip()
        self.st = None
        self.start_time = None
        self.end_time = None
        self.filetime = None
        self.network = None
        self.legacy = False

        self.wavpath2datetime()

    def find_sfile(self, mainclass="L"):
        """
        Try to find the corresponding S-file for this WAV file.
        """
        db = os.path.dirname(self.path).split("/")[-3]
        seisan_top = self.path.split("/WAV")[0]
        sfile = filetime2spath(
            self.filetime,
            mainclass=mainclass,
            db=db,
            seisan_top=seisan_top,
            fullpath=True,
        )
        return sfile, os.path.exists(sfile)

    def register(self, evtype, userid, overwrite=False, evtime=None):
        from obspy.io.nordic.core import blanksfile  # delayed import

        if not evtime:
            evtime = self.filetime
        return blanksfile(self.path, evtype, userid, overwrite=overwrite, evtime=evtime)

    def read(self, fixid=True, verbose=False):
        if not os.path.exists(self.path):
            if verbose:
                print(f"[Wavfile.read] File does not exist: {self.path}")
            return False

        self.legacy, self.network = legacy_or_not(self.path)

        try:
            self.st = read(self.path)

            if fixid:
                for tr in self.st:
                    tr.stats.original_id = tr.id

                    if self.legacy:
                        fix_trace_id(tr, legacy=True, netcode="MV")

                    else:
                        fix_trace_mvo(tr, verbose=verbose)

            if len(self.st) > 0:
                self.start_time = min(tr.stats.starttime for tr in self.st)
                self.end_time = max(tr.stats.endtime for tr in self.st)

            return True

        except Exception as e:
            if verbose:
                print(f"[Wavfile.read] Failed to read {self.path}: {e}")
            self.st = None
            return False

    def plot(self, equal_scale=False):
        if self.st is not None and len(self.st) > 0:
            self.st.plot(equal_scale=equal_scale)
            return True
        return False

    def to_dict(self):
        """
        Return a dictionary of all public attributes of the Wavfile object.
        """
        import json

        wavdict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
                wavdict[key] = value
            except (TypeError, ValueError):
                wavdict[key] = str(value)
        return wavdict

    def __str__(self):
        """
        Return a pretty-printed string summary of the WAVfile object.
        """
        from pprint import pformat

        summary = f"WAVfile object: {self.path}\n"
        try:
            summary += pformat(self.to_dict(), indent=4)
        except Exception as e:
            summary += f"[ERROR] Could not generate summary: {e}"
        return summary

    def wavpath2datetime(self):
        self.filetime = wavpath2datetime(self.path)


def wavpath2datetime(wavpath):
    """
    Extract datetime from a SEISAN WAV-file path.
    """
    try:
        basename = os.path.basename(wavpath)
        if not basename:
            raise ValueError("Empty filename")

        if "S." not in basename:
            raise ValueError("Filename does not contain 'S.' marker")

        parts = basename.split("S.")[0].split("-")

        if len(parts) == 5:
            # 4-digit year
            yyyy, mm = parts[0], parts[1]

        elif len(parts) == 4:
            # 2-digit year packed into first token
            yy = parts[0][0:2]
            yyyy = "19" + yy if yy.startswith("9") else "20" + yy
            mm = parts[0][2:4]

        else:
            raise ValueError(f"Unexpected token structure: {parts}")

        dd = parts[-3]
        HHMI = parts[-2]
        SS = parts[-1][0:2]
        HH, MI = HHMI[0:2], HHMI[2:4]

        return UTCDateTime(int(yyyy), int(mm), int(dd), int(HH), int(MI), int(SS))

    except Exception as e:
        raise IOError(f"[Wavfile] Failed to parse filetime from: {os.path.basename(wavpath)}: {e}")