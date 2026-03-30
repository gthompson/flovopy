from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Optional

from obspy import UTCDateTime
from obspy.io.nordic.core import (
    _is_sfile,
    _read_picks_nordic_old,
    readheader,
    readwavename,
)

from flovopy.enhanced.event import EnhancedEvent
from flovopy.seisanio.core.wavfile import Wavfile
from flovopy.seisanio.utils.helpers import filetime2wavpath, spath2datetime


class Sfile:
    """
    Generic wrapper around a Seisan S-file.

    Responsibilities
    ----------------
    - locate associated waveform files
    - parse Nordic event header into an ObsPy Event object
    - parse standard Nordic pick lines
    - provide convenience serialization/export helpers
    """

    def __init__(self, path, verbose: bool = False):
        self.path = str(path).strip()
        self.filetime: Optional[UTCDateTime] = None

        self.agency = None
        self.last_action = None
        self.action_time = None
        self.analyst = None
        self.analyst_delay = None

        self.wavfileobjs: list[Wavfile] = []
        self.eventobj = None

        if not self.path:
            return

        try:
            self.filetime = spath2datetime(self.path)
        except Exception:
            self.filetime = None

        if not os.path.exists(self.path):
            return

        if not _is_sfile(self.path):
            return

        self._read_basic(verbose=verbose)

    # ------------------------------------------------------------------
    # Basic parsing
    # ------------------------------------------------------------------

    def _read_basic(self, verbose: bool = False):
        """
        Read Nordic header and associated waveform references.
        """
        self.eventobj = readheader(self.path)

        try:
            self.agency = self.eventobj.creation_info.get("agency_id")
        except Exception:
            self.agency = None

        wavnames = readwavename(self.path)
        wavdir = os.path.dirname(self.path).replace("REA", "WAV")

        if not wavnames and self.filetime is not None:
            wavpattern = filetime2wavpath(self.filetime, self.path, y2kfix=False)
            potential = glob(wavpattern.split(".")[0] + ".*")
            if len(potential) == 1:
                wavnames = [os.path.basename(potential[0])]

        for wavname in wavnames:
            wavpath = os.path.join(wavdir, wavname)
            self.wavfileobjs.append(Wavfile(wavpath))

    # ------------------------------------------------------------------
    # Extended generic parsing
    # ------------------------------------------------------------------

    def parse_sfile(self, verbose: bool = False):
        """
        Parse additional generic metadata and standard Nordic old-format picks.
        """
        if verbose:
            print(f"Parsing {self.path}")

        with open(self.path, "r") as fptr:
            lines = fptr.readlines()

        picklines = []
        picklineheader = (
            " STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO "
            "SNR AR TRES W  DIS CAZ"
        )

        for rawline in lines:
            if not rawline.strip():
                continue

            if "STAT SP IPHASEW" in rawline:
                picklineheader = rawline

            elif rawline.strip()[0] == "M":
                picklines.append(rawline)

            line = rawline.strip()
            if not line:
                continue

            if line[-1] == "I":
                yy = line[11:13]
                yyyy = int("19" + yy) if yy > "90" else int("20" + yy)

                self.action_time = UTCDateTime(
                    yyyy,
                    int(line[14:16]),
                    int(line[17:19]),
                    int(line[20:22]),
                    int(line[23:25]),
                )
                self.filetime = UTCDateTime(line[59:73])
                self.last_action = line[7:10].strip()
                self.analyst = line[29:33].strip()

                if self.last_action and self.filetime:
                    analyst_delay = self.action_time - self.filetime
                    if analyst_delay < 7 * 86400:
                        self.analyst_delay = analyst_delay

        if picklines and self.eventobj and self.eventobj.origins:
            _read_picks_nordic_old(
                picklines,
                self.eventobj,
                picklineheader,
                self.eventobj.origins[0].time,
            )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Return a dictionary of all public attributes of the Sfile object.
        """
        sdict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue

            if key == "wavfileobjs":
                sdict[key] = [w.path for w in value if hasattr(w, "path")]
            else:
                try:
                    json.dumps(value)
                    sdict[key] = value
                except (TypeError, ValueError):
                    sdict[key] = str(value)
        return sdict

    def __str__(self):
        from pprint import pformat

        summary = f"Sfile object: {self.path}\n"
        try:
            summary += pformat(self.to_dict(), indent=4)
        except Exception as e:
            summary += f"[ERROR] Could not generate summary: {e}"
        return summary

    def to_css(self):
        raise NotImplementedError("to_css() is not implemented yet.")

    def to_csv(self, csvfile):
        import pandas as pd
        pd.DataFrame([self.to_dict()]).to_csv(csvfile, index=False)

    def to_pickle(self):
        import pickle

        picklefile = self.path.replace("REA", "PICKLE") + ".pickle"
        os.makedirs(os.path.dirname(picklefile), exist_ok=True)
        with open(picklefile, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def cat(self):
        with open(self.path, "r") as fptr:
            print(fptr.read())

    def printEvents(self):
        evobj = self.eventobj
        print(evobj)
        print(evobj.event_descriptions)
        for i, originobj in enumerate(evobj.origins):
            print(i, originobj)

    def to_enhancedevent(self, stream=None):
        """
        Convert Sfile object to an EnhancedEvent object.
        """
        if self.eventobj is None:
            raise ValueError("No ObsPy Event object found in Sfile.")

        wav_paths = [w.path for w in self.wavfileobjs if hasattr(w, "path")]

        metrics = {
            "filetime": self.filetime,
            "analyst": self.analyst,
            "analyst_delay": self.analyst_delay,
        }

        return EnhancedEvent(
            obspy_event=self.eventobj,
            metrics=metrics,
            sfile_path=self.path,
            wav_paths=wav_paths,
            stream=stream,
        )