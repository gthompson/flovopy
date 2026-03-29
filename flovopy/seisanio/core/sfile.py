from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Optional

from obspy import UTCDateTime
from obspy.core.event import WaveformStreamID
from obspy.io.nordic.core import (
    _is_sfile,
    _read_picks_nordic_old,
    readheader,
    readwavename,
)
from obspy.io.nordic.utils import _get_line_tags

from flovopy.enhanced.event import EnhancedEvent
from flovopy.seisanio.core.aeffile import AEFfile
from flovopy.seisanio.core.wavfile import Wavfile
from flovopy.seisanio.utils.helpers import filetime2wavpath, spath2datetime


class Sfile:
    """
    Wrapper around a Seisan S-file.

    Responsibilities
    ----------------
    - locate associated waveform / AEF files
    - parse Nordic event header into ObsPy event object
    - optionally parse MVO-specific extra metadata
    - optionally align pick waveform IDs to associated waveform files
    """

    def __init__(
        self,
        path,
        use_mvo_parser: bool = False,
        verbose: bool = False,
        parse_aef: bool = True,
        try_external_aeffile: bool = False,
    ):
        self.path = str(path).strip()
        self.filetime: Optional[UTCDateTime] = None
        self.mainclass = None
        self.subclass = None
        self.agency = None
        self.last_action = None
        self.action_time = None
        self.analyst = None
        self.analyst_delay = None

        self.dsnwavfileobj: Optional[Wavfile] = None
        self.asnwavfileobj: Optional[Wavfile] = None
        self.aeffileobj: Optional[AEFfile] = None
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

        self._read_basic()
        if use_mvo_parser:
            self.parse_sfile(
                verbose=verbose,
                parse_aef=parse_aef,
                try_external_aeffile=try_external_aeffile,
            )

    # ------------------------------------------------------------------
    # Basic parsing
    # ------------------------------------------------------------------

    def _read_basic(self):
        """
        Read the Nordic event header and associated waveform references.
        """
        self.eventobj = readheader(self.path)

        if self.eventobj.event_descriptions:
            try:
                self.mainclass = self.eventobj.event_descriptions[-1].get("text").strip()
            except Exception:
                self.mainclass = None

        try:
            self.agency = self.eventobj.creation_info.get("agency_id")
        except Exception:
            self.agency = None

        wavnames = readwavename(self.path)
        wavdir = os.path.dirname(self.path).replace("REA", "WAV")

        # Fallback: infer waveform file from event time if not explicitly listed
        if not wavnames and self.filetime is not None:
            wavpattern = filetime2wavpath(self.filetime, self.path, y2kfix=False)
            potential = glob(wavpattern.split(".")[0] + ".*")
            if len(potential) == 1:
                wavnames = [os.path.basename(potential[0])]

        for wavname in wavnames:
            wavpath = os.path.join(wavdir, wavname)
            wnu = wavname.upper()

            if "MVO" in wnu:
                self.dsnwavfileobj = Wavfile(wavpath)
            elif "ASN" in wnu or "SPN" in wnu:
                self.asnwavfileobj = Wavfile(wavpath)

    # ------------------------------------------------------------------
    # Extended / MVO-style parsing
    # ------------------------------------------------------------------

    def parse_sfile(self, verbose=False, parse_aef=True, try_external_aeffile=False):
        """
        Parse additional metadata and pick information from the S-file.

        This includes:
        - analyst/action metadata from type-I lines
        - MVO subclass from type-3 lines
        - embedded AEF blocks if present
        - Nordic old-format picks
        - optional alignment of pick waveform IDs to referenced waveform files
        """
        if verbose:
            print(f"Parsing {self.path}")

        with open(self.path, "r") as fptr:
            lines = fptr.readlines()

        with open(self.path, "r") as fptr:
            _ = _get_line_tags(fptr, report=verbose)

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

            # Embedded AEF marker
            if line == "STAT CMP   MAX AVG  TOTAL ENG             FREQUENCY BINS (Hz)       MAX  3":
                if parse_aef:
                    self.aeffileobj = AEFfile(self.path)
                    try_external_aeffile = False

            # Action / analyst metadata (type I)
            elif line[-1] == "I":
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

            # Type 3 lines
            elif line[-1] == "3":
                if line.startswith("VOLC MAIN"):
                    self.subclass = line[10]

        # External AEF fallback
        if self.dsnwavfileobj and try_external_aeffile:
            aeffullpath = self.dsnwavfileobj.path.replace("WAV", "AEF")
            if os.path.exists(aeffullpath):
                self.aeffileobj = AEFfile(aeffullpath)

        # Parse picks into ObsPy event object
        if picklines and self.eventobj and self.eventobj.origins:
            _read_picks_nordic_old(
                picklines,
                self.eventobj,
                picklineheader,
                self.eventobj.origins[0].time,
            )

        # After picks are parsed, try to align pick waveform IDs to the
        # associated waveform file(s) if available.
        self.align_pick_ids_to_waveforms(verbose=verbose)

    # ------------------------------------------------------------------
    # Waveform-aware pick ID alignment
    # ------------------------------------------------------------------

    def align_pick_ids_to_waveforms(self, verbose=False, netcode="MV"):
        """
        Align pick WaveformStreamIDs to the IDs present in the associated
        waveform file(s), if any.

        This is especially useful for legacy Seisan picks where only station
        and rough channel/orientation hints are present.
        """
        if not self.eventobj or not getattr(self.eventobj, "picks", None):
            return

        wav_objs = [w for w in (self.dsnwavfileobj, self.asnwavfileobj) if w is not None]

        available_ids = []
        for wobj in wav_objs:
            ok = wobj.read(fixid=True, verbose=verbose, netcode=netcode)
            if ok and wobj.st is not None:
                available_ids.extend([tr.id for tr in wobj.st])

        if not available_ids:
            return

        parsed_ids = [seed_id.split(".") for seed_id in available_ids]

        for pick in self.eventobj.picks:
            wf = pick.waveform_id
            if wf is None:
                continue

            pick_sta = (wf.station_code or "").upper()
            pick_cha = (wf.channel_code or "").upper()

            # Try to match on station and final component/orientation if present
            target_component = pick_cha[-1] if pick_cha else None

            best = None
            for net, sta, loc, cha in parsed_ids:
                if sta != pick_sta:
                    continue

                if target_component is None or not cha:
                    best = (net, sta, loc, cha)
                    break

                if cha[-1] == target_component:
                    best = (net, sta, loc, cha)
                    break

            if best is not None:
                net, sta, loc, cha = best
                pick.waveform_id = WaveformStreamID(
                    network_code=net,
                    station_code=sta,
                    location_code=loc,
                    channel_code=cha,
                )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Return a dictionary of all public attributes of the Sfile object.
        Nested objects like WAV and AEF files are handled appropriately.
        """
        sdict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue

            if key.endswith("obj"):
                if key == "aeffileobj":
                    if hasattr(value, "aef_ids"):
                        sdict[key] = getattr(value, "aef_ids", None)
                    elif hasattr(value, "path"):
                        sdict[key] = getattr(value, "path")
                    else:
                        sdict[key] = None
                elif hasattr(value, "path"):
                    sdict[key] = getattr(value, "path")
                else:
                    sdict[key] = None
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
        """Write CSS3.0 database event from an Sfile."""
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
        """Show contents of an S-file."""
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

        wav_paths = [w.path for w in (self.dsnwavfileobj, self.asnwavfileobj) if hasattr(w, "path")]
        aef_path = self.aeffileobj.path if self.aeffileobj else None
        trigger_window = getattr(self.aeffileobj, "trigger_window", None) if self.aeffileobj else None
        average_window = getattr(self.aeffileobj, "average_window", None) if self.aeffileobj else None

        metrics = {
            "filetime": self.filetime,
            "mainclass": self.mainclass,
            "subclass": self.subclass,
            "analyst": self.analyst,
            "analyst_delay": self.analyst_delay,
            "aefrows": getattr(self.aeffileobj, "aefrows", None) if self.aeffileobj else None,
        }

        return EnhancedEvent(
            obspy_event=self.eventobj,
            metrics=metrics,
            sfile_path=self.path,
            wav_paths=wav_paths,
            aef_path=aef_path,
            trigger_window=trigger_window,
            average_window=average_window,
            stream=stream,
        )


