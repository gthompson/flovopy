from __future__ import annotations

import os
from typing import Optional

from obspy import UTCDateTime
from obspy.core.event import WaveformStreamID
from obspy.io.nordic.core import _read_picks_nordic_old
from obspy.io.nordic.utils import _get_line_tags

from flovopy.enhanced.event import EnhancedEvent
from flovopy.research.mvo.aeffile import AEFfile
from flovopy.seisanio.core.sfile import Sfile
from flovopy.seisanio.core.wavfile import Wavfile


class MVOSfile(Sfile):
    """
    Montserrat-specific extension of a generic Seisan S-file.

    Responsibilities
    ----------------
    - classify referenced waveform files into DSN / ASN groups
    - parse MVO-specific metadata from type-3 and type-I lines
    - parse embedded or external AEF data
    - align pick waveform IDs to actual waveform trace IDs

    Notes
    -----
    This class assumes Montserrat-specific conventions such as:
    - `VOLC MAIN` lines in type-3 S-file records
    - external `.AEF` files associated with digital waveform files
    - DSN / ASN naming conventions in waveform filenames
    """

    def __init__(
        self,
        path,
        verbose: bool = False,
        parse_aef: bool = True,
        try_external_aeffile: bool = False,
    ):
        self.mainclass: Optional[str] = None
        self.subclass: Optional[str] = None

        self.dsnwavfileobj: Optional[Wavfile] = None
        self.asnwavfileobj: Optional[Wavfile] = None
        self.aeffileobj: Optional[AEFfile] = None

        super().__init__(path, verbose=verbose)

        if not self.path or self.eventobj is None:
            return

        self._classify_wavfiles()
        self.parse_mvo_sfile(
            verbose=verbose,
            parse_aef=parse_aef,
            try_external_aeffile=try_external_aeffile,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, verbose: bool = False) -> None:
        if verbose:
            print(msg)

    def _classify_wavfiles(self) -> None:
        """
        Classify generic waveform references into MVO DSN / ASN buckets.

        Any referenced waveform file is already present in ``self.wavfileobjs``.
        This method simply identifies the best-known DSN and ASN representatives.
        """
        for wobj in self.wavfileobjs:
            if not hasattr(wobj, "path"):
                continue

            wavname_upper = os.path.basename(wobj.path).upper()

            if "MVO" in wavname_upper:
                self.dsnwavfileobj = wobj
            elif "ASN" in wavname_upper or "SPN" in wavname_upper:
                self.asnwavfileobj = wobj

    def _parse_type_i_line(self, line: str, verbose: bool = False) -> None:
        """
        Parse analyst / action metadata from an S-file type-I line.
        """
        try:
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

        except Exception as e:
            self._log(f"[WARN] Failed to parse type-I line in {self.path}: {e}", verbose=verbose)

    def _parse_type_3_line(self, line: str, verbose: bool = False) -> None:
        """
        Parse MVO-specific type-3 lines.
        """
        if line.startswith("VOLC MAIN"):
            try:
                # Historical convention from prior code: subclass at index 10
                self.subclass = line[10]
            except Exception as e:
                self._log(f"[WARN] Failed to parse VOLC MAIN subclass in {self.path}: {e}", verbose=verbose)

    def _maybe_attach_external_aef(self, verbose: bool = False) -> None:
        """
        Attach external AEF file if a digital waveform file exists and the
        derived AEF path is present.
        """
        if not self.dsnwavfileobj or not getattr(self.dsnwavfileobj, "path", None):
            return

        aef_full_path = self.dsnwavfileobj.path.replace("WAV", "AEF")
        if os.path.exists(aef_full_path):
            self.aeffileobj = AEFfile(aef_full_path, verbose=verbose)

    # ------------------------------------------------------------------
    # MVO-specific parsing
    # ------------------------------------------------------------------

    def parse_mvo_sfile(
        self,
        verbose: bool = False,
        parse_aef: bool = True,
        try_external_aeffile: bool = False,
    ) -> None:
        """
        Parse additional MVO-specific metadata and picks from the S-file.

        This includes:
        - analyst/action metadata from type-I lines
        - MVO subclass from type-3 lines
        - embedded AEF blocks if present
        - old-format Nordic picks
        - optional external `.AEF` fallback
        - alignment of pick waveform IDs to referenced waveform files
        """
        if verbose:
            print(f"Parsing {self.path}")

        with open(self.path, "r", encoding="utf-8", errors="ignore") as fptr:
            lines = fptr.readlines()

        # Useful for debugging malformed line tags; side effects are internal to ObsPy.
        with open(self.path, "r", encoding="utf-8", errors="ignore") as fptr:
            _ = _get_line_tags(fptr, report=verbose)

        picklines = []
        picklineheader = (
            " STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO "
            "SNR AR TRES W  DIS CAZ"
        )

        embedded_aef_detected = False

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

            # Embedded AEF block marker
            if line == "STAT CMP   MAX AVG  TOTAL ENG             FREQUENCY BINS (Hz)       MAX  3":
                embedded_aef_detected = True
                if parse_aef:
                    self.aeffileobj = AEFfile(self.path, verbose=verbose)
                    try_external_aeffile = False
                continue

            # Type-I metadata
            if line[-1] == "I":
                self._parse_type_i_line(line, verbose=verbose)
                continue

            # Type-3 MVO-specific metadata
            if line[-1] == "3":
                self._parse_type_3_line(line, verbose=verbose)
                continue

        # Parse picks into ObsPy event object
        if picklines and self.eventobj and self.eventobj.origins:
            _read_picks_nordic_old(
                picklines,
                self.eventobj,
                picklineheader,
                self.eventobj.origins[0].time,
            )

        # Optional external AEF fallback
        if try_external_aeffile and not embedded_aef_detected and self.aeffileobj is None:
            self._maybe_attach_external_aef(verbose=verbose)

        # Align pick IDs to actual waveform IDs, if possible
        self.align_pick_ids_to_waveforms(verbose=verbose)

    # ------------------------------------------------------------------
    # Waveform-aware pick ID alignment
    # ------------------------------------------------------------------

    def align_pick_ids_to_waveforms(self, verbose: bool = False, netcode: str = "MV") -> None:
        """
        Align pick WaveformStreamIDs to the IDs present in associated waveform files.

        This is especially useful for older Seisan picks where station codes are
        present but channel/location information is incomplete or approximate.

        Parameters
        ----------
        verbose
            If True, print diagnostic output.
        netcode
            Network code passed through to Wavfile.read(fixid=True, netcode=...).
        """
        if not self.eventobj or not getattr(self.eventobj, "picks", None):
            return

        wav_objs = [w for w in (self.dsnwavfileobj, self.asnwavfileobj) if w is not None]

        available_ids: list[str] = []
        for wobj in wav_objs:
            ok = wobj.read(fixid=True, verbose=verbose, netcode=netcode)
            if ok and getattr(wobj, "st", None) is not None:
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
    # Convenience methods / export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a dictionary of all public attributes of the MVOSfile object.

        Nested WAV/AEF objects are represented compactly by path or selected IDs.
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

            elif key == "wavfileobjs":
                sdict[key] = [w.path for w in value if hasattr(w, "path")]

            else:
                try:
                    import json
                    json.dumps(value)
                    sdict[key] = value
                except (TypeError, ValueError):
                    sdict[key] = str(value)

        return sdict

    def to_enhancedevent(self, stream=None) -> EnhancedEvent:
        """
        Convert this MVOSfile object to an EnhancedEvent object.
        """
        if self.eventobj is None:
            raise ValueError("No ObsPy Event object found in MVOSfile.")

        wav_paths = [w.path for w in (self.dsnwavfileobj, self.asnwavfileobj) if hasattr(w, "path")]
        aef_path = self.aeffileobj.path if self.aeffileobj else None
        trigger_window = self.aeffileobj.trigger_window if self.aeffileobj else None
        average_window = self.aeffileobj.average_window if self.aeffileobj else None

        metrics = {
            "filetime": self.filetime,
            "mainclass": self.mainclass,
            "subclass": self.subclass,
            "analyst": self.analyst,
            "analyst_delay": self.analyst_delay,
            "aefrows": self.aeffileobj.aefrows if self.aeffileobj else None,
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

