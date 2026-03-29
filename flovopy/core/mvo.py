from __future__ import annotations

import struct
from pathlib import Path
from typing import Callable

import numpy as np
from obspy import Stream, Trace, read
from obspy.core.inventory import read_inventory
from obspy.core.utcdatetime import UTCDateTime

from flovopy.seisanio.archive import SeisanArchive
from flovopy.core.trace_utils import (
    remove_empty_traces,
    decompose_channel_code,
    fix_channel_code,
    fix_location_code,
)
from flovopy.seisanio.core.sfile import Sfile


# -------------------------------------------------------------------
# Montserrat constants
# -------------------------------------------------------------------

DOME_LOCATION = {"lat": 16.71060, "lon": -62.17747, "elev": 1000.0}

# (lon_min, lon_max, lat_min, lat_max)
REGION_DEFAULT = (-62.255, -62.135, 16.66, 16.84)


# -------------------------------------------------------------------
# Inventory helpers
# -------------------------------------------------------------------

def load_mvo_master_inventory(xmldir: str | Path):
    """
    Load the master StationXML file for the Montserrat digital seismic network.

    Parameters
    ----------
    xmldir
        Directory containing ``MontserratDigitalSeismicNetwork.xml``.

    Returns
    -------
    Inventory or None
        ObsPy Inventory if found, else None.
    """
    xmldir = Path(xmldir)
    master_station_xml = xmldir / "MontserratDigitalSeismicNetwork.xml"
    if master_station_xml.exists():
        print(f"Loading {master_station_xml}")
        return read_inventory(str(master_station_xml))

    print(f"Could not find {master_station_xml}")
    return None


# -------------------------------------------------------------------
# Generic small helpers
# -------------------------------------------------------------------

def change_last_sample(tr: Trace):
    """
    Remove the final sample from a trace.

    Useful for some legacy Montserrat SEISAN files where the last sample
    is known to be corrupt.
    """
    tr.data = tr.data[:-1]


def swap32(i: int) -> int:
    """
    Swap endianness of a 32-bit integer.
    """
    return struct.unpack("<i", struct.pack(">i", i))[0]


def infer_mvo_sample_rate_from_time(time):
    """
    Heuristic sample-rate choice for legacy MVO data.

    Before 2005:
        usually 75 Hz
    2005 onward:
        usually 100 Hz
    """
    if time is None:
        return 100.0
    return 75.0 if time < UTCDateTime(2005, 1, 1) else 100.0


def legacy_or_not(fname):
    """
    Heuristic classification for Montserrat waveform filenames.

    Parameters
    ----------
    fname
        Filename or path.

    Returns
    -------
    tuple[bool, str | None]
        (legacy, network_guess)

        network_guess is:
        - "ASN" for analog/legacy network files
        - "DSN" for digital MVO files
        - None if uncertain
    """
    fnamelower = Path(fname).name.lower()

    if "asne" in fnamelower or "spn" in fnamelower:
        return True, "ASN"

    if "mvo" in fnamelower or "dsn" in fnamelower:
        return False, "DSN"

    return False, None


# -------------------------------------------------------------------
# MVO-specific waveform fixes
# -------------------------------------------------------------------

def fix_sample_rate(obj, Fs=75.0, tol=0.01):
    """
    Snap near-matching sample rates to an exact target rate.

    Parameters
    ----------
    obj
        ObsPy Stream or Trace.
    Fs
        Target sample rate.
    tol
        Fractional tolerance, e.g. 0.01 means ±1%.
    """
    if isinstance(obj, Stream):
        for tr in obj:
            fix_sample_rate(tr, Fs=Fs, tol=tol)
    elif isinstance(obj, Trace):
        sr = float(obj.stats.sampling_rate)
        if Fs * (1 - tol) < sr < Fs * (1 + tol):
            obj.stats.sampling_rate = Fs
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace.")


def fix_y2k_times_mvo(obj):
    """
    Correct known MVO legacy year errors in trace start times.

    Rules
    -----
    - 1991, 1992, 1993 -> add 8 years
    - years < 1908     -> add 100 years

    Parameters
    ----------
    obj
        ObsPy Stream or Trace.
    """
    if isinstance(obj, Stream):
        for tr in obj:
            fix_y2k_times_mvo(tr)
    elif isinstance(obj, Trace):
        yyyy = obj.stats.starttime.year
        if yyyy in (1991, 1992, 1993):
            obj.stats.starttime._set_year(yyyy + 8)
        elif yyyy < 1908:
            obj.stats.starttime._set_year(yyyy + 100)
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace.")


def fix_trace_mvo(trace: Trace, verbose: bool = False):
    """
    Apply MVO-specific NSLC and timing cleanup to a trace.

    Parameters
    ----------
    trace
        Trace to modify in place.
    verbose
        Print diagnostics.

    Notes
    -----
    - Legacy analog-style traces are assumed to have already had their
      embedded station/channel info decoded elsewhere.
    - Digital traces are normalized via ``correct_nslc_mvo()``.
    """
    sta = (trace.stats.station or "").upper()
    fs = float(trace.stats.sampling_rate) if trace.stats.sampling_rate else np.nan

    legacy = True
    short_period = True

    if (
        (sta.startswith("MB") and sta != "MBET")
        or sta.startswith("MTB")
        or sta.startswith("BLV")
        or sta == "GHWS"
    ):
        legacy = False
        short_period = False

    if np.isfinite(fs) and 70.0 < fs < 80.0:
        fix_y2k_times_mvo(trace)
        fix_sample_rate(trace)

    if verbose:
        print(f"Fixing MVO trace ID: {trace.id} legacy={legacy} short_period={short_period}")

    if legacy:
        trace.stats.location = fix_location_code(trace.stats.location)
        trace.stats.channel = fix_channel_code(
            trace.stats.channel,
            trace.stats.sampling_rate,
            short_period=True,
        )
    else:
        trace.id = correct_nslc_mvo(
            trace.id,
            trace.stats.sampling_rate,
            short_period=short_period,
            net="MV",
            verbose=verbose,
        )


def correct_nslc_mvo(
    trace_id: str,
    Fs: float,
    short_period: bool | None = None,
    net: str = "MV",
    verbose: bool = False,
) -> str:
    """
    Standardize Montserrat trace IDs, handling legacy and special cases.

    Parameters
    ----------
    trace_id
        Input trace ID in NET.STA.LOC.CHA form.
    Fs
        Sampling rate in Hz.
    short_period
        If True, force short-period band-code logic where appropriate.
        If None, leave to existing channel/station heuristics.
    net
        Network code to use in output ID.
    verbose
        If True, print debug information.

    Returns
    -------
    str
        Corrected trace ID in NET.STA.LOC.CHA form.
    """

    def vprint(*args):
        if verbose:
            print(*args)

    def _safe_decompose(chan: str) -> tuple[str, str, str]:
        band, inst, orient = decompose_channel_code(chan)
        band = "" if band in (None, "x", "X") else band.upper()
        inst = "" if inst in (None, "x", "X") else inst.upper()
        orient = "" if orient in (None, "x", "X") else orient.upper()
        return band, inst, orient

    def _normalize_loc(loc: str) -> str:
        return fix_location_code((loc or "").strip().upper())

    def _normalize_pressure_channel(
        chan: str,
        loc: str,
        sta: str,
        Fs: float,
    ) -> tuple[str, str]:
        """
        Normalize pressure / infrasound channels to D?O / D?F style.
        """
        vprint(f"Handling microbarometer channel: {chan} at {sta} with Fs={Fs}")

        chan = (chan or "").strip().upper()
        loc = (loc or "").strip().upper()

        # Common legacy pressure names
        if chan in ("PR", "PRS"):
            chan = "BDO"

        # Legacy broadband/short-period acoustic prefixes
        if chan.startswith("E"):
            chan = "H" + chan[1:]
        elif chan.startswith("S"):
            chan = "B" + chan[1:]

        _, _, orient = _safe_decompose(chan)
        fixed = fix_channel_code(chan if chan else "BDO", Fs, short_period=False)
        band = fixed[0].upper() if fixed else "B"

        # Acoustic / pressure sensors use instrument code D
        if orient.isdigit():
            chan_out = f"{band}D{orient}"
        elif orient == "F":
            chan_out = f"{band}DF"
        else:
            chan_out = f"{band}DO"

        loc_out = _normalize_loc(loc)
        vprint(f"Converted microbarometer channel: {loc_out}.{chan_out}")
        return chan_out, loc_out

    def _normalize_acousticish_channel(
        chan: str,
        loc: str,
        Fs: float,
    ) -> tuple[str, str]:
        """
        Normalize already-acoustic-ish channels while preserving D/O semantics.
        """
        fixed = fix_channel_code(chan, Fs, short_period=False)
        band, inst, orient = _safe_decompose(fixed)

        if not band:
            band = "B"
        if not inst:
            inst = "D"
        if not orient:
            orient = "O"

        return f"{band}{inst}{orient}", _normalize_loc(loc)

    def _normalize_seismic_channel(
        chan: str,
        loc: str,
        sta: str,
        Fs: float,
        short_period: bool | None,
    ) -> tuple[str, str]:
        """
        Normalize MVO seismic channels.
        """
        vprint(f"Handling seismic channel: {chan} at {sta} with Fs={Fs}")

        chan = (chan or "").strip().upper()
        loc = (loc or "").strip().upper()
        sta = (sta or "").strip().upper()

        # Legacy short-period form like SBZ/SBN/SBE -> BHZ/BHN/BHE
        if chan.startswith("SB") and len(chan) == 3 and chan[2] in "ZNE":
            chan = "BH" + chan[2]
            short_period = False

        band, inst, orient = _safe_decompose(chan)

        # Sometimes orientation is carried in 1-char location code
        if not orient and len(loc) == 1 and loc in "ZNE":
            orient = loc
            loc = ""

        if not inst:
            inst = "H"
        if not orient:
            orient = "Z"
        if not band:
            fixed = fix_channel_code("BHZ", Fs, short_period=bool(short_period))
            band = fixed[0] if fixed else "S"

        chan_guess = f"{band}{inst}{orient}"

        if sta and sta[:2] != "MB" and ((short_period and "L" in chan_guess) or sta.endswith("L")):
            vprint(
                f"Warning: {trace_id} might be a legacy low-gain analog-network ID"
            )

        chan_out = fix_channel_code(chan_guess, Fs, short_period=bool(short_period))
        loc_out = _normalize_loc(loc)
        return chan_out, loc_out

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------
    trace_id = (trace_id or "").replace("?", "X").strip()
    parts = trace_id.split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid trace ID format: {trace_id!r}. Expected NET.STA.LOC.CHA")

    oldnet, oldsta, oldloc, oldcha = parts

    sta = (oldsta or "").strip().upper()
    loc = (oldloc or "").strip().upper()
    raw_chan = (oldcha or "").strip().upper()
    chan = raw_chan.replace(" ", "")
    net = (net or oldnet or "").strip().upper()

    # Weather station special case
    if sta == "GHWS":
        return trace_id

    # Legacy location codes sometimes contain timing / telemetry hints
    if "J" in loc or "I" in loc:
        Fs = 75.0
        loc = loc.replace("J", "").replace("I", "")

    # Additional 75 Hz hint embedded in channel code
    if "J" in chan:
        Fs = 75.0

    loc = _normalize_loc(loc)

    if chan == "DUM":
        return f"{net}.{sta}.{loc}.{chan}"

    # ------------------------------------------------------------------
    # AEF / legacy channel quirks
    # Preserve the historical "S J" form used on the original MVO digital network.
    # ------------------------------------------------------------------
    if raw_chan.startswith("S J") and len(raw_chan) >= 4:
        chan = "SH" + raw_chan[3:].replace(" ", "")
    elif chan.startswith("SJ") and len(chan) >= 3:
        chan = "SH" + chan[2:]
    elif chan.startswith("SBJ") and len(chan) >= 4:
        chan = "BH" + chan[3:]

    # ------------------------------------------------------------------
    # Channel family dispatch
    # ------------------------------------------------------------------
    if len(chan) >= 2 and chan[:2] in ("AP", "PR"):
        chan, loc = _normalize_pressure_channel(chan, loc, sta, Fs)

    elif len(chan) >= 2 and (
        chan[:2] in ("AH", "PH") or (chan.startswith("S") and chan.endswith("A"))
    ):
        chan, loc = _normalize_acousticish_channel(chan, loc, Fs)

    else:
        chan, loc = _normalize_seismic_channel(
            chan,
            loc,
            sta,
            Fs,
            short_period=short_period,
        )

    return f"{net}.{sta}.{loc}.{chan}"


# -------------------------------------------------------------------
# MVO waveform and event helpers
# -------------------------------------------------------------------

def read_mvo_waveform_file(
    wavpath: str | Path,
    verbose: bool = False,
    seismic_only: bool = False,
    vertical_only: bool = False,
) -> Stream:
    """
    Read a waveform file from the Montserrat archive and apply MVO-specific
    NSLC fixing and light cleanup.

    Parameters
    ----------
    wavpath
        Path to waveform file readable by ObsPy.
    verbose
        Print warnings / debug messages.
    seismic_only
        Keep only seismic channels (instrument codes H or L).
    vertical_only
        Keep only vertical seismic channels.

    Returns
    -------
    Stream
    """
    wavpath = Path(wavpath)
    if not wavpath.exists():
        if verbose:
            print(f"ERROR: {wavpath} not found.")
        return Stream()

    try:
        st = read(str(wavpath))
    except Exception as e:
        if verbose:
            print(f"ERROR reading {wavpath}: {e}")
        return Stream()

    if vertical_only:
        st = st.select(component="Z")
    elif seismic_only:
        tr_keep = Stream()
        for tr in st:
            chan = getattr(tr.stats, "channel", "")
            if len(chan) >= 2 and chan[1] in "HL":
                tr_keep.append(tr)
        st = tr_keep

    remove_empty_traces(st)

    for tr in st:
        fix_trace_mvo(tr, verbose=verbose)

    return st


def correct_mvo_waveformstreamid(waveform_id, time=None):
    """
    Correct a WaveformStreamID in place using MVO-specific NSLC rules.
    """
    if waveform_id is None:
        return

    original_id = waveform_id.get_seed_string()
    fs = infer_mvo_sample_rate_from_time(time)
    corrected_id = correct_nslc_mvo(original_id, fs)

    net, sta, loc, cha = corrected_id.split(".")
    waveform_id.network_code = net
    waveform_id.station_code = sta
    waveform_id.location_code = loc
    waveform_id.channel_code = cha


def correct_mvo_event_ids(eventobj):
    """
    Correct waveform IDs for all picks in an ObsPy Event using MVO-specific rules.
    """
    if eventobj is None or not hasattr(eventobj, "picks"):
        return eventobj

    for pick in eventobj.picks:
        correct_mvo_waveformstreamid(
            pick.waveform_id,
            time=getattr(pick, "time", None),
        )

    return eventobj


# -------------------------------------------------------------------
# MVO-specific archive wrapper
# -------------------------------------------------------------------

class MVOSeisanArchive(SeisanArchive):
    """
    Convenience wrapper around a generic SeisanArchive for Montserrat.

    Continuous DB:
        WAV/DSNC_/

    Event DB:
        REA/MVOE_/
    """

    def __init__(self, root: str | Path):
        super().__init__(root=root, db_cont="DSNC_", db_event="MVOE_")

    def iter_continuous_streams(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        verbose: bool = False,
        seismic_only: bool = False,
        vertical_only: bool = False,
    ):
        """
        Yield cleaned MVO waveform streams from the continuous WAV archive.
        """
        for wavpath in self.iter_waveform_files(starttime, endtime, db=self.db_cont):
            st = read_mvo_waveform_file(
                wavpath,
                verbose=verbose,
                seismic_only=seismic_only,
                vertical_only=vertical_only,
            )
            if len(st) > 0:
                yield wavpath, st

    def iter_event_streams(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        verbose: bool = False,
        use_mvo_parser: bool = True,
        parse_aef: bool = False,
        seismic_only: bool = False,
        vertical_only: bool = False,
        valid_subclasses: str = "",
    ):
        """
        Yield (Sfile, Stream) pairs from the MVO event archive.
        """
        for sfile_path in self.iter_sfiles(starttime, endtime, db=self.db_event):
            try:
                s = Sfile(str(sfile_path), use_mvo_parser=use_mvo_parser, parse_aef=parse_aef)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to parse Sfile {sfile_path}: {e}")
                continue

            if valid_subclasses:
                if not (s.mainclass == "LV" and s.subclass in valid_subclasses):
                    continue

            if not getattr(s, "dsnwavfileobj", None):
                continue

            wavpath = getattr(s.dsnwavfileobj, "path", None)
            if not wavpath:
                continue

            st = read_mvo_waveform_file(
                wavpath,
                verbose=verbose,
                seismic_only=seismic_only,
                vertical_only=vertical_only,
            )
            if len(st) > 0:
                yield s, st

    def apply_to_events(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        function: Callable | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Apply a user function to each event stream.

        The callable should accept:
            function(sfile, stream, **kwargs)
        """
        for s, st in self.iter_event_streams(starttime, endtime, verbose=verbose):
            if function is None:
                yield s, st
            else:
                function(s, st, **kwargs)