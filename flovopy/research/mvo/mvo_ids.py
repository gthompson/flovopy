from __future__ import annotations

import struct
from pathlib import Path
from typing import Callable

import numpy as np
from obspy import Stream, Trace, read

from obspy.core.utcdatetime import UTCDateTime

from flovopy.seisanio.core.seisanarchive import SeisanArchive
from flovopy.core.trace_utils import (
    decompose_channel_code,
    fix_channel_code,
    fix_location_code,
)

# -------------------------------------------------------------------
# Montserrat constants
# -------------------------------------------------------------------

DOME_LOCATION = {"lat": 16.71060, "lon": -62.17747, "elev": 1000.0}

# (lon_min, lon_max, lat_min, lat_max)
REGION_DEFAULT = (-62.255, -62.135, 16.66, 16.84)

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
    trace_id_in = trace.id

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
    if trace.stats.location in {"Z", "N", "E"} and len(trace.stats.channel) >= 2 and trace.stats.channel[-1]==trace.stats.location:
        trace.stats.location = ""

    print(f"{trace_id_in} -> {trace.id}")


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

        # Sometimes orientation is carried in the 1-character location code.
        # Prefer a valid location-code orientation over a missing or conflicting
        # channel orientation, then blank the location code.
        if len(loc) == 1 and loc in "ZNE12":
            if not orient:
                orient = loc
                loc = ""
            elif orient != loc:
                if verbose:
                    vprint(
                        f"Overriding conflicting orientation for {sta}: "
                        f"channel={chan!r} gives {orient!r}, location={loc!r}"
                    )
                orient = loc
                loc = ""
            else:
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

