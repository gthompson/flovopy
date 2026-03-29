from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable

from obspy import Stream, read
from obspy.core.inventory import read_inventory
from obspy.core.utcdatetime import UTCDateTime

from flovopy.seisanio.archive import SeisanArchive
from flovopy.core.trace_utils import remove_empty_traces
from flovopy.core.mvo import fix_trace_mvo
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
    """
    xmldir = Path(xmldir)
    master_station_xml = xmldir / "MontserratDigitalSeismicNetwork.xml"
    if master_station_xml.exists():
        print(f"Loading {master_station_xml}")
        return read_inventory(str(master_station_xml))

    print(f"Could not find {master_station_xml}")
    return None


# -------------------------------------------------------------------
# MVO waveform helpers
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
    obspy.Stream
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


def change_last_sample(tr):
    """
    For some legacy Montserrat SEISAN files, the last sample can be corrupt.
    Remove the final sample.
    """
    tr.data = tr.data[:-1]


def swap32(i: int) -> int:
    """
    Swap endianness of a 32-bit integer.
    """
    return struct.unpack("<i", struct.pack(">i", i))[0]


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
        Yield (Sfile, Stream) pairs from the event archive.
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



def infer_mvo_sample_rate_from_time(time):
    """
    Heuristic sample rate choice for legacy MVO picks/events.

    Before 2005:
        usually 75 Hz
    2005 onward:
        usually 100 Hz
    """
    if time is None:
        return 100.0
    return 75.0 if time < UTCDateTime(2005, 1, 1) else 100.0


def correct_mvo_waveformstreamid(waveform_id, time=None):
    """
    Correct a WaveformStreamID in place using MVO-specific NSLC rules.
    """
    from flovopy.core.trace_utils import correct_nslc_mvo

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
        correct_mvo_waveformstreamid(pick.waveform_id, time=getattr(pick, "time", None))

    return eventobj

def legacy_or_not(fname):
    """
    Heuristic classification for Montserrat waveform filenames.

    Returns
    -------
    legacy : bool
    network : str | None
        'ASN' for analog/legacy network files, 'MV' for digital MVO files
        when confidently inferred, else None.
    """
    fnamelower = os.path.basename(str(fname)).lower()

    if "asne" in fnamelower or "spn" in fnamelower:
        return True, "ASN"

    if "mvo" in fnamelower or "dsn" in fnamelower:
        return False, "DSN"

    return False, None