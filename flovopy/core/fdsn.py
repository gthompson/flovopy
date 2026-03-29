######################################################################
##   Additional tools for ObsPy FDSN client                         ##
######################################################################

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from obspy import Stream, UTCDateTime, read
from obspy.core.inventory import Inventory, read_inventory
from obspy.clients.fdsn import Client

from flovopy.core.miniseed_io import smart_merge


CACHE_DIR = Path(".") / "cache"


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _check_client_or_string(fdsn_thing) -> Client:
    """
    Return an ObsPy FDSN Client from either an existing Client or a base URL.
    """
    if isinstance(fdsn_thing, Client):
        return fdsn_thing
    if isinstance(fdsn_thing, str):
        return Client(base_url=fdsn_thing)
    raise TypeError("fdsn_thing must be an ObsPy FDSN Client or a base URL string")


def _make_cache_dir(cachedir) -> Path:
    """
    Ensure a cache directory exists and return it as a Path.
    """
    cachedir = Path(cachedir)
    cachedir.mkdir(parents=True, exist_ok=True)
    return cachedir


def _safe_trace_id_for_filename(trace_id: str) -> str:
    """
    Make a trace ID safer for use in cache filenames.
    """
    return trace_id.replace("*", "X").replace("?", "X").replace("/", "_")


def _get_stationxml_filename(
    startt,
    endt,
    centerlat,
    centerlon,
    search_radius_deg,
    cachedir=CACHE_DIR,
) -> Path:
    """
    Return a cache filename for a station query.
    """
    cachedir = _make_cache_dir(cachedir)
    return cachedir / (
        f"{UTCDateTime(startt).strftime('%Y%m%d%H%M')}_"
        f"{UTCDateTime(endt).strftime('%Y%m%d%H%M')}_"
        f"{centerlat:.4f}_{centerlon:.4f}_{search_radius_deg:.2f}.SML"
    )


def _get_mseed_filename(
    startt,
    endt,
    trace_ids: Iterable[str],
    cachedir=CACHE_DIR,
) -> Path:
    """
    Return a cache filename for a waveform query.
    """
    trace_ids = list(trace_ids)
    if not trace_ids:
        raise ValueError("trace_ids must not be empty")

    cachedir = _make_cache_dir(cachedir)
    return cachedir / (
        f"{UTCDateTime(startt).strftime('%Y%m%d%H%M')}_"
        f"{UTCDateTime(endt).strftime('%Y%m%d%H%M')}_"
        f"{_safe_trace_id_for_filename(trace_ids[0])}_"
        f"{_safe_trace_id_for_filename(trace_ids[-1])}.MSEED"
    )


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def get_inventory(
    fdsn_client,
    startt,
    endt,
    centerlat,
    centerlon,
    search_radius_deg,
    network="*",
    station="*",
    channel="*",
    overwrite: bool = False,
    cache: bool = False,
    cachedir=CACHE_DIR,
    verbose: bool = True,
) -> Inventory | None:
    """
    Get an inventory of available stations/channels for a circular query.

    Parameters
    ----------
    fdsn_client
        ObsPy FDSN Client or base URL string.
    startt, endt
        Query time range.
    centerlat, centerlon
        Query center coordinates in decimal degrees.
    search_radius_deg
        Maximum radius in degrees.
    network, station, channel
        FDSN selectors.
    overwrite
        If True, ignore any cached StationXML file.
    cache
        If True, cache downloaded inventory to StationXML.
    cachedir
        Cache directory.
    verbose
        If True, print progress and warnings.

    Returns
    -------
    Inventory or None
        ObsPy Inventory, or None if unavailable.
    """
    startt = UTCDateTime(startt)
    endt = UTCDateTime(endt)

    stationxml_file = _get_stationxml_filename(
        startt,
        endt,
        centerlat,
        centerlon,
        search_radius_deg,
        cachedir=cachedir,
    )

    if stationxml_file.is_file() and not overwrite:
        try:
            inv = read_inventory(str(stationxml_file))
            if verbose:
                print(f"Loaded cached inventory from {stationxml_file}")
            return inv
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to read cached inventory {stationxml_file}: {e}")

    if verbose:
        print(
            "Trying to load inventory from "
            f"{startt.strftime('%Y/%m/%d %H:%M')} to {endt.strftime('%Y/%m/%d %H:%M')}"
        )

    client = _check_client_or_string(fdsn_client)

    try:
        inv = client.get_stations(
            network=network,
            station=station,
            channel=channel,
            latitude=centerlat,
            longitude=centerlon,
            maxradius=search_radius_deg,
            starttime=startt,
            endtime=endt,
            level="response",
        )
    except Exception as e:
        if verbose:
            print(e)
            print("- no inventory available")
        return None

    if cache:
        try:
            inv.write(str(stationxml_file), format="STATIONXML")
            if verbose:
                print(f"Inventory saved to {stationxml_file}")
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to cache inventory to {stationxml_file}: {e}")

    return inv


def get_stream(
    fdsn_client,
    trace_ids,
    startt,
    endt,
    overwrite: bool = False,
    cache: bool = False,
    cachedir=CACHE_DIR,
    attach_response: bool = True,
    merge_strategy: str = "both",
    force_close_sampling_rates_on_merge_failure: bool = True,
    sampling_rate_tolerance_hz: float = 0.1,
    verbose: bool = True,
) -> Stream:
    """
    Load waveform data for all requested trace IDs over a time range.

    Parameters
    ----------
    fdsn_client
        ObsPy FDSN Client or base URL string.
    trace_ids
        Iterable of NET.STA.LOC.CHA trace IDs.
    startt, endt
        Query time range.
    overwrite
        If True, ignore cached MiniSEED and redownload.
    cache
        If True, cache downloaded waveform data as MiniSEED.
    cachedir
        Cache directory.
    attach_response
        Passed to ObsPy FDSN get_waveforms().
    merge_strategy
        Strategy passed to ``smart_merge()``.
    force_close_sampling_rates_on_merge_failure
        If True, allow smart_merge() to retry ObsPy merge after forcing
        near-equal sampling rates to a weighted mean.
    sampling_rate_tolerance_hz
        Tolerance used by the above fallback.
    verbose
        If True, print progress and warnings.

    Returns
    -------
    Stream
        ObsPy Stream, possibly empty.
    """
    startt = UTCDateTime(startt)
    endt = UTCDateTime(endt)
    trace_ids = list(trace_ids)

    if not trace_ids:
        return Stream()

    mseedfile = _get_mseed_filename(startt, endt, trace_ids, cachedir=cachedir)

    if mseedfile.is_file() and not overwrite:
        try:
            st = read(str(mseedfile))
            if verbose:
                print(f"Loaded cached waveform data from {mseedfile}")
            return st
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to read cached waveform file {mseedfile}: {e}")

    client = _check_client_or_string(fdsn_client)
    st = Stream()

    for trace_id in trace_ids:
        try:
            network, station, location, chancode = trace_id.split(".")
        except ValueError:
            if verbose:
                print(f"[WARN] Invalid trace ID skipped: {trace_id}")
            continue

        if verbose:
            print(
                f"net={network}, station={station}, "
                f"location={location}, chancode={chancode}"
            )

        try:
            this_st = client.get_waveforms(
                network,
                station,
                location,
                chancode,
                starttime=startt,
                endtime=endt,
                attach_response=attach_response,
            )
        except Exception as e:
            if verbose:
                print(f"- No waveform data available for {trace_id}: {e}")
            continue

        if len(this_st):
            this_st = smart_merge(
                this_st,
                strategy=merge_strategy,
                sanitize_before_merge=True,
                force_close_sampling_rates_on_merge_failure=force_close_sampling_rates_on_merge_failure,
                sampling_rate_tolerance_hz=sampling_rate_tolerance_hz,
                return_stream_only=True,
                verbose=verbose,
            )
            st += this_st

    if not len(st):
        if verbose:
            print(f"- No waveform data available for request {mseedfile.name}")
        return st

    st = smart_merge(
        st,
        strategy=merge_strategy,
        sanitize_before_merge=True,
        force_close_sampling_rates_on_merge_failure=force_close_sampling_rates_on_merge_failure,
        sampling_rate_tolerance_hz=sampling_rate_tolerance_hz,
        return_stream_only=True,
        verbose=verbose,
    )

    if cache:
        try:
            st.write(str(mseedfile), format="MSEED")
            if verbose:
                print(f"Waveform cache written to {mseedfile}")
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to cache waveform data to {mseedfile}: {e}")

    return st


def get_fdsn_identifier(fdsn_url: str) -> str:
    """
    Return a short prefix for a known FDSN service URL.
    """
    fdsn_url = fdsn_url.lower()

    if "iris" in fdsn_url:
        return "iris_"
    if "shake" in fdsn_url:
        return "rboom_"
    if "geonet" in fdsn_url:
        return "nz_"
    return ""