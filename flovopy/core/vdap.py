"""
vdap.py
-------

Utilities for working with legacy VDAP (Volcano Disaster Assistance Program)
data products, including:

- HYPO71 earthquake catalog files
- demultiplexed SUDS waveform files (DMX)
- fixed-width STATION0.HYP station coordinate files
- legacy VDAP-style waveform trace IDs

Main functionality
------------------
- parse_hypo71_line
- parse_hypo71_file
- read_dmx_file
- parse_station0_line
- read_station0_file
- subset_stations_within_radius
- fix_legacy_vdap_id
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude, Comment

from obspy.core.inventory import Inventory, Network, Station, Channel, Site


# -----------------------------------------------------------------------------
# Coordinate helpers
# -----------------------------------------------------------------------------

def dm_to_decimal(deg: int, minutes: float, hemisphere: str) -> float:
    """
    Convert degrees + decimal minutes to signed decimal degrees.

    Parameters
    ----------
    deg
        Integer degrees.
    minutes
        Decimal arc-minutes.
    hemisphere
        One of N, S, E, W.

    Returns
    -------
    float
        Signed decimal degrees.
    """
    value = float(deg) + float(minutes) / 60.0
    if hemisphere.upper() in {"S", "W"}:
        value *= -1.0
    return value


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in kilometers.
    """
    r_earth_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r_earth_km * math.asin(math.sqrt(a))


# -----------------------------------------------------------------------------
# HYPO71 parsing
# -----------------------------------------------------------------------------

def parse_hypo71_line(line: str, *, verbose: bool = False) -> Optional[dict]:
    """
    Parse one line of HYPO71 earthquake location output.

    Parameters
    ----------
    line
        One text line from a HYPO71 output file.
    verbose
        If True, print parse failures.

    Returns
    -------
    dict or None
        Dictionary containing parsed origin/magnitude information, or None if
        parsing fails.
    """
    try:
        year = int(line[0:2])
        month = int(line[2:4])
        day = int(line[4:6])
        hour = int(line[7:9]) if line[7:9].strip() else 0
        minute = int(line[9:11]) if line[9:11].strip() else 0
        seconds = float(line[12:17]) if line[12:17].strip() else 0.0

        lat_deg = int(line[17:20].strip())
        lat_hem = line[20].strip().upper()
        lat_min = float(line[21:26].strip())

        lon_deg = int(line[27:30].strip())
        lon_hem = line[30].strip().upper()
        lon_min = float(line[31:36].strip())

        depth_km = float(line[37:43].strip())
        magnitude = float(line[44:50].strip())
        n_ass = int(line[51:53].strip())
        time_residual = float(line[62:].strip())

        year = year + 1900 if year >= 70 else year + 2000

        add_seconds = 0.0
        if minute == 60:
            minute = 0
            add_seconds = 60.0

        origin_time = UTCDateTime(year, month, day, hour, minute, seconds) + add_seconds

        latitude = dm_to_decimal(lat_deg, lat_min, lat_hem)
        longitude = dm_to_decimal(lon_deg, lon_min, lon_hem)

        return {
            "origin_time": origin_time,
            "latitude": latitude,
            "longitude": longitude,
            "depth_km": depth_km,
            "magnitude": magnitude,
            "n_ass": n_ass,
            "time_residual": time_residual,
        }

    except Exception as e:
        if verbose:
            print(f"Failed to parse HYPO71 line: {line.rstrip()} | Error: {e}")
        return None


def parse_hypo71_file(file_path: str | Path, *, verbose: bool = True) -> tuple[Catalog, list[str]]:
    """
    Parse a HYPO71 file into an ObsPy Catalog.

    Parameters
    ----------
    file_path
        Path to the HYPO71 text file.
    verbose
        If True, print summary information.

    Returns
    -------
    (Catalog, list[str])
        Parsed ObsPy Catalog and list of unparsed lines.
    """
    file_path = Path(file_path)
    catalog = Catalog()
    unparsed_lines: list[str] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            event_data = parse_hypo71_line(line)
            if event_data is None:
                unparsed_lines.append(line)
                continue

            event = Event()

            origin = Origin(
                time=event_data["origin_time"],
                latitude=event_data["latitude"],
                longitude=event_data["longitude"],
                depth=event_data["depth_km"] * 1000.0,
            )
            origin.comments.append(Comment(text=f"n_ass: {event_data['n_ass']}"))
            origin.comments.append(Comment(text=f"time_residual: {event_data['time_residual']} sec"))

            magnitude = Magnitude(mag=event_data["magnitude"])

            event.origins.append(origin)
            event.magnitudes.append(magnitude)
            catalog.append(event)

    if verbose:
        print(f"Parsed: {len(catalog)} | Unparsed: {len(unparsed_lines)}")

    return catalog, unparsed_lines


# -----------------------------------------------------------------------------
# STATION0.HYP parsing
# -----------------------------------------------------------------------------

def parse_station0_line(line: str, expected_len: Optional[int] = 27) -> Optional[dict]:
    """
    Parse one fixed-width STATION0.HYP station line.

    Expected layout
    ---------------
      cols  0:2   -> two leading blanks
      cols  2:6   -> station code
      cols  6:8   -> latitude degrees
      cols  8:13  -> latitude minutes
      col  13     -> latitude hemisphere (N/S)
      cols 14:17  -> longitude degrees
      cols 17:22  -> longitude minutes
      col  22     -> longitude hemisphere (E/W)
      cols 23:27  -> elevation in meters

    Parameters
    ----------
    line
        Input text line.
    expected_len
        Exact expected line length. If not None, lines with other lengths
        are ignored.

    Returns
    -------
    dict or None
        Parsed station record, or None if parsing fails.
    """
    line = line.rstrip("\n")

    if expected_len is not None and len(line) != expected_len:
        return None

    try:
        station = line[2:6].strip()
        lat_deg = int(line[6:8].strip())
        lat_min = float(line[8:13].strip())
        lat_hemi = line[13].strip().upper()

        lon_deg = int(line[14:17].strip())
        lon_min = float(line[17:22].strip())
        lon_hemi = line[22].strip().upper()

        elevation_m = int(line[23:27].strip())

        if not station:
            return None
        if lat_hemi not in {"N", "S"}:
            return None
        if lon_hemi not in {"E", "W"}:
            return None

        latitude = dm_to_decimal(lat_deg, lat_min, lat_hemi)
        longitude = dm_to_decimal(lon_deg, lon_min, lon_hemi)

        return {
            "station": station,
            "lat_deg": lat_deg,
            "lat_min": lat_min,
            "lat_hemi": lat_hemi,
            "lon_deg": lon_deg,
            "lon_min": lon_min,
            "lon_hemi": lon_hemi,
            "elevation_m": elevation_m,
            "latitude": latitude,
            "longitude": longitude,
        }

    except (ValueError, IndexError):
        return None


def read_station0_file(filepath: str | Path, expected_len: int = 27) -> pd.DataFrame:
    """
    Read a fixed-width STATION0.HYP station file into a pandas DataFrame.

    Parameters
    ----------
    filepath
        Path to the input text file.
    expected_len
        Exact required line length. Lines with other lengths are ignored.

    Returns
    -------
    pandas.DataFrame
        DataFrame of successfully parsed station rows.
    """
    filepath = Path(filepath)
    records: list[dict] = []

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_station0_line(line, expected_len=expected_len)
            if parsed is not None:
                records.append(parsed)

    return pd.DataFrame(records)


def subset_stations_within_radius(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> pd.DataFrame:
    """
    Return stations within a given radius of a center point.

    Parameters
    ----------
    df
        DataFrame from `read_station0_file()`.
    center_lat, center_lon
        Center point in decimal degrees.
    radius_km
        Radius in kilometers.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with added `distance_km` column.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["distance_km"] = out.apply(
        lambda row: haversine_km(
            row["latitude"],
            row["longitude"],
            center_lat,
            center_lon,
        ),
        axis=1,
    )
    out = out.loc[out["distance_km"] <= radius_km].sort_values("distance_km")
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# DMX waveform reading
# -----------------------------------------------------------------------------

def read_dmx_file(dmx_file: str | Path, fix: bool = True, defaultnet: str = "") -> Stream:
    """
    Read a DMX waveform file into an ObsPy Stream.

    Parameters
    ----------
    dmx_file
        Path to DMX file.
    fix
        If True, apply basic metadata/data corrections.
    defaultnet
        Network code to assign if trace network is 'unk'.

    Returns
    -------
    obspy.Stream
    """
    dmx_file = Path(dmx_file)
    st = Stream()

    print(f"Reading {dmx_file}")
    try:
        st = read(str(dmx_file))
        print("- read successful")

        if fix:
            for tr in st:
                if getattr(tr.stats, "network", "") == "unk":
                    tr.stats.network = defaultnet
                tr.data = tr.data.astype(float) - 2048.0

    except Exception as e:
        print(f"- Failed to read DMX file: {e}")

    return st


# -----------------------------------------------------------------------------
# Legacy VDAP ID fixing
# -----------------------------------------------------------------------------

def fix_legacy_vdap_id(
    trace: Trace,
    network: Optional[str] = None,
    default_channel: str = "EHZ",
    default_lowgain_channel: str = "ELZ",
) -> None:
    """
    Decode and normalize a legacy VDAP-style trace ID in place.

    Parameters
    ----------
    trace
        Trace to modify.
    network
        Optional network code to assign.
    default_channel
        Default channel for ordinary legacy seismic traces.
        Currently retained for API compatibility.
    default_lowgain_channel
        Default low-gain channel.
        Currently retained for API compatibility.

    Notes
    -----
    This function interprets legacy cases where:
    - station code embeds orientation or low-gain suffixes
    - channel may be empty, partial, or one character long
    - `IRIG` station is treated as timing channel `ACE`
    """
    if network:
        trace.stats.network = network

    sta = (trace.stats.station or "").strip().upper()
    loc = (trace.stats.location or "").strip().upper()
    chan = (trace.stats.channel or "").strip().upper()

    if sta == "IRIG":
        trace.stats.channel = "ACE"
        return

    bandcode = "E"
    instrumentcode = "H"
    orientationcode = "Z"

    if len(chan) == 3:
        bandcode = chan[0]
        instrumentcode = chan[1]
        orientationcode = chan[2]

    elif len(chan) == 2:
        bandcode = chan[0]
        if chan[1] in "ZNE":
            instrumentcode = "H"
            orientationcode = chan[1]
        elif chan[1] in "HLD":
            instrumentcode = chan[1]
            orientationcode = "Z"

    elif len(chan) == 1:
        c = chan[0]
        if c in "VZ":
            orientationcode = "Z"
        elif c in "NE":
            orientationcode = c
        elif c == "L":
            instrumentcode = "L"
            orientationcode = "Z"

    if len(sta) >= 2:
        last2 = sta[-2:]
        if last2[0] == "L" and last2[1] in "ZNEV":
            instrumentcode = "L"
            orientationcode = "Z" if last2[1] == "V" else last2[1]
            sta = sta[:-2]

    if len(sta) >= 4:
        last1 = sta[-1]
        if last1 in "ZNEV":
            orientationcode = "Z" if last1 == "V" else last1
            sta = sta[:-1]
        elif last1 == "L":
            instrumentcode = "L"
            orientationcode = "Z"
            sta = sta[:-1]

    if not bandcode or bandcode == "X":
        bandcode = "E"
    if not instrumentcode or instrumentcode == "X":
        instrumentcode = "H"
    if not orientationcode or orientationcode == "X":
        orientationcode = "Z"
    if orientationcode == "V":
        orientationcode = "Z"
    if instrumentcode not in ("H", "L", "D"):
        instrumentcode = "H"

    trace.stats.network = (trace.stats.network or "").strip().upper()
    trace.stats.station = sta
    trace.stats.location = loc
    trace.stats.channel = f"{bandcode}{instrumentcode}{orientationcode}"




def station0_to_inventory(
    station0: str | Path | pd.DataFrame,
    *,
    network_code: str = "XX",
    source: str = "STATION0.HYP",
    sender: str = "",
    module: str = "flovopy.vdap",
    include_channels: bool = False,
    channel_codes: Optional[list[str]] = None,
    location_code: str = "",
    sample_rate: float = 100.0,
    start_date: UTCDateTime | str | None = None,
    end_date: UTCDateTime | str | None = None,
    station_type: str = "",
) -> Inventory:
    """
    Convert a STATION0.HYP station table into an ObsPy Inventory.

    Parameters
    ----------
    station0
        Either:
        - path to a STATION0.HYP file, or
        - pandas DataFrame returned by `read_station0_file()`

    network_code
        Network code to assign to all stations.
    source
        Inventory source string.
    sender
        Optional sender/creator string.
    module
        Optional module/software identifier.
    include_channels
        If True, create placeholder Channel entries for each station.
        If False, create Station objects only.
    channel_codes
        List of channel codes to create when `include_channels=True`.
        Default is ["BHZ"].
    location_code
        Location code to assign to generated channels.
    sample_rate
        Sample rate to assign to generated channels.
    start_date, end_date
        Optional start/end dates for stations and channels.
    station_type
        Optional free-text station description.

    Returns
    -------
    obspy.core.inventory.Inventory
        Inventory containing one Network with stations from the STATION0.HYP file.

    Notes
    -----
    This creates a minimal metadata representation suitable for:
    - plotting station maps
    - associating waveforms with approximate station coordinates
    - bootstrapping legacy metadata

    It does NOT create instrument responses.
    """
    if isinstance(station0, (str, Path)):
        df = read_station0_file(station0)
    elif isinstance(station0, pd.DataFrame):
        df = station0.copy()
    else:
        raise TypeError("station0 must be a file path or a pandas DataFrame")

    if df.empty:
        return Inventory(networks=[], source=source, sender=sender, module=module)

    if channel_codes is None:
        channel_codes = ["BHZ"]

    if start_date is not None:
        start_date = UTCDateTime(start_date)
    if end_date is not None:
        end_date = UTCDateTime(end_date)

    stations: list[Station] = []

    for _, row in df.iterrows():
        sta_code = str(row["station"]).strip().upper()
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        elev = float(row.get("elevation_m", 0.0))

        station = Station(
            code=sta_code,
            latitude=lat,
            longitude=lon,
            elevation=elev,
            creation_date=start_date,
            start_date=start_date,
            end_date=end_date,
            site=Site(name=sta_code),
        )

        if station_type:
            station.description = station_type

        if include_channels:
            for chan_code in channel_codes:
                channel = Channel(
                    code=chan_code,
                    location_code=location_code,
                    latitude=lat,
                    longitude=lon,
                    elevation=elev,
                    depth=0.0,
                    azimuth=0.0,
                    dip=-90.0 if chan_code.endswith("Z") else 0.0,
                    sample_rate=sample_rate,
                    start_date=start_date,
                    end_date=end_date,
                )
                station.channels.append(channel)

        stations.append(station)

    network = Network(
        code=network_code,
        stations=stations,
        description=f"Inventory created from {source}",
        start_date=start_date,
        end_date=end_date,
    )

    inv = Inventory(
        networks=[network],
        source=source,
        sender=sender,
        module=module,
    )
    return inv