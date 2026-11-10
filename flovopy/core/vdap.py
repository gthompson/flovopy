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
- convert_dmx_to_seisan
- detect_pretrigger_window (generic triggered-record onset/noise analysis)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude, Comment

from obspy.core.inventory import Inventory, Network, Station, Channel, Site


# ============================================================================
# DMX to SEISAN DB wrapper (with waveform files to MiniSEED)
# ============================================================================


def convert_dmx_to_seisan(
    rawtop: str | Path,
    seisan_wav_db: str | Path,
    *,
    network: str,
    database: str,
    sampling_rate: float | None = None,
    pattern: str = "**/*.DMX",
    max_files: int | None = None,
    remove_irig: bool = True,
    event_id_func=None,
    id_fixer=None,
    out_waveform_index: str | Path | None = None,
    out_trace_id_map: str | Path | None = None,
    out_trace_id_map_tex: str | Path | None = None,
    out_failures: str | Path | None = None,
    out_trace_id_provenance: str | Path | None = None,
    overwrite: bool = False,
    continue_on_error: bool = True,
    verbose: bool = False,
) -> dict:
    """Convert legacy DMX files to SEISAN-named MiniSEED and build indexes.

    ``sampling_rate`` corrects metadata only; it does not resample samples.
    Data correction is delegated to read_dmx_file (float conversion and
    subtraction of 2048). Low gain is preserved by fix_legacy_vdap_id.

    event_id_func(path) returns an event ID, or None to reject a filename.
    The default expects YYMMDDNN.DMX. id_fixer(trace, network=network)
    modifies each trace in place; the default is fix_legacy_vdap_id.

    Files are sorted, merged (method=1, zero-filled) and padded to the union
    of their time windows before writing. Existing files are reused unless
    overwrite=True. Their actual saved headers supply the waveform index;
    their IDs and sample rates must match the proposed conversion. Reused files
    must be regenerated explicitly to apply changed IDs or rate corrections.

    Returns DataFrames: waveform_index, trace_id_map (unique verified output-ID
    mappings), trace_id_provenance (per-file observed mappings and status),
    failures; plus a summary dictionary. Existing-file mappings are included
    only when saved IDs and sample rates match the proposed conversion.
    CSV/LaTeX exports are optional; empty tables retain their column names.
    Per-file exceptions are collected unless continue_on_error=False.
    """
    # Local imports avoid loading SEISAN helpers during ordinary VDAP imports.
    from flovopy.core.trace_utils import remove_empty_traces
    from flovopy.seisanio.utils.helpers import write_wavfile

    rawtop = Path(rawtop).expanduser()
    outroot = Path(seisan_wav_db).expanduser()
    if not rawtop.is_dir():
        raise NotADirectoryError(str(rawtop))
    if not network or not database:
        raise ValueError("network and database must be nonempty")
    if sampling_rate is not None and (
        not math.isfinite(sampling_rate) or sampling_rate <= 0
    ):
        raise ValueError("sampling_rate must be finite and positive")
    if max_files is not None and (
        isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0
    ):
        raise ValueError("max_files must be a positive integer or None")

    event_id_func = extract_event_id if event_id_func is None else event_id_func
    id_fixer = fix_legacy_vdap_id if id_fixer is None else id_fixer
    files = [p for p in discover_dmx_files(rawtop, pattern) if p.is_file()]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {rawtop}")

    index_columns = ["event_id", "dmx_file", "wav_file", "starttime", "endtime",
                     "stations", "channels", "ntraces", "status"]
    provenance_columns = ["dmx_file", "event_id", "original_id", "fixed_id", "status"]
    index_rows, provenance_rows, failures = [], [], []
    successful_map = {}
    claimed_paths = {}
    counts = dict(processed=len(files), written=0, existing=0, empty=0, failed=0)

    for i, dmx in enumerate(files, 1):
        event_id = None
        pairs = []
        status = "failed"
        stage = "event_id"
        try:
            event_id = event_id_func(dmx)
            if not event_id:
                raise ValueError(f"Cannot derive event ID from {dmx.name}")
            if verbose:
                print(f"[{i}/{len(files)}] {event_id}: {dmx}")
            stage = "read"
            st = read_dmx_file(dmx, defaultnet=network, verbose=verbose)
            if not st:
                raise ValueError("DMX reader returned an empty stream")
            stage = "prepare"
            st = remove_empty_traces(st)
            if remove_irig:
                remove_IRIG_channel(st)
            if not st:
                counts["empty"] += 1
                status = "empty"
                continue
            if sampling_rate is not None:
                fix_sampling_rate(st, sampling_rate)
            for tr in st:
                original = tr.stats.get("vdap_original_id", tr.id)
                id_fixer(tr, network=network)
                pairs.append((original, tr.id))

            proposed = {}
            for original, fixed in pairs:
                if original in proposed and proposed[original] != fixed:
                    raise ValueError(f"Conflicting mappings for {original}")
                proposed[original] = fixed
                if original in successful_map and successful_map[original] != fixed:
                    raise ValueError(f"Mapping changed between files for {original}")

            st.sort(keys=["station", "channel"])
            st.merge(method=1, fill_value=0)
            start = min(tr.stats.starttime for tr in st)
            end = max(tr.stats.endtime for tr in st)
            st.trim(start, end, pad=True, fill_value=0)
            t = min(tr.stats.starttime for tr in st)
            name = (f"{t.year:04d}-{t.month:02d}-{t.day:02d}-"
                    f"{t.hour:02d}{t.minute:02d}-{t.second:02d}M."
                    f"{str(database).ljust(5, '_')}_{len(st):03d}")
            target = outroot / f"{t.year:04d}" / f"{t.month:02d}" / name
            key = target.resolve()
            if key in claimed_paths:
                raise ValueError(f"Output filename collision with {claimed_paths[key]}: {target}")
            claimed_paths[key] = str(dmx)
            existed = target.exists()
            stage = "write"
            wavpath = Path(write_wavfile(
                st, outroot, database,
                skip_existing=not overwrite,
                sort_traces=False, merge_traces=False, align_traces=False,
            ))
            if wavpath.resolve() != key or not wavpath.is_file():
                raise OSError(f"Writer did not produce expected file: {target}")
            stage = "verify"
            # Index what is on disk, including when an existing file was reused.
            saved = read(str(wavpath), headonly=True)
            if not saved:
                raise OSError(f"Saved waveform has no traces: {wavpath}")
            expected_ids = {tr.id for tr in st}
            if {tr.id for tr in saved} != expected_ids:
                raise ValueError("Saved IDs differ from proposed IDs; regenerate with overwrite=True")
            expected_rates = {(tr.id, tr.stats.sampling_rate) for tr in st}
            if {(tr.id, tr.stats.sampling_rate) for tr in saved} != expected_rates:
                raise ValueError("Saved sample rates differ; regenerate with overwrite=True")
            status = "existing" if existed and not overwrite else "written"
            index_rows.append(dict(
                event_id=event_id, dmx_file=str(dmx), wav_file=str(wavpath),
                starttime=min(tr.stats.starttime for tr in saved).isoformat(),
                endtime=max(tr.stats.endtime for tr in saved).isoformat(),
                stations=",".join(sorted({tr.stats.station for tr in saved})),
                channels=",".join(sorted({tr.stats.channel for tr in saved})),
                ntraces=len(saved), status=status,
            ))
            successful_map.update(proposed)
            counts[status] += 1
        except Exception as exc:
            counts["failed"] += 1
            failures.append(dict(dmx_file=str(dmx), event_id=event_id,
                                 stage=stage, error=f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"Failed at {stage}: {dmx} | {exc}")
            if not continue_on_error:
                raise
        finally:
            provenance_rows.extend(dict(
                dmx_file=str(dmx), event_id=event_id, original_id=old,
                fixed_id=new, status=status,
            ) for old, new in pairs)

    index = pd.DataFrame(index_rows, columns=index_columns)
    mapping = pd.DataFrame(sorted(successful_map.items()),
                           columns=["original_id", "fixed_id"])
    provenance = pd.DataFrame(provenance_rows, columns=provenance_columns)
    failure_table = pd.DataFrame(failures, columns=["dmx_file", "event_id", "stage", "error"])
    for table, path in [(index, out_waveform_index), (mapping, out_trace_id_map),
                        (failure_table, out_failures), (provenance, out_trace_id_provenance)]:
        if path is not None:
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
    if out_trace_id_map_tex is not None:
        path = Path(out_trace_id_map_tex).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_latex(path, index=False, escape=True,
                         caption="Verified output DMX trace ID mappings.",
                         label="tab:trace_id_mapping")
    return dict(waveform_index=index, trace_id_map=mapping,
                trace_id_provenance=provenance, failures=failure_table, summary=counts)

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
    """Parse the supported fixed-column HYPO71 summary dialect.

    Normalize hour/minute/second overflow by arithmetic from midnight.
    Blank optional quality fields are allowed; nonnumeric quality text is
    retained in n_ass_raw/time_residual_raw without rejecting the origin.
    Return UTCDateTime origin_time and the existing VDAP numeric field names.
    Unsupported/malformed origin records return None (optionally logged).
    """
    try:
        year = int(line[0:2]); month = int(line[2:4]); day = int(line[4:6])
        year += 1900 if year >= 70 else 2000
        hour = int(line[7:9].strip() or 0)
        minute = int(line[9:11].strip() or 0)
        seconds = float(line[12:17].strip() or 0)
        if hour < 0 or minute < 0 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid negative/nonfinite clock field')
        origin_time = UTCDateTime(year, month, day) + hour*3600 + minute*60 + seconds
        lat_deg = int(line[17:20]); lat_hem = line[20].upper()
        lat_min = float(line[21:26])
        lon_deg = int(line[27:30]); lon_hem = line[30].upper()
        lon_min = float(line[31:36])
        if lat_hem not in 'NS' or lon_hem not in 'EW':
            raise ValueError('Missing/invalid hemisphere')
        if not (0 <= lat_min < 60 and 0 <= lon_min < 60):
            raise ValueError('Invalid coordinate minutes')
        latitude = dm_to_decimal(lat_deg, lat_min, lat_hem)
        longitude = dm_to_decimal(lon_deg, lon_min, lon_hem)
        depth = float(line[37:43]); magnitude = float(line[44:50])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError('Coordinates out of range')
        if not all(math.isfinite(v) for v in (depth, magnitude)):
            raise ValueError('Nonfinite depth/magnitude')
        nraw = line[51:53].strip(); rraw = line[62:].strip()
        def optional(text, converter):
            try:
                value = converter(text) if text else None
                return value if value is None or math.isfinite(value) else None
            except ValueError:
                return None
        return dict(origin_time=origin_time, latitude=latitude, longitude=longitude,
                    depth_km=depth, magnitude=magnitude,
                    n_ass=optional(nraw, int), time_residual=optional(rraw, float),
                    n_ass_raw=nraw or None, time_residual_raw=rraw or None)
    except (ValueError, IndexError, TypeError) as exc:
        if verbose:
            print(f'Failed to parse HYPO71 line: {line.rstrip()} | {exc}')
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
            if event_data["n_ass"] is not None:
                origin.comments.append(Comment(text=f"n_ass: {event_data['n_ass']}"))
            if event_data["time_residual"] is not None:
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

def read_dmx_file(
    dmx_file: str | Path,
    fix: bool = True,
    defaultnet: str = "",
    verbose: bool = False,
) -> Stream:
    """
    Read a legacy VDAP DMX waveform file.

    If fix=True, fill missing network codes with defaultnet and convert
    samples to float before subtracting the legacy 2048-count offset.

    Read failures are reported and return an empty Stream. Correction
    errors propagate to the caller; partially corrected data are not returned.

    Station and channel ID conversion must be applied separately.
    """
    dmx_file = Path(dmx_file)

    if verbose:
        print(f"Reading {dmx_file}")

    try:
        st = read(str(dmx_file))
    except Exception as exc:
        print(f"Failed to read DMX file: {dmx_file} | {exc}")
        return Stream()

    for tr in st:
        tr.stats.vdap_original_id = tr.id

    if fix:
        for tr in st:
            network = getattr(tr.stats, "network", None)
            if network is None or network.strip().lower() in ("", "unk"):
                tr.stats.network = defaultnet

            tr.data = tr.data.astype(float) - 2048.0

    if verbose:
        print(f"Read successful: {len(st)} trace(s)")

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

# ============================================================================
# Helpers
# ============================================================================

def remove_IRIG_channel(st):
    to_remove = [tr for tr in st if (tr.stats.station or "").upper().startswith("IRIG")]
    for tr in to_remove:
        st.remove(tr)


def fix_sampling_rate(st, fs):
    for tr in st:
        tr.stats.sampling_rate = fs

def discover_dmx_files(rawtop, pattern):
    return sorted(Path(rawtop).glob(pattern))


def extract_event_id(dmx_path: Path):
    """
    Extract YYMMDDNN from YYMMDDNN.DMX
    """
    m = re.match(r"(\d{6}[0-9A-Z]{2})\.DMX$", dmx_path.name.upper())
    return m.group(1) if m else None


# ============================================================================
# Station0 to Inventory
# ============================================================================


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




def parse_vdap_phase_line(line, *, network='', dialect='auto'):
    """Decode one single-event VDAP phase record into pick dictionaries.

    Supports compact 12-digit timestamps (YYMMDDHHMMSS.fraction) in tokenized
    records and the documented fixed-column VDAP P/S dialect. No multi-event
    delimiter interpretation, time clustering or outlier filtering occurs.
    Placeholder XXXX/XXX records return []; malformed records raise ValueError.
    Legacy station suffixes are decoded by fix_legacy_vdap_id (including L).
    S fields contain seconds within the minute, not a P-to-S duration.
    """
    if dialect not in ('auto', 'tokenized', 'fixed'):
        raise ValueError("dialect must be 'auto', 'tokenized', or 'fixed'")
    text = line.rstrip('\r\n')
    if not text.strip():
        return []
    tokens = text.split()
    if tokens[0].upper() in ('XXX', 'XXXX'):
        return []

    def timestamp(value):
        match = re.fullmatch(r'(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\.\d+)?', value)
        if not match:
            raise ValueError(f'Invalid VDAP timestamp: {value!r}')
        yy, mo, day, hour, minute, sec = map(int, match.groups()[:6])
        year = yy + (1900 if yy >= 70 else 2000)
        return UTCDateTime(year, mo, day) + hour*3600 + minute*60 + sec + float(match[7] or 0)

    def metadata(station):
        if not re.fullmatch(r'[A-Z0-9]{2,6}', station):
            raise ValueError(f'Invalid station identifier: {station!r}')
        trace = Trace(); trace.stats.station = station
        fix_legacy_vdap_id(trace, network=network or None)
        return dict(station=trace.stats.station, channel=trace.stats.channel, seed_id=trace.id)

    def pick(meta, phase, time, code='', weight=None):
        onset = next((c for c in code[:1] if c in 'IE'), None)
        motion = next((c for c in code if c in 'UD'), None)
        if weight is None and code and code[-1].isdigit():
            weight = int(code[-1])
        return dict(**meta, phase=phase, time=time, onset=onset,
                    first_motion=motion, weight=weight)

    quality_pattern = r'(?P<onset>[IE])?(?P<phase>[PS])?(?P<motion>[UD])?(?P<weight>\d)?'
    compact_quality = re.fullmatch(quality_pattern, tokens[1].upper()) if len(tokens) == 3 else None
    token_like = ((len(tokens) == 4 and tokens[1].upper() in ('P', 'S')) or
                  (compact_quality is not None and compact_quality['phase'] is not None))
    if dialect == 'tokenized' or (dialect == 'auto' and token_like):
        if len(tokens) == 3:
            station, code, stamp = tokens
            quality = re.fullmatch(quality_pattern, code.upper())
            phase = quality['phase'] if quality else None
        elif len(tokens) == 4:
            station, phase, code, stamp = tokens
            phase = phase.upper()
            quality = re.fullmatch(quality_pattern, code.upper())
        else:
            raise ValueError('Expected station [phase] quality timestamp')
        if not quality or not any(quality.groups()) or phase not in ('P', 'S'):
            raise ValueError('Unsupported phase/quality fields')
        if quality['phase'] and quality['phase'] != phase:
            raise ValueError('Phase/quality mismatch')
        return [pick(metadata(station.upper()), phase, timestamp(stamp),
                     code.upper(), int(quality['weight']) if quality['weight'] else None)]

    if len(text) < 23:
        raise ValueError('Short fixed-column phase record')
    meta = metadata(text[:4].strip().upper())
    pcode = text[4:8].upper().ljust(4)
    start = 9 if text[8:9] == ' ' else 8
    stamp = text[start:start+15].strip().replace(' ', '0')
    time = timestamp(stamp)
    picks = []
    if pcode[1:2] == 'P':
        picks.append(pick(meta, 'P', time, pcode))
    spos = [i for i, c in enumerate(text) if c.upper() == 'S' and 35 <= i <= 40]
    if len(spos) > 1:
        raise ValueError('Ambiguous S-phase fields')
    if spos:
        pos = spos[0]
        value = float(text[pos-7:pos-1].strip())
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid S seconds')
        minute_start = UTCDateTime(time.year, time.month, time.day, time.hour, time.minute)
        stime = minute_start + value
        while stime < time:
            stime += 60
        scode = text[max(0, pos-1):pos+3].upper()
        picks.append(pick(meta, 'S', stime, scode))
    if not picks:
        raise ValueError('No supported phase fields')
    return picks


def read_vdap_pha_file(path, *, network='', dialect='auto', strict=True,
                       return_diagnostics=False):
    """Read one event's VDAP PHA file; retain file/line provenance.

    Multi-event 10/100 separators are rejected, never merged into one event.
    With strict=False, malformed lines are returned as structured diagnostics;
    I/O errors always raise. No deduplication or outlier rejection is applied.
    Return picks, or (picks, diagnostics) if return_diagnostics=True.
    """
    if dialect not in ('auto', 'tokenized', 'fixed'):
        raise ValueError('Invalid dialect')
    path = Path(path)
    picks, diagnostics = [], []
    with path.open('r', encoding='utf-8', errors='strict') as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.rstrip('\r\n')
            if line.strip() in ('10', '100'):
                raise ValueError(f'{path}:{lineno}: multi-event delimiters are unsupported')
            try:
                parsed = parse_vdap_phase_line(line, network=network, dialect=dialect)
            except (ValueError, IndexError) as exc:
                diagnostic = dict(source_file=str(path), source_line=lineno,
                                  raw_line=line, error=str(exc))
                if strict:
                    raise ValueError(f'{path}:{lineno}: {exc}') from exc
                diagnostics.append(diagnostic)
                continue
            for item in parsed:
                item.update(raw_line=line, raw_lineno=lineno, raw_file=path.name)
                picks.append(item)
    return (picks, diagnostics) if return_diagnostics else picks


def detect_pretrigger_window(trace_or_stream, *, settings=None):
    """Propose quiet prefixes and onset support in an already triggered record.

    Discoverable VDAP entry point for the generic implementation in
    ``flovopy.processing.detection.detect_pretrigger_window``. Settings and
    results are identical: no independent VDAP algorithm/defaults are applied.
    Use ``PretriggerSettings`` from ``flovopy.processing.detection`` to change
    the initial assumed noise duration, guard or station support. In particular,
    ``network_min_stations=1`` explicitly enables single-station analysis.

    This estimates quiet waveform prefixes, not the recorder's trigger time or
    configured buffer length. All proposals need review, especially for emergent
    rockfall/PDC signals. No files are written and inputs are not modified.
    """
    from flovopy.processing.detection import detect_pretrigger_window as detect
    return detect(trace_or_stream, settings=settings)
