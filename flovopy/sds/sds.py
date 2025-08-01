# flovopy/core/sds.py
import os
import glob
import numpy as np
from obspy import read, Stream, Trace
from obspy.core.utcdatetime import UTCDateTime
import obspy.clients.filesystem.sds
from flovopy.core.trace_utils import remove_empty_traces #, fix_trace_id, _can_write_to_miniseed_and_read_back
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
from itertools import groupby
from operator import itemgetter
from flovopy.core.miniseed_io import smart_merge, read_mseed, write_mseed, downsample_stream_to_common_rate #, unmask_gaps
from math import ceil
from tqdm import tqdm
#from flovopy.core.trace_utils import ensure_float32
import re
import gc
import traceback
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed


def _compute_percent(args):
    self, trace_id, day, speed, merge_strategy, verbose = args
    net, sta, loc, chan = trace_id.split('.')
    percent = 0

    try:
        if speed < 3:
            sdsfile = self.client._get_filename(net, sta, loc, chan, day)
            if sdsfile and os.path.exists(sdsfile) and os.path.getsize(sdsfile) > 0:
                st = read(sdsfile)
                if len(st) > 0:
                    st = smart_merge(st, strategy=merge_strategy)
                    tr = st[0]
                    expected = tr.stats.sampling_rate * 86400
                    npts = np.count_nonzero(~np.isnan(tr.data)) if speed == 1 else tr.stats.npts
                    percent = min(100.0, 100 * npts / expected) if expected > 0 else 0
        else:
            percent = self.client.get_availability_percentage(
                net, sta, loc, chan, day, day + 86400)[0]
    except Exception as e:
        if verbose:
            print(f"Error for {trace_id} on {day.date()}: {e}")
        percent = 0

    return (day.date, trace_id, percent)


def safe_remove(filepath):
    """Remove file if it exists."""
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Failed to remove file {filepath}: {e}")

def split_trace_at_midnight(tr):
    """
    Split a Trace at UTC midnight boundaries. Return list of Trace objects.
    """
    out = []
    t1 = tr.stats.starttime
    t2 = tr.stats.endtime

    while t1 < t2:
        next_midnight = UTCDateTime(t1.date) + 86400
        trim_end = min(t2, next_midnight)
        tr_piece = tr.copy().trim(starttime=t1, endtime=trim_end, nearest_sample=True)
        out.append(tr_piece)
        t1 = trim_end
    return out

class SDSobj:
    """
    A class to manage an SDS (SeisComP Data Structure) archive.
    Allows reading, writing, checking availability, and plotting data from an SDS archive.
    """

    def __init__(self, basedir, sds_type='D', format='MSEED', streamobj=None, metadata=None):
        """
        Initialize SDSobj.

        Parameters:
        - basedir (str): Root directory of the SDS archive.
        - sds_type (str): SDS file type (default 'D' for daily files).
        - format (str): File format (default 'MSEED').
        - streamobj (obspy Stream): Optional preloaded stream object.
        """
        os.makedirs(basedir, exist_ok=True)
        self.client = obspy.clients.filesystem.sds.Client(basedir, sds_type=sds_type, format=format)
        self.stream = streamobj or Stream()
        self.basedir = basedir
        self.metadata = metadata # for supporting a dataframe of allowable SEED ids (from same Excel spreadsheet used to generate StationXML)

    def read(self, startt, endt, skip_low_rate_channels=True, trace_ids=None,
            speed=2, verbose=True, progress=False, max_sampling_rate=250.0, merge_strategy='obspy'):
        """
        Read data from the SDS archive into the internal stream.
        """
        if trace_ids is None:
            trace_ids = self._get_nonempty_traceids(
                startt, endt, skip_low_rate_channels, speed=speed
            )

        st = Stream()
        trace_iter = tqdm(trace_ids, desc="Reading traces") if progress else trace_ids

        for trace_id in trace_iter:
            net, sta, loc, chan = trace_id.split('.')
            if skip_low_rate_channels and chan.startswith('L'):
                continue

            if verbose:
                print(f"\n**************\nReading SDS for {trace_id}: {startt} – {endt}")

            try:
                if speed == 1:
                    sdsfiles = self.client._get_filenames(net, sta, loc, chan, startt, endt)
                    if verbose:
                        print(f"Found {len(sdsfiles)} matching SDS files")

                    for sdsfile in sdsfiles:
                        if os.path.isfile(sdsfile):
                            try:
                                if verbose:
                                    print(f"Reading {sdsfile}")
                                traces = read_mseed(sdsfile, starttime=startt, endtime=endt)
                                st += traces
                            except Exception as e:
                                if verbose:
                                    print(f"✘ Failed to read (v1) {sdsfile}: {e}")

                elif speed == 2:
                    traces = self.client.get_waveforms(net, sta, loc, chan, startt, endt, merge=-1)

                    ds_stream = downsample_stream_to_common_rate(traces, max_sampling_rate=max_sampling_rate)
                    smart_merge(ds_stream, strategy=merge_strategy)
                    st += ds_stream

            except Exception as e:
                if verbose:
                    print(f"✘ Failed to read (v2) {trace_id}: {e}")

        remove_empty_traces(st, inplace=True)
        if verbose:
            print(f"\nAfter removing empty traces:\n{st}")

        if len(st):
            st.trim(startt, endt)
            smart_merge(st, strategy=merge_strategy)
            if verbose:
                print(f"\nAfter final smart_merge:\n{st}")

        self.stream = st
        gc.collect()
        return 0 if len(st) else 1

    def write(self, fill_value=0.0, debug=False, merge_strategy='obspy'):
        """
        Writes a Stream or Trace to the SDS archive.

        Parameters
        ----------

        fill_value : float, optional
            Value to fill masked gap regions before writing.
        debug : bool, optional
            Print detailed debug output.

        Returns
        -------
        dict
            Results dictionary keyed by Trace ID with status, reason, and path.
        """

        results = {}
        all_ok = True
        stream = Stream([self.stream]) if isinstance(self.stream, Trace) else self.stream

        for tr in stream:
            trace_id = tr.id
            sdsfile = self.client._get_filename(
                tr.stats.network,
                tr.stats.station,
                tr.stats.location,
                tr.stats.channel,
                tr.stats.starttime,
                'D'
            )

            os.makedirs(os.path.dirname(sdsfile), exist_ok=True)

            if debug:
                print(f"→ Attempting to write: {trace_id} → {sdsfile}")

            try:
                if os.path.exists(sdsfile):
                    # Try merging with existing file
                    existing = read_mseed(sdsfile)
                    merged = existing + Stream([tr])
                    report = smart_merge(merged, debug=debug, strategy=merge_strategy)

                    if report['status'] == 'ok' and len(merged) == 1:
                        success = write_mseed(merged[0], sdsfile, fill_value=fill_value)
                        results[trace_id] = {
                            "status": "ok" if success else "exception",
                            "reason": "Merged and written" if success else "Failed to write merged stream",
                            "path": sdsfile,
                        }
                    else:
                        msg = "Merge conflict"
                        if report['status'] != 'ok':
                            msg += f" ({report['status']})"
                        if len(merged) != 1:
                            msg += f"; result has {len(merged)} traces (expected 1)"
                        results[trace_id] = {
                            "status": "conflict",
                            "reason": msg,
                            "path": sdsfile,
                        }
                        all_ok = False
                        if debug:
                            print(f"⚠️ Merge failed for {trace_id}: {msg}")
                else:
                    # file does not exist
                    success = write_mseed(tr, sdsfile, fill_value=fill_value)
                    results[trace_id] = {
                        "status": "ok" if success else "exception",
                        "reason": "Written (no existing file)" if success else "Failed to write",
                        "path": sdsfile,
                    }

            except Exception as e:
                results[trace_id] = {
                    "status": "exception",
                    "reason": str(e),
                    "path": sdsfile,
                }
                all_ok = False
                if debug:
                    print(f"✘ Exception while writing {trace_id} → {e}")

        results["all_ok"] = all_ok
        return results

    def _get_nonempty_traceids(self, startday, endday=None, skip_low_rate_channels=True, speed=1):
        import datetime
        endday = endday or startday + 86400
        trace_ids = set()
        thisday = startday

        while thisday < endday:
            print(thisday)
            try:
                # Try to get the NSLC list from the client
                nslc_list = self.client.get_all_nslc(sds_type='D', datetime=thisday)
            except Exception as e:
                #print(f"Warning: get_all_nslc() failed for {thisday} with error: {e}")
                # Fall back to manual walk if get_all_nslc() fails
                nslc_list = self._walk_sds_for_day(thisday)

            # If still no data found, just continue to next day
            if not nslc_list:
                print(f"No NSLC data found for {thisday}")
                thisday += 86400
                continue

            # Process the NSLC list to filter channels and check data presence
            for net, sta, loc, chan in nslc_list:
                if chan.startswith('L') and skip_low_rate_channels:
                    continue
                if speed == 1:
                    try:
                        if not self.client.has_data(net, sta, loc, chan):
                            continue
                    except Exception as e:
                        print(f"has_data() error for {net}.{sta}.{loc}.{chan}: {e}")
                        continue
                trace_ids.add(f"{net}.{sta}.{loc}.{chan}")

            thisday += 86400

        return sorted(trace_ids)


    def _walk_sds_for_day(self, day):
        """
        Scan SDS directory structure manually for the given day to build NSLC list.
        """
        base_path = Path(self.client.sds_root)
        year = day.strftime("%Y")
        jday = day.strftime("%j")  # Julian day, zero-padded 3 digits
        nslc_set = set()

        year_path = base_path / year
        if not year_path.exists():
            print(f"Missing SDS year directory: {year_path}")
            return []

        for net_dir in year_path.iterdir():
            if not net_dir.is_dir():
                continue
            for sta_dir in net_dir.iterdir():
                if not sta_dir.is_dir():
                    continue
                for chan_dir in sta_dir.iterdir():
                    if not chan_dir.is_dir() or not chan_dir.name.endswith(".D"):
                        continue
                    chan = chan_dir.name[:-2]  # Remove trailing '.D'
                    # Look for files matching pattern *.D.YEAR.JDAY
                    for file in chan_dir.glob(f"*.D.{year}.{jday}"):
                        parts = file.name.split(".")
                        if len(parts) >= 4:
                            n, s, l, c = parts[:4]
                            nslc_set.add((n, s, l, c))

        return sorted(nslc_set)


    def find_missing_days(self, stime, etime, net, sta=None):
        """
        Return list of days with no data for a given network (or station).

        Parameters:
        - stime, etime (UTCDateTime): Time range.
        - net (str): Network code.
        - sta (str): Optional station code. If None, checks all stations.

        Returns:
        - list of UTCDateTime: Days with no matching files.
        """
        missing_days = []
        dayt = stime

        while dayt < etime:
            year = f"{dayt.year:04d}"
            jday = f"{dayt.julday:03d}"
            station_glob = sta or '*'
            pattern = os.path.join(
                self.basedir, year, net, station_glob, '*.D',
                f"{net}.{station_glob}.*.*.{year}.{jday}"
            )
            existingfiles = glob.glob(pattern)
            if not existingfiles:
                missing_days.append(dayt)
            dayt += 86400

        return missing_days


    def get_percent_availability(self, startday, endday, skip_low_rate_channels=True,
                                trace_ids=None, speed=3, verbose=False,
                                progress=True, merge_strategy='obspy', max_workers=8):
        """
        Compute data availability percentage for each trace ID per day using parallel processing.
        """


        trace_ids = trace_ids or self._get_nonempty_traceids(startday, endday,
                                                            skip_low_rate_channels,
                                                            speed=speed)

        # Create list of (day, trace_id) tasks
        days = []
        t = startday
        while t < endday:
            days.append(t)
            t += 86400

        tasks = [(self, tid, day, speed, merge_strategy, verbose)
                for day in days for tid in trace_ids]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.map(_compute_percent, tasks))
            if progress:
                futures = tqdm(futures, total=len(tasks), desc="Availability")

            for result in futures:
                results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=["date", "trace_id", "percent"])
        df = df.pivot(index="date", columns="trace_id", values="percent").reset_index()
        df["date"] = pd.to_datetime(df["date"])

        return df, trace_ids


    def plot_availability(self, availabilityDF, outfile=None, figsize=(12, 8), fontsize=10, labels=None, cmap='viridis'):
        """
        Plot availability heatmap for SEED IDs across time.

        Parameters:
        - availabilityDF (DataFrame): output from get_percent_availability
        - outfile (str): optional path to save the figure
        - figsize (tuple): figure size in inches
        - fontsize (int): font size for labels
        - labels (list): optional list of trace IDs
        - cmap (str): matplotlib colormap (default 'viridis')
        """
        import matplotlib.pyplot as plt

        if availabilityDF.empty:
            print("No availability data to plot.")
            return

        Adf = availabilityDF.set_index('date').T / 100.0
        Adata = Adf.to_numpy()
        xticklabels = availabilityDF['date'].dt.strftime('%Y-%m-%d').tolist()
        yticklabels = labels or Adf.index.tolist()

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(Adata, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(xticklabels))[::max(1, len(xticklabels) // 25)])
        ax.set_xticklabels(xticklabels[::max(1, len(xticklabels) // 25)], rotation=90, fontsize=fontsize)

        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=fontsize)

        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel("SEED ID", fontsize=fontsize)
        ax.set_title("SDS Data Availability (%)", fontsize=fontsize + 2)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Availability", fontsize=fontsize)

        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=300)
            print(f"Saved availability plot to: {outfile}")

    def __str__(self):
        return f"client={self.client}, stream={self.stream}"

    def get_fullpath(self, trace):
        """
        Build the correct SDS file path for a given trace.

        Parameters:
        - trace (Trace): ObsPy Trace object

        Returns:
        - str: Correct file path
        """
        net = trace.stats.network
        sta = trace.stats.station
        loc = trace.stats.location or "--"
        if len(loc) == 1:
            loc = '0' + loc
        chan = trace.stats.channel
        year = str(trace.stats.starttime.year)
        day_of_year = str(trace.stats.starttime.julday).zfill(3)
        filename = f"{net}.{sta}.{loc}.{chan}.D.{year}.{day_of_year}"
        sds_subdir = os.path.join(year, net, sta, f"{chan}.D")
        return os.path.join(self.basedir, sds_subdir, filename)
    
    def load_metadata_from_excel(self, excel_path, sheet_name=0):
        """
        Load metadata from an Excel file into the SDSobj, including
        on/off dates and multi-channel expansion.

        Parameters
        ----------
        excel_path : str
            Path to the Excel file.
        sheet_name : str or int, optional
            Sheet name or index to load (default is first sheet).
        """
        df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype={"location": str})
        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = {'network', 'station', 'location', 'channel'}
        id_cols = {'id', 'seedid', 'traceid'}

        if not required_cols.issubset(df.columns) and not id_cols.intersection(df.columns):
            raise ValueError("Excel file must contain either full IDs ('id', 'seedid') or network/station/location/channel columns")

        # i think this turns a column headed seedid or traceid into id
        if 'id' not in df.columns:
            if id_cols.intersection(df.columns):
                df = df.rename(columns={list(id_cols.intersection(df.columns))[0]: 'id'})
            else:
                df['id'] = df.apply(
                    lambda row: f"{row['network']}.{row['station']}.{str(row['location']).zfill(2)}.{row['channel']}",
                    axis=1
                )

        # Convert ondate/offdate to UTCDateTime
        if 'ondate' in df.columns:
            df['ondate'] = pd.to_datetime(df['ondate'], errors='coerce')
            df['ondate'] = df['ondate'].apply(lambda x: UTCDateTime(x) if pd.notnull(x) else None)
        if 'offdate' in df.columns:
            df['offdate'] = pd.to_datetime(df['offdate'], errors='coerce')
            df['offdate'] = df['offdate'].apply(lambda x: UTCDateTime(x) if pd.notnull(x) else None)

        # Expand multi-character channel strings (e.g. 'EHZNEZ') into multiple 3-char channels
        expanded_rows = []
        for _, row in df.iterrows():
            chan = row["channel"]
            if isinstance(chan, str) and len(chan) > 3:
                basechan = chan[0:2]
                for ch in chan[2:]:
                    new_row = row.copy()
                    new_row["channel"] = basechan + ch
                    expanded_rows.append(new_row)

        if expanded_rows:
            expanded_df = pd.DataFrame(expanded_rows)
            df = df[df["channel"].apply(lambda x: isinstance(x, str) and len(x) == 3)]
            df = pd.concat([df, expanded_df], ignore_index=True)

        self.metadata = df
    '''
    def match_metadata(self, trace):
        """
        Match trace metadata and update trace.stats.location if needed.

        Returns
        -------
        bool : True if metadata was matched and updated, False otherwise.
        """
        if self.metadata is None or self.metadata.empty:
            return False

        net = trace.stats.network
        sta = trace.stats.station
        cha = trace.stats.channel
        start = trace.stats.starttime
        end = trace.stats.endtime

        df = self.metadata

        # Match only on net, sta, cha, and date overlap (ignore location)
        match = df[
            (df["network"] == net) &
            (df["station"] == sta) &
            (df["channel"] == cha) &
            (df["ondate"] <= start) &
            (df["offdate"] >= end - 86400)
        ]

        if not match.empty:
            loc = match.iloc[0]["location"]
            trace.stats.location = str(loc).zfill(2)
            return True
        else:
            return False
    '''
    def match_metadata(self, trace):
        """
        Match trace metadata and update trace.stats.location if needed.

        Matching strategy:
        ------------------
        1. Try to match on (network, station, channel) and on/off date overlap.
        2. If no match found, try to match on (das_serial) and on/off date overlap.

        Returns
        -------
        bool : True if metadata was matched and updated, False otherwise.
        """
        if self.metadata is None or self.metadata.empty:
            return False

        net = trace.stats.network
        sta = trace.stats.station
        cha = trace.stats.channel
        start = trace.stats.starttime
        end = trace.stats.endtime

        df = self.metadata

        # -- First attempt: match by network/station/channel/date overlap
        match = df[
            (df["network"] == net) &
            (df["station"] == sta) &
            (df["channel"] == cha) &
            (df["ondate"] <= start) &
            (df["offdate"] >= end - 86400)
        ]

        if not match.empty:
            loc = match.iloc[0]["location"]
            trace.stats.location = str(loc).zfill(2)
            return True

        # -- Second attempt: match by das_serial/date overlap (fallback)
        das_serial = trace.stats.station
        if das_serial is not None and "das_serial" in df.columns:
            match = df[
                (df["das_serial"] == das_serial) &
                (df["ondate"] <= start) &
                (df["offdate"] >= end - 86400)
            ]
            if not match.empty:
                row = match.iloc[0]
                trace.stats.network = row["network"]
                trace.stats.station = row["station"]
                trace.stats.channel = row["channel"]
                trace.stats.location = str(row["location"]).zfill(2)
                return True

        return False



    def build_file_list(self, return_failed_list_too=False, parameters=None, starttime=None, endtime=None):
        """
        Construct a list of file paths to process.
        Optionally filters by network/station/channel/location and time window.

        Parameters
        ----------
        return_failed_list_too : bool
            Whether to return a list of files that failed validation.
        parameters : dict, optional
            Dictionary of filtering parameters, e.g.:
            {
                'network': ['XA', '1R', 'AM', 'FL'],
                'station': ['SHZ1', 'ABC2'],
                'channel': ['EHZ'],
                'location': ['00', '10', '--']
            }
        starttime : UTCDateTime, optional
            Inclusive start of time window.
        endtime : UTCDateTime, optional
            Inclusive end of time window.

        Returns
        -------
        list
            Valid file paths (and optionally failed ones).
        """
        file_list = []
        failed_list = []

        for root, dirs, files in os.walk(self.basedir, topdown=True):
            dirs.sort()
            files.sort()

            if not is_valid_sds_dir(root):
                continue

            for filename in files:
                full_path = os.path.join(root, filename)

                if not is_valid_sds_filename(filename):
                    failed_list.append(full_path)
                    continue

                parsed = parse_sds_filename(filename)
                if not parsed:
                    failed_list.append(full_path)
                    continue

                network, station, location, channel, dtype, year, jday = parsed

                # Apply filtering by parameters
                if parameters:
                    if 'network' in parameters and network not in parameters['network']:
                        continue
                    if 'station' in parameters and station not in parameters['station']:
                        continue
                    if 'channel' in parameters and channel not in parameters['channel']:
                        continue
                    if 'location' in parameters and location not in parameters['location']:
                        continue

                # Apply filtering by time
                if starttime or endtime:
                    try:
                        file_date = UTCDateTime(year=int(year), julday=int(jday))
                        if (starttime and file_date < starttime) or (endtime and file_date > endtime):
                            continue
                    except Exception:
                        failed_list.append(full_path)
                        continue

                file_list.append(full_path)

        if return_failed_list_too:
            return file_list, failed_list
        else:
            return file_list





def parse_sds_dirname(dir_path):
    """
    Parse and extract components from an SDS directory path.
    Expected format: .../YEAR/NET/STA/CHAN.D

    Returns
    -------
    tuple or None
        (year, network, station, channel) if valid format, else None
    """
    parts = os.path.normpath(dir_path).split(os.sep)[-4:]
    if len(parts) != 4:
        return None

    year, net, sta, chanD = parts

    # Validate each component
    if not (year.isdigit() and len(year) == 4):
        return None
    if not re.match(r"^[A-Z0-9]{1,8}$", net):
        return None
    if not re.match(r"^[A-Z0-9]{1,8}$", sta):
        return None
    chan_match = re.match(r"^([A-Z0-9]{3})\.D$", chanD)
    if not chan_match:
        return None

    chan = chan_match.group(1)
    return year, net, sta, chan

def is_valid_sds_dir(dir_path):
    """
    Validate that a directory follows the SDS structure: YEAR/NET/STA/CHAN.D

    Returns
    -------
    bool
        True if valid SDS directory format, else False.
    """
    return parse_sds_dirname(dir_path) is not None

def parse_sds_filename(filename):
    """
    Parses an SDS-style MiniSEED filename and extracts its components.
    Assumes filenames follow: NET.STA.LOC.CHAN.TYPE.YEAR.DAY
    """
    if '/' in filename:
        filename = os.path.basename(filename)
    pattern = r"^(\w*)\.(\w*)\.(\w*)\.(\w*)\.(\w*)\.(\d{4})\.(\d{3})"
    match = re.match(pattern, filename)
    if match:
        network, station, location, channel, dtype, year, jday = match.groups()
        location = location if location else "--"
        return network, station, location, channel, dtype, year, jday
    return None

def is_valid_sds_filename(filename):
    """
    Validate SDS MiniSEED filename using parsing logic.
    Accepts only files matching NET.STA.LOC.CHAN.D.YEAR.DAY format
    with dtype == 'D' (daily MiniSEED).
    """
    parsed = parse_sds_filename(filename)
    if parsed is None:
        return False

    _, _, _, _, dtype, _, _ = parsed
    return dtype == 'D'

