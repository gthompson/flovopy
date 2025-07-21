# flovopy/core/sds.py
import os
import glob
import numpy as np
from obspy import read, Stream, Trace
from obspy.core.utcdatetime import UTCDateTime
import obspy.clients.filesystem.sds
from flovopy.core.preprocessing import remove_empty_traces #, fix_trace_id, _can_write_to_miniseed_and_read_back
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
from itertools import groupby
from operator import itemgetter
from flovopy.core.miniseed_io import smart_merge, read_mseed, write_mseed, decimate
from math import ceil
#from flovopy.core.trace_utils import ensure_float32
import re

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
                speed=2, verbose=True, progress=False, max_sampling_rate=250.0):
            """
            Read data from the SDS archive into the internal stream.
            """
            if not trace_ids:
                trace_ids = self._get_nonempty_traceids(startt, endt, skip_low_rate_channels, speed=speed)

            st = Stream()
            #trace_iter = tqdm(trace_ids, desc="Reading traces") if progress else trace_ids

            #for trace_id in trace_iter:
            
            for trace_id in trace_ids:
                net, sta, loc, chan = trace_id.split('.')
                if chan.startswith('L') and skip_low_rate_channels:
                    continue

                print(f'\n**************\nReading SDS for {trace_id}: {startt}-{endt}')
                try:
                    if speed == 1:
                        sdsfiles = self.client._get_filenames(net, sta, loc, chan, startt, endt)
                        if verbose:
                            print(f'Found {len(sdsfiles)} matching SDS files')
                        for sdsfile in sdsfiles:
                            if os.path.isfile(sdsfile):
                                try:
                                    if verbose:
                                        print(f'Reading {sdsfile}')
                                    traces = read_mseed(sdsfile, starttime=startt, endtime=endt)
                                    if verbose:
                                        print(traces)
                                    for tr in traces:
                                        st += tr
                                except Exception as e:
                                    if verbose:
                                        print(f"Failed to read (v1) {sdsfile}: {e}")
                    elif speed == 2:
                        traces = self.client.get_waveforms(net, sta, loc, chan, startt, endt, merge=-1)
                        for tr in traces:
                            decimate(tr, max_sampling_rate=max_sampling_rate)
                        traces, report = smart_merge(traces)
                        if verbose:
                            print(traces)
                        for tr in traces:
                            st += tr

                except Exception as e:
                    if verbose:
                        print(f"Failed to read (v2) {trace_id}: {e}")

            st = remove_empty_traces(st)
            print(f'\nAfter remove blank traces:\n{st}')

            if len(st):
                st.trim(startt, endt)
                st, report = smart_merge(st)
                print(f'\nAfter final smart_merge:\n{st}')

            self.stream = st
            return 0 if len(st) else 1

    def write(self, force_overwrite=False, fallback_to_indexed=True, debug=False):
        """
        Write internal stream to SDS archive, marking gaps with 0.0 and preserving metadata.

        Parameters:
        - force_overwrite (bool): Overwrite existing files if True.
        - fallback_to_indexed (bool): If True, write .01, .02 files on merge conflict. If False, raise error.
        - debug (bool): Print debug messages if True.

        Returns:
        - bool: True if all writes succeed, False otherwise.
        """
        if isinstance(self.stream, Trace):
            self.stream = Stream(traces=[self.stream])

        write_status = {}  # Per-trace success flag

        # Setup subdirectories
        tempdir = os.path.join(self.basedir, 'temporarily_move_while_merging')
        unmergeddir = os.path.join(self.basedir, 'unable_to_merge')
        obsoletedir = os.path.join(self.basedir, 'obsolete')
        unwrittendir = os.path.join(self.basedir, 'failed_to_write_to_sds')
        multitracedir = os.path.join(self.basedir, 'multitrace')

        for d in [tempdir, unmergeddir, obsoletedir, unwrittendir, multitracedir]:
            os.makedirs(d, exist_ok=True)

        if debug:
            print('> SDSobj.write()')

        for tr_unsplit in self.stream:
            split_traces = split_trace_at_midnight(tr_unsplit)

            for tr in split_traces:
                trace_id = tr.id

                sdsfile = self.client._get_filename(
                    tr.stats.network, tr.stats.station,
                    tr.stats.location, tr.stats.channel,
                    tr.stats.starttime, 'D'
                )

                basename = os.path.basename(sdsfile)
                tempfile = os.path.join(tempdir, basename)
                unmergedfile = os.path.join(unmergeddir, basename)
                obsoletefile = os.path.join(obsoletedir, basename)
                unwrittenfile = os.path.join(unwrittendir, basename)
                multitracefile = os.path.join(multitracedir, basename)

                os.makedirs(os.path.dirname(sdsfile), exist_ok=True)

                if debug:
                    print(f'- Attempting to write {trace_id} to {sdsfile}')

                if force_overwrite or not os.path.isfile(sdsfile):
                    ok = write_mseed(tr, sdsfile, overwrite_ok=True)
                    write_status[trace_id] = ok
                    if debug:
                        if ok:
                            print(f"- ✔ New file written: {sdsfile}")
                        else:
                            print(f"- ✘ Failed to write {sdsfile} even in overwrite mode")                   

                else: # output file already exists
                    try:
                        existing = read(sdsfile)
                    except Exception as e:
                        print(f"- Error reading existing file {sdsfile}: {e}")
                        existing = Stream()

                    shutil.copy2(sdsfile, tempfile)

                    merged, merge_info = smart_merge(Stream([tr]) + existing)

                    if merge_info["status"] == "identical":
                        if debug:
                            print("- Duplicate of existing SDS file — skipping")
                        safe_remove(tempfile)
                        write_status[trace_id] = True
                        continue

                    elif merge_info["status"] == "conflict":
                        if debug:
                            print(f"- ✘ Cannot merge {trace_id} — conflict found. Writing to {unmergedfile}")
                        write_mseed(tr, unmergedfile)
                        safe_remove(tempfile)
                        write_status[trace_id] = False
                        continue

                    elif merge_info["status"] == "ok":
                        if len(merged) == 1:
                            ok = write_mseed(merged[0], sdsfile, overwrite_ok=True)
                            write_status[trace_id] = ok
                            if ok:
                                shutil.move(tempfile, obsoletefile)
                                if debug:
                                    print(f"- ✔ Merged and wrote: {sdsfile}")
                            else:
                                print(f"- ✘ Failed to write merged trace to {sdsfile}")
                        else:
                            print(f"- ✘ Merge produced multiple traces for {trace_id}, saving to {multitracefile}")
                            write_mseed(merged, multitracefile)
                            safe_remove(tempfile)
                            write_status[trace_id] = False
                    else:
                        print(f"- ✘ Unexpected merge status: {merge_info['status']}")
                        write_status[trace_id] = False


        if debug:
            print('< SDSobj.write()>')

        return all(write_status.values())


    def _get_nonempty_traceids(self, startday, endday=None, skip_low_rate_channels=True, speed=1):
        """
        Get a list of trace IDs that have data between two dates.

        Parameters:
        - startday (UTCDateTime)
        - endday (UTCDateTime): Optional. Defaults to startday + 1 day.
        - skip_low_rate_channels (bool)
        - speed (int): If 1, confirm using has_data(); if >1, trust get_all_nslc().

        Returns:
        - list: Sorted list of trace IDs.
        """
        endday = endday or startday + 86400
        trace_ids = set()
        thisday = startday

        while thisday < endday:
            try:
                for net, sta, loc, chan in self.client.get_all_nslc(sds_type='D', datetime=thisday):
                    if chan.startswith('L') and skip_low_rate_channels:
                        continue
                    if speed == 1:
                        if not self.client.has_data(net, sta, loc, chan):
                            continue
                    trace_ids.add(f"{net}.{sta}.{loc}.{chan}")
            except Exception as e:
                print(f"Error on {thisday.date()}: {e}")
            thisday += 86400

        return sorted(trace_ids)

    def find_missing_days(self, stime, etime, net, sta=None):
        """
        Return list of days with no data for a given network (or station).

        Parameters:
        - stime, etime (UTCDateTime): Start and end time range.
        - net (str): Network code.
        - sta (str): Optional station code. If None, checks any station in net.

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
                f"{net}*.{year}.{jday}"
            )
            existingfiles = glob.glob(pattern)
            if not existingfiles:
                missing_days.append(dayt)
            dayt += 86400

        return missing_days

    def get_percent_availability(self, startday, endday, skip_low_rate_channels=True,
                                trace_ids=None, speed=3, verbose=False, progress=True):
        """
        Compute data availability percentage for each trace ID per day.

        Parameters:
        - startday, endday (UTCDateTime): Date range.
        - skip_low_rate_channels (bool): Skip L-prefixed channels.
        - trace_ids (list): Optional list of SEED IDs.
        - speed (int): Mode (1 = count non-NaN, 2 = use .npts, 3 = SDS client).
        - verbose (bool): Print errors.
        - progress (bool): Show progress bar.

        Returns:
        - (DataFrame, list): Availability DataFrame and SEED IDs.
        """
        from tqdm import tqdm
        import pandas as pd

        trace_ids = trace_ids or self._get_nonempty_traceids(startday, endday, skip_low_rate_channels, speed=speed)
        lod = []

        day_list = []
        t = startday
        while t < endday:
            day_list.append(t)
            t += 86400

        day_iter = tqdm(day_list, desc="Computing availability") if progress else day_list

        for thisday in day_iter:
            row = {'date': thisday.date()}
            for trace_id in trace_ids:
                net, sta, loc, chan = trace_id.split('.')
                percent = 0
                try:
                    if speed < 3:
                        sdsfile = self.client._get_filename(net, sta, loc, chan, thisday)
                        if os.path.isfile(sdsfile):
                            st = read(sdsfile)
                            if len(st) > 0:
                                st = smart_merge(st)
                                if len(st)==1:
                                    tr = st[0]
                                    expected = tr.stats.sampling_rate * 86400
                                    npts = np.count_nonzero(~np.isnan(tr.data)) if speed == 1 else tr.stats.npts
                                    percent = min(100.0, 100 * npts / expected) if expected > 0 else 0
                                else:
                                    percent = np.nan
                    else:
                        percent = self.client.get_availability_percentage(net, sta, loc, chan,
                                                                        thisday, thisday + 86400)[0]
                except Exception as e:
                    if verbose:
                        print(f"Error for {trace_id} on {thisday.date()}: {e}")
                    percent = 0
                row[trace_id] = percent
            lod.append(row)

        df = pd.DataFrame(lod)
        df['date'] = pd.to_datetime(df['date'])
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


def merge_two_sds_archives(source1_sds_dir, source2_sds_dir, dest_sds_dir):
    """
    Merge two SDS archives into a destination SDS archive.

    - If overlapping files are found, waveform data are merged using `smart_merge()`.
    - Merged files are written only if the result differs from what's already in `dest_sds_dir`.
    - Unresolved conflicts are logged using pandas to a CSV file.

    Parameters
    ----------
    source1_sds_dir : str
        Path to the first SDS archive.
    source2_sds_dir : str
        Path to the second SDS archive.
    dest_sds_dir : str
        Path to the destination SDS archive (created if it doesn't exist).
    """
    final_sds = SDSobj(dest_sds_dir)
    conflicts_resolved = 0
    conflicts_remaining = 0
    unresolved_conflicts = []

    for source in [source1_sds_dir, source2_sds_dir]:
        for root, _, files in os.walk(source):
            if not is_valid_sds_dir(root):
                continue

            for file in files:
                if not is_valid_sds_filename(file):
                    continue

                source_file = os.path.join(root, file)
                rel_path = os.path.relpath(source_file, source)
                dest_file = os.path.join(dest_sds_dir, rel_path)

                if os.path.exists(dest_file):
                    try:
                        st1 = read_mseed(dest_file)
                        st2 = read_mseed(source_file)
                        merged, report = smart_merge(st1 + st2)

                        if report["status"] == "ok":
                            # Only write if merged result differs from original
                            if not _streams_equal(merged, st1):
                                final_sds.write(merged)
                                conflicts_resolved += 1
                        else:
                            conflicts_remaining += 1
                            unresolved_conflicts.append({
                                "relative_path": rel_path,
                                "source_file": source_file,
                                "dest_file": dest_file,
                                "reason": report.get("reason", "merge failed")
                            })
                    except Exception as e:
                        conflicts_remaining += 1
                        unresolved_conflicts.append({
                            "relative_path": rel_path,
                            "source_file": source_file,
                            "dest_file": dest_file,
                            "reason": str(e)
                        })
                else:
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    shutil.copy2(source_file, dest_file)

    # Save unresolved conflicts to CSV using pandas
    if unresolved_conflicts:
        log_path = os.path.join(dest_sds_dir, "conflicts_unresolved.csv")
        df_conflicts = pd.DataFrame(unresolved_conflicts)
        df_conflicts.to_csv(log_path, index=False)
        print(f"⚠️ Logged {len(unresolved_conflicts)} unresolved conflicts to {log_path}")

    print("✅ SDS merge complete.")
    print(f"✔️ Conflicts resolved and merged: {conflicts_resolved}")
    print(f"⚠️ Conflicts remaining:           {conflicts_remaining}")


def _streams_equal(st1: Stream, st2: Stream) -> bool:
    """
    Determine whether two ObsPy Streams are effectively equal.

    Uses:
    - Number of traces
    - Trace IDs
    - Start/end times
    - Number of samples
    """
    if len(st1) != len(st2):
        return False

    for tr1, tr2 in zip(st1, st2):
        if tr1.id != tr2.id:
            return False
        if tr1.stats.starttime != tr2.stats.starttime:
            return False
        if tr1.stats.endtime != tr2.stats.endtime:
            return False
        if len(tr1.data) != len(tr2.data):
            return False

    return True

def merge_multiple_sds_archives(source_sds_dirs, dest_sds_dir):
    """
    Merge multiple SDS archives into a single destination SDS archive.

    This function uses `merge_two_sds_archives()` repeatedly to combine each source
    archive into the growing destination archive.

    Parameters
    ----------
    source_sds_dirs : list of str
        List of SDS archive directories to merge.
    dest_sds_dir : str
        Path to destination SDS archive. Created if it doesn't exist.
    """
    if not source_sds_dirs:
        print("❌ No source directories provided.")
        return

    # Step 1: Copy the first archive directly into the destination (if different)
    first = source_sds_dirs[0]
    if os.path.abspath(first) != os.path.abspath(dest_sds_dir):
        print(f"📂 Copying initial archive from {first} to {dest_sds_dir}...")
        shutil.copytree(first, dest_sds_dir, dirs_exist_ok=True)

    # Step 2: Merge the rest
    for src in source_sds_dirs[1:]:
        print(f"\n🔄 Merging {src} into {dest_sds_dir}...")
        merge_two_sds_archives(src, dest_sds_dir, dest_sds_dir)

    print("\n✅ All SDS archives merged successfully.")