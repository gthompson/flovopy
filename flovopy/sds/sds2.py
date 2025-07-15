# flovopy/core/sds.py
import os
import glob
import numpy as np
from obspy import read, Stream, Trace
from obspy.core.utcdatetime import UTCDateTime
import obspy.clients.filesystem.sds
from flovopy.core.preprocessing import remove_empty_traces, fix_trace_id, _can_write_to_miniseed_and_read_back
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
from itertools import groupby
from operator import itemgetter
from flovopy.core.merge import sanitize_trace, smart_merge, trace_equals, ensure_float32, mask_gaps, read_mseed_with_gap_masking, write_safely

'''
def write_safely(tr, mseedfile, fill_value=0, overwrite_ok=False):
    try:
        if isinstance(tr, Stream):
            marked = Stream()
            for real_tr in tr:
                marked.append(mark_gaps_and_fill(real_tr, fill_value=fill_value))
        elif isinstance(tr, Trace):
            marked = mark_gaps_and_fill(tr, fill_value=fill_value)

        if overwrite_ok:
            marked.write(mseedfile, format='MSEED')
            return True
        else:
            index = 1
            while True:
                indexed = f"{mseedfile}.{index:02d}"
                if not os.path.isfile(indexed):
                    marked.write(indexed, format='MSEED')
                    print(f"✔ Indexed file written due to merge conflict: {indexed}")
                    break
                index += 1
            return False
    except Exception as e:
        print(e)
        pklfile=mseedfile + '.pkl'
        if not os.path.isfile(pklfile):
            try:
                tr.write(pklfile, format='PICKLE')
            except:
                print(f'Could not write {pklfile}')
        return False


def detect_zero_gaps(tr, threshold_samples=100):
    """
    Annotate long zero spans in Trace.data as heuristic gaps using stats.processing,
    without modifying the data directly.
    """
    data = tr.data
    zero_mask = data == 0.0

    if not np.any(zero_mask):
        return tr

    indices = np.where(zero_mask)[0]
    groups = []
    for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        if len(group) >= threshold_samples:
            groups.append((group[0], group[-1]))

    for start_idx, end_idx in groups:
        start_time = tr.stats.starttime + (start_idx / tr.stats.sampling_rate)
        end_time = tr.stats.starttime + (end_idx / tr.stats.sampling_rate)
        note = f"Heuristic gap: {start_time} to {end_time} (zero padding)"
        if note not in tr.stats.processing:
            tr.stats.processing.append(note)

    return tr

'''
def safe_remove(filepath):
    """Remove file if it exists."""
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Failed to remove file {filepath}: {e}")
'''

def ensure_float32(tr):
    """Convert trace data to float32 if not already."""
    if not np.issubdtype(tr.data.dtype, np.floating) or tr.data.dtype != np.float32:
        tr.data = tr.data.astype(np.float32)
'''
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
'''

def mark_gaps_and_fill(trace, fill_value=0.0):
    trace = trace.copy()
    
    # Ensure float dtype to avoid merge fill errors
    ensure_float32(tr)

    st = Stream([trace])
    st.merge(method=1, fill_value=fill_value)
    gaps = st.get_gaps()
    tr = st[0]
    if gaps:
        if not hasattr(tr.stats, "processing"):
            tr.stats.processing = []
        tr.stats.processing.append(f"Filled {len(gaps)} gaps with {fill_value}")
        tr.stats.processing.extend(
            [f"GAP {gap[4]}s from {gap[2]} to {gap[3]}" for gap in gaps]
        )
    if np.ma.isMaskedArray(tr.data):
        tr.data = tr.data.filled(fill_value)
    return tr


def restore_gaps(trace, fill_value=0.0):
    """
    Masks regions filled by `fill_value` based on stored gap metadata in trace.stats.processing.
    """
    if not hasattr(trace.stats, "processing"):
        return trace

    processing_lines = [p for p in trace.stats.processing if p.startswith("GAP")]
    if not processing_lines:
        return trace

    data = np.ma.masked_array(trace.data, mask=False)

    for line in processing_lines:
        try:
            parts = line.split()
            t1 = UTCDateTime(parts[3])
            t2 = UTCDateTime(parts[5])
            idx1 = int((t1 - trace.stats.starttime) * trace.stats.sampling_rate)
            idx2 = int((t2 - trace.stats.starttime) * trace.stats.sampling_rate)
            data.mask[idx1:idx2] = True
        except Exception as e:
            print(f"✘ Could not parse gap line '{line}': {e}")
'''
class SDSobj:
    """
    A class to manage an SDS (SeisComP Data Structure) archive.
    Allows reading, writing, checking availability, and plotting data from an SDS archive.
    """

    def __init__(self, SDS_TOP, sds_type='D', format='MSEED', streamobj=None, metadata=None):
        """
        Initialize SDSobj.

        Parameters:
        - SDS_TOP (str): Root directory of the SDS archive.
        - sds_type (str): SDS file type (default 'D' for daily files).
        - format (str): File format (default 'MSEED').
        - streamobj (obspy Stream): Optional preloaded stream object.
        """
        os.makedirs(SDS_TOP, exist_ok=True)
        self.client = obspy.clients.filesystem.sds.Client(SDS_TOP, sds_type=sds_type, format=format)
        self.stream = streamobj or Stream()
        self.topdir = SDS_TOP
        self.metadata = metadata # for supporting a dataframe of allowable SEED ids (from same Excel spreadsheet used to generate StationXML)


    def read(self, startt, endt, skip_low_rate_channels=True, trace_ids=None,
                speed=1, verbose=True, merge_method=0, progress=False,
                fill_value=0.0, detect_zero_padding=True):
            """
            Read data from the SDS archive into the internal stream.
            """
            if not trace_ids:
                trace_ids = self._get_nonempty_traceids(startt, endt, skip_low_rate_channels, speed=speed)

            st = Stream()
            trace_iter = tqdm(trace_ids, desc="Reading traces") if progress else trace_ids

            for trace_id in trace_iter:
                net, sta, loc, chan = trace_id.split('.')
                if chan.startswith('L') and skip_low_rate_channels:
                    continue

                try:
                    if speed == 1:
                        sdsfiles = self.client._get_filenames(net, sta, loc, chan, startt, endt)
                        for sdsfile in sdsfiles:
                            if os.path.isfile(sdsfile):
                                try:
                                    traces = read_mseed_with_gap_masking(sdsfile)
                                    for tr in traces:
                                        st += tr
                                except Exception as e:
                                    if verbose:
                                        print(f"Failed to read (v1) {sdsfile}: {e}")
                    elif speed == 2:
                        traces = self.client.get_waveforms(net, sta, loc, chan, startt, endt, merge=-1)
                        for tr in traces:
                            tr = mask_gaps(tr, fill_value=fill_value)
                            st += tr

                except Exception as e:
                    if verbose:
                        print(f"Failed to read (v2) {trace_id}: {e}")

            st = remove_empty_traces(st)

            if len(st):
                st.trim(startt, endt)
                for tr in st:
                    ensure_float32(tr)
                st.merge(method=merge_method)

            self.stream = st
            return 0 if len(st) else 1

    def write(self, overwrite=False, fallback_to_indexed=True, debug=False, fill_value=0.0):
        """
        Write internal stream to SDS archive, marking gaps with fill_value and preserving metadata.

        Parameters:
        - overwrite (bool): Overwrite existing files if True.
        - fallback_to_indexed (bool): If True, write .01, .02 files on merge conflict. If False, raise error.
        - debug (bool): Print debug messages if True.
        - fill_value (float): Value to insert in gaps when merging (default 0.0).

        Returns:
        - bool: True if all writes succeed, False otherwise.
        """
        if isinstance(self.stream, Trace):
            self.stream=Stream(traces=[self.stream])
        success = False  

        # Setup subdirectories
        tempdir = os.path.join(self.topdir, 'temporarily_move_while_merging')
        unmergeddir = os.path.join(self.topdir, 'unable_to_merge')
        obsoletedir = os.path.join(self.topdir, 'obsolete')
        unwrittendir = os.path.join(self.topdir, 'failed_to_write_to_sds')
        multitracedir = os.path.join(self.topdir, 'multitrace')

        for d in [tempdir, unmergeddir, obsoletedir, unwrittendir, multitracedir]:
            os.makedirs(d, exist_ok=True)

        if debug:
            print(f'SDSobj.write(): Processing stream with {len(self.stream)} traces')

        print(self.stream)
        for tr_unsplit in self.stream:
            try:
                split_traces = split_trace_at_midnight(tr_unsplit)
            except Exception as e:
                print(e)
                print(f'self.stream={self.stream}')
                raise e

            for tr in split_traces:
                tr = ensure_float32(tr)
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
                    print(f'SDSobj.write(): Attempting to write {tr.id} to {sdsfile}')

                if not overwrite and os.path.isfile(sdsfile):
                    try:
                        existing = read(sdsfile)
                    except Exception as e:
                        print(f"⚠ Error reading existing file {sdsfile}: {e}")
                        existing = Stream()

                    shutil.copy2(sdsfile, tempfile)

                    merged, merge_info = smart_merge([tr], existing)

                    if merge_info["status"] == "identical":
                        print("Duplicate of existing SDS record — skipping")
                        safe_remove(tempfile)
                        continue

                    elif merge_info["status"] == "conflict":
                        print(f"✘ Cannot merge {tr.id} — conflict found.")
                        write_safely(tr, unmergedfile)
                        safe_remove(tempfile)
                        success = False
                        continue

                    elif merge_info["status"] == "merged":
                        if len(merged) == 1:
                            success = write_safely(merged[0], sdsfile, overwrite_ok=True)
                            shutil.move(tempfile, obsoletefile)
                            if debug:
                                print(f"✔ Merged and wrote: {sdsfile}")
                        else:
                            write_safely(merged, multitracefile)
                            safe_remove(tempfile)
                            print(f"✘ Merge produced multiple traces: {tr.id}. Written to {multitracefile}")
                            success = False
                    else:
                        print(f"✘ Unexpected merge status: {merge_info['status']}")
                        write_safely(tr, unwrittenfile)
                        safe_remove(tempfile)
                        success = False
                else:
                    success = write_safely(tr, sdsfile, overwrite_ok=True)
                    if debug:
                        print(f"✔ New file written: {sdsfile}")


        return success


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
                self.topdir, year, net, station_glob, '*.D',
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
                                # Ensure all Traces are in float32 dtype to avoid merge errors
                                for tr in st:
                                    ensure_float32(tr)

                                st.merge(method=0)
                                tr = st[0]
                                expected = tr.stats.sampling_rate * 86400
                                npts = np.count_nonzero(~np.isnan(tr.data)) if speed == 1 else tr.stats.npts
                                percent = min(100.0, 100 * npts / expected) if expected > 0 else 0
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
        return os.path.join(self.topdir, sds_subdir, filename)
    
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


def parse_sds_filename(filename):
    """
    Parses an SDS-style MiniSEED filename and extracts its components.
    Assumes filenames follow: NET.STA.LOC.CHAN.TYPE.YEAR.DAY
    """
    pattern = r"^(\w*)\.(\w*)\.(\w*)\.(\w*)\.(\w*)\.(\d{4})\.(\d{3})$"
    match = re.match(pattern, filename)
    if match:
        network, station, location, channel, dtype, year, jday = match.groups()
        location = location if location else "--"
        if len(location) == 1:
            location = '0' + location
        return network, station, location, channel, dtype, year, jday
    return None
