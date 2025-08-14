# --- Standard library ---
import gc
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path

# --- Third-party ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from obspy import read, Stream, Trace
from obspy.core.inventory import Inventory
from obspy.core.utcdatetime import UTCDateTime
import obspy.clients.filesystem.sds
from tqdm import tqdm

# --- Local imports ---
from flovopy.core.miniseed_io import (
    smart_merge,
    read_mseed,
    write_mseed,
    downsample_stream_to_common_rate,
)
from flovopy.core.trace_utils import remove_empty_traces
from flovopy.sds.sds_utils import (
    is_valid_sds_dir,
    is_valid_sds_filename,
    parse_sds_filename,
)
from flovopy.stationmetadata.utils import build_dataframe_from_table

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

    def read(self, startt, endt,
            skip_low_rate_channels=True,
            trace_ids=None,
            speed=2,
            verbose=True,
            progress=False,
            min_sampling_rate=50.0,
            max_sampling_rate=250.0,
            merge_strategy='obspy'):
        """
        Read data from the SDS archive into the internal stream.

        Guarantees (on success):
        - traces culled below `min_sampling_rate` (if not None),
        - downsampled to a common rate (<= `max_sampling_rate`),
        - merged per-id with `smart_merge` when needed,
        - trimmed to [startt, endt],
        - tagged in stats.processing: "flovopy:smart_merge_v1".
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
                    # Read per-file using read_mseed() (which already sanitizes/merges per-file)
                    sdsfiles = self.client._get_filenames(net, sta, loc, chan, startt, endt)
                    if verbose:
                        print(f"Found {len(sdsfiles)} matching SDS files")

                    for sdsfile in sdsfiles:
                        if os.path.isfile(sdsfile):
                            try:
                                if verbose:
                                    print(f"Reading {sdsfile}")
                                traces = read_mseed(
                                    sdsfile,
                                    starttime=startt, endtime=endt,
                                    min_sampling_rate=min_sampling_rate,
                                    max_sampling_rate=max_sampling_rate,
                                    merge=True, merge_strategy=merge_strategy
                                )
                                st += traces
                            except Exception as e:
                                if verbose:
                                    print(f"✘ Failed to read (v1) {sdsfile}: {e}")

                elif speed == 2:
                    # Read a whole window from the SDS client, then normalize like read_mseed()
                    traces = self.client.get_waveforms(net, sta, loc, chan, startt, endt, merge=-1)

                    # Cull below min_sampling_rate (match read_mseed policy)
                    if min_sampling_rate is not None:
                        for tr in list(traces):
                            if tr.stats.sampling_rate < float(min_sampling_rate):
                                traces.remove(tr)

                    # Downsample to common rate (cap by max_sampling_rate), then merge
                    ds_stream = downsample_stream_to_common_rate(traces, max_sampling_rate=max_sampling_rate)
                    smart_merge(ds_stream, strategy=merge_strategy)
                    st += ds_stream
                    del ds_stream

                else:
                    raise ValueError(f"Unknown speed value: {speed}")

            except Exception as e:
                if verbose:
                    print(f"✘ Failed to read (v2) {trace_id}: {e}")

        remove_empty_traces(st, inplace=True)
        if verbose:
            print(f"\nAfter removing empty traces:\n{st}")

        if len(st):
            st.trim(startt, endt)

            # Merge only if needed: gaps exist or duplicate IDs present
            needs_merge = bool(st.get_gaps()) or (len({tr.id for tr in st}) != len(st))
            if needs_merge:
                smart_merge(st, strategy=merge_strategy)
                if verbose:
                    print(f"\nAfter final smart_merge:\n{st}")

            # Tag traces so downstream can trust the merge contract
            for tr in st:
                tr.stats.processing = (tr.stats.processing or []) + ["flovopy:smart_merge_v1"]

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

    def get_availability(self, startday, endday, skip_low_rate_channels=True,
                                trace_ids=None, speed=3, verbose=False,
                                progress=True, merge_strategy='obspy', max_workers=8):
        """
        Return (DataFrame, ordered_trace_ids) with one row per date and one column per SEED id.
        Values are FRACTIONS in [0, 1] representing availability.

        Usage:
            df, ids = sds.get_availability(startday, endday)
            sds.plot_availability(df, outfile="avail.png")
        """
        trace_ids = trace_ids or self._get_nonempty_traceids(startday, endday, skip_low_rate_channels, speed=speed)

        days, t = [], startday
        while t < endday:
            days.append(t)
            t += 86400

        tasks = [(self, tid, day, speed, merge_strategy, verbose) for day in days for tid in trace_ids]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_compute_percent, task) for task in tasks]
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Availability") if progress else as_completed(futures)
            for fut in iterator:
                results.append(fut.result())  # expecting (date_epoch_or_dt, trace_id, value)

        if not results:
            df = pd.DataFrame({"date": pd.to_datetime([int(d) for d in days], unit='s')})
            for tid in trace_ids:
                df[tid] = np.nan
            return df, trace_ids

        df = pd.DataFrame(results, columns=["date", "trace_id", "value"])

        # Dates -> naive datetime (localize/strip tz for plotting)
        try:
            df["date"] = pd.to_datetime(df["date"], unit='s', utc=True).dt.tz_convert(None)
        except (ValueError, TypeError):
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

        # Coerce to float and normalize to fraction if needed (robust to 0–100 or 0–1 inputs)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        if df["value"].max(skipna=True) > 1.5:
            df["value"] = df["value"] / 100.0  # convert percent -> fraction

        wide = (df.pivot_table(index="date", columns="trace_id", values="value", aggfunc="mean")
                .sort_index())

        all_dates = pd.to_datetime([int(d) for d in days], unit='s', utc=True).tz_convert(None)
        wide = wide.reindex(index=all_dates, columns=trace_ids)

        wide = wide.reset_index().rename(columns={"index": "date"})
        return wide, trace_ids


    # --- helper: build daily availability (fractions) from audit ranges csv ---

    def _availability_from_audit_csv(self, csv_path, start=None, end=None, ids=None):
        """
        Convert audit_sds_archive() CSV into a wide daily availability DataFrame:
        one row per date, one column per SEED id (fixed_id), values in [0..1].
        """
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(columns=["date"])

        # Parse times to UTC
        df["segment_start"] = pd.to_datetime(df["segment_start"], utc=True, errors="coerce")
        df["segment_end"]   = pd.to_datetime(df["segment_end"],   utc=True, errors="coerce")

        # Discover IDs if none provided
        if ids is None:
            ids = sorted(df["fixed_id"].dropna().unique().tolist())

        # Filter to requested IDs if given
        df = df[df["fixed_id"].isin(ids)].copy()
        if df.empty:
            return pd.DataFrame(columns=["date"] + list(ids))

        # Establish overall date span
        seg_min = df["segment_start"].min()
        seg_max = df["segment_end"].max()

        # Optional external bounds
        if start is not None:
            start = pd.to_datetime(start, utc=True, errors="coerce")
            seg_min = max(seg_min, start) if seg_min is not None else start
        if end is not None:
            end = pd.to_datetime(end, utc=True, errors="coerce")
            seg_max = min(seg_max, end) if seg_max is not None else end

        if (seg_min is pd.NaT) or (seg_max is pd.NaT) or (seg_min >= seg_max):
            return pd.DataFrame(columns=["date"] + list(ids))

        # Build midnight-aligned day bins
        days_utc = pd.date_range(seg_min.normalize(), seg_max.normalize(), freq="D", tz="UTC")
        day_starts = days_utc
        day_ends   = day_starts + pd.Timedelta(days=1)

        # Accumulate overlap seconds
        day_id_seconds = {}
        use_sr = (df.get("sampling_rate", 0) > 0) & (df.get("total_npts", 0) > 0)
        npts_seconds = pd.Series(np.nan, index=df.index, dtype="float64")
        npts_seconds.loc[use_sr] = df.loc[use_sr, "total_npts"] / df.loc[use_sr, "sampling_rate"]

        for idx, row in df.iterrows():
            fid = row["fixed_id"]
            t0  = row["segment_start"]
            t1  = row["segment_end"]
            if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
                continue

            target_total = npts_seconds.iloc[idx] if not np.isnan(npts_seconds.iloc[idx]) else None
            remaining = target_total

            first_day_idx = max(0, day_starts.searchsorted(t0, side="right") - 1)
            last_day_idx  = min(len(day_starts)-1, day_starts.searchsorted(t1, side="left"))

            for di in range(first_day_idx, last_day_idx + 1):
                d0 = day_starts[di]
                d1 = day_ends[di]
                ov_start = max(t0, d0)
                ov_end   = min(t1, d1)
                overlap  = (ov_end - ov_start).total_seconds()
                if overlap <= 0:
                    continue

                if remaining is not None:
                    if overlap > remaining:
                        overlap = remaining
                    remaining -= overlap

                key = (d0.tz_convert(None).to_pydatetime().date(), fid)
                day_id_seconds[key] = day_id_seconds.get(key, 0.0) + float(overlap)

        if not day_id_seconds:
            return pd.DataFrame(columns=["date"] + list(ids))

        recs = [{"date": k[0], "fixed_id": k[1], "seconds": v} for k, v in day_id_seconds.items()]
        tall = pd.DataFrame(recs)

        secs_per_day = 86400.0
        tall["value"] = np.clip(tall["seconds"] / secs_per_day, 0.0, 1.0)

        wide = tall.pivot_table(index="date", columns="fixed_id", values="value", aggfunc="sum").sort_index()
        wide = wide.clip(upper=1.0)

        # Reindex to full date span and discovered/requested IDs
        all_dates = pd.date_range(days_utc.min(), days_utc.max(), freq="D", tz="UTC").tz_convert(None).date
        wide = wide.reindex(index=all_dates, columns=ids)

        return wide.reset_index().rename(columns={"index": "date"})


    # --- plotting: now accepts either availabilityDF or audit CSV ---
    '''
    def plot_availability(
        self,
        availabilityDF=None,
        outfile=None,
        figsize=(12, 9),                 # a bit taller to fit the counts plot
        fontsize=10,
        labels=None,
        cmap='gray_r',
        audit_ranges_csvfile=None,
        start=None,
        end=None,
        ids=None,
        grouping=None,                   # None | 'component' | 'location' | 'station' | 'similarity'
        station_prefix_len=4,
        group_agg="mean",                # 'mean' (default) or 'max'
        show_counts=True,                # NEW: show counts axis above heatmap
        count_min_fraction=0.0           # NEW: threshold for counting a row as reporting
    ):
        """
        Plot availability heatmap from FRACTIONS (0..1). Darker = more available.
        If show_counts=True, adds a panel above showing number of reporting rows per day.
        """


        # Build availabilityDF if only CSV is provided
        if availabilityDF is None and audit_ranges_csvfile:
            availabilityDF = self._availability_from_audit_csv(audit_ranges_csvfile, start=start, end=end, ids=ids)

        if availabilityDF is None or availabilityDF.empty:
            print("No availability data to plot.")
            return

        # Optional grouping (now with mean-aggregation by default)
        if grouping:
            availabilityDF = _apply_grouping(availabilityDF, mode=grouping, station_prefix_len=station_prefix_len, agg=group_agg)

        # Arrange: rows = ids, cols = dates; values in 0..1
        A_df = availabilityDF.set_index('date').T
        # Keep date index as datetime index for nicer formatting
        A_df.columns = pd.to_datetime(A_df.columns)

        yticklabels = labels or list(A_df.index)
        xdates = A_df.columns

        if show_counts:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 4], hspace=0.15, left=0.08, right=0.98, top=0.95, bottom=0.08)
            ax_top = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # --- Top counts plot (optional) ---
        if show_counts:
            # Count how many rows have fraction > threshold each day (ignoring NaNs)
            counts = (A_df > count_min_fraction).sum(axis=0)
            ax_top.plot(xdates, counts)                 # default style; no explicit color/style
            ax_top.set_ylabel("# reporting", fontsize=fontsize)
            ax_top.grid(True, axis='y', alpha=0.3)
            # Keep x tick labels only on bottom axis
            plt.setp(ax_top.get_xticklabels(), visible=False)

        # --- Heatmap ---
        im = ax.imshow(A_df.to_numpy(), aspect='auto', interpolation='nearest',
                    cmap=cmap, vmin=0, vmax=1)

        # X ticks: subsample for readability
        if len(xdates) > 0:
            step = max(1, len(xdates) // 25)
            xlab = pd.to_datetime(pd.Series(xdates)).dt.strftime('%Y-%m-%d').tolist()
            ax.set_xticks(np.arange(len(xdates))[::step])
            ax.set_xticklabels(xlab[::step], rotation=90, fontsize=fontsize)

        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=fontsize)

        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel("SEED ID" if not grouping else f"Group ({grouping})", fontsize=fontsize)
        ax.set_title("SDS Data Availability (%)", fontsize=fontsize + 2)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cbar.set_label("Availability", fontsize=fontsize)

        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=300)
            print(f"Saved availability plot to: {outfile}")
        return fig  
    # --- plotting: now accepts either availabilityDF or audit CSV ---
    '''  
    
    def plot_availability(
        self,
        availabilityDF=None,
        outfile=None,
        figsize=(12, 9),
        fontsize=10,
        labels=None,
        cmap='gray_r',
        audit_ranges_csvfile=None,
        start=None,
        end=None,
        ids=None,
        grouping=None,
        station_prefix_len=4,
        group_agg="mean",
        show_counts=True,
        count_min_fraction=0.0
    ):


        # Build availabilityDF if only CSV is provided
        if availabilityDF is None and audit_ranges_csvfile:
            availabilityDF = self._availability_from_audit_csv(audit_ranges_csvfile, start=start, end=end, ids=ids)

        if availabilityDF is None or availabilityDF.empty:
            print("No availability data to plot.")
            return

        # Optional grouping
        if grouping:
            availabilityDF = _apply_grouping(
                availabilityDF, mode=grouping, station_prefix_len=station_prefix_len, agg=group_agg
            )

        # rows = ids, cols = dates; values in 0..1
        A_df = availabilityDF.set_index('date').T
        # normalize/ensure datetime for labels (not used for x coords)
        xdates = pd.to_datetime(A_df.columns)
        yticklabels = labels or list(A_df.index)

        # numeric x coordinates so both axes align
        n_cols = len(xdates)
        n_rows = len(yticklabels)
        idx = np.arange(n_cols)

        if show_counts:
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 4])
            ax_top = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0])
        else:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            ax_top = None

        # --- Heatmap (use extent to align with idx coordinates) ---
        A = A_df.to_numpy()
        im = ax.imshow(
            A,
            aspect='auto',
            interpolation='nearest',
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5]  # x: columns, y: rows; top-left origin
        )

        # X ticks: indices with date labels
        if n_cols > 0:
            step = max(1, n_cols // 25)
            ax.set_xticks(idx[::step])
            ax.set_xticklabels(xdates.strftime('%Y-%m-%d')[::step], rotation=90, fontsize=fontsize)

        # Y ticks: rows
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(yticklabels, fontsize=fontsize)

        ax.set_xlabel("Date", fontsize=fontsize)
        ax.set_ylabel("SEED ID" if not grouping else f"Group ({grouping})", fontsize=fontsize)
        ax.set_title("SDS Data Availability (%)", fontsize=fontsize + 2)

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cbar.set_label("Availability", fontsize=fontsize)

        # --- Counts on top (share the same idx coordinates) ---
        if show_counts:
            # Count “reporting” rows per day (post-grouping); treat NaN as False
            counts = (A_df.fillna(0.0) > count_min_fraction).sum(axis=0)
            ax_top.plot(idx, counts.values)
            ax_top.set_xlim(-0.5, n_cols - 0.5)  # match heatmap
            ax.set_xlim(-0.5, n_cols - 0.5)      # explicit, too
            ax_top.set_ylabel("# reporting", fontsize=fontsize)
            ax_top.grid(True, axis='y', alpha=0.3)
            # Hide top x tick labels; bottom handles them
            ax_top.tick_params(labelbottom=False)

        # Do NOT call plt.tight_layout(); constrained_layout handles this
        if outfile:
            fig.savefig(outfile, dpi=300)
            print(f"Saved availability plot to: {outfile}")
        return fig


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
    

    def sds2eventStream(self, eventtime,
                        pretrig=3600, posttrig=3600,
                        networks=['*'], bandcodes=['GHDCESB'],
                        show_available=True):
        """
        Load waveform data from an SDS archive around a trigger time.

        Parameters:
        -----------
        eventtime : UTCDateTime or ISO time string
            Center time of the event window.

        pretrig : float
            Seconds before the trigger to include.

        posttrig : float
            Seconds after the trigger to include.

        networks : list of str
            Network codes to include (use ['*'] for wildcard).

        bandcodes : list of str
            Channel bandcodes or wildcards (e.g., ['HH', 'GH', 'BH']).

        show_available : bool
            If True, print available NSLC structure at midpoint of the window.

        Returns:
        --------
        obspy.Stream
            Combined waveform stream for requested time window and filters.
        """

        from flovopy.core.trace_utils import print_nslc_tree
        startt = UTCDateTime(eventtime) - pretrig
        endt = UTCDateTime(eventtime) + posttrig

        if show_available:
            mid = UTCDateTime((startt.timestamp + endt.timestamp) / 2)
            nslc_list = self.client.get_all_nslc(datetime=mid)
            print_nslc_tree(nslc_list)

        st = Stream()
        for network in networks:
            bc_filter = f"[{''.join(bandcodes)}]*" if bandcodes else "*"
            try:
                this_st = self.client.get_waveforms(network, "*", "*", bc_filter, startt, endt)
                st += this_st
            except Exception as e:
                print(f"[WARN] Failed to load waveforms for network {network}: {e}")

        return st
    
    '''
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
      # or your actual import path

    def load_metadata_from_excel(self, excel_path, sheet_name='ksc_stations_master'):
        """
        Load metadata from an Excel file into the SDSobj, including
        on/off dates and multi-channel expansion.

        Parameters
        ----------
        excel_path : str
            Path to the Excel file.
        sheet_name : str or int, optional
            Sheet name or index to load (default is 'ksc_stations_master').
        """
        df = build_dataframe_from_table(excel_path, sheet_name=sheet_name)

        required_cols = {'network', 'station', 'location', 'channel'}
        id_cols = {'id', 'seedid', 'traceid'}

        if not required_cols.issubset(df.columns) and not id_cols.intersection(df.columns):
            raise ValueError("Excel file must contain either full IDs ('id', 'seedid') or network/station/location/channel columns")

        if 'id' not in df.columns:
            if id_cols.intersection(df.columns):
                # Rename one of the ID columns to 'id'
                df = df.rename(columns={list(id_cols.intersection(df.columns))[0]: 'id'})
            else:
                df['id'] = df.apply(
                    lambda row: f"{row['network']}.{row['station']}.{str(row['location']).zfill(2)}.{row['channel']}",
                    axis=1
                )

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



    def load_metadata_from_stationxml(self, xml_path):
        """
        Populate SDSobj.metadata from a StationXML file.

        Parameters
        ----------
        xml_path : str
            Path to StationXML file.
        """
        inv = Inventory.read(xml_path, format="stationxml")
        rows = []

        for net in inv:
            for sta in net:
                for chan in sta:
                    row = {
                        "network": net.code,
                        "station": sta.code,
                        "location": chan.location_code,
                        "channel": chan.code,
                        "latitude": sta.latitude,
                        "longitude": sta.longitude,
                        "elevation_m": sta.elevation,
                        "depth_m": chan.depth,
                        "starttime": chan.start_date.isoformat() if chan.start_date else None,
                        "endtime": chan.end_date.isoformat() if chan.end_date else None,
                        "samplerate": chan.sample_rate
                    }
                    rows.append(row)

        self.metadata = pd.DataFrame(rows)

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




_SEED_RE = re.compile(r"^(?P<net>[^.]+)\.(?P<sta>[^.]+)\.(?P<loc>[^.]+)\.(?P<cha>[^.]+)$")

def _parse_seed_id(fid: str):
    m = _SEED_RE.match(fid)
    if not m:
        # Fallback: treat everything as station-level bucket
        return None, fid, None, None
    return m.group("net"), m.group("sta"), m.group("loc"), m.group("cha")

def _component_suffix(ch: str):
    """Return the component discriminator (last char), e.g. Z/N/E or 1/2/3; None if unknown."""
    return ch[-1] if ch else None

def _component_sort_key(c):
    # canonical order: Z,N,E then 1,2,3 then letters then digits
    order = {c:i for i,c in enumerate(list("ZNE123"))}
    return order.get(c, 100 + ord(str(c)[0]))

def _longest_common_prefix(strings: list[str]) -> str:
    """Return the full longest common prefix across all strings."""
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)  # lexicographic bound trick
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]

def _group_map_for_columns(cols, mode="component", station_prefix_len=4, station_prefix_min=2):
    """
    Build a mapping {original_col -> group_key} and, for component mode,
    a pretty renamer {group_key -> label_with_suffixes}.
    """
    mapping = {}
    renamer = {}

    if mode == "location":
        for c in cols:
            net, sta, loc, cha = _parse_seed_id(c)
            if sta is None:
                mapping[c] = c  # fallback
            else:
                mapping[c] = f"{net}.{sta}.{loc}"
        return mapping, renamer

    if mode == "station":
        for c in cols:
            net, sta, loc, cha = _parse_seed_id(c)
            if sta is None:
                mapping[c] = c
            else:
                mapping[c] = f"{net}.{sta}"
        return mapping, renamer

    if mode == "similarity":
        # 1) coarse bucket by (net, first station_prefix_min chars) to keep it efficient
        buckets = defaultdict(list)  # (net, base_min) -> [full id cols]
        parsed = {}
        for c in cols:
            net, sta, loc, cha = _parse_seed_id(c)
            parsed[c] = (net, sta, loc, cha)
            if not sta:
                mapping[c] = c  # fallback
                continue
            base_min = sta[:max(1, station_prefix_min)]
            buckets[(net, base_min)].append(c)

        # 2) for each bucket, compute full LCP of station codes and use that as the final label
        #    (this allows 5+ characters when truly shared; singletons use their full station code)
        for (net, _), members in buckets.items():
            stas = [parsed[m][1] for m in members if parsed[m][1]]
            lcp = _longest_common_prefix(stas) or stas[0]  # fallback to first if no common prefix
            gkey = f"{net}.{lcp}"
            for m in members:
                mapping[m] = gkey

        # no pretty renamer needed; gkey is already the desired label
        return mapping, renamer


    # component mode (default)
    buckets = defaultdict(list)  # key -> list of original cols
    key2parts = {}               # key -> (net,sta,loc,prefix)
    for c in cols:
        net, sta, loc, cha = _parse_seed_id(c)
        if sta is None or not cha:
            mapping[c] = c
            continue
        prefix = cha[:2]  # e.g., 'BH','EH','DH','DD'
        key = (net, sta, loc, prefix)
        buckets[key].append(c)
        key2parts[key] = (net, sta, loc, prefix)

    for key, members in buckets.items():
        net, sta, loc, prefix = key2parts[key]
        gkey = f"{net}.{sta}.{loc}.{prefix}"  # internal group key
        for m in members:
            mapping[m] = gkey

        # build pretty label with concatenated, ordered suffixes present
        suffixes = []
        for m in members:
            _, _, _, mcha = _parse_seed_id(m)
            s = _component_suffix(mcha)
            if s:
                suffixes.append(s)
        # de-dup & order
        seen = []
        for s in sorted(set(suffixes), key=_component_sort_key):
            seen.append(s)
        suffix_concat = "".join(seen) if seen else ""
        renamer[gkey] = f"{net}.{sta}.{loc}.{prefix}{suffix_concat}"

    return mapping, renamer

'''
def _apply_grouping(wide_df: pd.DataFrame, mode="component", station_prefix_len=4):
    """
    wide_df: 'date' column + one column per id (values are fractions 0..1).
    Returns grouped wide_df with columns grouped and aggregated by max.
    """
    if wide_df is None or wide_df.empty:
        return wide_df

    id_cols = [c for c in wide_df.columns if c != "date"]
    if not id_cols:
        return wide_df

    mapping, renamer = _group_map_for_columns(id_cols, mode=mode, station_prefix_len=station_prefix_len)

    # group by mapping (aggregate with max: “available if any component is available”)
    M = pd.Series(mapping)
    grouped_vals = (
        wide_df[id_cols]
        .T.groupby(M)  # group columns by target key
        .max()         # or .mean() if you prefer average across components
        .T
    )

    out = pd.concat([wide_df[["date"]], grouped_vals], axis=1)

    # pretty names for component mode
    if renamer:
        out = out.rename(columns=renamer)

    return out
'''

def _apply_grouping(wide_df: pd.DataFrame, mode="component", station_prefix_len=4, agg="mean"):
    """
    wide_df: 'date' + id columns (fractions 0..1).
    mode: 'component' | 'location' | 'station' | 'similarity' | None
    agg: 'mean' or 'max'
    """
    if wide_df is None or wide_df.empty:
        return wide_df

    id_cols = [c for c in wide_df.columns if c != "date"]
    if not id_cols or not mode:
        return wide_df

    mapping, renamer = _group_map_for_columns(id_cols, mode=mode, station_prefix_len=station_prefix_len)

    grouped_vals = (
        wide_df[id_cols]
        .T.groupby(pd.Series(mapping))   # map original ids -> group key
        .mean() if agg == "mean" else
        wide_df[id_cols].T.groupby(pd.Series(mapping)).max()
    ).T

    out = pd.concat([wide_df[["date"]], grouped_vals], axis=1)
    if renamer:
        out = out.rename(columns=renamer)
    return out

if __name__ == "__main__":
    #sdsobj = SDSobj(basedir="/data/remastered/SDS_KSC")
    sdsobj = SDSobj(basedir="SDS_KSC")
    sdsobj.plot_availability(
        availabilityDF=None,
        audit_ranges_csvfile="/Users/glennthompson/Dropbox/audit_ksc_ranges.csv",
        start="2016-02-20",
        end="2022-12-03",
        ids=None, #["1R.BCHH.00.DD1","1R.BCHH.00.DD2","1R.BCHH.00.DD3"],
        group_agg="mean",
        show_counts=True,
        count_min_fraction=0.0,        
        outfile="avail_from_audit.png"
    )
    
    sdsobj.plot_availability(
        audit_ranges_csvfile="/Users/glennthompson/Dropbox/audit_ksc_ranges.csv",
        grouping="component",
        group_agg="mean",
        show_counts=True,
        count_min_fraction=0.0,  
        outfile="avail_component.png"
    )

    sdsobj.plot_availability(
        audit_ranges_csvfile="/Users/glennthompson/Dropbox/audit_ksc_ranges.csv",
        grouping="location",
        group_agg="mean",
        show_counts=True,
        count_min_fraction=0.0,         
        outfile="avail_location.png"
    )
    
    sdsobj.plot_availability(
        audit_ranges_csvfile="/Users/glennthompson/Dropbox/audit_ksc_ranges.csv",
        grouping="station",
        group_agg="mean",
        show_counts=True,
        count_min_fraction=0.0,  
        outfile="avail_station.png"
    )
    
    sdsobj.plot_availability(
        audit_ranges_csvfile="/Users/glennthompson/Dropbox/audit_ksc_ranges.csv",
        grouping="similarity",
        station_prefix_len=3,
        group_agg="mean",
        show_counts=True,
        count_min_fraction=0.0,          
        outfile="avail_similarity.png"
    )