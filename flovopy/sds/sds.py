# flovopy/core/sds.py

import os
import re
import glob
import shutil
import numpy as np
import pandas as pd
from obspy import read, Stream
import obspy.clients.filesystem.sds
from flovopy.core.preprocessing import remove_empty_traces, fix_trace_id, _can_write_to_miniseed_and_read_back

class SDSobj:
    """
    A class to manage an SDS (SeisComP Data Structure) archive.
    Allows reading, writing, checking availability, and plotting data from an SDS archive.
    """

    def __init__(self, SDS_TOP, sds_type='D', format='MSEED', streamobj=None):
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

    def read(self, startt, endt, skip_low_rate_channels=True, trace_ids=None, speed=1, verbose=True):
        """
        Read data from the SDS archive into the internal stream.

        Parameters:
        - startt, endt: Start and end times (UTCDateTime).
        - skip_low_rate_channels (bool): Skip channels with 'L' prefix.
        - trace_ids (list): Optional list of trace IDs to read.
        - speed (int): Reading method (1=filename-based, 2=SDS client).
        - verbose (bool): Print messages if True.

        Returns:
        - int: 0 if successful, 1 if stream is empty.
        """
        if not trace_ids:
            trace_ids = self._get_nonempty_traceids(startt, endt, skip_low_rate_channels)

        st = Stream()
        for trace_id in trace_ids:
            net, sta, loc, chan = trace_id.split('.')
            if chan.startswith('L') and skip_low_rate_channels:
                continue
            try:
                if speed == 1:
                    sdsfiles = self.client._get_filenames(net, sta, loc, chan, startt, endt)
                    for sdsfile in sdsfiles:
                        if os.path.isfile(sdsfile):
                            that_st = read(sdsfile)
                            that_st.merge(method=0)
                            st += that_st
                elif speed == 2:
                    st += self.client.get_waveforms(net, sta, loc, chan, startt, endt, merge=-1)
            except Exception as e:
                if verbose:
                    print(f"Failed to read {trace_id}: {e}")

        st.trim(startt, endt)
        st = remove_empty_traces(st)
        st.merge(method=0)
        self.stream = st
        return 0 if len(st) else 1

    def write(self, overwrite=False, debug=False):
        """
        Write internal stream to SDS archive.

        Parameters:
        - overwrite (bool): Overwrite existing files if True.
        - debug (bool): Currently unused.

        Returns:
        - bool: True if successful, False otherwise.
        """
        successful = True
        for tr in self.stream:
            try:
                sdsfile = self.client._get_filename(tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, tr.stats.starttime, 'D')
                os.makedirs(os.path.dirname(sdsfile), exist_ok=True)

                if not overwrite and os.path.isfile(sdsfile):
                    existing = read(sdsfile)
                    new_st = existing.copy().append(tr).merge(method=1, fill_value=0)
                    if len(new_st) == 1:
                        new_st[0].write(sdsfile, format='MSEED')
                    else:
                        raise ValueError("Cannot write Stream with more than 1 trace to a single SDS file")
                else:
                    tr.write(sdsfile, format='MSEED')

            except Exception as e:
                print(f"Write failed for {tr.id}: {e}")
                successful = False

        return successful

    def _get_nonempty_traceids(self, startday, endday=None, skip_low_rate_channels=True):
        """
        Get a list of trace IDs that have data between two dates.

        Parameters:
        - startday (UTCDateTime)
        - endday (UTCDateTime): Optional. Defaults to startday + 1 day.
        - skip_low_rate_channels (bool)

        Returns:
        - list: Sorted list of trace IDs.
        """
        endday = endday or startday + 86400
        trace_ids = set()
        thisday = startday
        while thisday < endday:
            for net, sta, loc, chan in self.client.get_all_nslc(sds_type='D', datetime=thisday):
                if chan.startswith('L') and skip_low_rate_channels:
                    continue
                if self.client.has_data(net, sta, loc, chan):
                    trace_ids.add(f"{net}.{sta}.{loc}.{chan}")
            thisday += 86400
        return sorted(trace_ids)

    def find_missing_days(self, stime, etime, net):
        """
        Return list of days with no data for a given network.

        Parameters:
        - stime, etime: Start and end time range (UTCDateTime).
        - net (str): Network code.

        Returns:
        - list of datetime: Missing days.
        """
        missing_days = []
        dayt = stime
        while dayt < etime:
            jday = dayt.strftime('%j')
            yyyy = dayt.strftime('%Y')
            pattern = os.path.join(self.topdir, yyyy, net, '*', '*.D', f"{net}*.{yyyy}.{jday}")
            existingfiles = glob.glob(pattern)
            if not existingfiles:
                missing_days.append(dayt)
            dayt += 86400
        return missing_days

    def get_percent_availability(self, startday, endday, skip_low_rate_channels=True, trace_ids=None, speed=3):
        """
        Compute data availability percentage for each trace ID per day.

        Parameters:
        - startday, endday (UTCDateTime)
        - skip_low_rate_channels (bool)
        - trace_ids (list): Optional trace ID list.
        - speed (int): Speed method for calculating availability (1, 2, or 3).

        Returns:
        - (DataFrame, list): Availability DataFrame and trace ID list
        """
        trace_ids = trace_ids or self._get_nonempty_traceids(startday, endday, skip_low_rate_channels)
        lod = []
        thisday = startday
        while thisday < endday:
            row = {'date': thisday.date()}
            for trace_id in trace_ids:
                net, sta, loc, chan = trace_id.split('.')
                percent = 0
                try:
                    if speed < 3:
                        sdsfile = self.client._get_filename(net, sta, loc, chan, thisday)
                        if os.path.isfile(sdsfile):
                            st = read(sdsfile)
                            st.merge(method=0)
                            tr = st[0]
                            expected = tr.stats.sampling_rate * 86400
                            npts = np.count_nonzero(~np.isnan(tr.data)) if speed == 1 else tr.stats.npts
                            percent = 100 * npts / expected
                    else:
                        percent = 100 * self.client.get_availability_percentage(net, sta, loc, chan, thisday, thisday + 86400)[0]
                except:
                    percent = 0
                row[trace_id] = percent
            lod.append(row)
            thisday += 86400
        return pd.DataFrame(lod), trace_ids

    def plot_availability(self, availabilityDF, outfile=None, FS=12, labels=None):
        """
        Plot availability DataFrame as a heatmap.

        Parameters:
        - availabilityDF (DataFrame): Availability data.
        - outfile (str): Optional path to save the figure.
        - FS (int): Font size and figure size.
        - labels (list): Optional list of y-axis labels.
        """
        import matplotlib.pyplot as plt
        Adf = availabilityDF.iloc[:, 1:] / 100
        Adata = Adf.to_numpy()
        xticklabels = labels or Adf.columns
        yticks = list(range(len(availabilityDF)))
        yticklabels = availabilityDF['date'].astype(str)

        step = max(1, len(yticks) // 25)
        yticks = yticks[::step]
        yticklabels = yticklabels[::step]

        plt.figure(figsize=(FS, FS))
        ax = plt.gca()
        ax.imshow(1.0 - Adata.T, aspect='auto', cmap='gray', interpolation='nearest')
        ax.set_xticks(yticks)
        ax.set_xticklabels(yticklabels, rotation=90, fontsize=FS)
        ax.set_yticks(np.arange(len(xticklabels)))
        ax.set_yticklabels(xticklabels, fontsize=FS)
        ax.set_xlabel('Date')
        ax.set_ylabel('NSLC')
        ax.grid(True)
        if outfile:
            plt.savefig(outfile, dpi=300)

    def __str__(self):
        return f"client={self.client}, stream={self.stream}"


class SDSFileManager:
    """
    Class for managing SDS archive operations including filename correction,
    file relocation, and metadata-based path resolution.
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_processed_dirs(self, log_file):
        """Reads the log file and returns a set of already processed directories."""
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                return set(line.strip() for line in f)
        return set()

    def save_processed_dir(self, directory, log_file):
        """Logs the processed directory to a file."""
        with open(log_file, "a") as f:
            f.write(directory + "\n")

    def fix_bad_filenames(self, write=False, networks=None, backup=True, log_file="fix_sds_filenames.log"):
        """
        Scan the SDS archive, rename files if necessary.
        Walks the directory tree in alphanumeric order, resuming from log if interrupted.
        """
        processed_dirs = self.get_processed_dirs(log_file)
        for root, dirs, files in os.walk(self.base_dir, topdown=True):
            dirs.sort()
            files.sort()

            if root in processed_dirs:
                print(f"Skipping already processed: {root}")
                continue

            print(f"Processing: {root}")
            for filename in files:
                if (networks and filename[0:2] in networks) or not networks:
                    file_path = os.path.join(root, filename)
                    parts = filename.split('.')
                    if len(parts) != 7:
                        print('\n', f'Bad filename {filename}')
                        if len(parts) > 7:
                            newfilename = '.'.join(parts[0:7])
                            newfile_path = os.path.join(root, newfilename)

                            if any(key in parts[7] for key in ['old', 'seed', 'ms', 'part']):
                                if os.path.isfile(newfile_path):
                                    self.mergefile(root, file_path, newfile_path, write=write, backup=backup)
                                else:
                                    self.movefile(file_path, newfile_path, write=write)

            self.save_processed_dir(root, log_file)

    def get_trace_directory(self, trace):
        """Returns the correct directory for a trace."""
        return os.path.dirname(self._trace_to_full_path(trace))

    def get_trace_filename(self, trace):
        """Returns the correct filename for a trace."""
        return os.path.basename(self._trace_to_full_path(trace))

    def _trace_to_full_path(self, trace):
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
        return os.path.join(self.base_dir, sds_subdir, filename)

    def get_directory_from_filename(self, filename):
        """Returns the correct directory path from an SDS-style filename."""
        parts = parse_sds_filename(filename)
        if parts:
            network, station, location, channel, dtype, year, jday = parts
            return os.path.join(self.base_dir, year, network, station, f"{channel}.{dtype}")
        return None

    def movefile(src, dst, write=False, logfile='movedfiles.log'):
        """
        Move a file to a new location.

        Parameters:
        - src (str): Source path
        - dst (str): Destination path
        - write (bool): If True, perform the move. Else, simulate.
        """
        print(f'- Will move {src} to {dst}') 
        if write:
            os.makedirs(os.path.dirname(dst), exist_ok=True)                                    
            shutil.move(src, dst)
            if not os.path.isfile(dst):
                print('move failed')
        else:
            print(f'- Would move {src} to {dst}')
        with open(logfile, "a") as f:
            f.write(f'{src} to {dst}' + "\n")           

    def mergefile(self, root, src, dst, write=False, backup=False, logfile='mergedfiles.log', backupdir='.'):
        """
        Merge two MiniSEED files if possible. Logs result.

        Parameters:
        - root (str): Root directory (used for logging, e.g., SOH directory check).
        - file_path (str): Path to the file to merge from.
        - newfile_path (str): Path to the existing file to merge into.
        - write (bool): Whether to actually write files.
        - backup (bool): Whether to backup files before overwriting.
        - logfile (str): Path to the log file.
        - BACKUPDIR (str): Directory to save backups in.
        """
        print(f'- Will try to merge {src} with {dst}')
        try:
            oldst = read(dst)
            partst = read(src)
            for tr in partst:
                try:
                    oldst.append(tr)
                    oldst.merge(fill_value=0)
                except:
                    pass

            if len(oldst) == 1 or 'soh' in root.lower():
                if _can_write_to_miniseed_and_read_back(oldst[0]):
                    if write:
                        if backup:
                            shutil.copy2(dst, os.path.join(backupdir, os.path.basename(dst)))
                            shutil.copy2(src, os.path.join(backupdir, os.path.basename(src)))
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        oldst.write(dst, format='MSEED')
                        os.remove(src)
                    else:
                        print(f'- would write merged file to {dst} and remove {src}')
                    with open(logfile, "a") as f:
                        f.write(f'{src} to {dst} SUCCESS\n')
                else:
                    print('- merge/write/read failed')
                    with open(logfile, "a") as f:
                        f.write(f'{src} to {dst} FAILED\n')
            else:
                print(f'- got {len(oldst)} Trace objects from merging. should be 1')
                with open(logfile, "a") as f:
                    f.write(f'{src} to {dst} but got {len(oldst)} traces\n')
        except Exception as e:
            print(f'- merge failed for {dst} and {src}?')
            print(e)
            with open(logfile, "a") as f:
                f.write(f'{src} to {dst} CRASHED\n')

    def fix_sds_ids(self, write=False, networks=None, log_file='sdsfilemanager.log'):
        """
        Scan the SDS archive, correct band codes, and rename files if necessary.
        Walks the directory tree in alphanumeric order, resuming from log if interrupted.
        """
            
        processed_dirs = self.get_processed_dirs(log_file)

        for root, dirs, files in os.walk(self.base_dir, topdown=True):
            dirs.sort()
            files.sort()

            if root in processed_dirs:
                print(f"Skipping already processed: {root}")
                continue

            print(f"Processing: {root}")
            for filename in files:
                if (networks and filename[0:2] in networks) or not networks:
                    file_path = os.path.join(root, filename)
                    parts = filename.split('.')
                    if len(parts) != 7:
                        continue
                    try:
                        stream = read(file_path)
                        for trace in stream:
                            if fix_trace_id(trace):
                                new_file_path = trace2correct_sdsfullpath(self.base_dir, trace)
                                if write:
                                    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                                    stream.write(new_file_path, format="MSEED")
                                    os.remove(file_path)
                                    print(f"Renamed to: {new_file_path}")
                                else:
                                    print(f'Would write {file_path} to {new_file_path}')
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

            self.save_processed_dir(root, log_file)

    def move_files_to_correct_directory(self, write=False):
        """ 
        Walks through the SDS archive and moves MiniSEED files to the correct channel directory.
        Walks the directory tree in alphanumeric order, resuming from log if interrupted.
        """        
        log_file = self._get_log_file()
        processed_dirs = self._get_processed_dirs(log_file)

        for root, dirs, files in os.walk(self.base_dir, topdown=True):
            dirs.sort()
            files.sort()

            if root in processed_dirs:
                print(f"Skipping already processed: {root}")
                continue

            print(f"Processing: {root}")
            current_channel_dir = find_channel_directory(root)
            if not current_channel_dir:
                continue

            for filename in files:
                file_path = os.path.join(root, filename)
                sds_tuple = parse_sds_filename(filename)
                if sds_tuple:
                    network, station, location, channel, dtype, year, jday = sds_tuple
                    expected_dir = f'{channel}.{dtype}'
                    if current_channel_dir != expected_dir:
                        print(f"Moving: {file_path} -> {expected_dir}")
                        if write:
                            os.makedirs(expected_dir, exist_ok=True)
                            shutil.move(file_path, os.path.join(expected_dir, filename))

            self.save_processed_dir(root, log_file)

    @staticmethod
    def find_channel_directory(path):
        """
        Extracts the channel directory (e.g., 'HHZ.D') from a file path.
        Assumes the directory structure is in SDS format.
        """
        parts = path.split(os.sep)
        for part in parts:
            if part.endswith(".D"):
                return part
        return None

    def move_files_to_structure(self, source_dir, write=False, backup=False, fix_filename_only=False):
        """
        Moves MiniSEED files from a flat directory into the correct SDS archive structure.
        If `fix_filename_only` is True, it just corrects the filename in the current directory.
        """
        for filename in os.listdir(source_dir):
            if parse_sds_filename(filename) or fix_filename_only:
                print(f'Processing {filename}')
                file_path = os.path.join(source_dir, filename)
                try:
                    this_st = read(file_path, format='MSEED')
                except Exception as e:
                    print(e)
                    print('Cannot read ', file_path)
                    continue

                if len(this_st) == 1:
                    fix_trace_id(this_st[0])
                    target_path = self._trace_to_full_path(this_st[0]) if not fix_filename_only else \
                                  SDSFileManager(self.base_dir)._trace_to_full_path(this_st[0])
                    if file_path != target_path:
                        if os.path.isfile(target_path):
                            mergefile(self.base_dir, file_path, target_path, write=write, backup=backup)
                        else:
                            movefile(file_path, target_path, write=write)
                else:
                    print(f'got {len(this_st)} traces')
            else:
                print(f"Skipping file (does not match SDS format): {filename}")

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
