import os
import shutil
from obspy import read
from obspy.core.trace import Trace

# These helper functions must also be defined or imported elsewhere:
# - parse_sds_filename
# - fix_trace_id
# - trace2correct_sdsfullpath
# - _can_write_to_miniseed_and_read_back

# If they're part of your project, you could import them like this:
# from your_module import parse_sds_filename, fix_trace_id, trace2correct_sdsfullpath, _can_write_to_miniseed_and_read_back

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