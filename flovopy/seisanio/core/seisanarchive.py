from __future__ import annotations

from pathlib import Path
import glob
from typing import Callable, Iterator

from obspy import UTCDateTime, read
from collections import defaultdict
from flovopy.enhanced.sdsclient import EnhancedSDSClient

class SeisanArchive:
    """
    Navigate a Seisan archive (WAV + REA).

    Supports:
    - continuous waveform iteration
    - event-based waveform retrieval via S-files
    - Seisan WAV filename parsing
    - optional Seisan/MVO trace-id fixing
    - optional shared waveform preprocessing
    - conversion of WAV archives to SDS
    """

    def __init__(self, root, db_cont=None, db_event=None):
        self.root = Path(root)
        self.db_cont = db_cont
        self.db_event = db_event

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _wav_dir(self, db, time):
        if db is None:
            raise ValueError("No continuous Seisan database name was provided.")
        yyyy = time.strftime("%Y")
        mm = time.strftime("%m")
        return self.root / "WAV" / db / yyyy / mm

    def _rea_dir(self, db, time):
        if db is None:
            raise ValueError("No event Seisan database name was provided.")
        yyyy = time.strftime("%Y")
        mm = time.strftime("%m")
        return self.root / "REA" / db / yyyy / mm

    # ------------------------------------------------------------------
    # WAV filename / metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wavpath2datetime(wavpath) -> UTCDateTime:
        """
        Extract a file time from a SEISAN WAV filename.

        Supports both modern and legacy-style SEISAN naming conventions.

        Examples
        --------
        Modern:
            1996-11-18-1716-44S.MVO___014
        Legacy:
            9611-18-1716-44S.MVO___014
        """
        wavpath = Path(wavpath)
        basename = wavpath.name

        if not basename:
            raise IOError("Empty WAV filename")
        if "S." not in basename:
            raise IOError(f"Filename does not contain 'S.' marker: {basename}")

        stem = basename.split("S.")[0]
        parts = stem.split("-")

        try:
            if len(parts) == 5:
                # YYYY-MM-DD-HHMM-SS
                yyyy = int(parts[0])
                mm = int(parts[1])
                dd = int(parts[2])
                hhmi = parts[3]
                ss = int(parts[4][0:2])
            elif len(parts) == 4:
                # YYMM-DD-HHMM-SS
                yy = int(parts[0][0:2])
                yyyy = 1900 + yy if yy >= 70 else 2000 + yy
                mm = int(parts[0][2:4])
                dd = int(parts[1])
                hhmi = parts[2]
                ss = int(parts[3][0:2])
            else:
                raise ValueError(f"Unexpected token structure: {parts}")

            hh = int(hhmi[0:2])
            mi = int(hhmi[2:4])

            return UTCDateTime(yyyy, mm, dd, hh, mi, ss)

        except Exception as e:
            raise IOError(f"Failed to parse filetime from WAV filename '{basename}': {e}")

    def get_waveform_file_info(self, wavpath, read_data=False, fixid=False,
                               preprocess=False, preprocess_fn=None,
                               preprocess_kwargs=None, verbose=False):
        """
        Return metadata for a waveform file.

        Parameters
        ----------
        wavpath : str or Path
            Path to SEISAN WAV file.
        read_data : bool
            If True, also read waveform data.
        fixid : bool
            If True, apply Seisan/MVO trace-id fixing when reading.
        preprocess : bool
            If True, apply shared preprocessing after read.
        preprocess_fn : callable or None
            Optional preprocessing function. Signature:
                preprocess_fn(stream, **kwargs) -> stream
        preprocess_kwargs : dict or None
            Additional kwargs for preprocess_fn.

        Returns
        -------
        dict
        """
        wavpath = Path(wavpath)

        info = {
            "path": wavpath,
            "exists": wavpath.exists(),
            "filetime": None,
            "legacy": None,
            "network": None,
            "start_time": None,
            "end_time": None,
            "stream": None,
        }

        if not wavpath.exists():
            return info

        try:
            info["filetime"] = self._wavpath2datetime(wavpath)
        except Exception:
            pass

        try:
            from flovopy.seisanio.utils.helpers import legacy_or_not
            legacy, network = legacy_or_not(str(wavpath))
            info["legacy"] = legacy
            info["network"] = network
        except Exception:
            pass

        if read_data:
            st = self.read_waveform_file(
                wavpath,
                fixid=fixid,
                preprocess=preprocess,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs=preprocess_kwargs,
                verbose=verbose,
            )
            info["stream"] = st
            if st and len(st) > 0:
                info["start_time"] = min(tr.stats.starttime for tr in st)
                info["end_time"] = max(tr.stats.endtime for tr in st)

        return info

    def find_sfile_for_wav(self, wavpath, mainclass="L"):
        """
        Try to find the corresponding REA S-file for a WAV file.

        Returns
        -------
        tuple[Path, bool]
            (candidate_sfile_path, exists_flag)
        """
        from flovopy.seisanio.utils.helpers import filetime2spath

        wavpath = Path(wavpath)
        filetime = self._wavpath2datetime(wavpath)

        parts = wavpath.parts
        try:
            wav_idx = parts.index("WAV")
            db = parts[wav_idx + 1]
            seisan_top = Path(*parts[:wav_idx])
        except Exception as e:
            raise ValueError(f"Could not infer Seisan DB/root from WAV path '{wavpath}': {e}")

        sfile = filetime2spath(
            filetime,
            mainclass=mainclass,
            db=db,
            seisan_top=str(seisan_top),
            fullpath=True,
        )
        sfile = Path(sfile)
        return sfile, sfile.exists()

    # ------------------------------------------------------------------
    # Waveform loading / preprocessing
    # ------------------------------------------------------------------

    def _apply_trace_id_fixes(self, st, wavpath=None, verbose=False):
        """
        Apply Seisan/MVO trace-id fixes to a Stream in place.
        """
        from flovopy.seisanio.utils.helpers import legacy_or_not
        from flovopy.core.trace_utils import fix_trace_id

        try:
            from flovopy.research.mvo.mvo_ids import fix_trace_mvo
        except Exception:
            fix_trace_mvo = None

        legacy = False
        network = None
        if wavpath is not None:
            try:
                legacy, network = legacy_or_not(str(wavpath))
            except Exception:
                legacy, network = False, None

        for tr in st:
            try:
                tr.stats.original_id = tr.id
            except Exception:
                pass

            if legacy:
                try:
                    fix_trace_id(tr, legacy=True, netcode="MV")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] fix_trace_id failed for {tr.id}: {e}")
            else:
                if fix_trace_mvo is not None:
                    try:
                        fix_trace_mvo(tr, verbose=verbose)
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] fix_trace_mvo failed for {tr.id}: {e}")

        return st

    def read_waveform_file(
        self,
        wavpath,
        *,
        reader: Callable = read,
        fixid: bool = False,
        preprocess: bool = False,
        preprocess_fn=None,
        preprocess_kwargs=None,
        verbose: bool = False,
    ):
        """
        Read a single waveform file, optionally applying:
        - Seisan/MVO trace-id fixes
        - shared preprocessing

        Parameters
        ----------
        wavpath : str or Path
            Waveform file path.
        reader : callable
            Reader function, default ObsPy read().
        fixid : bool
            If True, apply Seisan/MVO trace-id fixes.
        preprocess : bool
            If True, apply preprocess_fn after reading/fixing.
        preprocess_fn : callable or None
            Shared waveform preprocessing function.
            Signature: preprocess_fn(stream, **kwargs) -> stream
        preprocess_kwargs : dict or None
            Optional kwargs passed to preprocess_fn.
        verbose : bool
            Print warnings.

        Returns
        -------
        Stream | None
        """
        wavpath = Path(wavpath)

        if not wavpath.exists():
            if verbose:
                print(f"[WARN] File does not exist: {wavpath}")
            return None

        try:
            st = reader(str(wavpath))
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to read waveform file {wavpath}: {e}")
            return None

        if fixid:
            try:
                st = self._apply_trace_id_fixes(st, wavpath=wavpath, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Trace-id fixing failed for {wavpath}: {e}")

        if preprocess:
            if preprocess_fn is None:
                if verbose:
                    print(f"[WARN] preprocess=True but no preprocess_fn was supplied for {wavpath}")
            else:
                try:
                    kwargs = preprocess_kwargs or {}
                    st = preprocess_fn(st, **kwargs)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Preprocessing failed for {wavpath}: {e}")
                    return None

        return st

    # ------------------------------------------------------------------
    # WAV (continuous)
    # ------------------------------------------------------------------

    def iter_waveform_files(self, starttime, endtime, db=None) -> Iterator[Path]:
        """
        Yield waveform file paths whose filename-derived start times satisfy

            starttime <= filetime < endtime

        Parameters
        ----------
        starttime, endtime
            Half-open time interval.
        db
            Continuous Seisan database name. Defaults to ``self.db_cont``.
        """
        if db is None:
            db = self.db_cont

        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        if endtime <= starttime:
            return

        t = UTCDateTime(starttime.date)
        last_day = UTCDateTime((endtime - 1e-6).date)

        while t <= last_day:
            yyyy, mm, dd = t.strftime("%Y %m %d").split()

            current = self._wav_dir(db, t)
            prev = self._wav_dir(db, t - 86400)

            pattern_today = f"{yyyy}-{mm}-{dd}*"
            pattern_prev = f"{yyyy}-{mm}-{dd}-23[45]*"

            candidates = sorted({
                *glob.glob(str(prev / pattern_prev)),
                *glob.glob(str(current / pattern_today)),
            })

            for f in candidates:
                fpath = Path(f)
                try:
                    filetime = self._wavpath2datetime(fpath)
                except Exception:
                    continue

                if starttime <= filetime < endtime:
                    yield fpath

            t += 86400

    def get_file_list(self, starttime, endtime, db=None):
        """
        Return a list of waveform file paths across a time range.
        """
        return [str(p) for p in self.iter_waveform_files(starttime, endtime, db=db)]

    def read_waveforms(
        self,
        starttime,
        endtime,
        db=None,
        reader: Callable = read,
        return_paths: bool = False,
        fixid: bool = False,
        preprocess: bool = False,
        preprocess_fn=None,
        preprocess_kwargs=None,
        verbose: bool = False,
    ):
        """
        Generator yielding ObsPy Streams, or (path, Stream) pairs if return_paths=True.

        Parameters
        ----------
        fixid : bool
            Apply Seisan/MVO trace-id fixes.
        preprocess : bool
            Apply preprocess_fn after reading/fixing.
        preprocess_fn : callable or None
            Shared waveform preprocessing function.
        preprocess_kwargs : dict or None
            Optional kwargs for preprocess_fn.
        """
        for path in self.iter_waveform_files(starttime, endtime, db=db):
            st = self.read_waveform_file(
                path,
                reader=reader,
                fixid=fixid,
                preprocess=preprocess,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs=preprocess_kwargs,
                verbose=verbose,
            )
            if st is None:
                continue

            if return_paths:
                yield path, st
            else:
                yield st

    # ------------------------------------------------------------------
    # REA (event-based)
    # ------------------------------------------------------------------

    def iter_sfiles(self, starttime, endtime, db=None, mainclass="L", verbose=False):
        """
        Yield S-file paths across a time range, filtered by actual S-file time.

        Parameters
        ----------
        mainclass : str, list, or None
            Examples:
                "L"         -> only L
                "LV"        -> LV
                ["L", "R"]  -> multiple classes
                "*" or None -> all classes
        """
        from flovopy.seisanio.utils.helpers import spath2datetime

        def _normalize_mainclass(mainclass):
            if mainclass is None or mainclass == "*":
                return ["*.S*"]
            if isinstance(mainclass, str):
                return [f"*{mainclass}.S*"]
            return [f"*{mc}.S*" for mc in mainclass]

        if db is None:
            db = self.db_event
        if db is None:
            raise ValueError("No event Seisan database name was provided.")

        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        reapath = self.root / "REA" / db
        years = range(starttime.year, endtime.year + 1)
        patterns = _normalize_mainclass(mainclass)

        for year in years:
            if year == starttime.year == endtime.year:
                months = range(starttime.month, endtime.month + 1)
            elif year == starttime.year:
                months = range(starttime.month, 13)
            elif year == endtime.year:
                months = range(1, endtime.month + 1)
            else:
                months = range(1, 13)

            for month in months:
                yearmonthdir = reapath / f"{year:04d}" / f"{month:02d}"
                if not yearmonthdir.exists():
                    continue

                seen = set()

                for pattern in patterns:
                    for f in yearmonthdir.glob(pattern):
                        if f in seen:
                            continue
                        seen.add(f)

                        try:
                            fdt = spath2datetime(str(f))
                        except Exception as e:
                            if verbose:
                                print(f"[WARN] Could not parse S-file time from {f}: {e}")
                            continue

                        if starttime <= fdt < endtime:
                            yield f

    def iter_events(self, starttime, endtime, db=None, sfile_parser=None, verbose=False):
        """
        Yield parsed Sfile objects.
        """
        if sfile_parser is None:
            raise ValueError("iter_events requires an sfile_parser callable.")

        for path in self.iter_sfiles(starttime, endtime, db=db, verbose=verbose):
            try:
                yield sfile_parser(path)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to parse S-file {path}: {e}")
                continue

    def iter_event_waveforms(
        self,
        starttime,
        endtime,
        db=None,
        sfile_parser=None,
        waveform_reader=None,
        fixid: bool = False,
        preprocess: bool = False,
        preprocess_fn=None,
        preprocess_kwargs=None,
        verbose=False,
    ):
        """
        Yield (Sfile, Stream) pairs.

        If waveform_reader is supplied, it is used directly.
        Otherwise this method uses self.read_waveform_file().
        """
        if sfile_parser is None:
            raise ValueError("iter_event_waveforms requires an sfile_parser callable.")

        for s in self.iter_events(starttime, endtime, db=db, sfile_parser=sfile_parser, verbose=verbose):
            if not s or not getattr(s, "dsnwavfileobj", None):
                continue

            wavpath = getattr(s.dsnwavfileobj, "path", None)
            if not wavpath:
                continue

            try:
                if waveform_reader is not None:
                    st = waveform_reader(wavpath)
                else:
                    st = self.read_waveform_file(
                        wavpath,
                        fixid=fixid,
                        preprocess=preprocess,
                        preprocess_fn=preprocess_fn,
                        preprocess_kwargs=preprocess_kwargs,
                        verbose=verbose,
                    )

                if st is not None:
                    yield s, st

            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to read waveform referenced by {s}: {wavpath}: {e}")
                continue

    # ------------------------------------------------------------------
    # Conversion / export
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Conversion / export
    # ------------------------------------------------------------------

    def to_sds(
        self,
        starttime,
        endtime,
        sds_root,
        db=None,
        reader=read,
        fixid: bool = False,
        preprocess: bool = False,
        preprocess_fn=None,
        preprocess_kwargs=None,
        write_mode: str = "merge",
        write_preprocess: bool = True,
        return_counts: bool = False,
        verbose: bool = False,
        **write_kwargs,
    ):
        """
        Convert a Seisan waveform archive to SDS format.

        This method groups Seisan waveform files by UTC day before writing,
        so that each SDS day file is merged/re-written at most once per call
        per day group.

        Parameters
        ----------
        starttime, endtime : UTCDateTime or compatible
            Time range to process.
        sds_root : str or Path
            Output SDS root directory.
        db : str or None
            Seisan database name (defaults to self.db_cont).
        reader : callable
            Function used to read waveform files (default: obspy.read).
        fixid : bool
            If True, apply Seisan/MVO trace-id fixing during read.
        preprocess : bool
            If True, apply `preprocess_fn` during read.
        preprocess_fn : callable or None
            Optional read-side preprocessing function.
            Signature:
                preprocess_fn(stream, **kwargs) -> stream
        preprocess_kwargs : dict or None
            Optional kwargs for `preprocess_fn`.
        write_mode : str
            SDS write mode passed to EnhancedSDSClient.write_stream():
                - "fail"
                - "overwrite"
                - "merge"   (recommended; default)
        write_preprocess : bool
            If True, apply FLOVOpy pre-write processing inside
            EnhancedSDSClient.write_stream() before writing to SDS.
        return_counts : bool
            If True, return summary statistics.
        verbose : bool
            Print progress messages.
        **write_kwargs
            Additional kwargs passed to EnhancedSDSClient.write_stream().

            Examples:
                merge=True
                merge_strategy="both"
                harmonize_rates=True
                max_sampling_rate=100.0
                fill_value=0.0
                encoding="STEIM2"
                reclen=4096

        Returns
        -------
        dict or None
            If return_counts=True, returns a summary dictionary with keys:
                - files_read
                - files_failed
                - traces_read
                - days_written
                - sds_files_written
                - failed_files
        """


        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        sds_root = Path(sds_root)
        sds_root.mkdir(parents=True, exist_ok=True)

        client = EnhancedSDSClient(str(sds_root))

        # ------------------------------------------------------------
        # Group Seisan waveform files by UTC day using filename time
        # ------------------------------------------------------------
        files_by_day = defaultdict(list)
        failed_files = []

        for path in self.iter_waveform_files(starttime, endtime, db=db):
            try:
                filetime = self._wavpath2datetime(path)
                day = UTCDateTime(filetime.year, julday=filetime.julday)
                files_by_day[day].append(Path(path))
            except Exception as e:
                failed_files.append(str(path))
                if verbose:
                    print(f"[WARN] Could not determine day for {path}: {e}")

        # ------------------------------------------------------------
        # Process one day at a time
        # ------------------------------------------------------------
        n_files_read = 0
        n_files_failed = len(failed_files)
        n_traces_read = 0
        n_days_written = 0
        n_sds_files_written = 0

        for day in sorted(files_by_day.keys()):
            day_files = files_by_day[day]
            day_stream = Stream()

            if verbose:
                print(f"[INFO] Processing {len(day_files)} file(s) for day {day.date}")

            # --------------------------------------------------------
            # Read all Seisan files for this day into one Stream
            # --------------------------------------------------------
            for path in day_files:
                st = self.read_waveform_file(
                    path,
                    reader=reader,
                    fixid=fixid,
                    preprocess=preprocess,
                    preprocess_fn=preprocess_fn,
                    preprocess_kwargs=preprocess_kwargs,
                    verbose=verbose,
                )

                if st is None or len(st) == 0:
                    n_files_failed += 1
                    failed_files.append(str(path))
                    continue

                day_stream += st
                n_files_read += 1
                n_traces_read += len(st)

            if len(day_stream) == 0:
                if verbose:
                    print(f"[WARN] No usable traces for day {day.date}")
                continue

            # --------------------------------------------------------
            # Write once per day-group into SDS
            # --------------------------------------------------------
            try:
                written = client.write_stream(
                    day_stream,
                    mode=write_mode,
                    preprocess=write_preprocess,
                    verbose=verbose,
                    **write_kwargs,
                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to write day {day.date} to SDS: {e}")
                continue

            if written:
                n_days_written += 1
                n_sds_files_written += len(written)

            if verbose:
                print(
                    f"[INFO] Day {day.date}: "
                    f"{len(day_stream)} trace(s) in memory, "
                    f"{len(written) if written else 0} SDS file(s) written"
                )

        if verbose:
            print(
                f"[DONE] Read {n_files_read} Seisan file(s), "
                f"{n_traces_read} trace(s); "
                f"wrote {n_sds_files_written} SDS file(s) across {n_days_written} day(s)"
            )

        if return_counts:
            return {
                "files_read": n_files_read,
                "files_failed": n_files_failed,
                "traces_read": n_traces_read,
                "days_written": n_days_written,
                "sds_files_written": n_sds_files_written,
                "failed_files": failed_files,
            }
    


def get_file_list(SEISAN_DATA, DB, startdate, enddate):
    """
    Return a list of waveform file paths in a continuous Seisan archive.
    """
    archive = SeisanArchive(SEISAN_DATA, db_cont=DB)
    return archive.get_file_list(startdate, enddate)


def get_sfile_list(SEISAN_DATA, DB, startdate, enddate, mainclass="L"):
    """
    Return a list of S-file paths in an event Seisan archive.
    """
    archive = SeisanArchive(SEISAN_DATA, db_event=DB)
    return [str(p) for p in archive.iter_sfiles(startdate, enddate, mainclass=mainclass)]