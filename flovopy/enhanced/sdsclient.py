from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Set
import pandas as pd


from obspy import read, Stream, UTCDateTime
from obspy.clients.filesystem.sds import Client

from flovopy.core.miniseed_io import postprocess_stream_after_read, smart_merge
import re

class EnhancedSDSClient(Client):
    """
    Enhanced SDS Client extending ObsPy's filesystem SDS Client.

    Responsibilities
    -----------------
    - Robust NSLC discovery (client + filesystem fallback)
    - Day-based SDS reads (browser-friendly)
    - Safe file discovery and reading
    - Availability / presence primitives
    - SDS filename and path helpers

    Non-responsibilities
    --------------------
    - Plotting
    - Smart merging / downsampling
    - Metadata reconciliation
    - Workflow orchestration
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        sds_root: str | Path,
        sds_type: str = "D",
        format: str = "MSEED",
    ):
        self.sds_root = Path(sds_root).expanduser().resolve()
        super().__init__(str(self.sds_root), sds_type=sds_type, format=format)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def day_bounds(day: UTCDateTime) -> Tuple[UTCDateTime, UTCDateTime]:
        """
        Return (start, end) UTCDateTime for a full UTC day.
        """
        t0 = UTCDateTime(day.year, julday=day.julday)
        return t0, t0 + 86400

    # ------------------------------------------------------------------
    # SDS filename helpers
    # ------------------------------------------------------------------

    def build_sds_filename(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        day: UTCDateTime,
    ) -> Path:
        """
        Build full SDS daily file path without touching disk.
        """
        root = Path(self.sds_root)
        year = f"{day.year:04d}"
        jday = f"{day.julday:03d}"
        loc = loc or "--"
        if len(loc) == 1:
            loc = f"0{loc}"

        fname = f"{net}.{sta}.{loc}.{chan}.D.{year}.{jday}"
        return (
            root
            / year
            / net
            / sta
            / f"{chan}.D"
            / fname
        )

    # ------------------------------------------------------------------
    # NSLC discovery
    # ------------------------------------------------------------------

    def get_nslc_for_day(
        self,
        day: UTCDateTime,
        skip_low_rate: bool = True,
    ) -> Set[Tuple[str, str, str, str]]:
        """
        Return set of (net, sta, loc, chan) tuples reporting on a given day.

        Tries ObsPy client first; falls back to filesystem walk if needed.
        """
        try:
            try:
                nslc = self.get_all_nslc(sds_type="D", datetime=day)
            except TypeError:
                nslc = self.get_all_nslc(sds_type="D", datetime=day.datetime)
            return {
                (n, s, l, c)
                for n, s, l, c in nslc
                if not (skip_low_rate and c.startswith("L"))
            }
        except Exception:
            return self._walk_sds_for_day(day, skip_low_rate=skip_low_rate)

    def iter_trace_ids(
        self,
        startday: UTCDateTime,
        endday: Optional[UTCDateTime] = None,
        skip_low_rate: bool = True,
    ) -> List[str]:
        if endday is None:
            endday = startday + 86400

        nslc = self.iter_nslc(startday, endday)
        ids = {
            f"{n}.{s}.{l}.{c}"
            for n, s, l, c in nslc
            if not (skip_low_rate and c.startswith("L"))
        }
        return sorted(ids)

    
    def iter_nslc(
        self,
        start: UTCDateTime,
        end: UTCDateTime,
    ) -> Set[Tuple[str, str, str, str]]:
        """
        Return set of (net, sta, loc, chan) tuples that exist at any time
        between start and end (inclusive).

        Pure SDS discovery: no filtering, no policy.
        """
        out: Set[Tuple[str, str, str, str]] = set()
        day = UTCDateTime(start.year, julday=start.julday)

        while day < end:
            out |= self.get_nslc_for_day(day, skip_low_rate=False)
            day += 86400

        return out

    # ------------------------------------------------------------------
    # Filesystem fallback
    # ------------------------------------------------------------------

    def _walk_sds_for_day(
        self,
        day: UTCDateTime,
        skip_low_rate: bool = True,
    ) -> Set[Tuple[str, str, str, str]]:
        """
        Walk SDS directory tree manually for a given day.
        """
        year = f"{day.year:04d}"
        jday = f"{day.julday:03d}"

        out: Set[Tuple[str, str, str, str]] = set()
        year_dir = self.sds_root / year
        if not year_dir.exists():
            return out

        for net_dir in year_dir.iterdir():
            if not net_dir.is_dir():
                continue
            for sta_dir in net_dir.iterdir():
                if not sta_dir.is_dir():
                    continue
                for chan_dir in sta_dir.iterdir():
                    if not chan_dir.is_dir() or not chan_dir.name.endswith(".D"):
                        continue
                    chan = chan_dir.name[:-2]
                    if skip_low_rate and chan.startswith("L"):
                        continue

                    for f in chan_dir.glob(f"*.D.{year}.{jday}"):
                        parts = f.name.split(".")
                        if len(parts) >= 4:
                            n, s, l, c = parts[:4]
                            out.add((n, s, l, c))

        return out

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    """
    Reading interface overview
    ==========================

    EnhancedSDSClient provides multiple read methods with different levels
    of abstraction. These are intentionally layered, from low-level/raw
    access to higher-level, fully processed workflows.

    The design principle is:
        raw data access → optional normalization → analysis-ready stream

    ----------------------------------------------------------------------
    1. get_waveforms()  (inherited from ObsPy SDS client)
    ----------------------------------------------------------------------

    Lowest-level SDS access.

    - Reads waveform data for a single NSLC selector and time window.
    - Returns raw ObsPy Stream.
    - No fault tolerance across multiple selectors.
    - No preprocessing or normalization.

    Use when:
    - You need direct control over a single request.
    - You want completely raw data.

    ----------------------------------------------------------------------
    2. read_day()
    ----------------------------------------------------------------------

    Thin wrapper around get_waveforms() for a single UTC day.

    - Reads exactly one UTC day (00:00–24:00).
    - No preprocessing.
    - Minimal error handling.
    - Intended for internal use and building higher-level methods.

    Use when:
    - You are looping over days manually.
    - You want raw daily chunks.

    ----------------------------------------------------------------------
    3. read_files()
    ----------------------------------------------------------------------

    Read a list of MiniSEED (or similar) files.

    - Safe file-by-file reading with exception handling.
    - Optional merge and postprocessing.
    - Not tied to SDS structure.

    Use when:
    - Working with arbitrary file lists.
    - Reprocessing exported or external data.

    ----------------------------------------------------------------------
    4. read()
    ----------------------------------------------------------------------

    Primary high-level read interface.

    - Wraps get_waveforms() and read_day().
    - Supports:
        - selector expansion (net/sta/loc/chan lists)
        - explicit trace_ids
        - optional day-wise reading
        - fault-tolerant multi-request loops
    - Applies optional FLOVOpy postprocessing:
        - dtype normalization
        - sanitization
        - empty-trace removal
        - sampling-rate filtering
        - rate harmonization
        - optional smart_merge
    - Can annotate processing history.

    This is the recommended default entry point for most workflows.

    Use when:
    - You want analysis-ready data.
    - You need robustness across many channels/stations.
    - You want consistent preprocessing.

    ----------------------------------------------------------------------
    5. read_days()
    ----------------------------------------------------------------------

    Convenience wrapper for full-day ranges.

    - Expands start/end to full UTC days.
    - Delegates to read().
    - Ensures:
        - daywise reading
        - no trimming (full days preserved)

    Use when:
    - Working with daily archives.
    - Preparing SDS → downstream pipelines.
    - Bulk processing over multiple days.

    ----------------------------------------------------------------------
    Summary
    ----------------------------------------------------------------------

    Method         Level        Preprocessing   Typical use
    ----------------------------------------------------------------------
    get_waveforms  lowest       none            single request, raw access
    read_day       low          none            internal daily reads
    read_files     medium       optional        arbitrary file ingestion
    read           high         optional        main workflow interface
    read_days      high         optional        full-day bulk workflows

    ----------------------------------------------------------------------
    Guidance
    ----------------------------------------------------------------------

    - Prefer read() for most applications.
    - Use read_days() for daily batch workflows.
    - Use read_files() for non-SDS or ad hoc inputs.
    - Use read_day() and get_waveforms() only when you need fine control.

    """
    def read_day(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        year: str | int,
        jday: str | int,
        merge: int = 0,
        verbose: bool = False,
    ) -> Stream:
        """
        Read one full UTC day from SDS for the given selectors.

        This is a thin wrapper around get_waveforms() for a single UTC day.
        It intentionally does not apply FLOVOpy postprocessing.
        """
        day = UTCDateTime(int(year), julday=int(jday))
        t0, t1 = self.day_bounds(day)

        try:
            return self.get_waveforms(
                network=net,
                station=sta,
                location=loc,
                channel=chan,
                starttime=t0,
                endtime=t1,
                merge=merge,
            )
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to read {net}.{sta}.{loc}.{chan} for {year}:{int(jday):03d}: {e}")
            return Stream()
        
    def read_files(
        self,
        files: Sequence[Path],
        postprocess: bool = False,
        postprocess_kwargs: Optional[dict] = None,
        merge: int | None = None,
        verbose: bool = False,
    ) -> Stream:
        """
        Read a list of MiniSEED files safely.

        Parameters
        ----------
        files
            Sequence of file paths.
        postprocess
            If True, apply postprocess_stream_after_read() to the combined Stream.
        postprocess_kwargs
            Optional kwargs for postprocess_stream_after_read().
        merge
            If not None, apply Stream.merge(method=merge) before postprocessing.
        verbose
            Print warnings.

        Returns
        -------
        Stream
        """
        st = Stream()

        for f in files:
            try:
                st += read(str(f))
            except Exception as e:
                if verbose:
                    print(f"⚠️ Failed to read {f}: {e}")

        if len(st) and merge is not None:
            try:
                st.merge(method=merge)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Merge failed after read_files(): {e}")

        if postprocess and len(st):
            from flovopy.core.miniseed_io import postprocess_stream_after_read

            kwargs = postprocess_kwargs or {}
            try:
                st = postprocess_stream_after_read(
                    st,
                    copy=False,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️ Postprocessing failed after read_files(): {e}")

        return st
        

    def read(
        self,
        starttime: UTCDateTime | str,
        endtime: UTCDateTime | str,
        net: str | Sequence[str] = "*",
        sta: str | Sequence[str] = "*",
        loc: str | Sequence[str] = "*",
        chan: str | Sequence[str] = "*",
        trace_ids: Optional[Sequence[str]] = None,
        skip_low_rate_channels: bool = True,
        merge: int | None = 0,
        trim: bool = True,
        daywise: Optional[bool] = None,
        progress: bool = False,
        verbose: bool = False,
        postprocess: bool = True,
        postprocess_kwargs: Optional[dict] = None,
        final_smart_merge: bool = False,
        annotate_processing: bool = True,
    ) -> Stream:
        """
        High-level SDS read wrapper.

        This wraps raw SDSClient.get_waveforms() / read_day() calls and optionally
        applies FLOVOpy post-read normalization.

        Parameters
        ----------
        starttime, endtime
            Requested time window.
        net, sta, loc, chan
            Selector strings or lists of selector strings.
            Ignored if trace_ids is provided.
        trace_ids
            Optional explicit NSLC list, e.g. ["MV.MBWH..BHZ", ...].
            If None, selector cartesian product is used.
        skip_low_rate_channels
            If True, skip channel codes beginning with "L" before reading.
        merge
            ObsPy merge method applied before postprocessing.
            Use 0 for no merge attempt, or None to skip merge entirely.
        trim
            If True, trim final stream to [starttime, endtime].
        daywise
            If True, read day-by-day using read_day().
            If False, call get_waveforms() directly.
            If None, auto-enable when crossing a UTC day boundary.
        progress
            If True and trace_ids is used, show a tqdm progress bar.
        verbose
            Print warnings for failed reads/processing.
        postprocess
            If True, apply postprocess_stream_after_read() to final stream.
        postprocess_kwargs
            Extra kwargs for postprocess_stream_after_read().
        final_smart_merge
            If True, apply FLOVOpy smart_merge() after postprocessing.
        annotate_processing
            If True, append processing notes to trace.stats.processing.

        Returns
        -------
        Stream
        """
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        if endtime <= starttime:
            return Stream()

        if daywise is None:
            daywise = self._crosses_day_boundary(starttime, endtime)

        st = Stream()

        # ------------------------------------------------------------
        # Build request list
        # ------------------------------------------------------------
        requests = []

        if trace_ids is not None:
            iterator = trace_ids
            if progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(trace_ids, desc="Reading SDS")
                except Exception:
                    iterator = trace_ids

            for tid in iterator:
                try:
                    n, s, l, c = tid.split(".")
                except ValueError:
                    if verbose:
                        print(f"⚠️ Invalid trace_id '{tid}', expected NET.STA.LOC.CHA")
                    continue

                if skip_low_rate_channels and c.startswith("L"):
                    continue

                requests.append((n, s, l, c))

        else:
            nets = self._as_list(net)
            stas = self._as_list(sta)
            locs = self._as_list(loc)
            chans = self._as_list(chan)

            for n in nets:
                for s in stas:
                    for l in locs:
                        for c in chans:
                            if skip_low_rate_channels and c.startswith("L"):
                                continue
                            requests.append((n, s, l, c))

        # ------------------------------------------------------------
        # Raw read loop
        # ------------------------------------------------------------
        for n, s, l, c in requests:
            try:
                if daywise:
                    day = UTCDateTime(starttime.year, julday=starttime.julday)
                    lastday = UTCDateTime(endtime.year, julday=endtime.julday)

                    while day <= lastday:
                        try:
                            day_st = self.read_day(
                                net=n,
                                sta=s,
                                loc=l,
                                chan=c,
                                year=day.year,
                                jday=day.julday,
                                merge=0,
                            )
                            st += day_st
                        except Exception as e:
                            if verbose:
                                print(
                                    f"⚠️ Failed {n}.{s}.{l}.{c} "
                                    f"for {day.year}:{day.julday:03d}: {e}"
                                )
                        day += 86400
                else:
                    st += self.get_waveforms(
                        network=n,
                        station=s,
                        location=l,
                        channel=c,
                        starttime=starttime,
                        endtime=endtime,
                        merge=0,
                    )

            except Exception as e:
                if verbose:
                    print(
                        f"⚠️ Failed to read {n}.{s}.{l}.{c} "
                        f"from {starttime} to {endtime}: {e}"
                    )

        # ------------------------------------------------------------
        # Optional trim before postprocessing
        # ------------------------------------------------------------
        if trim and len(st):
            try:
                st.trim(starttime, endtime)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Trim failed: {e}")

        # ------------------------------------------------------------
        # Optional ordinary ObsPy merge before postprocessing
        # ------------------------------------------------------------
        if len(st) and merge is not None:
            try:
                st.merge(method=merge)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Merge failed: {e}")

        # ------------------------------------------------------------
        # FLOVOpy postprocessing
        # ------------------------------------------------------------
        if postprocess and len(st):
            kwargs = dict(postprocess_kwargs or {})
            try:
                st = postprocess_stream_after_read(
                    st,
                    copy=False,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️ Postprocessing failed: {e}")

        # ------------------------------------------------------------
        # Optional final smart merge
        # ------------------------------------------------------------
        if final_smart_merge and len(st) > 1:
            try:
                strategy = "obspy"
                if postprocess_kwargs:
                    strategy = postprocess_kwargs.get("merge_strategy", "obspy")
                smart_merge(st, strategy=strategy, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"⚠️ final smart_merge failed: {e}")

        # ------------------------------------------------------------
        # Annotate processing
        # ------------------------------------------------------------
        if annotate_processing and len(st):
            for tr in st:
                proc = list(getattr(tr.stats, "processing", []) or [])
                if postprocess:
                    proc.append("flovopy:postprocess_stream_after_read")
                if final_smart_merge:
                    proc.append("flovopy:smart_merge_v1")
                tr.stats.processing = proc

        return st
    
    def read_days(
        self,
        net: str | Sequence[str] = "*",
        sta: str | Sequence[str] = "*",
        loc: str | Sequence[str] = "*",
        chan: str | Sequence[str] = "*",
        startday: UTCDateTime | str = None,
        endday: UTCDateTime | str = None,
        merge: int | None = 0,
        skip_low_rate_channels: bool = True,
        postprocess: bool = True,
        postprocess_kwargs: Optional[dict] = None,
        final_smart_merge: bool = False,
        verbose: bool = False,
    ) -> Stream:
        """
        Read one or more full UTC days for the given selectors.

        Parameters
        ----------
        net, sta, loc, chan
            Selector strings or sequences of strings.
        startday, endday
            Day range. Times, if present, are ignored; full UTC days are read.
        merge
            ObsPy merge method passed through to self.read().
        skip_low_rate_channels
            If True, skip channel codes beginning with "L".
        postprocess
            If True, apply postprocess_stream_after_read() via self.read().
        postprocess_kwargs
            Optional kwargs for postprocessing.
        final_smart_merge
            If True, apply smart_merge after postprocessing.
        verbose
            Print warnings.

        Returns
        -------
        Stream
        """
        startday = UTCDateTime(startday)
        endday = UTCDateTime(endday)

        t0 = UTCDateTime(startday.year, julday=startday.julday)
        t1 = UTCDateTime(endday.year, julday=endday.julday) + 86400

        return self.read(
            starttime=t0,
            endtime=t1,
            net=net,
            sta=sta,
            loc=loc,
            chan=chan,
            merge=merge,
            trim=False,            # full UTC days
            daywise=True,          # explicit daily reads
            skip_low_rate_channels=skip_low_rate_channels,
            postprocess=postprocess,
            postprocess_kwargs=postprocess_kwargs,
            final_smart_merge=final_smart_merge,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Availability primitives
    # ------------------------------------------------------------------

    def _availability_seconds_waveform(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        start: UTCDateTime,
        end: UTCDateTime,
        merge_strategy: str = "obspy",
        verbose: bool = False,
    ) -> float:
        """
        Slow waveform-based estimate of coverage duration in seconds.

        This reads waveform data, trims to the requested interval, merges
        overlapping segments, and sums the resulting durations.
        """
        from flovopy.core.miniseed_io import smart_merge

        start = UTCDateTime(start)
        end = UTCDateTime(end)

        if end <= start:
            return 0.0

        try:
            st = self.get_waveforms(
                network=net,
                station=sta,
                location=loc,
                channel=chan,
                starttime=start,
                endtime=end,
                merge=0,
            )
        except Exception as e:
            if verbose:
                print(
                    f"⚠️ get_waveforms failed for "
                    f"{net}.{sta}.{loc}.{chan} {start}–{end}: {e}"
                )
            return 0.0

        if len(st) == 0:
            return 0.0

        try:
            st.trim(start, end)
        except Exception as e:
            if verbose:
                print(f"⚠️ Trim failed in _availability_seconds_waveform: {e}")

        if len(st) > 1:
            try:
                smart_merge(st, strategy=merge_strategy, verbose=verbose)
            except Exception:
                try:
                    st.merge(method=0)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Merge failed in _availability_seconds_waveform: {e}")

        seconds = 0.0
        for tr in st:
            try:
                seconds += max(0.0, float(tr.stats.endtime - tr.stats.starttime))
            except Exception:
                try:
                    if tr.stats.sampling_rate > 0:
                        seconds += tr.stats.npts / tr.stats.sampling_rate
                except Exception:
                    pass

        return seconds

    def _availability_fraction_waveform(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        start: UTCDateTime,
        end: UTCDateTime,
        merge_strategy: str = "obspy",
        verbose: bool = False,
    ) -> float:
        """
        Slow waveform-based availability fraction in [0, 1].
        """
        start = UTCDateTime(start)
        end = UTCDateTime(end)

        total = max(0.0, float(end - start))
        if total == 0.0:
            return 0.0

        present = self._availability_seconds_waveform(
            net=net,
            sta=sta,
            loc=loc,
            chan=chan,
            start=start,
            end=end,
            merge_strategy=merge_strategy,
            verbose=verbose,
        )
        return min(1.0, present / total)

    def availability_fraction(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        start: UTCDateTime,
        end: UTCDateTime,
        method: str = "fast",
        merge_strategy: str = "obspy",
        verbose: bool = False,
    ) -> float:
        """
        Availability fraction in [0, 1] for the given NSLC and time window.

        Parameters
        ----------
        method
            'fast'     -> use ObsPy's get_availability_percentage()
            'waveform' -> read waveform data and estimate true coverage
        merge_strategy
            Only used when method='waveform'.
        verbose
            If True, print warnings.

        Returns
        -------
        float
            Availability fraction in [0, 1].
        """
        start = UTCDateTime(start)
        end = UTCDateTime(end)

        if end <= start:
            return 0.0

        method = str(method).lower()

        if method == "fast":
            try:
                frac, _ = self.get_availability_percentage(
                    network=net,
                    station=sta,
                    location=loc,
                    channel=chan,
                    starttime=start,
                    endtime=end,
                )
                return float(frac)
            except Exception as e:
                if verbose:
                    print(
                        f"⚠️ get_availability_percentage failed for "
                        f"{net}.{sta}.{loc}.{chan} {start}–{end}: {e}"
                    )
                return 0.0

        if method in ("waveform", "slow"):
            return self._availability_fraction_waveform(
                net=net,
                sta=sta,
                loc=loc,
                chan=chan,
                start=start,
                end=end,
                merge_strategy=merge_strategy,
                verbose=verbose,
            )

        raise ValueError(f"Unknown availability method: {method!r}")

    def availability_fraction_for_day(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        day: UTCDateTime,
        method: str = "fast",
        merge_strategy: str = "obspy",
        verbose: bool = False,
    ) -> float:
        """
        Availability fraction for one UTC day.

        Parameters
        ----------
        day
            Any time within the day of interest.
        method
            'fast' or 'waveform'
        merge_strategy
            Only used when method='waveform'.
        verbose
            If True, print warnings.

        Returns
        -------
        float
            Availability fraction in [0, 1].
        """
        t0, t1 = self.day_bounds(UTCDateTime(day))
        return self.availability_fraction(
            net=net,
            sta=sta,
            loc=loc,
            chan=chan,
            start=t0,
            end=t1,
            method=method,
            merge_strategy=merge_strategy,
            verbose=verbose,
        )

    def has_data_for_day(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        day: UTCDateTime,
    ) -> bool:
        """
        Return True if any data exist for this NSLC on the given day.

        Uses ObsPy's has_data() first, then falls back to checking whether
        the corresponding SDS file exists and is non-empty.
        """
        t0, t1 = self.day_bounds(UTCDateTime(day))

        try:
            return self.has_data(
                network=net,
                station=sta,
                location=loc,
                channel=chan,
                starttime=t0,
                endtime=t1,
            )
        except Exception:
            try:
                path = self.build_sds_filename(net, sta, loc, chan, t0)
                return path.exists() and path.stat().st_size > 0
            except Exception:
                return False

    def get_availability(
        self,
        startday: UTCDateTime,
        endday: UTCDateTime,
        trace_ids: Optional[Sequence[str]] = None,
        skip_low_rate_channels: bool = True,
        progress: bool = True,
        method: str = "fast",
        merge_strategy: str = "obspy",
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Return (availability_df, trace_ids) for a day range.

        Parameters
        ----------
        startday, endday
            Day range. Rows are generated for each UTC day from startday
            up to but not including endday.
        trace_ids
            Explicit trace ID list. If None, discovered via iter_trace_ids().
        skip_low_rate_channels
            If True, skip channel codes beginning with "L" during discovery.
        progress
            If True, show tqdm progress bar if available.
        method
            'fast'     -> use ObsPy get_availability_percentage()
            'waveform' -> read waveform data to estimate coverage
        merge_strategy
            Only used when method='waveform'.
        verbose
            If True, print warnings.

        Returns
        -------
        (availability_df, trace_ids)
            availability_df is a wide DataFrame:
                columns = ['date'] + trace_ids
                values in [0, 1]
        """
        startday = UTCDateTime(startday)
        endday = UTCDateTime(endday)

        if trace_ids is None:
            trace_ids = list(
                self.iter_trace_ids(
                    startday,
                    endday,
                    skip_low_rate=skip_low_rate_channels,
                )
            )
        else:
            trace_ids = list(trace_ids)

        days = []
        t = UTCDateTime(startday.year, julday=startday.julday)
        endday0 = UTCDateTime(endday.year, julday=endday.julday)

        while t < endday0:
            days.append(t)
            t += 86400

        iterator = days
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(days, desc=f"Availability ({method})")
            except Exception:
                pass

        records = []
        for day in iterator:
            for tid in trace_ids:
                try:
                    net, sta, loc, chan = tid.split(".")
                except ValueError:
                    if verbose:
                        print(f"⚠️ Invalid trace_id skipped in get_availability(): {tid}")
                    continue

                frac = self.availability_fraction_for_day(
                    net=net,
                    sta=sta,
                    loc=loc,
                    chan=chan,
                    day=day,
                    method=method,
                    merge_strategy=merge_strategy,
                    verbose=verbose,
                )
                records.append((day.date, tid, frac))

        if not records:
            return pd.DataFrame(columns=["date"]), trace_ids

        df = pd.DataFrame(records, columns=["date", "trace_id", "value"])
        wide = (
            df.pivot_table(
                index="date",
                columns="trace_id",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
        )

        return wide, trace_ids

    def plot_availability(
        self,
        availability_df: Optional[pd.DataFrame] = None,
        *,
        startday: Optional[UTCDateTime] = None,
        endday: Optional[UTCDateTime] = None,
        trace_ids: Optional[Sequence[str]] = None,
        skip_low_rate_channels: bool = True,
        method: str = "fast",
        merge_strategy: str = "obspy",
        progress: bool = True,
        outfile: Optional[str] = None,
        figsize=(12, 9),
        fontsize: int = 10,
        cmap: str = "gray_r",
        show_counts: bool = True,
        count_min_fraction: float = 0.0,
        verbose: bool = False,
    ):
        """
        Plot an availability heatmap.

        Parameters
        ----------
        availability_df
            Wide DataFrame with a 'date' column and one column per trace_id.
            If None, availability is computed internally using startday/endday.
        startday, endday
            Required if availability_df is None.
        trace_ids
            Optional explicit trace ID list for internal availability calculation.
        skip_low_rate_channels
            Used only if availability_df is None and trace_ids is None.
        method
            'fast' or 'waveform'
        merge_strategy
            Only used when method='waveform'.
        progress
            Show progress bar during internal availability calculation.
        outfile
            If given, save figure to this path.
        figsize
            Matplotlib figure size.
        fontsize
            Axis tick label font size.
        cmap
            Matplotlib colormap.
        show_counts
            If True, add a simple per-day count summary to the title.
        count_min_fraction
            Threshold for counting channels "available enough".
        verbose
            Print warnings.

        Returns
        -------
        matplotlib.figure.Figure | None
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        if availability_df is None:
            if startday is None or endday is None:
                raise ValueError(
                    "plot_availability requires either availability_df or both startday and endday."
                )

            availability_df, trace_ids = self.get_availability(
                startday=startday,
                endday=endday,
                trace_ids=trace_ids,
                skip_low_rate_channels=skip_low_rate_channels,
                progress=progress,
                method=method,
                merge_strategy=merge_strategy,
                verbose=verbose,
            )

        if availability_df is None or availability_df.empty:
            print("No availability data to plot.")
            return None

        A_df = availability_df.set_index("date").T
        dates = pd.to_datetime(A_df.columns)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            A_df.to_numpy(),
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(np.arange(len(dates)))
        ax.set_xticklabels(
            dates.strftime("%Y-%m-%d"),
            rotation=90,
            fontsize=fontsize,
        )
        ax.set_yticks(np.arange(len(A_df.index)))
        ax.set_yticklabels(A_df.index, fontsize=fontsize)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        title = f"Availability ({method})"
        if show_counts:
            try:
                counts = (A_df >= count_min_fraction).sum(axis=0)
                title += (
                    f" (median channels/day ≥ {count_min_fraction:.2f}: "
                    f"{int(np.median(counts))})"
                )
            except Exception:
                pass

        ax.set_title(title)
        fig.tight_layout()

        if outfile:
            fig.savefig(outfile, dpi=300)
            print(f"Saved availability plot to {outfile}")

        return fig

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _split_trace_to_days(self, trace: Trace) -> Stream:
        """
        Split a Trace into one or more traces, each confined to a single UTC day.
        """
        out = Stream()

        t0 = trace.stats.starttime
        t1 = trace.stats.endtime

        if t1 <= t0:
            return out

        day = UTCDateTime(t0.year, julday=t0.julday)

        while day <= t1:
            next_day = day + 86400
            seg_start = max(t0, day)
            seg_end = min(t1, next_day)

            if seg_start < seg_end:
                tr_day = trace.copy()
                tr_day.trim(seg_start, seg_end, nearest_sample=False)
                if len(tr_day.data) > 0:
                    out.append(tr_day)

            day = next_day

        return out

    def _split_stream_to_days(self, stream: Stream) -> Stream:
        """
        Split all traces in a Stream into UTC-day-confined traces.
        """
        out = Stream()
        for tr in stream:
            out.extend(self._split_trace_to_days(tr))
        return out

    def _group_stream_by_sds_path(self, stream: Stream) -> dict[Path, Stream]:
        """
        Group a Stream by target SDS output file path.

        Assumes traces are already split to UTC-day-confined segments.
        """
        from collections import defaultdict

        grouped = defaultdict(Stream)

        for tr in stream:
            path = self.build_sds_filename(
                tr.stats.network,
                tr.stats.station,
                tr.stats.location,
                tr.stats.channel,
                tr.stats.starttime,
            )
            grouped[path].append(tr)

        return dict(grouped)

    def _split_pipeline_and_writer_kwargs(self, kwargs: dict) -> tuple[dict, dict]:
        """
        Split kwargs into:
        - preprocess_stream_before_write() kwargs
        - write_mseed() kwargs

        Notes
        -----
        `fill_value` is meaningful for both, so it is passed to both.
        Unknown kwargs are passed through to write_mseed(), where they are
        harmless in low-level mode and reserved for future expansion.
        """
        pipeline_keys = {
            "bypass_processing",
            "copy",
            "ensure_float32_data",
            "ensure_masked_data",
            "sanitize",
            "drop_empty",
            "fix_ids",
            "legacy",
            "netcode",
            "trace_fixer",
            "merge",
            "merge_strategy",
            "allow_timeshift",
            "max_shift_seconds",
            "harmonize_rates",
            "max_sampling_rate",
            "unmask_before_write",
            "fill_value",
            "verbose",
        }

        writer_keys = {
            "fill_value",
            "pickle_fallback",
            "encoding",
            "reclen",
        }

        pipeline_kwargs = {k: v for k, v in kwargs.items() if k in pipeline_keys}
        writer_kwargs = {k: v for k, v in kwargs.items() if k in writer_keys}

        # Pass through anything else to write_mseed() for future compatibility.
        for k, v in kwargs.items():
            if k not in pipeline_keys and k not in writer_keys:
                writer_kwargs[k] = v

        return pipeline_kwargs, writer_kwargs

    def _prepare_stream_for_sds_write(
        self,
        stream: Stream,
        *,
        apply_pipeline: bool,
        force_merge: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Stream:
        """
        Prepare a stream for SDS writing.

        This applies the FLOVOpy pre-write pipeline if requested, but handles
        smart_merge explicitly here because current smart_merge() returns a
        report dict rather than operating in-place.

        Parameters
        ----------
        stream
            Input stream.
        apply_pipeline
            If True, call preprocess_stream_before_write().
        force_merge
            If True, perform smart_merge even if pipeline kwargs did not
            explicitly request merge=True. This is useful when combining
            incoming data with an existing SDS day file.
        verbose
            Print diagnostics.
        **kwargs
            Mixed pipeline/write kwargs. Only pipeline-relevant kwargs are used.

        Returns
        -------
        Stream
            Prepared stream, possibly empty.
        """
        from flovopy.core.miniseed_io import preprocess_stream_before_write, smart_merge
        from flovopy.core.trace_utils import remove_empty_traces

        if stream is None or len(stream) == 0:
            return Stream()

        pipeline_kwargs, _ = self._split_pipeline_and_writer_kwargs(kwargs)
        st = stream.copy()

        # Extract merge controls explicitly so we can apply smart_merge correctly.
        merge_requested = bool(pipeline_kwargs.get("merge", False)) or force_merge
        merge_strategy = pipeline_kwargs.get("merge_strategy", "obspy")
        allow_timeshift = pipeline_kwargs.get("allow_timeshift", False)
        max_shift_seconds = pipeline_kwargs.get("max_shift_seconds", 2)

        if apply_pipeline:
            # Disable merge inside preprocess_stream_before_write() itself,
            # because smart_merge() returns a dict and is not in-place.
            pp_kwargs = dict(pipeline_kwargs)
            pp_kwargs["merge"] = False
            pp_kwargs["verbose"] = verbose

            try:
                st = preprocess_stream_before_write(st, **pp_kwargs)
            except Exception as e:
                if verbose:
                    print(f"⚠️ preprocess_stream_before_write failed: {e}")
                return Stream()

            if st is None or len(st) == 0:
                return Stream()

        if merge_requested and len(st) > 1:
            try:
                report = smart_merge(
                    st,
                    strategy=merge_strategy,
                    allow_timeshift=allow_timeshift,
                    max_shift_seconds=max_shift_seconds,
                    verbose=verbose,
                )
                st = report["merged_stream"]
            except Exception as e:
                if verbose:
                    print(f"⚠️ smart_merge failed during SDS write preparation: {e}")

        try:
            remove_empty_traces(st, inplace=True)
        except Exception:
            pass

        return st if st is not None else Stream()

    def _atomic_write_mseed(
        self,
        stream: Stream | Trace,
        path: Path,
        *,
        overwrite: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> Path | None:
        """
        Write MiniSEED atomically via temporary file and os.replace().

        Parameters
        ----------
        stream
            Stream or Trace to write.
        path
            Final destination path.
        overwrite
            If False and path exists, raise FileExistsError.
        verbose
            Print diagnostics.
        **kwargs
            Passed to write_mseed().

        Returns
        -------
        Path | None
            Final path written, or None on failure.
        """
        import os
        import uuid
        from flovopy.core.miniseed_io import write_mseed

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(f"SDS file exists: {path}")

        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

        try:
            out = write_mseed(
                stream,
                str(tmp),
                overwrite_ok=True,
                pickle_fallback=False,
                use_preprocess_pipeline=False,   # stream should already be prepared
                verbose=verbose,
                **kwargs,
            )
            if out is None:
                if verbose:
                    print(f"⚠️ write_mseed failed for temporary file {tmp}")
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                return None

            outpath = Path(out)
            os.replace(outpath, path)
            return path

        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def write_trace(
        self,
        trace: Trace,
        mode: str = "merge",
        preprocess: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> List[Path]:
        """
        Write a single Trace to SDS.

        This is a thin wrapper around write_stream(Stream([trace]), ...).

        Parameters
        ----------
        trace
            Trace to write.
        mode
            One of:
                - "fail"      : raise if target SDS file already exists
                - "overwrite" : replace existing SDS file
                - "merge"     : read existing SDS file, merge with new data, rewrite
        preprocess
            If True, apply FLOVOpy pre-write processing to the incoming trace.
        verbose
            Print diagnostics.
        **kwargs
            Additional kwargs forwarded through the SDS write pipeline.

        Returns
        -------
        list[Path]
            Paths written.
        """
        if trace is None:
            return []

        return self.write_stream(
            Stream([trace]),
            mode=mode,
            preprocess=preprocess,
            verbose=verbose,
            **kwargs,
        )

    def write_stream(
        self,
        stream: Stream,
        mode: str = "merge",
        preprocess: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> List[Path]:
        """
        Carefully write a Stream into an SDS archive.

        This method is safe for archive conversion workflows where many short
        waveform files (e.g. from Seisan or SUDS) are incrementally ingested
        into an existing SDS archive.

        Workflow
        --------
        1. Optionally preprocess the *incoming* stream once as a whole.
        2. Split traces at UTC day boundaries.
        3. Group incoming traces by target SDS output path.
        4. For each target SDS day file:
           - if absent: write new data
           - if present and mode="fail": raise
           - if present and mode="overwrite": replace file
           - if present and mode="merge": read existing file, combine with new
             data, merge/sanitize, and rewrite safely

        Parameters
        ----------
        stream
            Input stream to write.
        mode
            One of:
                - "fail"
                - "overwrite"
                - "merge"
        preprocess
            If True, apply preprocess_stream_before_write() to the incoming
            stream before splitting/grouping.
        verbose
            Print diagnostics.
        **kwargs
            Mixed kwargs for:
                - preprocess_stream_before_write()
                - write_mseed()

            Common examples include:
                merge=True
                merge_strategy="obspy"
                harmonize_rates=True
                max_sampling_rate=100.0
                fill_value=0.0
                encoding="STEIM2"
                reclen=4096

        Returns
        -------
        list[Path]
            SDS file paths written.
        """
        from obspy import read as obspy_read

        mode = str(mode).lower()
        if mode not in {"fail", "overwrite", "merge"}:
            raise ValueError(f"Unknown SDS write mode: {mode!r}")

        if stream is None or len(stream) == 0:
            return []

        # ------------------------------------------------------------
        # Split kwargs into pipeline vs writer kwargs
        # ------------------------------------------------------------
        pipeline_kwargs, writer_kwargs = self._split_pipeline_and_writer_kwargs(kwargs)

        # ------------------------------------------------------------
        # Step 1: preprocess incoming stream once (optional)
        # ------------------------------------------------------------
        incoming = self._prepare_stream_for_sds_write(
            stream,
            apply_pipeline=preprocess,
            force_merge=False,
            verbose=verbose,
            **pipeline_kwargs,
        ) if preprocess else stream.copy()

        if incoming is None or len(incoming) == 0:
            return []

        # ------------------------------------------------------------
        # Step 2: split incoming stream to UTC-day traces
        # ------------------------------------------------------------
        incoming_day = self._split_stream_to_days(incoming)
        if len(incoming_day) == 0:
            return []

        # ------------------------------------------------------------
        # Step 3: group incoming traces by target SDS file
        # ------------------------------------------------------------
        grouped_incoming = self._group_stream_by_sds_path(incoming_day)

        written: List[Path] = []

        # ------------------------------------------------------------
        # Step 4: process each target SDS day file exactly once
        # ------------------------------------------------------------
        for path, new_group in grouped_incoming.items():
            path = Path(path)

            if len(new_group) == 0:
                continue

            if path.exists():
                if mode == "fail":
                    raise FileExistsError(f"SDS file exists: {path}")

                elif mode == "overwrite":
                    final_stream = new_group

                elif mode == "merge":
                    try:
                        existing = obspy_read(str(path))
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Failed to read existing SDS file {path}: {e}")
                        existing = Stream()

                    combined = existing + new_group

                    # For on-disk merge mode, always force a merge pass on the
                    # combined stream before rewriting.
                    final_stream = self._prepare_stream_for_sds_write(
                        combined,
                        apply_pipeline=True,
                        force_merge=True,
                        verbose=verbose,
                        **pipeline_kwargs,
                    )
                else:
                    # defensive, should never get here
                    raise ValueError(f"Unknown mode: {mode!r}")

            else:
                # No existing SDS file
                final_stream = new_group

                # If preprocess was False on input, but kwargs request e.g.
                # merge/harmonize/fix_ids, we should still honor those here.
                if not preprocess and len(final_stream):
                    final_stream = self._prepare_stream_for_sds_write(
                        final_stream,
                        apply_pipeline=bool(pipeline_kwargs),
                        force_merge=False,
                        verbose=verbose,
                        **pipeline_kwargs,
                    )

            if final_stream is None or len(final_stream) == 0:
                if verbose:
                    print(f"⚠️ Nothing to write for SDS file {path}")
                continue

            out = self._atomic_write_mseed(
                final_stream,
                path,
                overwrite=True,   # safe here because merge/fail/overwrite already handled
                verbose=verbose,
                **writer_kwargs,
            )

            if out is None:
                if verbose:
                    print(f"⚠️ Failed to finalize SDS file {path}")
                continue

            written.append(out)

        return written



    



    @staticmethod
    def _crosses_day_boundary(starttime: UTCDateTime, endtime: UTCDateTime) -> bool:
        """
        Return True if the time window spans more than one UTC day.
        """
        s0 = UTCDateTime(starttime.year, julday=starttime.julday)
        e0 = UTCDateTime(endtime.year, julday=endtime.julday)
        return e0 > s0
    

    def ingest_files(
        self,
        files: Sequence[str | Path],
        *,
        reader: Callable = read,
        batch_size: int = 100,
        continue_on_error: bool = True,
        preprocess_on_read: bool = False,
        postprocess_read_kwargs: Optional[dict] = None,
        write_mode: str = "merge",
        preprocess_before_write: bool = True,
        verbose: bool = False,
        progress: bool = True,
        **write_kwargs,
    ) -> dict:
        """
        Ingest many waveform files into the SDS archive.

        This is intended for archive-conversion workflows where many short
        waveform files (e.g. Seisan, SUDS, or MiniSEED files) are read and
        merged into SDS day files safely.

        Workflow
        --------
        1. Read source files safely, one by one.
        2. Optionally apply post-read normalization to each file.
        3. Accumulate traces in batches.
        4. Write each batch into SDS using write_stream(..., mode="merge").

        Parameters
        ----------
        files
            Sequence of file paths to ingest.
        reader
            Reader function, default `obspy.read`.
        batch_size
            Number of successfully read files to accumulate before writing.
            Smaller values reduce memory use; larger values may reduce repeated
            SDS day-file rewrites.
        continue_on_error
            If True, skip unreadable/unwritable files and continue.
            If False, raise immediately on first error.
        preprocess_on_read
            If True, apply FLOVOpy post-read normalization to each file after
            reading and before adding it to the batch.
        postprocess_read_kwargs
            Optional kwargs passed to `postprocess_stream_after_read()`.
        write_mode
            Passed to `write_stream()`:
                - "fail"
                - "overwrite"
                - "merge"   (recommended for archive building)
        preprocess_before_write
            If True, apply FLOVOpy pre-write normalization to each batch before
            SDS writing.
        verbose
            Print diagnostics.
        progress
            If True, show a tqdm progress bar if available.
        **write_kwargs
            Additional kwargs forwarded to `write_stream()`, and ultimately to
            the preprocessing pipeline / `write_mseed()`.

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
        dict
            Summary report with keys:
                - files_total
                - files_read_ok
                - files_read_failed
                - files_write_failed
                - batches_written
                - sds_files_written
                - failed_files

        Notes
        -----
        - This method is intentionally batch-oriented to avoid repeatedly
        reading/writing the same SDS day file for every tiny source file.
        - If your source files are strongly interleaved in time, increasing
        `batch_size` may reduce repeated rewrites of the same SDS day files.

        Simple example:
        ---------------
        summary = client.ingest_files(
            file_list,
            batch_size=200,
            write_mode="merge",
            preprocess_before_write=True,
            merge=True,
            merge_strategy="both",
            harmonize_rates=True,
            max_sampling_rate=100.0,
            encoding="STEIM2",
            reclen=4096,
            verbose=True,
        )

        Advanced example:
        -----------------
        # basic processing on read
        summary = client.ingest_files(
            file_list,
            preprocess_on_read=True,
            postprocess_read_kwargs=dict(
                sanitize=True,
                drop_empty=True,
                fix_ids=True,
                netcode="MV",
                min_sampling_rate=50.0,
            ),
            write_mode="merge",
            preprocess_before_write=True,
            merge=True,
            merge_strategy="both",
            verbose=True,
        )


        """
        from flovopy.core.miniseed_io import postprocess_stream_after_read

        files = [Path(f) for f in files]

        summary = {
            "files_total": len(files),
            "files_read_ok": 0,
            "files_read_failed": 0,
            "files_write_failed": 0,
            "batches_written": 0,
            "sds_files_written": 0,
            "failed_files": [],
        }

        if len(files) == 0:
            return summary

        iterator = files
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(files, desc="Ingesting files")
            except Exception:
                pass

        batch = Stream()
        n_in_batch = 0

        def _flush_batch() -> int:
            """
            Write current batch to SDS and reset it.

            Returns
            -------
            int
                Number of SDS files written for this batch.
            """
            nonlocal batch, n_in_batch

            if len(batch) == 0:
                n_in_batch = 0
                return 0

            try:
                written = self.write_stream(
                    batch,
                    mode=write_mode,
                    preprocess=preprocess_before_write,
                    verbose=verbose,
                    **write_kwargs,
                )
                n_written = len(written)
                summary["batches_written"] += 1
                summary["sds_files_written"] += n_written
                batch = Stream()
                n_in_batch = 0
                return n_written

            except Exception as e:
                summary["files_write_failed"] += n_in_batch
                if verbose:
                    print(f"⚠️ Failed to write batch of {n_in_batch} files: {e}")
                batch = Stream()
                n_in_batch = 0

                if continue_on_error:
                    return 0
                raise

        for f in iterator:
            try:
                st = reader(str(f))
            except Exception as e:
                summary["files_read_failed"] += 1
                summary["failed_files"].append(str(f))
                if verbose:
                    print(f"⚠️ Failed to read {f}: {e}")
                if continue_on_error:
                    continue
                raise

            if preprocess_on_read:
                try:
                    st = postprocess_stream_after_read(
                        st,
                        copy=False,
                        verbose=verbose,
                        **(postprocess_read_kwargs or {}),
                    )
                except Exception as e:
                    summary["files_read_failed"] += 1
                    summary["failed_files"].append(str(f))
                    if verbose:
                        print(f"⚠️ Post-read processing failed for {f}: {e}")
                    if continue_on_error:
                        continue
                    raise

            if st is None or len(st) == 0:
                summary["files_read_failed"] += 1
                summary["failed_files"].append(str(f))
                if verbose:
                    print(f"⚠️ No usable traces in {f}")
                if continue_on_error:
                    continue
                raise RuntimeError(f"No usable traces in {f}")

            batch += st
            n_in_batch += 1
            summary["files_read_ok"] += 1

            if n_in_batch >= batch_size:
                _flush_batch()

        # flush final partial batch
        _flush_batch()

        return summary
    
    # ------------------------------------------------------------------
    # SDS path / filename parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_sds_dirname(dir_path) -> tuple[str, str, str, str] | None:
        """
        Parse an SDS directory path.

        Expected layout
        ---------------
        ... / YEAR / NET / STA / CHAN.D

        Parameters
        ----------
        dir_path : str or Path
            Directory path to parse.

        Returns
        -------
        tuple or None
            (year, network, station, channel) if the path matches the
            expected SDS directory structure, else None.

        Examples
        --------
        For a path like:
            /data/SDS/2024/MV/MBWH/BHZ.D

        returns:
            ("2024", "MV", "MBWH", "BHZ")
        """
        import re
        from pathlib import Path

        parts = Path(dir_path).parts[-4:]
        if len(parts) != 4:
            return None

        year, net, sta, chan_d = parts

        if not (year.isdigit() and len(year) == 4):
            return None
        if not re.match(r"^[A-Z0-9]{1,8}$", net, re.IGNORECASE):
            return None
        if not re.match(r"^[A-Z0-9]{1,8}$", sta, re.IGNORECASE):
            return None

        m = re.match(r"^([A-Z0-9]{3})\.D$", chan_d, re.IGNORECASE)
        if not m:
            return None

        chan = m.group(1).upper()
        return year, net.upper(), sta.upper(), chan

    @classmethod
    def is_valid_sds_dir(cls, dir_path) -> bool:
        """
        Return True if a directory path matches SDS layout:

            ... / YEAR / NET / STA / CHAN.D
        """
        return cls.parse_sds_dirname(dir_path) is not None

    @staticmethod
    def parse_sds_filename(
        filename,
        normalize_empty_loc: bool = True,
    ) -> tuple[str, str, str, str, str, str, str] | None:
        """
        Parse an SDS MiniSEED filename.

        Expected format
        ---------------
            NET.STA.LOC.CHAN.TYPE.YEAR.DAY

        where TYPE is typically 'D' for daily waveform data.

        Parameters
        ----------
        filename : str or Path
            Filename or full path.
        normalize_empty_loc : bool
            If True, normalize an empty location code ('') to '--'.

        Returns
        -------
        tuple or None
            (network, station, location, channel, type, year, day),
            or None if the filename does not match SDS format.

        Notes
        -----
        - Location codes of length 0–2 are accepted.
        - Returned strings are uppercased.
        """
        import re
        from pathlib import Path

        filename = Path(filename).name

        pattern = (
            r"^([A-Z0-9]+)\."          # NET
            r"([A-Z0-9]+)\."           # STA
            r"([A-Z0-9\-]{0,2})\."     # LOC (0-2 chars, may be empty or --)
            r"([A-Z0-9]+)\."           # CHAN
            r"([A-Z])\."               # TYPE
            r"(\d{4})\."               # YEAR
            r"(\d{3})$"                # DAY
        )

        m = re.match(pattern, filename, re.IGNORECASE)
        if not m:
            return None

        net, sta, loc, chan, dtype, year, day = m.groups()

        net = net.upper()
        sta = sta.upper()
        loc = loc.upper()
        chan = chan.upper()
        dtype = dtype.upper()

        if normalize_empty_loc and loc == "":
            loc = "--"

        return net, sta, loc, chan, dtype, year, day

    @classmethod
    def is_valid_sds_filename(cls, filename) -> bool:
        """
        Return True if a filename matches SDS MiniSEED naming and has
        TYPE == 'D'.

        Accepted format:
            NET.STA.LOC.CHAN.D.YEAR.DAY
        """
        parsed = cls.parse_sds_filename(filename)
        if parsed is None:
            return False

        _, _, _, _, dtype, _, _ = parsed
        return dtype == "D"

    @classmethod
    def parse_sds_path(
        cls,
        path,
        normalize_empty_loc: bool = True,
        check_consistency: bool = True,
    ) -> dict | None:
        """
        Parse a full SDS file path into both directory and filename components.

        Parameters
        ----------
        path : str or Path
            Full SDS file path.
        normalize_empty_loc : bool
            Passed to parse_sds_filename().
        check_consistency : bool
            If True, verify that directory and filename components agree.

        Returns
        -------
        dict or None
            Dictionary with keys:
                - path
                - year
                - network
                - station
                - location
                - channel
                - type
                - day

            Returns None if parsing fails.

        Notes
        -----
        For a canonical SDS path like:

            /SDS/2024/MV/MBWH/BHZ.D/MV.MBWH.--.BHZ.D.2024.123

        this method verifies both:
            - directory structure
            - filename structure

        and optionally checks that they are internally consistent.
        """
        from pathlib import Path

        path = Path(path)

        dir_info = cls.parse_sds_dirname(path.parent)
        file_info = cls.parse_sds_filename(
            path.name,
            normalize_empty_loc=normalize_empty_loc,
        )

        if dir_info is None or file_info is None:
            return None

        dir_year, dir_net, dir_sta, dir_chan = dir_info
        net, sta, loc, chan, dtype, year, day = file_info

        if check_consistency:
            if year != dir_year:
                return None
            if net != dir_net:
                return None
            if sta != dir_sta:
                return None
            if chan != dir_chan:
                return None
            if dtype != "D":
                return None

        return {
            "path": path,
            "year": year,
            "network": net,
            "station": sta,
            "location": loc,
            "channel": chan,
            "type": dtype,
            "day": day,
        }
    
    # ------------------------------------------------------------------
    # Export: SDS → BUD (via symlinks)
    # ------------------------------------------------------------------

    def to_bud_symlinks(
        self,
        bud_root,
        *,
        starttime: UTCDateTime | str | None = None,
        endtime: UTCDateTime | str | None = None,
        net: str | Sequence[str] = "*",
        sta: str | Sequence[str] = "*",
        chan: str | Sequence[str] = "*",
        overwrite: bool = False,
        dry_run: bool = False,
        loc_blank_as: str = "..",
        verbose: bool = True,
    ) -> dict:
        """
        Create a BUD-style archive of symlinks pointing to SDS files.

        This does NOT rewrite MiniSEED data. Instead, it creates a BUD
        directory tree populated with symlinks to SDS daily files.

        Parameters
        ----------
        bud_root : str or Path
            Destination root directory for BUD archive.
        starttime, endtime : UTCDateTime, str, or None
            Optional time range filter. Files are filtered by the UTC day
            encoded in their SDS filename. A file is included if its day
            intersects the requested interval.
        net, sta, chan : str or sequence of str
            Optional selectors supporting shell-style wildcards via fnmatch.
            Examples:
                net="MV"
                sta=["MBWH", "M*"]
                chan="BH?"
        overwrite : bool
            If True, overwrite existing files/symlinks.
        dry_run : bool
            If True, print actions without creating links.
        loc_blank_as : str
            Representation of blank location codes in BUD filenames
            (default: '..').
        verbose : bool
            Print progress messages.

        Returns
        -------
        dict
            Summary:
                {
                    "created": int,
                    "skipped": int,
                    "considered": int,
                    "matched": int,
                }

        Notes
        -----
        SDS layout:
            <SDS_ROOT>/<YYYY>/<NET>/<STA>/<CHAN>.D/
                NET.STA.LOC.CHAN.D.YYYY.DDD

        BUD layout:
            <BUD_ROOT>/<NET>/<STA>/<CHAN>/
                STA.NET.LOC.CHAN.YYYY.DDD[ext]

        - Location code:
            SDS: '--' or ''
            BUD: typically '..'
        - Symlinks are relative paths for portability.
        

        Examples
        --------
        Convert entire SDS archive:
        >>> client.to_bud_symlinks("/data/BUD")

        Filter by network and station pattern:
        >>> client.to_bud_symlinks(
        ...     "/data/BUD",
        ...     net="MV",
        ...     sta="MB*",
        ... )

        Filter by time range and channels:
        >>> client.to_bud_symlinks(
        ...     "/data/BUD",
        ...     starttime="2024-05-01T00:00:00",
        ...     endtime="2024-06-01T00:00:00",
        ...     net=["MV", "XB"],
        ...     chan=["BHZ", "HHZ"],
        ... )

        Dry run (no files created):
        >>> client.to_bud_symlinks(
        ...     "/data/BUD",
        ...     net="MV",
        ...     dry_run=True,
        ... )
        """


    
        import os
        import fnmatch


        sds_root = Path(self.sds_root)
        bud_root = Path(bud_root)

        def _as_list(x):
            if isinstance(x, (list, tuple, set)):
                return list(x)
            return [x]

        net_patterns = [str(x).upper() for x in _as_list(net)]
        sta_patterns = [str(x).upper() for x in _as_list(sta)]
        chan_patterns = [str(x).upper() for x in _as_list(chan)]

        def _match_any(value: str, patterns: list[str]) -> bool:
            value = value.upper()
            return any(fnmatch.fnmatchcase(value, pat) for pat in patterns)

        def _bud_filename(net, sta, loc, chan, year, doy, ext):
            loc_field = loc_blank_as if (loc in ("", "--")) else loc
            return f"{sta}.{net}.{loc_field}.{chan}.{year}.{doy}{ext}"

        def _create_symlink(src: Path, dest: Path):
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() or dest.is_symlink():
                if overwrite:
                    dest.unlink()
                else:
                    return False

            rel_src = os.path.relpath(src, dest.parent)

            if not dry_run:
                os.symlink(rel_src, dest)

            return True

        # Normalize time window
        if starttime is not None:
            starttime = UTCDateTime(starttime)
        if endtime is not None:
            endtime = UTCDateTime(endtime)

        def _time_ok(year: str, doy: str) -> bool:
            if starttime is None and endtime is None:
                return True

            day_start = UTCDateTime(int(year), julday=int(doy))
            day_end = day_start + 86400

            if starttime is not None and day_end <= starttime:
                return False
            if endtime is not None and day_start >= endtime:
                return False
            return True

        created = 0
        skipped = 0
        considered = 0
        matched = 0

        for year_dir in sds_root.iterdir():
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            if not (len(year) == 4 and year.isdigit()):
                continue

            for net_dir in year_dir.iterdir():
                if not net_dir.is_dir():
                    continue

                net_name = net_dir.name.upper()
                if not _match_any(net_name, net_patterns):
                    continue

                for sta_dir in net_dir.iterdir():
                    if not sta_dir.is_dir():
                        continue

                    sta_name = sta_dir.name.upper()
                    if not _match_any(sta_name, sta_patterns):
                        continue

                    for chanD_dir in sta_dir.iterdir():
                        if not chanD_dir.is_dir() or not chanD_dir.name.endswith(".D"):
                            continue

                        chan_name = chanD_dir.name[:-2].upper()  # strip ".D"
                        if not _match_any(chan_name, chan_patterns):
                            continue

                        for f in chanD_dir.iterdir():
                            if not f.is_file():
                                continue

                            considered += 1

                            parsed = self.parse_sds_filename(f.name)
                            if parsed is None:
                                continue

                            net0, sta0, loc0, chan0, dtype, fyear, doy = parsed

                            if dtype != "D":
                                continue
                            if fyear != year:
                                continue
                            if not _time_ok(fyear, doy):
                                continue

                            # Preserve suffixes like .mseed, .gz if present
                            suffixes = Path(f.name).suffixes
                            ext = ""
                            if len(suffixes) > 0:
                                # Remove the core SDS suffixes from the name calculation;
                                # Path.suffixes only returns actual filesystem suffixes,
                                # so for typical SDS names ext will often be "".
                                ext = "".join(suffixes)

                            matched += 1

                            dest_dir = bud_root / net0 / sta0 / chan0
                            dest_name = _bud_filename(
                                net0, sta0, loc0, chan0, fyear, doy, ext
                            )
                            dest = dest_dir / dest_name

                            if dry_run and verbose:
                                rel_src = os.path.relpath(f, dest_dir)
                                print(f"LINK: {dest} -> {rel_src}")

                            try:
                                if _create_symlink(f, dest):
                                    created += 1
                                else:
                                    skipped += 1
                            except Exception as e:
                                skipped += 1
                                if verbose:
                                    print(f"[WARN] Failed: {dest} ({e})")

        if verbose:
            print(
                f"[DONE] Considered {considered} SDS files, matched {matched}, "
                f"created {created} link(s), skipped {skipped}"
            )

        return {
            "created": created,
            "skipped": skipped,
            "considered": considered,
            "matched": matched,
        }