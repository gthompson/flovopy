from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Set

from obspy import read, Stream, UTCDateTime
from obspy.clients.filesystem.sds import Client

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

    def read_day(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        year: str | int,
        jday: str | int,
        merge: int = 0,
    ) -> Stream:
        """
        Read one full UTC day from SDS for the given selectors.

        Parameters may include wildcards if supported by the underlying
        ObsPy SDS client get_waveforms().
        """
        day = UTCDateTime(int(year), julday=int(jday))
        t0, t1 = self.day_bounds(day)
        return self.get_waveforms(
            network=net,
            station=sta,
            location=loc,
            channel=chan,
            starttime=t0,
            endtime=t1,
            merge=merge,
        )

    def read_files(
        self,
        files: Sequence[Path],
        verbose: bool = False,
    ) -> Stream:
        """
        Read a list of MiniSEED files safely.
        """
        st = Stream()
        for f in files:
            try:
                st += read(str(f))
            except Exception as e:
                if verbose:
                    print(f"⚠️ Failed to read {f}: {e}")
        return st

    # ------------------------------------------------------------------
    # Availability primitives
    # ------------------------------------------------------------------

    def availability_seconds(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        start: UTCDateTime,
        end: UTCDateTime,
    ) -> float:
        """
        Return total seconds of data present between start and end.

        Pure mechanical measure based on sample count.
        """
        try:
            st = self.get_waveforms(
                net, sta, loc, chan,
                start, end,
                merge=0
            )
        except Exception:
            return 0.0

        seconds = 0.0
        for tr in st:
            if tr.stats.sampling_rate > 0:
                seconds += tr.stats.npts / tr.stats.sampling_rate
        return seconds

    def availability_fraction(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        start: UTCDateTime,
        end: UTCDateTime,
    ) -> float:
        """
        Fraction in [0, 1] of expected time covered by samples.
        """
        total = max(0.0, float(end - start))
        if total == 0:
            return 0.0

        present = self.availability_seconds(net, sta, loc, chan, start, end)
        return min(1.0, present / total)

    def availability_fraction_for_day(
        self,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        day: UTCDateTime,
    ) -> float:
        """
        Convenience wrapper for a UTC day.
        """
        t0, t1 = self.day_bounds(day)
        return self.availability_fraction(net, sta, loc, chan, t0, t1)

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
        """
        t0, t1 = self.day_bounds(day)

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
            # Fallback to filesystem existence
            try:
                path = self.build_sds_filename(net, sta, loc, chan, day)
                return path.exists() and path.stat().st_size > 0
            except Exception:
                return False

    def has_data_for_browser_day(
        self,
        *,
        net: str,
        sta: str,
        loc: str,
        chan: str,
        day: UTCDateTime,
    ) -> bool:
        """
        Browser-facing convenience wrapper.
        """
        return self.has_data_for_day(
            net=net or "*",
            sta=sta or "*",
            loc=loc or "*",
            chan=chan or "*",
            day=day,
        )


    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_trace(
        self,
        trace,
        overwrite: bool = False,
    ) -> Path:
        """
        Write a single ObsPy Trace to SDS.

        No merging, no gap logic, no smart behavior.
        """
        path = self.build_sds_filename(
            trace.stats.network,
            trace.stats.station,
            trace.stats.location,
            trace.stats.channel,
            trace.stats.starttime,
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(f"SDS file exists: {path}")

        trace.write(str(path), format=self.format)
        return path


    def write_stream(
        self,
        stream: Stream,
        overwrite: bool = False,
    ) -> List[Path]:
        """
        Write each trace in a Stream independently.
        """
        paths: List[Path] = []
        for tr in stream:
            paths.append(self.write_trace(tr, overwrite=overwrite))
        return paths



    
    def read_days(
        self,
        net: str | Sequence[str] = "*",
        sta: str | Sequence[str] = "*",
        loc: str | Sequence[str] = "*",
        chan: str | Sequence[str] = "*",
        startday: UTCDateTime | str = None,
        endday: UTCDateTime | str = None,
        merge: int = 0,
        verbose: bool = False,
    ) -> Stream:
        """
        Read one or more full UTC days for the given selectors.
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
            trim=False,     # full days
            daywise=True,   # explicit
            verbose=verbose,
        )
    

    def read(
        self,
        starttime: UTCDateTime | str,
        endtime: UTCDateTime | str,
        net: str | Sequence[str] = "*",
        sta: str | Sequence[str] = "*",
        loc: str | Sequence[str] = "*",
        chan: str | Sequence[str] = "*",
        merge: int = 0,
        trim: bool = True,
        daywise: Optional[bool] = None,
        verbose: bool = False,
    ) -> Stream:
        """
        Read waveform data for an arbitrary time window.

        This is a flexible wrapper around ObsPy SDSClient.get_waveforms() that
        additionally supports selector lists (e.g. multiple stations) and optional
        day-wise reading across UTC day boundaries.

        Parameters
        ----------
        starttime, endtime
            Requested time window.
        net, sta, loc, chan
            Selectors. Each may be:
                - a string, including wildcards supported by ObsPy
                - or a sequence of strings
            Sequences are expanded as a cartesian product.
        merge
            If 0, do not merge after reading.
            If nonzero, call Stream.merge(method=merge) after concatenation.
        trim
            If True, trim final output to exactly [starttime, endtime].
        daywise
            If True, read by looping over UTC days and concatenating results.
            If False, call get_waveforms() directly for the full interval.
            If None, automatically use day-wise reading only when the request
            crosses a UTC day boundary.
        verbose
            If True, print warnings for failed reads/merges.

        Returns
        -------
        Stream
            Combined Stream of all matching traces.
        """
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)

        if endtime <= starttime:
            return Stream()

        nets = self._as_list(net)
        stas = self._as_list(sta)
        locs = self._as_list(loc)
        chans = self._as_list(chan)

        if daywise is None:
            daywise = self._crosses_day_boundary(starttime, endtime)

        st = Stream()

        for n in nets:
            for s in stas:
                for l in locs:
                    for c in chans:
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

        if trim and len(st):
            try:
                st.trim(starttime, endtime)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Trim failed: {e}")

        if len(st) and merge is not None:
            try:
                st.merge(method=merge)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Merge failed: {e}")

        return st
    
    @staticmethod
    def _as_list(x) -> List[str]:
        """
        Normalize selector input to a list of strings.
        """
        if x is None:
            return ["*"]
        if isinstance(x, str):
            return [x]
        return [str(v) for v in x]


    @staticmethod
    def _crosses_day_boundary(starttime: UTCDateTime, endtime: UTCDateTime) -> bool:
        """
        Return True if the time window spans more than one UTC day.
        """
        s0 = UTCDateTime(starttime.year, julday=starttime.julday)
        e0 = UTCDateTime(endtime.year, julday=endtime.julday)
        return e0 > s0