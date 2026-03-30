from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from obspy import Stream, Trace, read
from obspy.core.inventory import read_inventory
from obspy.core.utcdatetime import UTCDateTime

from flovopy.core.miniseed_io import postprocess_stream_after_read
from flovopy.core.trace_utils import remove_empty_traces
from flovopy.research.mvo.mvo_ids import fix_trace_mvo
from flovopy.seisanio.core.seisanarchive import SeisanArchive


# -----------------------------------------------------------------------------
# Inventory helpers
# -----------------------------------------------------------------------------


def load_mvo_master_inventory(xmldir: str | Path):
    """
    Load the master StationXML file for the Montserrat digital seismic network.

    Parameters
    ----------
    xmldir
        Directory containing ``MontserratDigitalSeismicNetwork.xml``.

    Returns
    -------
    obspy.Inventory | None
        The loaded inventory, or ``None`` if the file is not present.
    """
    xmldir = Path(xmldir)
    master_station_xml = xmldir / "MontserratDigitalSeismicNetwork.xml"
    if master_station_xml.exists():
        return read_inventory(str(master_station_xml))
    return None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _keep_seismic_only(st: Stream) -> Stream:
    """
    Return only seismic traces.

    Here, "seismic" is defined as channels whose second character is ``H`` or
    ``L`` after MVO trace-ID fixing, e.g. ``BHZ``, ``HHN``, ``SHZ``, ``LHZ``.
    """
    st_out = Stream()
    for tr in st:
        chan = str(getattr(tr.stats, "channel", "") or "")
        if len(chan) >= 2 and chan[1] in "HL":
            st_out.append(tr)
    return st_out


# -----------------------------------------------------------------------------
# MVO waveform reader
# -----------------------------------------------------------------------------


def read_mvo_waveform_file(
    wavpath: str | Path,
    *,
    verbose: bool = False,
    seismic_only: bool = False,
    vertical_only: bool = False,
    bypass_processing: bool = False,
    copy: bool = True,
    ensure_float32_data: bool = True,
    ensure_masked_data: bool = False,
    sanitize: bool = True,
    drop_empty: bool = True,
    merge: bool = False,
    merge_strategy: str = "both",
    allow_timeshift: bool = False,
    max_shift_seconds: int = 2,
    harmonize_rates: bool = False,
    min_sampling_rate: float | None = None,
    max_sampling_rate: float | None = None,
) -> Stream:
    """
    Read one Montserrat waveform file and apply the standard FLOVOpy post-read
    normalization using the MVO-specific trace fixer.

    This is the preferred high-level file reader for the MVO archive. It wraps
    ObsPy ``read()`` and then applies ``postprocess_stream_after_read()`` with
    ``trace_fixer=fix_trace_mvo`` so that:

    - temporary MVO/SEISAN component information stored in the location code is
      moved into the channel/component fields,
    - the location code is blanked after that information has been consumed,
    - optional sanitization, merge, and sample-rate harmonization are handled in
      a consistent FLOVOpy way.

    Parameters
    ----------
    wavpath
        Path to a waveform file readable by ObsPy.
    verbose
        Print warnings and debug messages.
    seismic_only
        Keep only seismic channels after MVO ID fixing.
    vertical_only
        Keep only vertical-component traces after MVO ID fixing.
    bypass_processing
        Passed to ``postprocess_stream_after_read()``.
    copy
        Passed to ``postprocess_stream_after_read()``.
    ensure_float32_data
        Passed to ``postprocess_stream_after_read()``.
    ensure_masked_data
        Passed to ``postprocess_stream_after_read()``.
    sanitize
        Passed to ``postprocess_stream_after_read()``.
    drop_empty
        Passed to ``postprocess_stream_after_read()``.
    merge
        If True, merge traces within this file using FLOVOpy's merge logic.
    merge_strategy
        Merge strategy passed to ``postprocess_stream_after_read()``.
        Use ``"both"`` to request FLOVOpy smart merging.
    allow_timeshift
        Passed to ``postprocess_stream_after_read()``.
    max_shift_seconds
        Passed to ``postprocess_stream_after_read()``.
    harmonize_rates
        Passed to ``postprocess_stream_after_read()``.
    min_sampling_rate
        Passed to ``postprocess_stream_after_read()``.
    max_sampling_rate
        Passed to ``postprocess_stream_after_read()``.

    Returns
    -------
    obspy.Stream
        A processed stream. Returns an empty stream if the file is missing or
        unreadable.
    """
    wavpath = Path(wavpath)

    if not wavpath.exists():
        if verbose:
            print(f"[WARN] File does not exist: {wavpath}")
        return Stream()

    try:
        st = read(str(wavpath))
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to read waveform file {wavpath}: {e}")
        return Stream()

    try:
        st = postprocess_stream_after_read(
            st,
            bypass_processing=bypass_processing,
            copy=copy,
            ensure_float32_data=ensure_float32_data,
            ensure_masked_data=ensure_masked_data,
            sanitize=sanitize,
            drop_empty=drop_empty,
            trace_fixer=lambda tr: fix_trace_mvo(tr, verbose=verbose),
            merge=merge,
            merge_strategy=merge_strategy,
            allow_timeshift=allow_timeshift,
            max_shift_seconds=max_shift_seconds,
            harmonize_rates=harmonize_rates,
            min_sampling_rate=min_sampling_rate,
            max_sampling_rate=max_sampling_rate,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"[WARN] Post-read processing failed for {wavpath}: {e}")
        return Stream()

    if not isinstance(st, Stream):
        st = Stream([st]) if isinstance(st, Trace) else Stream()

    if vertical_only:
        st = st.select(component="Z")
    elif seismic_only:
        st = _keep_seismic_only(st)

    if drop_empty:
        remove_empty_traces(st, inplace=True)

    return st


# -----------------------------------------------------------------------------
# MVO-specific Seisan archive wrapper
# -----------------------------------------------------------------------------


class MVOSeisanArchive(SeisanArchive):
    """
    Convenience wrapper around :class:`SeisanArchive` for Montserrat.

    Continuous database
    -------------------
    ``WAV/DSNC_/``

    Event database
    --------------
    ``REA/MVOE_/``

    Notes
    -----
    This class provides both iterator-style methods that yield one stream per
    file/event, and collector methods that assemble many streams into a single
    stream and optionally merge them with FLOVOpy's smart merge logic.
    """

    def __init__(self, root: str | Path):
        super().__init__(root=root, db_cont="DSNC_", db_event="MVOE_")

    # ------------------------------------------------------------------
    # Continuous archive
    # ------------------------------------------------------------------

    def iter_continuous_streams(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        *,
        verbose: bool = False,
        seismic_only: bool = False,
        vertical_only: bool = False,
        bypass_processing: bool = False,
        copy: bool = True,
        ensure_float32_data: bool = True,
        ensure_masked_data: bool = False,
        sanitize: bool = True,
        drop_empty: bool = True,
        merge: bool = False,
        merge_strategy: str = "both",
        allow_timeshift: bool = False,
        max_shift_seconds: int = 2,
        harmonize_rates: bool = False,
        min_sampling_rate: float | None = None,
        max_sampling_rate: float | None = None,
    ) -> Iterator[tuple[Path, Stream]]:
        """
        Yield processed streams from the MVO continuous waveform archive.

        Each yielded stream corresponds to one waveform file returned by
        ``iter_waveform_files()``.
        """
        for wavpath in self.iter_waveform_files(starttime, endtime, db=self.db_cont):
            st = read_mvo_waveform_file(
                wavpath,
                verbose=verbose,
                seismic_only=seismic_only,
                vertical_only=vertical_only,
                bypass_processing=bypass_processing,
                copy=copy,
                ensure_float32_data=ensure_float32_data,
                ensure_masked_data=ensure_masked_data,
                sanitize=sanitize,
                drop_empty=drop_empty,
                merge=merge,
                merge_strategy=merge_strategy,
                allow_timeshift=allow_timeshift,
                max_shift_seconds=max_shift_seconds,
                harmonize_rates=harmonize_rates,
                min_sampling_rate=min_sampling_rate,
                max_sampling_rate=max_sampling_rate,
            )
            if len(st) > 0:
                yield wavpath, st

    def read_continuous_stream(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        *,
        verbose: bool = False,
        seismic_only: bool = False,
        vertical_only: bool = False,
        bypass_processing: bool = False,
        copy: bool = True,
        ensure_float32_data: bool = True,
        ensure_masked_data: bool = False,
        sanitize: bool = True,
        drop_empty: bool = True,
        merge: bool = True,
        merge_strategy: str = "both",
        allow_timeshift: bool = False,
        max_shift_seconds: int = 2,
        harmonize_rates: bool = False,
        min_sampling_rate: float | None = None,
        max_sampling_rate: float | None = None,
        trim: bool = True,
    ) -> Stream:
        """
        Read and assemble a continuous MVO stream across a time window.

        This is the main high-level convenience method for continuous data.
        It reads all matching waveform files, concatenates them, and optionally
        applies FLOVOpy smart merge and trimming.
        """
        st_all = Stream()

        for _, st in self.iter_continuous_streams(
            starttime,
            endtime,
            verbose=verbose,
            seismic_only=seismic_only,
            vertical_only=vertical_only,
            bypass_processing=bypass_processing,
            copy=copy,
            ensure_float32_data=ensure_float32_data,
            ensure_masked_data=ensure_masked_data,
            sanitize=sanitize,
            drop_empty=drop_empty,
            merge=False,
            merge_strategy=merge_strategy,
            allow_timeshift=allow_timeshift,
            max_shift_seconds=max_shift_seconds,
            harmonize_rates=False,
            min_sampling_rate=min_sampling_rate,
            max_sampling_rate=max_sampling_rate,
        ):
            st_all += st

        if len(st_all) == 0:
            return st_all

        st_all = postprocess_stream_after_read(
            st_all,
            bypass_processing=bypass_processing,
            copy=False,
            ensure_float32_data=ensure_float32_data,
            ensure_masked_data=ensure_masked_data,
            sanitize=sanitize,
            drop_empty=drop_empty,
            trace_fixer=lambda tr: fix_trace_mvo(tr, verbose=verbose),
            merge=merge,
            merge_strategy=merge_strategy,
            allow_timeshift=allow_timeshift,
            max_shift_seconds=max_shift_seconds,
            harmonize_rates=harmonize_rates,
            min_sampling_rate=min_sampling_rate,
            max_sampling_rate=max_sampling_rate,
            verbose=verbose,
        )

        if not isinstance(st_all, Stream):
            st_all = Stream([st_all]) if isinstance(st_all, Trace) else Stream()

        if vertical_only:
            st_all = st_all.select(component="Z")
        elif seismic_only:
            st_all = _keep_seismic_only(st_all)

        if trim and len(st_all) > 0:
            st_all.trim(UTCDateTime(starttime), UTCDateTime(endtime))

        if drop_empty:
            remove_empty_traces(st_all, inplace=True)

        return st_all

    # ------------------------------------------------------------------
    # Event archive
    # ------------------------------------------------------------------

    def iter_event_streams(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        *,
        verbose: bool = False,
        parse_aef: bool = False,
        seismic_only: bool = False,
        vertical_only: bool = False,
        valid_subclasses: str = "",
        bypass_processing: bool = False,
        copy: bool = True,
        ensure_float32_data: bool = True,
        ensure_masked_data: bool = False,
        sanitize: bool = True,
        drop_empty: bool = True,
        merge: bool = True,
        merge_strategy: str = "both",
        allow_timeshift: bool = False,
        max_shift_seconds: int = 2,
        harmonize_rates: bool = False,
        min_sampling_rate: float | None = None,
        max_sampling_rate: float | None = None,
    ):
        """
        Yield ``(sfile, stream)`` pairs from the MVO event archive.

        Parameters
        ----------
        valid_subclasses
            If non-empty, only yield events where ``mainclass == 'LV'`` and the
            subclass is in this string.
        """
        from flovopy.research.mvo.mvosfile import MVOSfile

        for sfile_path in self.iter_sfiles(starttime, endtime, db=self.db_event):
            try:
                s = MVOSfile(
                    str(sfile_path),
                    verbose=verbose,
                    parse_aef=parse_aef,
                    try_external_aeffile=False,
                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to parse Sfile {sfile_path}: {e}")
                continue

            if valid_subclasses:
                if not (getattr(s, "mainclass", None) == "LV" and getattr(s, "subclass", "") in valid_subclasses):
                    continue

            if not getattr(s, "dsnwavfileobj", None):
                continue

            wavpath = getattr(s.dsnwavfileobj, "path", None)
            if not wavpath:
                continue

            st = read_mvo_waveform_file(
                wavpath,
                verbose=verbose,
                seismic_only=seismic_only,
                vertical_only=vertical_only,
                bypass_processing=bypass_processing,
                copy=copy,
                ensure_float32_data=ensure_float32_data,
                ensure_masked_data=ensure_masked_data,
                sanitize=sanitize,
                drop_empty=drop_empty,
                merge=merge,
                merge_strategy=merge_strategy,
                allow_timeshift=allow_timeshift,
                max_shift_seconds=max_shift_seconds,
                harmonize_rates=harmonize_rates,
                min_sampling_rate=min_sampling_rate,
                max_sampling_rate=max_sampling_rate,
            )
            if len(st) > 0:
                yield s, st

    def apply_to_events(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        function: Callable | None = None,
        *,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Apply a user function to each event stream.

        The callable should have signature::

            function(sfile, stream, **kwargs)

        If ``function`` is ``None``, yield ``(sfile, stream)`` pairs instead.
        Additional keyword arguments are passed both to
        ``iter_event_streams()`` and, when applicable, to ``function()``.
        """
        for s, st in self.iter_event_streams(
            starttime,
            endtime,
            verbose=verbose,
            **kwargs,
        ):
            if function is None:
                yield s, st
            else:
                function(s, st, **kwargs)