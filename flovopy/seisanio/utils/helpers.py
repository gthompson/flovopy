import os
from glob import glob
#import datetime as dt
from obspy import UTCDateTime, Stream
from flovopy.core.miniseed_io import write_mseed
from pathlib import Path


# need to leave this here to prevent circular imports between Wavfile and Sfile
def filetime2spath(filetime, mainclass='L', db=None, seisan_top=None, fullpath=True):
    """Create full path to a SEISAN S-file from a datetime object."""
    spath = '%02d-%02d%02d-%02d%s.S%4d%02d' % (
        filetime.day, filetime.hour, filetime.minute, filetime.second,
        mainclass, filetime.year, filetime.month
    )
    if fullpath:
        spath = os.path.join(f"{filetime.year}", f"{filetime.month:02d}", spath)
        if db:
            spath = os.path.join("REA", db, spath)
            if seisan_top:
                spath = os.path.join(seisan_top, spath)
    return spath

def spath2datetime(spath):
    """Extract datetime from SEISAN S-file path."""
    basename = os.path.basename(spath)
    if '.S' in spath: # 
        parts = basename.split('.S')
        yyyy = int(parts[1][0:4])
        mm = int(parts[1][4:6])
        parts = parts[0].split('-')
        dd = int(parts[0])
        HH = int(parts[1][0:2])
        MM = int(parts[1][2:4])
        SS = float(parts[2][0:2])
        return UTCDateTime(yyyy, mm, dd, HH, MM, SS)
    else:
        return None    




def filetime2wavpath(filetime, sfilepath, y2kfix=False, numchans=0, dbstring='MVO__'):
    """Create full path to a WAV file from datetime object."""

    if len(dbstring) < 5:
        dbstring += '_' * (5 - len(dbstring))

    wavdir = os.path.dirname(sfilepath).replace('REA', 'WAV')


    if not y2kfix and filetime.year < 2000:
        basename = '%2d%02d-%02d-%02d%02d-%02dS.%s_%03d' % (
            filetime.year - 1900, filetime.month, filetime.day,
            filetime.hour, filetime.minute, filetime.second,
            dbstring, numchans
        )
    else:
        basename = '%4d-%02d-%02d-%02d%02d-%02dS.%s_%03d' % (
            filetime.year, filetime.month, filetime.day,
            filetime.hour, filetime.minute, filetime.second,
            dbstring, numchans
        )

    return os.path.join(wavdir, basename)

def find_matching_wavfiles(filetime, sfilepath, y2kfix=False):
    # Should match any wavfile (even SPN) within 1-s of filetime
    matching_wavfiles = []
    for timeshift in [0, -1, 1]:
        wavpattern = filetime2wavpath(filetime + timeshift, sfilepath, y2kfix=y2kfix)
        potentialwavfiles = glob(wavpattern.split('.')[0]+".*")
        matching_wavfiles.extend(potentialwavfiles)
    return matching_wavfiles


def write_wavfile(
    st: Stream,
    out_root: str | Path,
    dbstring: str,
    numchans: int | None = None,
    *,
    year_month_dirs: bool = True,
    fmt: str = "MSEED",
    y2kfix: bool = True,
    preprocess: bool = False,
    preprocess_fn=None,
    preprocess_kwargs: dict | None = None,
    write_kwargs: dict | None = None,
    skip_existing: bool = True,
    sort_traces: bool = True,
    merge_traces: bool = True,
    align_traces: bool = True,
    merge_method: int = 1,
    fill_value: float = 0.0,
) -> str:
    """
    Write a Stream using a SEISAN-style waveform filename.

    Processing operates on a copy, leaving the caller's stream unchanged.

    By default:
    - skip an existing output file;
    - sort by station and channel;
    - merge matching trace IDs;
    - pad all traces to the full stream time range;
    - write MiniSEED through write_mseed().

    numchans controls the filename suffix. If omitted, it is calculated
    after merging and alignment.

    Optional preprocessing runs before sorting, merging and alignment.
    The filename timestamp is taken from the earliest prepared trace.

    Returns
    -------
    str
        Path written, or the existing path when skip_existing=True.

    Raises
    ------
    ValueError
        For an empty stream or invalid options.
    RuntimeError
        If preprocessing fails.
    OSError
        If the MiniSEED writer reports failure.
    """
    if not st:
        raise ValueError("Empty stream given to write_wavfile")

    if preprocess and preprocess_fn is None:
        raise ValueError("preprocess=True but no preprocess_fn provided")

    if numchans is not None and numchans < 1:
        raise ValueError("numchans must be positive")

    # Never modify the caller's stream.
    prepared = st.copy()

    if preprocess:
        try:
            prepared = preprocess_fn(
                prepared,
                **dict(preprocess_kwargs or {}),
            )
        except Exception as exc:
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc

    if not isinstance(prepared, Stream):
        raise TypeError("Preprocessing must return an ObsPy Stream")

    if not prepared:
        raise ValueError("Stream empty after preprocessing")

    if any(tr.stats.npts == 0 for tr in prepared):
        raise ValueError("Stream contains a zero-length trace")

    if sort_traces:
        prepared.sort(keys=["station", "channel"])

    if merge_traces:
        prepared.merge(
            method=merge_method,
            fill_value=fill_value,
        )

    if not prepared:
        raise ValueError("Stream empty after merging")

    # Use the union of all trace windows, rather than the first trace.
    start = min(tr.stats.starttime for tr in prepared)
    end = max(tr.stats.endtime for tr in prepared)

    if align_traces:
        prepared.trim(
            starttime=start,
            endtime=end,
            pad=True,
            fill_value=fill_value,
        )

    filetime = min(tr.stats.starttime for tr in prepared)
    channel_count = len(prepared) if numchans is None else numchans

    dbstring = str(dbstring).ljust(5, "_")

    if not y2kfix and filetime.year < 2000:
        date_prefix = (
            f"{filetime.year - 1900:02d}{filetime.month:02d}"
        )
    else:
        date_prefix = f"{filetime.year:04d}-{filetime.month:02d}"

    basename = (
        f"{date_prefix}-{filetime.day:02d}-"
        f"{filetime.hour:02d}{filetime.minute:02d}-"
        f"{filetime.second:02d}M."
        f"{dbstring}_{channel_count:03d}"
    )

    out_dir = Path(out_root)
    if year_month_dirs:
        out_dir /= f"{filetime.year:04d}/{filetime.month:02d}"

    out_path = out_dir / basename

    if out_path.exists():
        if not out_path.is_file():
            raise IsADirectoryError(str(out_path))
        if skip_existing:
            return str(out_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(write_kwargs or {})

    if fmt.upper() == "MSEED":
        # Renamed outputs and pickle fallbacks would invalidate the
        # SEISAN filename returned to the waveform index.
        if not kwargs.get("overwrite_ok", True):
            raise ValueError(
                "Use skip_existing=True instead of overwrite_ok=False"
            )
        if kwargs.get("pickle_fallback", False):
            raise ValueError(
                "pickle_fallback is incompatible with a MiniSEED WAV file"
            )

        kwargs.setdefault("fill_value", fill_value)
        written = write_mseed(prepared, str(out_path), **kwargs)

        if written is None:
            raise OSError(f"Failed to write MiniSEED file: {out_path}")

        if Path(written) != out_path:
            raise OSError(
                f"Writer returned unexpected path: {written}; "
                f"expected {out_path}"
            )
    else:
        prepared.write(str(out_path), format=fmt, **kwargs)

    return str(out_path)