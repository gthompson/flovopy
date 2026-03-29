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
) -> str:
    """
    Write a Stream using a SEISAN-style waveform filename.

    Optionally applies preprocessing before writing.

    Parameters
    ----------
    st
        Input waveform stream.
    preprocess : bool
        If True, apply preprocess_fn before writing.
    preprocess_fn : callable or None
        Signature: preprocess_fn(stream, **kwargs) -> stream
    preprocess_kwargs : dict or None
        Passed to preprocess_fn.
    """

    if not st or len(st) == 0:
        raise ValueError("Empty stream given to write_wavfile")

    out_root = Path(out_root)
    write_kwargs = write_kwargs or {}

    # -------------------------------------------------
    # Preprocessing stage (NEW)
    # -------------------------------------------------
    if preprocess:
        if preprocess_fn is None:
            raise ValueError("preprocess=True but no preprocess_fn provided")

        try:
            kwargs = preprocess_kwargs or {}
            st = preprocess_fn(st, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")

        if not st or len(st) == 0:
            raise ValueError("Stream empty after preprocessing")

    # -------------------------------------------------
    # Filename construction
    # -------------------------------------------------
    filetime: UTCDateTime = st[0].stats.starttime

    if numchans is None:
        numchans = len(st)

    dbstring = str(dbstring)
    if len(dbstring) < 5:
        dbstring = dbstring + "_" * (5 - len(dbstring))

    if not y2kfix and filetime.year < 2000:
        basename = (
            f"{filetime.year - 1900:02d}{filetime.month:02d}-"
            f"{filetime.day:02d}-{filetime.hour:02d}{filetime.minute:02d}-"
            f"{filetime.second:02d}M.{dbstring}_{numchans:03d}"
        )
    else:
        basename = (
            f"{filetime.year:04d}-{filetime.month:02d}-{filetime.day:02d}-"
            f"{filetime.hour:02d}{filetime.minute:02d}-{filetime.second:02d}"
            f"M.{dbstring}_{numchans:03d}"
        )

    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------
    if year_month_dirs:
        out_dir = out_root / f"{filetime.year:04d}" / f"{filetime.month:02d}"
    else:
        out_dir = out_root

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / basename

    # -------------------------------------------------
    # Write
    # -------------------------------------------------
    if fmt.lower() == "mseed":
        write_mseed(st, str(out_path), **write_kwargs)
    else:
        st.write(str(out_path), format=fmt, **write_kwargs)

    return str(out_path)