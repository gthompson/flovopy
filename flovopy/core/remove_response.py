from __future__ import annotations
import numpy as np
from typing import Tuple
from obspy import Trace, UTCDateTime
from collections import defaultdict

# Reuse the local helpers (or import from core if colocated)
from flovopy.core.gaputils import _generate_spectrally_matched_noise
from flovopy.core.trace_utils import add_processing_step

def safe_pad_taper_filter(
    tr: Trace,
    *,
    taper_fraction: float,
    filter_type: str,
    freq,
    corners: int,
    zerophase: bool,
    inv=None,
    output_type: str = "VEL",
    verbose: bool = False,
) -> bool:
    """
    Perform pad→taper→filter→(response)→unpad safely on a single Trace.
    - Uses time-based unpad when possible (robust to resampling).
    - Updates tr.stats.filter and tr.stats.processing.
    """
    if tr.stats.npts == 0:
        return False

    try:
        # Decide padding seconds based on filter characteristics
        dur = tr.stats.npts * tr.stats.delta
        if filter_type == "bandpass":
            low = float(freq[0])
            pad_seconds = max(taper_fraction * dur, 1.0 / max(low, 1e-6))
        elif filter_type == "highpass":
            low = float(freq)
            pad_seconds = max(taper_fraction * dur, 1.0 / max(low, 1e-6))
        elif filter_type == "lowpass":
            # No low cutoff; rely on taper_fraction only
            pad_seconds = max(taper_fraction * dur, 0.5)
        else:
            pad_seconds = max(taper_fraction * dur, 0.5)
        pad_trace(tr, pad_seconds, method="mirror")

        # Taper (Hann) with the same fraction
        max_fraction = min(0.99, taper_fraction)
        tr.taper(max_percentage=max_fraction, type="hann")
        add_processing_step(tr, "tapered")

        # Filter and/or response
        if inv is not None:
            nyq = 0.5 / tr.stats.delta
            if filter_type == "bandpass":
                fmin, fmax = float(freq[0]), float(freq[1])
            else:
                fmin = float(freq)
                fmax = nyq * 0.95
            pre_filt = (
                max(0.01, fmin * 0.5),
                fmin,
                min(fmax, nyq * 0.95),
                min(fmax * 1.5, nyq * 0.99),
            )
            ok = handle_instrument_response(tr, inv, pre_filt, output_type, verbose)
            if not ok:
                unpad_trace(tr)
                return False
        else:
            if filter_type == "bandpass":
                tr.filter("bandpass", freqmin=freq[0], freqmax=freq[1], corners=corners, zerophase=zerophase)
            elif filter_type == "highpass":
                tr.filter("highpass", freq=freq, corners=corners, zerophase=zerophase)
            elif filter_type == "lowpass":
                tr.filter("lowpass", freq=freq, corners=corners, zerophase=zerophase)
            update_trace_filter_meta(tr, filter_type, freq, zerophase)
            add_processing_step(tr, f"filtered:{filter_type}")

        unpad_trace(tr)
        return True
    except Exception as e:
        print(f"[ERROR] safe_pad_taper_filter failed for {tr.id}: {e}")
        try:
            unpad_trace(tr)
        except Exception:
            pass
        return False


def pad_trace(tr: Trace, seconds: float, method: str = "mirror") -> None:
    if seconds is None or seconds <= 0:
        return
    sr = float(tr.stats.sampling_rate or 0.0)
    if sr <= 0:
        raise ValueError("Cannot pad trace with non-positive sampling_rate.")
    npts_pad = int(round(seconds * sr))
    if npts_pad == 0:
        return

    tr.stats["originalStartTime"] = tr.stats.starttime
    tr.stats["originalEndTime"] = tr.stats.endtime

    data = tr.data
    if isinstance(data, np.ma.MaskedArray):
        y = np.asarray(data.filled(0.0))
    else:
        y = np.asarray(data)

    n = y.size
    if n == 0:
        if method == "noise":
            pre = _generate_spectrally_matched_noise(tr, taper_percentage=0.2)[:npts_pad]
            post = _generate_spectrally_matched_noise(tr, taper_percentage=0.2)[:npts_pad]
        else:
            pre = np.zeros(npts_pad, dtype=y.dtype)
            post = np.zeros(npts_pad, dtype=y.dtype)
    else:
        if method == "mirror":
            k = min(npts_pad, n)
            left = np.flip(y[:k]); right = np.flip(y[-k:])
            if k < npts_pad:
                reps = int(np.ceil(npts_pad / k))
                left = np.resize(np.tile(left, reps), npts_pad)
                right = np.resize(np.tile(right, reps), npts_pad)
            pre, post = left.astype(y.dtype, copy=False), right.astype(y.dtype, copy=False)
        elif method == "zeros":
            pre = np.zeros(npts_pad, dtype=y.dtype)
            post = np.zeros(npts_pad, dtype=y.dtype)
        elif method == "noise":
            noise = _generate_spectrally_matched_noise(tr, taper_percentage=0.2)
            if noise.size < npts_pad:
                reps = int(np.ceil(npts_pad / max(1, noise.size)))
                noise = np.tile(noise, reps)
            pre = noise[:npts_pad].astype(y.dtype, copy=False)
            post = noise[:npts_pad].astype(y.dtype, copy=False)
        else:
            raise ValueError("method must be one of {'mirror','zeros','noise'}")

    tr.data = np.concatenate([pre, y.astype(pre.dtype, copy=False), post])
    tr.stats.starttime -= npts_pad * tr.stats.delta
    tr.stats["_pad"] = {"seconds": float(seconds), "npts": int(npts_pad), "method": str(method)}
    add_processing_step(tr, f"padded:{method}({seconds:.3f}s)")


def unpad_trace(tr: Trace) -> None:
    t0 = tr.stats.get("originalStartTime", None)
    t1 = tr.stats.get("originalEndTime", None)
    if t0 is not None and t1 is not None:
        try:
            tr.trim(starttime=t0, endtime=t1, pad=False)
        except Exception:
            pass
        else:
            for k in ("_pad", "originalStartTime", "originalEndTime"):
                if k in tr.stats:
                    try:
                        del tr.stats[k]
                    except Exception:
                        pass
            add_processing_step(tr, "unpadded")
            return
    meta = getattr(tr.stats, "_pad", None)
    if meta and isinstance(meta, dict):
        npts_pad = int(meta.get("npts", 0))
        if npts_pad > 0 and tr.stats.npts >= 2 * npts_pad:
            tr.data = tr.data[npts_pad : tr.stats.npts - npts_pad]
            if t0 is not None:
                tr.stats.starttime = t0
            else:
                tr.stats.starttime += npts_pad * tr.stats.delta
            for k in ("_pad", "originalStartTime", "originalEndTime"):
                if k in tr.stats:
                    try:
                        del tr.stats[k]
                    except Exception:
                        pass
            add_processing_step(tr, "unpadded")


def update_trace_filter_meta(tr: Trace, filtertype: str, freq, zerophase: bool) -> None:
    if not hasattr(tr.stats, "filter") or tr.stats.filter is None:
        tr.stats.filter = {"freqmin": 0.0, "freqmax": tr.stats.sampling_rate / 2.0, "zerophase": False}
    if filtertype == "highpass":
        tr.stats.filter["freqmin"] = max(float(freq), tr.stats.filter["freqmin"])
    elif filtertype == "bandpass":
        fmin, fmax = float(freq[0]), float(freq[1])
        if fmax < fmin:
            fmin, fmax = fmax, fmin
        tr.stats.filter["freqmin"] = max(fmin, tr.stats.filter["freqmin"])
        tr.stats.filter["freqmax"] = min(fmax, tr.stats.filter["freqmax"])
    elif filtertype == "lowpass":
        tr.stats.filter["freqmax"] = min(float(freq), tr.stats.filter["freqmax"])
    tr.stats.filter["zerophase"] = bool(zerophase)


def handle_instrument_response(tr: Trace, inv, pre_filt, outputType: str, verbose: bool) -> bool:
    #from flovopy.core.preprocessing import _get_calib  # reuse existing helper
    if inv is None:
        return True
    try:
        if verbose:
            print("- removing instrument response")
        out = outputType
        if tr.stats.channel[1] == 'D':
            out = 'DEF'
        tr.remove_response(
            inventory=inv,
            output=out,
            pre_filt=pre_filt,
            water_level=60,
            zero_mean=True,
            taper=False,
            taper_fraction=0.0,
            plot=False,
            fig=None,
        )
        tr.stats.calib = 1.0
        #tr.stats["calib_applied"] = _get_calib(tr, inv)
        if tr.stats.channel[1] == 'H':
            tr.stats["units"] = 'm/s' if out == 'VEL' else ('m' if out == 'DISP' else tr.stats.get('units', ''))
        elif tr.stats.channel[1] == 'N':
            tr.stats["units"] = 'm/s2'
        elif tr.stats.channel[1] == 'D':
            tr.stats["units"] = 'Pa'
        else:
            tr.stats["units"] = tr.stats.get("units", "")
        add_processing_step(tr, f"response_removed({out})")
        return True
    except Exception as e:
        print(f"Error removing response for {tr.id}: {e}")
        return False



def stationxml_match_report(st: Stream, inv, t1: UTCDateTime | None = None):
    """
    Report which traces in `st` are represented in the StationXML `inv`.

    A trace is a TRUE match iff:
      • Channel exists in inventory (same NSLC)
      • A channel epoch covers `t1` (defaults to stream starttime)
      • Channel has a Response with ≥1 stage

    Prints a summary and returns a structured dict.
    """
    if inv is None or len(st) == 0:
        return {"matched": [], "misses": {}}

    if t1 is None:
        t1 = st[0].stats.starttime

    matched = []
    misses = defaultdict(list)

    def _epoch_covers(ch, when):
        sd, ed = ch.start_date, ch.end_date
        if sd and when < sd:
            return False
        if ed and when > ed:
            return False
        return True

    for tr in st:
        n, s, l, c = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel

        # 1) Any channel objects with this NSLC?
        sel = inv.select(network=n, station=s, location=l, channel=c)
        chans = [ch for net in sel for sta in net.stations for ch in sta.channels]
        if not chans:
            misses["no_channel"].append(tr.id)
            continue

        # 2) Any of those channels cover t1?
        chans_covering = [ch for ch in chans if _epoch_covers(ch, t1)]
        if not chans_covering:
            ranges = []
            for ch in chans:
                sd = ch.start_date.isoformat() if ch.start_date else "None"
                ed = ch.end_date.isoformat() if ch.end_date else "None"
                ranges.append(f"[{sd} → {ed}]")
            misses["epoch_mismatch"].append(f"{tr.id} (have {', '.join(ranges)})")
            continue

        # 3) Response present and non-empty?
        ok_resp = False
        for ch in chans_covering:
            resp = getattr(ch, "response", None)
            stages = getattr(resp, "response_stages", None) if resp else None
            if resp and stages and len(stages) > 0:
                ok_resp = True
                break

        if not ok_resp:
            any_resp_objs = any(getattr(ch, "response", None) is not None for ch in chans_covering)
            key = "empty_response_stages" if any_resp_objs else "no_response"
            misses[key].append(tr.id)
            continue

        matched.append(tr.id)

    # ---- Printing summary ----
    total = len(st)
    nm = len(matched)
    print(f"[INV] matched {nm}/{total} traces ({100.0*nm/total:.1f}%): {matched}")
    for reason, ids in misses.items():
        print(f"[INV] missing ({reason}): {ids}")

    return {"matched": matched, "misses": dict(misses)}