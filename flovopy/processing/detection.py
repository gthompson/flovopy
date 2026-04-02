from __future__ import annotations

"""
flovopy.processing.detection

Core, non-interactive helpers for seismic event detection and event catalogue
building.

This module merges and refactors the older ``detection.py`` and the
``event_triggering.py`` teaching helpers into a single non-GUI module.

Included functionality
----------------------
- network coincidence triggering with ObsPy
- trigger consolidation / filtering
- dataframe-first event catalogue generation
- export of event windows to MiniSEED and PNG
- non-interactive Stream plotting helpers
- simple RMS/SNR and spectral summary helpers
- per-trace trigger overlays
- lightweight automatic phase-picking helpers
- classic STA/LTA convenience wrappers

Interactive / GUI workflows belong in:
    flovopy.processing.detection_gui
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from obspy import Stream, Trace, UTCDateTime, read
from obspy.signal.trigger import (
    aic_simple,
    ar_pick,
    classic_sta_lta,
    coincidence_trigger,
    pk_baer,
    trigger_onset,
    z_detect,
)

from flovopy.core.spectral import (
    compute_amplitude_ratios,
    get_bandwidth,
    plot_amplitude_ratios,
)
from flovopy.core.trace_qc import estimate_snr

try:
    from flovopy.core.remove_response import pad_trace
except Exception:  # pragma: no cover
    pad_trace = None


# =============================================================================
# Basic RMS / SNR helpers
# =============================================================================


def trace_rms(tr: Trace) -> float:
    """Return RMS amplitude of a single Trace, or NaN if empty."""
    x = np.asarray(tr.data, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))



def stream_median_rms(st: Stream) -> float:
    """Return the median RMS amplitude over all valid traces in a Stream."""
    vals = [trace_rms(tr) for tr in st]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


# =============================================================================
# Time / filename helpers
# =============================================================================


def _coerce_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a Series to pandas datetimes, coercing invalid values to NaT."""
    return pd.to_datetime(series, errors="coerce")



def _safe_timestamp_for_filename(dt: pd.Timestamp) -> str:
    """Return a filename-safe ISO-like timestamp string."""
    return dt.isoformat().replace(":", "")



def _event_day_dir(base_outdir: str, on_time: pd.Timestamp) -> str:
    """Return a YYYY/MM/DD directory path for an event time."""
    return os.path.join(
        base_outdir,
        f"{on_time.year:04d}",
        f"{on_time.month:02d}",
        f"{on_time.day:02d}",
    )


# =============================================================================
# Plotting helpers
# =============================================================================


def plot_stream_with_event_markers(
    st: Stream,
    df: pd.DataFrame,
    *,
    on_col: str = "on_time",
    off_col: str = "off_time",
    show: bool = True,
    label_lines: bool = True,
    shade_events: bool = False,
    shade_alpha: float = 0.15,
    linewidth: float = 1.5,
    equal_scale: bool = False,
):
    """
    Plot a Stream and superimpose trigger ON/OFF times on all trace axes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if on_col not in df.columns or off_col not in df.columns:
        raise ValueError(f"df must contain '{on_col}' and '{off_col}'.")
    if len(st) == 0:
        raise ValueError("Stream is empty.")

    dff = df[[on_col, off_col]].copy()
    dff[on_col] = _coerce_datetime_series(dff[on_col])
    dff[off_col] = _coerce_datetime_series(dff[off_col])
    dff = dff.dropna(subset=[on_col, off_col]).sort_values(on_col)

    fig = st.plot(show=False, handle=True, equal_scale=equal_scale)
    axes = [ax for ax in fig.axes if ax.has_data()]

    if len(dff) == 0:
        if show:
            plt.show()
        return fig, axes

    on_nums = mdates.date2num(dff[on_col].dt.to_pydatetime())
    off_nums = mdates.date2num(dff[off_col].dt.to_pydatetime())

    for i, ax in enumerate(axes):
        ymin, ymax = ax.get_ylim()
        if label_lines and i == 0:
            ax.vlines(on_nums, ymin, ymax, color="r", lw=linewidth, label="Trigger On")
            ax.vlines(off_nums, ymin, ymax, color="b", lw=linewidth, label="Trigger Off")
            ax.legend()
        else:
            ax.vlines(on_nums, ymin, ymax, color="r", lw=linewidth)
            ax.vlines(off_nums, ymin, ymax, color="b", lw=linewidth)

        if shade_events:
            for on_num, off_num in zip(on_nums, off_nums):
                ax.axvspan(on_num, off_num, alpha=shade_alpha)

    fig.canvas.draw()
    if show:
        plt.show()
    return fig, axes



def plot_stream_with_on_off(
    st: Stream,
    on_time: Union[UTCDateTime, pd.Timestamp],
    off_time: Union[UTCDateTime, pd.Timestamp],
    *,
    show: bool = True,
    equal_scale: bool = False,
):
    """Convenience wrapper for plotting a single ON/OFF pair on a Stream."""
    on_dt = on_time.datetime if hasattr(on_time, "datetime") else pd.to_datetime(on_time)
    off_dt = off_time.datetime if hasattr(off_time, "datetime") else pd.to_datetime(off_time)
    df = pd.DataFrame({"on_time": [on_dt], "off_time": [off_dt]})
    return plot_stream_with_event_markers(st, df, show=show, equal_scale=equal_scale)



def save_event_plot_png(
    st_event: Stream,
    on_time: UTCDateTime,
    off_time: UTCDateTime,
    png_path: str,
):
    """Save a PNG plot for an event segment with ON/OFF markers."""
    fig, _ = plot_stream_with_on_off(st_event, on_time, off_time, show=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_detected_stream(
    detected_st: Stream,
    best_trig: Dict[str, Any],
    outfile: Optional[str] = None,
    *,
    equal_scale: bool = False,
    show: bool = True,
):
    """Plot a detected stream with trigger start and end markers on every axis."""
    trig_time = UTCDateTime(best_trig["time"]).matplotlib_date
    trig_end = (UTCDateTime(best_trig["time"]) + float(best_trig["duration"])).matplotlib_date

    fig = detected_st.plot(show=False, equal_scale=equal_scale)
    axes = [ax for ax in fig.axes if ax.has_data()]
    for i, ax in enumerate(axes):
        if i == 0:
            ax.axvline(trig_time, color="r", linestyle="--", label="Trigger Start")
            ax.axvline(trig_end, color="b", linestyle="--", label="Trigger End")
            ax.legend()
        else:
            ax.axvline(trig_time, color="r", linestyle="--")
            ax.axvline(trig_end, color="b", linestyle="--")

    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig



def plot_snr_windows_on_stream(
    st: Stream,
    split_time: Union[UTCDateTime, Tuple[UTCDateTime, UTCDateTime]],
    window_length: float = 1.0,
    pre_noise_gap: float = 2.0,
    title: str = "Signal/Noise windows for SNR",
    outfile: Optional[str] = None,
    show: bool = True,
):
    """Shade the signal and noise windows used for SNR estimation on each trace."""
    if isinstance(split_time, (list, tuple)) and len(split_time) == 2:
        t_on, t_off = split_time
        if not isinstance(t_on, UTCDateTime):
            t_on = UTCDateTime(t_on)
        if not isinstance(t_off, UTCDateTime):
            t_off = UTCDateTime(t_off)
        dur = float(t_off - t_on)
    else:
        t_on = split_time if isinstance(split_time, UTCDateTime) else UTCDateTime(split_time)
        dur = float(window_length)
        t_off = t_on + dur

    if dur <= 0:
        raise ValueError("Non-positive signal duration")

    s1, s2 = t_on, t_off
    n2 = t_on - pre_noise_gap
    n1 = n2 - dur

    fig = st.plot(show=False, equal_scale=False)
    fig.suptitle(title)
    axes = [ax for ax in fig.axes if ax.has_data()]

    s1d, s2d = s1.matplotlib_date, s2.matplotlib_date
    n1d, n2d = n1.matplotlib_date, n2.matplotlib_date

    for tr, ax in zip(st, axes):
        sig_ok = (tr.stats.starttime <= s1) and (tr.stats.endtime >= s2)
        noi_ok = (tr.stats.starttime <= n1) and (tr.stats.endtime >= n2)

        if noi_ok:
            ax.axvspan(n1d, n2d, color="0.7", alpha=0.3, label="noise")
        if sig_ok:
            ax.axvspan(s1d, s2d, color="g", alpha=0.25, label="signal")

        if sig_ok and noi_ok:
            sr = float(tr.stats.sampling_rate or 0.0)
            sig_ns = int(round(dur * sr)) if sr > 0 else 0
            ax.text(s2d, 0.9 * ax.get_ylim()[1], f"{tr.id}\nlen={sig_ns} samp", fontsize=8, ha="right", va="top")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, l in zip(handles, labels):
                if l in seen:
                    continue
                seen.add(l)
                uniq_handles.append(h)
                uniq_labels.append(l)
            ax.legend(uniq_handles, uniq_labels, loc="upper right", fontsize=8)

    fig.autofmt_xdate()
    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
# Trigger consolidation / network detection helpers
# =============================================================================


def _event_window(ev: Dict[str, Any]) -> Tuple[UTCDateTime, UTCDateTime]:
    """Return trigger ON and OFF times for a trigger dictionary."""
    t_on = UTCDateTime(ev["time"])
    t_off = t_on + float(ev["duration"])
    return t_on, t_off



def _union_list(a: List[Any], b: List[Any]) -> List[Any]:
    """Return an order-preserving union of two lists."""
    seen = set(a)
    out = list(a)
    for x in b:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out



def consolidate_triggers(
    trig: List[Dict[str, Any]],
    *,
    join_within: float = 90.0,
) -> Tuple[List[Dict[str, Any]], List[UTCDateTime], List[UTCDateTime]]:
    """Merge adjacent or overlapping triggers into larger event windows."""
    if not trig:
        return [], [], []

    trig_sorted = sorted(trig, key=lambda e: _event_window(e)[0])
    merged: List[Dict[str, Any]] = []
    cur_members: List[Dict[str, Any]] = [trig_sorted[0]]
    _, cur_off = _event_window(trig_sorted[0])

    def _flush(members: List[Dict[str, Any]]) -> Dict[str, Any]:
        t_on = min(_event_window(e)[0] for e in members)
        t_off = max(_event_window(e)[1] for e in members)
        dur = float(t_off - t_on)

        stations: List[str] = []
        trace_ids: List[str] = []
        cft_peaks_all: List[float] = []
        cft_peak_wmean_vals: List[float] = []
        coinc_vals: List[float] = []

        for e in members:
            stations = _union_list(stations, e.get("stations") or [])
            trace_ids = _union_list(trace_ids, e.get("trace_ids") or [])
            cft_peaks_all.extend([float(x) for x in (e.get("cft_peaks") or []) if x is not None])
            cpwm = e.get("cft_peak_wmean")
            if cpwm is not None:
                try:
                    cft_peak_wmean_vals.append(float(cpwm))
                except Exception:
                    pass
            cs = e.get("coincidence_sum")
            if cs is not None:
                try:
                    coinc_vals.append(float(cs))
                except Exception:
                    pass

        return {
            "time": UTCDateTime(t_on),
            "duration": dur,
            "stations": stations or None,
            "trace_ids": trace_ids or None,
            "cft_peaks": cft_peaks_all or None,
            "cft_peak_wmean": max(cft_peak_wmean_vals) if cft_peak_wmean_vals else None,
            "coincidence_sum": max(coinc_vals) if coinc_vals else None,
            "cluster_size": len(members),
            "cluster_members": [{"time": e["time"], "duration": e["duration"]} for e in members],
        }

    for e in trig_sorted[1:]:
        e_on, e_off = _event_window(e)
        if e_on <= (cur_off + join_within):
            cur_members.append(e)
            if e_off > cur_off:
                cur_off = e_off
        else:
            merged.append(_flush(cur_members))
            cur_members = [e]
            _, cur_off = _event_window(e)

    merged.append(_flush(cur_members))
    ontimes = [m["time"] for m in merged]
    offtimes = [UTCDateTime(m["time"]) + float(m["duration"]) for m in merged]
    return merged, ontimes, offtimes



def drop_short_triggers(
    trig: List[Dict[str, Any]],
    *,
    min_duration: float = 60.0,
) -> Tuple[List[Dict[str, Any]], List[UTCDateTime], List[UTCDateTime]]:
    """Drop triggers shorter than ``min_duration`` seconds."""
    kept: List[Dict[str, Any]] = []
    for e in trig:
        try:
            dur = float(e["duration"])
        except Exception:
            continue
        if dur >= float(min_duration):
            kept.append(e)

    ontimes = [UTCDateTime(e["time"]) for e in kept]
    offtimes = [UTCDateTime(e["time"]) + float(e["duration"]) for e in kept]
    return kept, ontimes, offtimes



def detect_network_event(
    st_in: Stream,
    minchans: Optional[int] = None,
    threshon: float = 3.5,
    threshoff: float = 1.0,
    sta: float = 0.5,
    lta: float = 5.0,
    pad: float = 0.0,
    best_only: bool = False,
    verbose: bool = False,
    freq: Optional[List[float]] = None,
    algorithm: str = "recstalta",
    criterion: str = "longest",
    *,
    join_within: Optional[float] = None,
    min_duration: Optional[float] = None,
):
    """
    Detect and optionally associate seismic events across a network.

    Returns
    -------
    dict or tuple
        If ``best_only=True``, returns one trigger dict or ``None``.
        Otherwise returns ``(trig, ontimes, offtimes)``, or
        ``(None, None, None)`` if nothing is detected.
    """
    st = st_in.copy()

    if pad > 0.0:
        for tr in st:
            try:
                if pad_trace is not None:
                    pad_trace(tr, pad)
                else:
                    tr.trim(tr.stats.starttime - pad, tr.stats.endtime + pad, pad=True, fill_value=0.0)
            except Exception:
                pass

    if freq:
        if verbose:
            print(
                f"[detect_network_event] bandpass {freq[0]}–{freq[1]} Hz; "
                f"sta={sta} lta={lta} on={threshon} off={threshoff}"
            )
        st.filter("bandpass", freqmin=freq[0], freqmax=freq[1], corners=4, zerophase=True)

    if minchans is None:
        minchans = max(int(len(st) / 2), 2)
    if verbose:
        print("[detect_network_event] minchans =", minchans)

    if algorithm == "zdetect":
        trig = coincidence_trigger(algorithm, threshon, threshoff, st, minchans, sta=sta, details=True)
    elif algorithm == "carlstatrig":
        trig = coincidence_trigger(
            algorithm,
            threshon,
            threshoff,
            st,
            minchans,
            sta=sta,
            lta=lta,
            ratio=1,
            quiet=True,
            details=True,
        )
    else:
        trig = coincidence_trigger(algorithm, threshon, threshoff, st, minchans, sta=sta, lta=lta, details=True)

    if not trig:
        return None if best_only else (None, None, None)

    ontimes = [UTCDateTime(e["time"]) for e in trig]
    offtimes = [UTCDateTime(e["time"]) + float(e["duration"]) for e in trig]

    if join_within is not None:
        trig, ontimes, offtimes = consolidate_triggers(trig, join_within=float(join_within))

    if min_duration is not None:
        trig, ontimes, offtimes = drop_short_triggers(trig, min_duration=float(min_duration))

    if not trig:
        return None if best_only else (None, None, None)

    if best_only:
        best_ev = None
        best_score = -1.0
        for ev in trig:
            dur = float(ev["duration"])
            if criterion == "cft":
                score = sum(ev.get("cft_peaks") or []) or 0.0
            elif criterion == "cft_duration":
                score = (sum(ev.get("cft_peaks") or []) or 0.0) * dur
            else:
                score = (float(ev.get("coincidence_sum") or 0.0)) * dur
            if score > best_score:
                best_score = score
                best_ev = ev
        return best_ev

    return trig, ontimes, offtimes


# =============================================================================
# Dataframe-first trigger catalogue helpers
# =============================================================================


def extract_triggers_to_dataframe(
    st: Stream,
    trig: Sequence[Dict[str, Any]],
    *,
    pretrigger_seconds: float = 10.0,
    posttrigger_seconds: float = 15.0,
    write_mseed: bool = False,
    outdir: str = ".",
    max_events: Optional[int] = None,
    make_plots: bool = False,
    filename_safe: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert coincidence-trigger results into a dataframe catalogue."""
    os.makedirs(outdir, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    if not trig:
        return pd.DataFrame(rows)

    for i, t in enumerate(trig):
        if (max_events is not None) and (i >= max_events):
            break

        on_time = UTCDateTime(t["time"])
        duration = float(t["duration"])
        off_time = on_time + duration

        st_event = st.copy().trim(starttime=on_time - pretrigger_seconds, endtime=off_time + posttrigger_seconds)
        st_signal = st.copy().trim(starttime=on_time, endtime=off_time)
        st_noise = st.copy().trim(starttime=on_time - pretrigger_seconds, endtime=on_time)

        signal_rms = stream_median_rms(st_signal)
        noise_rms = stream_median_rms(st_noise)
        snr_rms = float(signal_rms / noise_rms) if np.isfinite(signal_rms) and np.isfinite(noise_rms) and noise_rms > 0 else float("nan")

        seed_ids = sorted({tr.id for tr in st_event})

        ts = on_time.isoformat()
        if filename_safe:
            ts = ts.replace(":", "")
        filename = f"{ts}.mseed"
        filepath = os.path.join(outdir, filename)

        if make_plots:
            plot_stream_with_on_off(st_event, on_time, off_time, show=True)

        if write_mseed:
            st_event.write(filepath, format="MSEED")

        if verbose:
            print(f"Event {i + 1}/{len(trig)}  ON={on_time}  dur={duration:.2f}s  SNR~{snr_rms:.2f}")

        rows.append(
            {
                "event_index": i,
                "on_time": on_time.datetime,
                "off_time": off_time.datetime,
                "duration_s": duration,
                "seed_ids": seed_ids,
                "n_seed_ids": len(seed_ids),
                "pretrigger_s": float(pretrigger_seconds),
                "posttrigger_s": float(posttrigger_seconds),
                "outdir": outdir,
                "filename": filename,
                "filepath": filepath,
                "wrote_file": bool(write_mseed),
                "coincidence_sum": t.get("coincidence_sum", np.nan),
                "signal_rms": signal_rms,
                "noise_rms": noise_rms,
                "snr_rms": snr_rms,
                "stations": t.get("stations", None),
                "trace_ids": t.get("trace_ids", None),
                "cft_peaks": t.get("cft_peaks", None),
                "cft_peak_wmean": t.get("cft_peak_wmean", None),
                "cluster_size": t.get("cluster_size", None),
            }
        )

    return pd.DataFrame(rows)



def run_coincidence_trigger_dataframe(
    st: Stream,
    *,
    trigger_type: str = "recstalta",
    sta_seconds: float = 1.0,
    lta_seconds: float = 10.0,
    threshold_on: float = 2.5,
    threshold_off: float = 0.5,
    min_channels: int = 1,
    pretrigger_seconds: float = 10.0,
    posttrigger_seconds: float = 15.0,
    write_mseed: bool = False,
    outdir: str = ".",
    max_events: Optional[int] = None,
    make_plots: bool = False,
    filename_safe: bool = True,
    details: Optional[bool] = True,
    event_templates: Optional[dict] = None,
    similarity_threshold: float = 0.7,
    **kwargs,
) -> pd.DataFrame:
    """Run ObsPy ``coincidence_trigger()`` on a Stream and return a dataframe catalogue."""
    trigger_kwargs = dict(
        sta=sta_seconds,
        lta=lta_seconds,
        details=details,
        **kwargs,
    )

    if event_templates is not None:
        trigger_kwargs["event_templates"] = event_templates
        trigger_kwargs["similarity_threshold"] = similarity_threshold

    trig = coincidence_trigger(
        trigger_type,
        threshold_on,
        threshold_off,
        st,
        min_channels,
        **trigger_kwargs,
    )

    return extract_triggers_to_dataframe(
        st,
        trig,
        pretrigger_seconds=pretrigger_seconds,
        posttrigger_seconds=posttrigger_seconds,
        write_mseed=write_mseed,
        outdir=outdir,
        max_events=max_events,
        make_plots=make_plots,
        filename_safe=filename_safe,
    )



def run_trigger_wrapper_df(
    st: Stream,
    sta_seconds: float = 1.0,
    lta_seconds: float = 10.0,
    threshold_on: float = 2.5,
    threshold_off: float = 0.5,
    min_channels: int = 1,
    pretrigger_seconds: float = 10.0,
    posttrigger_seconds: float = 15.0,
    write_mseed: bool = False,
    outdir: str = ".",
    max_events: Optional[int] = None,
    make_plots: bool = False,
    filename_safe: bool = True,
) -> pd.DataFrame:
    """Backward-compatible wrapper for a dataframe-based coincidence-trigger run."""
    return run_coincidence_trigger_dataframe(
        st,
        trigger_type="recstalta",
        sta_seconds=sta_seconds,
        lta_seconds=lta_seconds,
        threshold_on=threshold_on,
        threshold_off=threshold_off,
        min_channels=min_channels,
        pretrigger_seconds=pretrigger_seconds,
        posttrigger_seconds=posttrigger_seconds,
        write_mseed=write_mseed,
        outdir=outdir,
        max_events=max_events,
        make_plots=make_plots,
        filename_safe=filename_safe,
    )



def filter_events_df(
    df: pd.DataFrame,
    *,
    min_snr: Optional[float] = None,
    min_coincidence: Optional[float] = None,
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    min_seed_ids: Optional[int] = None,
) -> pd.DataFrame:
    """Return a filtered copy of an event-catalogue dataframe."""
    out = df.copy()

    if min_snr is not None and "snr_rms" in out.columns:
        out = out[out["snr_rms"] >= float(min_snr)]
    if min_coincidence is not None and "coincidence_sum" in out.columns:
        out = out[out["coincidence_sum"] >= float(min_coincidence)]
    if min_duration_s is not None and "duration_s" in out.columns:
        out = out[out["duration_s"] >= float(min_duration_s)]
    if max_duration_s is not None and "duration_s" in out.columns:
        out = out[out["duration_s"] <= float(max_duration_s)]
    if min_seed_ids is not None and "n_seed_ids" in out.columns:
        out = out[out["n_seed_ids"] >= int(min_seed_ids)]

    return out.reset_index(drop=True)



def export_events_from_catalogue(
    df: pd.DataFrame,
    *,
    base_outdir: str = "exported_events",
    write_mseed: bool = True,
    write_png: bool = True,
    st_continuous: Optional[Stream] = None,
    use_existing_mseed_if_present: bool = True,
    max_events: Optional[int] = None,
) -> pd.DataFrame:
    """Export event windows from a catalogue into ``YYYY/MM/DD`` directories."""
    if "on_time" not in df.columns:
        raise ValueError("df must contain an 'on_time' column (datetime).")

    out = df.copy()
    out["export_dir"] = ""
    out["export_mseed_path"] = ""
    out["export_png_path"] = ""

    n = len(out)
    if max_events is not None:
        n = min(n, int(max_events))

    for i in range(n):
        row = out.iloc[i]
        on_dt = pd.to_datetime(row["on_time"])
        if "off_time" in out.columns:
            off_dt = pd.to_datetime(row["off_time"])
        else:
            off_dt = on_dt + pd.to_timedelta(float(row["duration_s"]), unit="s")

        export_dir = _event_day_dir(base_outdir, on_dt)
        os.makedirs(export_dir, exist_ok=True)
        stem = _safe_timestamp_for_filename(on_dt)
        mseed_path = os.path.join(export_dir, f"{stem}.mseed")
        png_path = os.path.join(export_dir, f"{stem}.png")

        st_event: Optional[Stream] = None
        if use_existing_mseed_if_present and "filepath" in out.columns:
            fp = row.get("filepath", "")
            if isinstance(fp, str) and fp and os.path.exists(fp):
                st_event = read(fp)

        if st_event is None:
            if st_continuous is None:
                raise ValueError(
                    "No waveform source available for export. Provide st_continuous=... or ensure df['filepath'] points to existing MiniSEED files."
                )

            pre = float(row["pretrigger_s"]) if "pretrigger_s" in out.columns else 0.0
            post = float(row["posttrigger_s"]) if "posttrigger_s" in out.columns else 0.0
            on = UTCDateTime(on_dt.to_pydatetime())
            off = UTCDateTime(off_dt.to_pydatetime())
            st_event = st_continuous.copy().trim(starttime=on - pre, endtime=off + post)

        if write_mseed:
            st_event.write(mseed_path, format="MSEED")
        if write_png:
            on = UTCDateTime(on_dt.to_pydatetime())
            off = UTCDateTime(off_dt.to_pydatetime())
            save_event_plot_png(st_event, on, off, png_path)

        out.at[out.index[i], "export_dir"] = export_dir
        out.at[out.index[i], "export_mseed_path"] = mseed_path if write_mseed else ""
        out.at[out.index[i], "export_png_path"] = png_path if write_png else ""

    return out



def plot_event_rate(
    df: pd.DataFrame,
    *,
    time_col: str = "on_time",
    bin_size: str = "1h",
    min_snr: Optional[float] = None,
    min_coincidence: Optional[float] = None,
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    min_seed_ids: Optional[int] = None,
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """Plot event counts in fixed-width time bins over the full catalogue."""
    if time_col not in df.columns:
        raise ValueError(f"df must contain '{time_col}' column.")

    dff = df.copy()
    dff[time_col] = pd.to_datetime(dff[time_col])
    dff = dff.sort_values(time_col)
    dff = filter_events_df(
        dff,
        min_snr=min_snr,
        min_coincidence=min_coincidence,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        min_seed_ids=min_seed_ids,
    )

    if len(dff) == 0:
        print("No events after filtering.")
        return None

    counts = dff.set_index(time_col).resample(bin_size).size()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(counts.index, counts.values)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Events per {bin_size}")
    ax.set_title(title or "Event rate")
    fig.autofmt_xdate()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# SNR / spectral summary helpers
# =============================================================================


def detection_snr(
    detected_st: Stream,
    best_trig: Dict[str, Any],
    outfile: Optional[str] = None,
    method: str = "std",
    spectral_smooth: int = 5,
    spectral_average: str = "geometric",
    spectral_threshold: float = 0.5,
    freq_band: Optional[Tuple[float, float]] = None,
):
    """Estimate time-domain and spectral-domain SNR from a triggered waveform stream."""
    trig_stime = UTCDateTime(best_trig["time"])
    trig_etime = trig_stime + float(best_trig["duration"])

    signal_st = detected_st.copy().trim(starttime=trig_stime, endtime=trig_etime)
    noise_st = detected_st.copy().trim(endtime=trig_stime)

    snr_list, _, _ = estimate_snr(
        detected_st,
        method=method,
        split_time=(trig_stime, trig_etime),
        spectral_kwargs={"smooth_window": spectral_smooth, "average": spectral_average},
        freq_band=freq_band,
    )

    avg_freqs, avg_spectral_ratio, indiv_ratios, freqs_list, trace_ids, _ = compute_amplitude_ratios(
        signal_st,
        noise_st,
        smooth_window=spectral_smooth,
        average=spectral_average,
        verbose=True,
    )

    if isinstance(avg_freqs, np.ndarray) and isinstance(avg_spectral_ratio, np.ndarray):
        if outfile:
            plot_amplitude_ratios(
                avg_freqs,
                avg_spectral_ratio,
                individual_ratios=indiv_ratios,
                freqs_list=freqs_list,
                trace_ids=trace_ids,
                log_scale=True,
                outfile=outfile,
                threshold=spectral_threshold,
            )
        fmetrics_dict = get_bandwidth(avg_freqs, avg_spectral_ratio, threshold=spectral_threshold)
        return snr_list, fmetrics_dict

    return snr_list, None



def real_time_optimization(band: str = "all") -> Tuple[float, float, float, float, float, float, int]:
    """Return suggested real-time detection parameters for broad event classes."""
    corners = 2
    if band == "VT":
        sta_secs, lta_secs, thresh_on, thresh_off, freqmin, freqmax = 1.4, 7.0, 2.4, 1.2, 3.0, 18.0
    elif band == "LP":
        sta_secs, lta_secs, thresh_on, thresh_off, freqmin, freqmax = 2.3, 11.5, 2.4, 1.2, 0.8, 10.0
    else:
        sta_secs, lta_secs, thresh_on, thresh_off, freqmin, freqmax = 2.3, 11.5, 2.4, 1.2, 1.5, 12.0

    thresh_off = thresh_off / thresh_on
    return sta_secs, lta_secs, thresh_on, thresh_off, freqmin, freqmax, corners


# =============================================================================
# Single-channel trigger utilities
# =============================================================================


def add_channel_detections(
    st: Stream,
    lta: float = 5.0,
    threshon: float = 0.5,
    threshoff: float = 0.0,
    max_duration: float = 120.0,
) -> None:
    """Run single-channel z_detect triggering on each Trace in a Stream."""
    for tr in st:
        tr.stats["triggers"] = []
        fs = float(tr.stats.sampling_rate)
        cft = z_detect(tr.data, int(lta * fs))
        triggerlist = trigger_onset(cft, threshon, threshoff, max_len=max_duration * fs)
        for trigpair in triggerlist:
            trigpair_utc = [tr.stats.starttime + samplenum / fs for samplenum in trigpair]
            tr.stats.triggers.append(trigpair_utc)



def plot_stalta_triggers_on_stream(
    st: Stream,
    normalize: bool = False,
    equal_scale: bool = False,
    shade: bool = True,
    span_alpha: float = 0.18,
    on_color: str = "tab:red",
    off_color: str = "tab:blue",
    line_style: str = "--",
    line_width: float = 1.5,
    title: str = "STA/LTA Triggers",
    outfile: Optional[str] = None,
    show: bool = True,
):
    """Overlay trigger-on/off times stored in ``tr.stats.triggers`` on a Stream plot."""
    st_to_plot = st.copy()
    if normalize:
        for tr in st_to_plot:
            peak = float(np.max(np.abs(tr.data))) if len(tr.data) else 0.0
            if peak > 0.0:
                tr.data = tr.data / peak

    fig = st_to_plot.plot(show=False, equal_scale=equal_scale)
    fig.suptitle(title)
    axes = [ax for ax in fig.axes if ax.has_data()]

    for tr, ax in zip(st, axes):
        triggers = getattr(tr.stats, "triggers", []) or []
        if not triggers:
            continue

        for pair in triggers:
            if len(pair) != 2 or pair[0] is None or pair[1] is None:
                continue
            on_t, off_t = pair
            on_num = UTCDateTime(on_t).matplotlib_date
            off_num = UTCDateTime(off_t).matplotlib_date
            ax.axvline(on_num, color=on_color, linestyle=line_style, linewidth=line_width, label="Trigger On")
            ax.axvline(off_num, color=off_color, linestyle=line_style, linewidth=line_width, label="Trigger Off")
            if shade:
                ax.axvspan(on_num, off_num, color=on_color, alpha=span_alpha)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, l in zip(handles, labels):
                if l in seen:
                    continue
                seen.add(l)
                uniq_handles.append(h)
                uniq_labels.append(l)
            ax.legend(uniq_handles, uniq_labels, loc="upper right", fontsize=8)

    fig.autofmt_xdate()
    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig



def get_event_window(
    st: Stream,
    pretrig: float = 30.0,
    posttrig: float = 30.0,
) -> Tuple[Optional[UTCDateTime], Optional[UTCDateTime]]:
    """Determine a representative event window from per-trace trigger lists."""
    mintime: List[UTCDateTime] = []
    maxtime: List[UTCDateTime] = []

    for tr in st:
        if "triggers" not in tr.stats or len(tr.stats.triggers) == 0:
            continue
        trigons = [thistrig[0] for thistrig in tr.stats.triggers]
        trigoffs = [thistrig[1] for thistrig in tr.stats.triggers]
        mintime.append(min(trigons))
        maxtime.append(max(trigoffs))

    n = int(len(mintime) / 2)
    if len(mintime) > 0:
        return sorted(mintime)[n], sorted(maxtime)[n]
    return None, None



def trim_to_event(
    st: Stream,
    mintime: UTCDateTime,
    maxtime: UTCDateTime,
    pretrig: float = 10.0,
    posttrig: float = 10.0,
) -> None:
    """Trim a Stream in place to an event window with pre/post padding."""
    st.trim(starttime=mintime - pretrig, endtime=maxtime + posttrig)


# =============================================================================
# Automatic phase-picking helpers
# =============================================================================


def picker(st: Stream) -> None:
    """Run quick automatic pickers station-by-station and print their output."""
    station_groups = defaultdict(list)
    for tr in st:
        station_groups[tr.stats.station].append(tr)

    station_streams = {sta: Stream(traces) for sta, traces in station_groups.items()}
    for station, stream in station_streams.items():
        if len(stream) == 3:
            p_pick, s_pick = ar_pick(
                stream.select(component="Z")[0].data,
                stream.select(component="N")[0].data,
                stream.select(component="E")[0].data,
                stream[0].stats.sampling_rate,
                1.0,
                20.0,
                1.0,
                0.1,
                4.0,
                1.0,
                2,
                8,
                0.1,
                0.2,
            )
            print(f"ar_pick P for {station}: {p_pick} S: {s_pick}")
        else:
            for tr in stream:
                p_pick, phase_info = pk_baer(tr.data, tr.stats.sampling_rate, 20, 60, 7.0, 12.0, 100, 100)
                print(f"pk_baer P for {tr.id}: {p_pick} {phase_info}")
                aic_f = aic_simple(tr.data)
                p_idx = aic_f.argmin()
                print(f"aic_simple P for {tr.id}: {tr.stats.starttime + p_idx / tr.stats.sampling_rate}")



def filter_picks(
    tr: Trace,
    p_sec: Optional[float],
    s_sec: Optional[float],
    starttime: UTCDateTime,
    min_sec: float = 0.0,
) -> Dict[str, Dict[str, UTCDateTime]]:
    """Convert relative P/S picks in seconds into absolute UTCDateTime picks."""
    picks: Dict[str, Dict[str, UTCDateTime]] = {}

    if p_sec is not None and p_sec < min_sec:
        p_sec = None
    if s_sec is not None and s_sec < min_sec:
        s_sec = None

    if p_sec is not None:
        p_time = starttime + p_sec
    if s_sec is not None:
        s_time = starttime + s_sec

    if (p_sec is not None) and (s_sec is not None):
        if s_sec > p_sec:
            picks[tr.id] = {"P": p_time, "S": s_time}
    elif p_sec is not None:
        picks[tr.id] = {"P": p_time}
    elif s_sec is not None:
        picks[tr.id] = {"S": s_time}
    return picks



def _component_alias(code: str) -> str:
    """Normalize alternate component codes like 1/2 to E/N."""
    c = code[-1].upper()
    return {"1": "E", "2": "N"}.get(c, c)



def stream_picker(stream: Stream) -> Dict[str, Dict[str, UTCDateTime]]:
    """Run automatic phase picking on one station stream."""
    picks: Dict[str, Dict[str, UTCDateTime]] = {}

    if len(stream) == 3:
        try:
            z = stream.select(component="Z")[0]
            n = stream.select(component="N")[0]
            e = stream.select(component="E")[0]
        except Exception:
            return picks

        p_sec, s_sec = ar_pick(
            z.data,
            n.data,
            e.data,
            z.stats.sampling_rate,
            1.0,
            20.0,
            1.0,
            0.1,
            4.0,
            1.0,
            2,
            8,
            0.1,
            0.2,
        )
        picks.update(filter_picks(z, p_sec, s_sec, z.stats.starttime))
    else:
        try:
            tr = stream.select(component="Z")[0]
        except Exception:
            return picks

        p_sec, s_sec = ar_pick(
            tr.data,
            tr.data,
            tr.data,
            tr.stats.sampling_rate,
            1.0,
            20.0,
            1.0,
            0.1,
            4.0,
            1.0,
            2,
            8,
            0.1,
            0.2,
        )
        picks.update(filter_picks(tr, p_sec, s_sec, tr.stats.starttime))

    return picks



def identify_outlier_picks(
    picks: Dict[str, Dict[str, UTCDateTime]],
    phase: str = "P",
    threshold: float = 2.0,
) -> Tuple[List[str], Optional[float]]:
    """Identify pick outliers relative to the median pick time for a given phase."""
    if len(picks) < 3:
        return [], None

    pick_times = []
    pick_ids = []
    for tr_id, phases in picks.items():
        if phase in phases:
            pick_times.append(phases[phase].timestamp)
            pick_ids.append(tr_id)

    if len(pick_times) < 3:
        return [], None

    pick_times_arr = np.array(pick_times)
    median_time = np.median(pick_times_arr)
    abs_diff = np.abs(pick_times_arr - median_time)
    outliers = [pick_ids[i] for i, diff in enumerate(abs_diff) if diff > threshold]
    return outliers, float(median_time)



def plot_picks_on_stream(
    stream: Stream,
    picks: Dict[str, Dict[str, UTCDateTime]],
    title: str = "Waveform with Picks (Relative Time)",
):
    """Plot traces with vertical lines at pick times in relative seconds."""
    fig, axes = plt.subplots(len(stream), 1, figsize=(12, 2.5 * len(stream)), sharex=True)
    if len(stream) == 1:
        axes = [axes]

    for i, tr in enumerate(stream):
        t = tr.times("relative")
        axes[i].plot(t, tr.data, "k-", linewidth=0.8)
        axes[i].set_ylabel(tr.id, fontsize=9)

        if tr.id in picks:
            tr_picks = picks[tr.id]
            if "P" in tr_picks:
                p_rel = tr_picks["P"] - tr.stats.starttime
                axes[i].axvline(p_rel, color="b", linestyle="--", label="P")
            if "S" in tr_picks:
                s_rel = tr_picks["S"] - tr.stats.starttime
                axes[i].axvline(s_rel, color="r", linestyle="--", label="S")
            axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s since trace start)", fontsize=10)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig



def picker_with_plot(st: Stream, plot: bool = False) -> Dict[str, Dict[str, UTCDateTime]]:
    """Run automatic picks station-by-station, re-pick obvious outliers, and optionally plot."""
    picks: Dict[str, Dict[str, UTCDateTime]] = {}

    station_groups = defaultdict(list)
    for tr in st:
        station_groups[tr.stats.station].append(tr)

    station_streams = {sta: Stream(traces) for sta, traces in station_groups.items()}
    for _, stream in station_streams.items():
        picks.update(stream_picker(stream))

    p_threshold = 3.0
    s_threshold = p_threshold * 1.7
    p_outliers, p_median_time = identify_outlier_picks(picks, phase="P", threshold=p_threshold)
    s_outliers, s_median_time = identify_outlier_picks(picks, phase="S", threshold=s_threshold)
    all_outliers = list(set(p_outliers + s_outliers))

    if all_outliers:
        for trace_id in all_outliers:
            stime = UTCDateTime(p_median_time) - p_threshold if p_median_time is not None else st[0].stats.starttime
            etime = UTCDateTime(s_median_time) + s_threshold if s_median_time is not None else st[0].stats.endtime
            this_stream = st.select(id=trace_id).copy().trim(starttime=stime, endtime=etime)
            picks.update(stream_picker(this_stream))

    if plot:
        plot_picks_on_stream(st, picks, title="P & S Picks Overlayed on Waveform")
    return picks


# =============================================================================
# STA/LTA convenience helpers
# =============================================================================


def run_sta_lta_on_trace(
    tr: Trace,
    sta_s: float,
    lta_s: float,
    on: float,
    off: float,
    max_trigs: int = 2,
    t0_hint: Optional[UTCDateTime] = None,
    min_sep_s: float = 30.0,
    min_dur_s: float = 0.0,
    max_dur_s: Optional[float] = None,
    preprocess: bool = True,
    save_to_stats: bool = True,
) -> List[Dict[str, Any]]:
    """Run classic STA/LTA on a single Trace and optionally persist ranked triggers."""
    sr = float(tr.stats.sampling_rate or 0.0)
    if sr <= 0:
        if save_to_stats:
            tr.stats.triggers = []
            tr.stats.trigger_meta = {"algo": "classic_sta_lta", "windows": []}
        return []

    data = tr.data.astype(float, copy=False)
    if preprocess:
        bad = ~np.isfinite(data)
        if bad.any():
            data = data.copy()
            data[bad] = 0.0
        try:
            tr_tmp = tr.copy()
            tr_tmp.detrend("linear")
            tr_tmp.taper(0.01, type="cosine")
            data = tr_tmp.data.astype(float, copy=False)
        except Exception:
            pass

    nsta = max(1, int(round(sta_s * sr)))
    nlta = max(nsta + 1, int(round(lta_s * sr)))
    if nlta >= tr.stats.npts:
        if save_to_stats:
            tr.stats.triggers = []
            tr.stats.trigger_meta = {"algo": "classic_sta_lta", "windows": []}
        return []

    cft = classic_sta_lta(data, nsta, nlta)
    on_off = trigger_onset(cft, on, off)
    if not on_off:
        if save_to_stats:
            tr.stats.triggers = []
            tr.stats.trigger_meta = {"algo": "classic_sta_lta", "windows": []}
        return []

    merged: List[List[int]] = []
    min_sep_samples = int(round(min_sep_s * sr))
    for i_on, i_off in on_off:
        if not merged:
            merged.append([i_on, i_off])
            continue
        j_on, j_off = merged[-1]
        if i_on <= j_off + max(1, min_sep_samples):
            merged[-1][1] = max(j_off, i_off)
        else:
            merged.append([i_on, i_off])

    windows: List[Dict[str, Any]] = []
    for i_on, i_off in merged:
        i_on = int(max(0, min(i_on, tr.stats.npts - 1)))
        i_off = int(max(0, min(i_off, tr.stats.npts - 1)))
        if i_off <= i_on:
            continue

        dur_s = (i_off - i_on) / sr
        if dur_s < min_dur_s:
            continue
        if max_dur_s is not None and dur_s > max_dur_s:
            i_off = min(i_on + int(round(max_dur_s * sr)), tr.stats.npts - 1)
            dur_s = (i_off - i_on) / sr
            if dur_s <= 0:
                continue

        t_on = tr.stats.starttime + (i_on / sr)
        t_off = tr.stats.starttime + (i_off / sr)
        w0 = max(0, i_on - int(1 * sr))
        w1 = min(len(cft), i_on + int(5 * sr))
        peak_val = float(np.nanmax(cft[w0:w1])) if w1 > w0 else float(np.nanmax(cft))
        dt_hint = abs(t_on - t0_hint) if t0_hint else None

        windows.append(
            {
                "i_on": i_on,
                "i_off": i_off,
                "t_on": t_on,
                "t_off": t_off,
                "peak_cft": peak_val,
                "dt_from_hint": float(dt_hint) if dt_hint is not None else None,
                "dur_s": dur_s,
            }
        )

    if t0_hint is not None:
        windows.sort(key=lambda w: (w["dt_from_hint"], -w["peak_cft"]))
    else:
        windows.sort(key=lambda w: (-w["peak_cft"], w["t_on"]))

    windows = windows[:max_trigs]

    if save_to_stats:
        tr.stats.triggers = [[w["t_on"], w["t_off"]] for w in windows]
        tr.stats.trigger_meta = {
            "algo": "classic_sta_lta",
            "sta_s": sta_s,
            "lta_s": lta_s,
            "on": on,
            "off": off,
            "windows": windows,
        }

    return windows



def add_sta_lta_triggers_to_stream(
    st: Stream,
    sta_s: float,
    lta_s: float,
    on: float,
    off: float,
    max_trigs: int = 2,
    t0_hint: Optional[UTCDateTime] = None,
    min_sep_s: float = 30.0,
    min_dur_s: float = 0.0,
    max_dur_s: Optional[float] = None,
    preprocess: bool = True,
    save_to_stats: bool = True,
) -> Tuple[Dict[str, List[dict]], pd.DataFrame]:
    """Run classic STA/LTA on each trace in a Stream and optionally persist triggers."""
    picks_by_trace: Dict[str, List[dict]] = {}
    rows: List[Dict[str, Any]] = []

    for tr in st:
        try:
            picks = run_sta_lta_on_trace(
                tr,
                sta_s,
                lta_s,
                on,
                off,
                max_trigs=max_trigs,
                t0_hint=t0_hint,
                min_sep_s=min_sep_s,
                min_dur_s=min_dur_s,
                max_dur_s=max_dur_s,
                preprocess=preprocess,
                save_to_stats=save_to_stats,
            )
        except Exception:
            if save_to_stats:
                tr.stats.triggers = []
            picks = []

        picks_by_trace[tr.id] = picks
        for p in picks:
            rows.append(
                {
                    "trace_id": tr.id,
                    "t_on": p["t_on"],
                    "t_off": p["t_off"],
                    "dur_s": p["dur_s"],
                    "peak_cft": p["peak_cft"],
                    "dt_from_hint": p["dt_from_hint"],
                }
            )

    summary_df = pd.DataFrame(rows, columns=["trace_id", "t_on", "t_off", "dur_s", "peak_cft", "dt_from_hint"])
    return picks_by_trace, summary_df
