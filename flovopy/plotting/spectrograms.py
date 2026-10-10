"""UTC-aligned multichannel waveform/spectrogram plots (IceWeb layout).

Rendering is independent of numerical spectral estimation. No input mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from obspy import Stream
from flovopy.processing.spectrograms import (
    SpectrogramResult, compute_stream_spectrograms,
)


def _edges(centres, fallback_width):
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 0:
        return np.empty(0)
    if len(centres) == 1:
        width = float(fallback_width)
        return np.array([centres[0] - width / 2, centres[0] + width / 2])
    mid = (centres[:-1] + centres[1:]) / 2
    return np.r_[centres[0] - (mid[0] - centres[0]), mid,
                 centres[-1] + (centres[-1] - mid[-1])]


def _datetime_x(unix_seconds):
    # Matplotlib's floating-day representation; UTC is explicit.
    return mdates.epoch2num(unix_seconds) if hasattr(mdates, 'epoch2num') else np.asarray(unix_seconds) / 86400.0 + mdates.date2num(__import__('datetime').datetime(1970, 1, 1, tzinfo=__import__('datetime').timezone.utc))


def plot_waveform_spectrograms(
    stream: Stream,
    spectrograms: Mapping[str, SpectrogramResult] | None = None,
    *,
    fft_seconds: float = 2.56,
    step_seconds: float | None = None,
    quantity: str = 'psd',
    frequency_limits: tuple[float, float] | None = (0.5, 25.0),
    frequency_scale: str = 'linear',
    color_scale: str = 'individual',
    clim: tuple[float, float] | None = None,
    percentiles: tuple[float, float] = (5.0, 99.0),
    db: bool = True,
    reference: float = 1.0,
    cmap: str = 'viridis',
    channel_order: list[str] | None = None,
    title: str | None = None,
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
    outfile: str | None = None,
):
    """Plot one waveform and spectrogram pair per ObsPy trace.

    ``spectrograms`` may contain precomputed results keyed by SEED trace ID.
    If omitted, compute PSD spectrograms from the input Stream. All panels
    share absolute UTC time. Gaps remain masked, never filled with zeros.

    ``db=True`` uses 10 log10(PSD/reference) or 20 log10(ASD/reference).
    ``legacy_magnitude`` is also amplitude-like and uses 20 log10.
    ``clim`` is expressed in *display units* (dB when db=True).
    Returns a Matplotlib Figure. Does not call plt.show() or close the figure.
    """
    if not isinstance(stream, Stream):
        raise TypeError('stream must be an ObsPy Stream')
    if len(stream) == 0:
        raise ValueError('stream must contain at least one trace')
    if frequency_scale not in ('linear', 'log'):
        raise ValueError("frequency_scale must be 'linear' or 'log'")
    if color_scale not in ('individual', 'shared', 'robust'):
        raise ValueError("color_scale must be individual, shared, or robust")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('reference must be positive')
    if frequency_limits is not None:
        lo, hi = frequency_limits
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo < hi):
            raise ValueError('invalid frequency_limits')
        if frequency_scale == 'log' and lo <= 0:
            raise ValueError('logarithmic frequency limits must be positive')
    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError('invalid percentiles')
    if clim is not None and not (np.isfinite(clim).all() and clim[0] < clim[1]):
        raise ValueError('clim must be an increasing finite pair')

    traces = list(stream)
    ids = [tr.id for tr in traces]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate trace IDs; merge/select segments before plotting')
    if channel_order is not None:
        if len(set(channel_order)) != len(channel_order):
            raise ValueError('duplicate IDs in channel_order')
        missing = set(channel_order) - set(ids)
        if missing:
            raise ValueError(f'unknown channels: {sorted(missing)}')
        by_id = {tr.id: tr for tr in traces}
        traces = [by_id[key] for key in channel_order]
    if not traces:
        raise ValueError('channel_order selects no traces')
    if spectrograms is None:
        kwargs = {'fft_seconds': fft_seconds, 'quantity': quantity}
        if step_seconds is not None:
            kwargs['step_seconds'] = step_seconds
        spectrograms = compute_stream_spectrograms(Stream(traces), **kwargs)
    for tr in traces:
        if tr.id not in spectrograms:
            raise ValueError(f'missing spectrogram for {tr.id}')
        if not isinstance(spectrograms[tr.id], SpectrogramResult):
            raise TypeError(f'{tr.id}: expected SpectrogramResult')
        if spectrograms[tr.id].trace_id != tr.id:
            raise ValueError(f'{tr.id}: spectrogram trace_id mismatch')

    prepared = []
    for tr in traces:
        result = spectrograms[tr.id]
        freqs = result.frequencies
        select = np.ones(len(freqs), dtype=bool)
        if frequency_limits is not None:
            select &= (freqs >= frequency_limits[0]) & (freqs <= frequency_limits[1])
        if frequency_scale == 'log':
            select &= freqs > 0
        f = freqs[select]
        z = np.asarray(result.values[select], dtype=float).copy()
        z[~np.isfinite(z)] = np.nan
        if db:
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (10 if result.quantity == 'psd' else 20) * np.log10(z / reference)
            z[~np.isfinite(z)] = np.nan
        prepared.append((tr, result, f, z))

    finite = [z[np.isfinite(z)] for _, _, _, z in prepared if np.isfinite(z).any()]
    combined = np.concatenate(finite) if finite else np.array([])
    def limits(values, robust):
        if values.size == 0:
            return (0.0, 1.0)
        low, high = (np.percentile(values, percentiles) if robust
                     else (float(np.min(values)), float(np.max(values))))
        if low == high:
            pad = max(abs(low) * 0.01, 1e-6)
            return (low - pad, high + pad)
        return (float(low), float(high))
    common = (clim if clim is not None else
              limits(combined, color_scale == 'robust'))

    n = len(prepared)
    fig = plt.figure(figsize=figsize or (12, max(3.0, 2.7 * n)), constrained_layout=True)
    grid = fig.add_gridspec(2 * n, 1, height_ratios=[v for _ in range(n) for v in (1, 2)], hspace=0.07)
    axes = []
    for i, (tr, result, f, z) in enumerate(prepared):
        ax_wave = fig.add_subplot(grid[2*i, 0], sharex=axes[0][0] if axes else None)
        ax_spec = fig.add_subplot(grid[2*i+1, 0], sharex=ax_wave)
        axes.append((ax_wave, ax_spec))
        raw = np.ma.asarray(tr.data, dtype=float)
        y = np.asarray(raw.filled(np.nan), dtype=float)
        y[~np.isfinite(y)] = np.nan
        t = float(tr.stats.starttime.timestamp) + np.arange(len(y)) / float(tr.stats.sampling_rate)
        ax_wave.plot(_datetime_x(t), y, lw=0.55, color='black')
        ax_wave.set_ylabel(tr.id, fontsize=8, rotation=0, ha='right', va='center', labelpad=6)
        ax_wave.grid(alpha=0.2)
        ax_wave.tick_params(labelbottom=False)
        norm_limits = (clim if clim is not None else
                       common if color_scale != 'individual' else
                       limits(z[np.isfinite(z)], False))
        if f.size and result.times.size:
            te = _datetime_x(_edges(result.times, result.metadata.get('step_seconds', 1.0)))
            fe = _edges(f, float(np.median(np.diff(f))) if f.size > 1 else 1.0)
            if frequency_scale == 'log':
                fe = np.maximum(fe, np.finfo(float).tiny)
            mesh = ax_spec.pcolormesh(te, fe, np.ma.masked_invalid(z),
                                      shading='flat', cmap=cmap,
                                      norm=Normalize(*norm_limits))
            if colorbar:
                fig.colorbar(mesh, ax=ax_spec, pad=0.005, fraction=0.018,
                             label=f"{result.quantity.upper()}" + (' (dB)' if db else
                             f" ({result.units})" if result.units else ''))
        if frequency_scale == 'log':
            ax_spec.set_yscale('log')
        if frequency_limits is not None:
            ax_spec.set_ylim(*frequency_limits)
        ax_spec.set_ylabel('Hz')
        ax_spec.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=__import__('datetime').timezone.utc))
        if i != n-1:
            ax_spec.tick_params(labelbottom=False)
    axes[-1][1].set_xlabel('Time (UTC)')
    if title:
        fig.suptitle(title)
    fig._flovopy_axes = axes  # convenient access; not part of stable public API
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
    return fig
