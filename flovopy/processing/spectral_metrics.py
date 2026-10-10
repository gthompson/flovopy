"""Quantitative measurements from FLOVOpy SpectrogramResult objects.

All band powers integrate *PSD* over frequency in Hz. Spectral moments
are PSD-weighted. Invalid spectrogram columns yield NaN, not zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psd(result):
    f = np.asarray(result.frequencies, dtype=float)
    s = np.asarray(result.values, dtype=float)
    if result.quantity not in ('psd', 'asd'):
        raise ValueError('Spectral metrics require PSD or ASD; legacy magnitude is not calibrated power')
    if f.ndim != 1 or len(f) < 2 or not np.all(np.diff(f) > 0):
        raise ValueError('Frequency axis must have at least two increasing bins')
    if result.quantity == 'asd':
        s = s ** 2
    if np.any(np.isfinite(s) & (s < 0)):
        raise ValueError('PSD must be nonnegative')
    return f, s


def _weights(f):
    # Midpoint frequency-bin boundaries, with DC and Nyquist bins
    # assigned half-bin width on a uniform FFT frequency grid.
    d = np.diff(f)
    if not np.allclose(d, d[0], rtol=1e-6, atol=1e-12):
        raise ValueError('Only uniform FFT frequency grids are supported')
    w = np.full(len(f), d[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _valid(s):
    return np.all(np.isfinite(s), axis=0)


def band_power(result, fmin=None, fmax=None):
    """Integrated PSD over [fmin, fmax] (Hz), using bin-edge overlap.

    Bin edges are midway between adjacent frequencies, clipped to the
    first and last frequency. Partial bins are weighted by their overlap.
    """
    f, s = _psd(result)
    lo = f[0] if fmin is None else float(fmin)
    hi = f[-1] if fmax is None else float(fmax)
    if not (np.isfinite(lo) and np.isfinite(hi) and f[0] <= lo < hi <= f[-1]):
        raise ValueError('Band limits must lie within the frequency grid and satisfy fmin < fmax')
    edges = np.r_[f[0], (f[:-1] + f[1:]) / 2, f[-1]]
    widths = np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    result_power = np.sum(np.where(np.isfinite(s), s, 0) * widths[:, None], axis=0)
    result_power[~_valid(s)] = np.nan
    return result_power


def peak_frequency(result, fmin=None, fmax=None):
    f, s = _psd(result)

    mask = (
        (f >= (f[0] if fmin is None else fmin))
        & (f <= (f[-1] if fmax is None else fmax))
    )

    if not mask.any():
        raise ValueError("No frequency bins in requested range")

    subset = s[mask]
    safe = np.where(np.isfinite(subset), subset, -np.inf)

    indices = np.argmax(safe, axis=0)
    maxima = np.max(safe, axis=0)

    out = f[mask][indices].astype(float)

    valid = _valid(s) & (maxima > 0)
    out[~valid] = np.nan

    return out

def spectral_centroid(result):
    f, s = _psd(result)
    w = _weights(f)[:, None]
    denom = np.sum(np.where(np.isfinite(s), s, 0) * w, axis=0)
    numer = np.sum(np.where(np.isfinite(s), s, 0) * w * f[:, None], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = numer / denom
    out[~_valid(s) | (denom <= 0)] = np.nan
    return out


def spectral_bandwidth(result):
    """PSD-weighted standard deviation in Hz around spectral centroid."""
    f, s = _psd(result)
    mu = spectral_centroid(result)
    w = _weights(f)[:, None]
    safe = np.where(np.isfinite(s), s, 0) * w
    denom = safe.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.sqrt((safe * (f[:, None] - mu) ** 2).sum(axis=0) / denom)
    out[~_valid(s) | (denom <= 0)] = np.nan
    return out


def spectral_ratio(result, low_band, high_band, *, log2=True):
    """High-band/low-band integrated PSD ratio; optionally log2."""
    low = band_power(result, *low_band)
    high = band_power(result, *high_band)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = high / low
        if log2:
            ratio = np.log2(ratio)
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


#def compute_spectral_metrics(result, *, low_band=(0.5, 4.0), high_band=(4.0, 18.0)):
def compute_spectral_metrics(
    result,
    low_band=None,
    high_band=None,
):
    """Return one row per FFT time, including coverage and PSD metrics."""
    return pd.DataFrame({
        'time': result.times.copy(), 'coverage': result.coverage.copy(),
        'peak_frequency': peak_frequency(result),
        'spectral_centroid': spectral_centroid(result),
        'spectral_bandwidth': spectral_bandwidth(result),
        'low_band_power': band_power(result, *low_band),
        'high_band_power': band_power(result, *high_band),
        'fratio': spectral_ratio(result, low_band, high_band),
    })
