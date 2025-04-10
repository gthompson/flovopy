"""
Python version of the original ampengfft.c program.

This module emulates the core functionality of ampengfft:
- Computes amplitude, energy, and dominant frequency for seismic traces.
- Calculates single-station amplitude metrics (SSAM) using fixed frequency bands.
- Intended for use with ObsPy Trace and Stream objects.

Assumptions:
- Input traces are velocity (for energy estimates) or displacement (for amplitude), depending on use.
- Pre-trigger and post-trigger windows can be handled via slicing prior to calling this function.
- FFT is performed with zero-padding to next power of 2.

Frequency bands match ampengfft.c:
  [0.1, 1.0, 2.0, ..., 10.0, 30.0] Hz (11 bins between 12 edges).
"""
import numpy as np
from obspy import Stream, Trace
from scipy.signal import savgol_filter
from scipy.stats import describe
from flovopy.core.preprocessing import add_to_trace_history

def compute_ampengfft(trace, freq_bins=None, amp_avg_window=2.0):
    if freq_bins is None:
        freq_bins = np.array([0.1, 1.0, 2.0, 3.0, 4.0, 5.0,
                              6.0, 7.0, 8.0, 9.0, 10.0, 30.0])

    tr = trace.copy()
    tr.detrend('linear')
    tr.taper(0.01)
    dt = tr.stats.delta
    fs = tr.stats.sampling_rate
    N = len(tr.data)

    # Apply gain if present
    calib = tr.stats.get('calib') or tr.stats.get('gain_factor')
    if calib:
        tr.data = tr.data * calib

    # Pad to next power of 2
    nfft = int(2 ** np.ceil(np.log2(N)))
    fft_vals = np.fft.rfft(tr.data, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    amps = np.abs(fft_vals)

    # Amplitude spectrum
    spectral = {'freqs': freqs, 'amplitudes': amps}

    # Peak amplitude
    peakamp = np.max(np.abs(tr.data))

    # Energy: sum of squared velocity
    energy = np.sum(tr.data ** 2) * dt

    # Peak frequency
    peakf = freqs[np.argmax(amps)]

    # Spectral slice energy percentages
    band_energy = []
    total = np.sum(amps ** 2)
    for i in range(len(freq_bins) - 1):
        fmin, fmax = freq_bins[i], freq_bins[i+1]
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        frac = np.sum(amps[idx] ** 2) / total if total > 0 else 0
        band_energy.append(frac)

    ssam = {
        'f': 0.5 * (freq_bins[:-1] + freq_bins[1:]),
        'E': np.array(band_energy),
        'edges': freq_bins
    }

    # Max average amplitude using sliding window
    nwin = int(amp_avg_window * fs)
    if nwin % 2 == 0:
        nwin += 1
    avg_amp = savgol_filter(np.abs(tr.data), nwin, 2, mode='interp')
    max_avg_amp = np.nanmax(avg_amp)

    # Populate metrics
    if not hasattr(tr.stats, 'metrics'):
        tr.stats.metrics = {}

    tr.stats.spectral = spectral
    tr.stats.metrics['peakamp'] = peakamp
    tr.stats.metrics['energy'] = energy
    tr.stats.metrics['peakf'] = peakf
    tr.stats.metrics['ssam'] = ssam
    tr.stats.metrics['avg_amp_max'] = max_avg_amp

    try:
        stats = describe(tr.data)
        tr.stats.metrics['skewness'] = stats.skewness
        tr.stats.metrics['kurtosis'] = stats.kurtosis
    except Exception:
        pass

    add_to_trace_history(tr, 'ampengfft_py')
    return tr

def compute_ampengfft_stream(stream, **kwargs):
    return Stream([compute_ampengfft(tr, **kwargs) for tr in stream])

def write_aef_file(stream, filepath, trigger_window=None, average_window=None):
    """
    Write a stream to an .AEF file in the legacy ampengfft format.
    Assumes each trace has ampengfft metrics already computed.
    """
    with open(filepath, 'w') as f:
        f.write("      STAT CMP   MAX AVG  TOTAL ENG             FREQUENCY BINS (Hz)       MAX  3\n")
        f.write("                AMP (m/s)   (J/kg)   0.1  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 10.0 30.0 (Hz) 3\n")

        for tr in stream:
            m = tr.stats.metrics
            sta = tr.stats.station
            cha = tr.stats.channel

            ssam_fracs = m['ssam']['E'] if 'ssam' in m and 'E' in m['ssam'] else [0.0]*11
            volc_line = f"VOLC {sta:<4} {cha:<4} A{m['avg_amp_max']:.2e} E{m['energy']:.2e} "
            volc_line += ' '.join(f"{int(round(x * 100)):2d}" for x in ssam_fracs)
            volc_line += f" {m['peakf']:.2f} 3\n"
            f.write(volc_line)

        # Add trigger and average window lines if specified
        if trigger_window is not None:
            f.write(f" trigger window={trigger_window:.0f}s\n")
        if average_window is not None:
            f.write(f" average window={average_window:.0f}s\n")
        if trigger_window or average_window:
            f.write("                                          3\n")

