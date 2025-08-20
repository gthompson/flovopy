import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import read, Stream, Trace, UTCDateTime
from obspy.core.event import Event, Origin, Magnitude, Comment, ResourceIdentifier
from obspy.core.event.magnitude import StationMagnitude, StationMagnitudeContribution
from obspy.core.event.base import WaveformStreamID

from collections import defaultdict
from math import pi


"""
Functions for computing data quality metrics and statistical metrics (such as amplitude, energy and frequency) 
on Stream/Trace objects.
"""
def estimate_snr(trace_or_stream, method='std', window_length=1.0, split_time=None,
                 verbose=True, spectral_kwargs=None, freq_band=None):
    """
    Estimate the signal-to-noise ratio (SNR) from a Trace or Stream.

    Parameters
    ----------
    trace_or_stream : obspy.Trace or obspy.Stream
        Input signal(s).
    method : str
        SNR method: 'max', 'std', 'rms', or 'spectral'.
    window_length : float
        Window length in seconds for window-based methods.
    split_time : UTCDateTime or tuple/list of UTCDateTime
        Time or (start, end) tuple to define signal/noise intervals.
    verbose : bool
        Print detailed output.
    spectral_kwargs : dict, optional
        Arguments passed to compute_amplitude_ratios().
    freq_band : tuple(float, float), optional
        Frequency range (Hz) to compute average SNR in 'spectral' mode.

    Returns
    -------
    snr, signal_val, noise_val : float
        SNR and underlying values.
    """
    if isinstance(trace_or_stream, Stream):
        results = [estimate_snr(tr, method, window_length, split_time, verbose, spectral_kwargs, freq_band)
                for tr in trace_or_stream]
        snrs, signals, noises = zip(*results)
        return list(snrs), list(signals), list(noises)


    trace = trace_or_stream
    fs = trace.stats.sampling_rate
    snr = signal_val = noise_val = np.nan

    try:
        # ---- Get signal and noise data ----
        # ---- Get signal and noise data (no padding; well-defined noise) ----
        def _extract_win(tr, t1, t2):
            trc = tr.copy().trim(starttime=t1, endtime=t2, pad=False)
            return trc.data if trc.stats.npts > 0 else None

        pre_noise_gap = 2.0  # seconds to back off noise from onset (adjustable)

        if split_time:
            if isinstance(split_time, (list, tuple)) and len(split_time) == 2:
                t_on, t_off = split_time
                dur = float(t_off - t_on)
                if not np.isfinite(dur) or dur <= 0:
                    return np.nan, np.nan, np.nan

                # signal = [t_on, t_off]
                sig = _extract_win(trace, t_on, t_off)

                # noise = same duration before t_on with a small guard
                n2 = t_on - pre_noise_gap
                n1 = n2 - dur
                noi = _extract_win(trace, n1, n2)

            else:
                # split_time is an onset; use window_length for both windows
                t_on = split_time
                dur = float(window_length)
                if not np.isfinite(dur) or dur <= 0:
                    return np.nan, np.nan, np.nan

                sig = _extract_win(trace, t_on, t_on + dur)
                n2  = t_on - pre_noise_gap
                n1  = n2 - dur
                noi = _extract_win(trace, n1, n2)
        else:
            # auto-mode (unchanged): loudest vs quietest equal windows of length window_length
            data = trace.data
            fs = trace.stats.sampling_rate
            samples_per_window = int(fs * window_length)
            num_windows = int(trace.stats.npts // samples_per_window)
            if num_windows < 2:
                if verbose: print("Trace too short for SNR estimation.")
                return np.nan, np.nan, np.nan
            reshaped = data[:samples_per_window * num_windows].reshape((num_windows, samples_per_window))
            sig = reshaped[np.argmax(np.nanstd(reshaped, axis=1))]
            noi = reshaped[np.argmin(np.nanstd(reshaped, axis=1))]

        # Bail if either window is missing (do NOT pad with zeros)
        if sig is None or noi is None:
            if verbose: print("[SNR] window not covered by trace; skipping (no padding).")
            return np.nan, np.nan, np.nan
        signal_data, noise_data = sig, noi


        # ---- Compute SNR based on method ----
        if method == 'max':
            signal_val = np.nanmax(np.abs(signal_data))
            noise_val = np.nanmax(np.abs(noise_data))
        elif method in ('std', 'rms'):
            signal_val = np.nanstd(signal_data)
            noise_val = np.nanstd(noise_data)
        elif method == 'spectral':
            spectral_kwargs = spectral_kwargs or {}
            freqs, avg_ratio, *_ = compute_amplitude_ratios(
                signal_data, noise_data, **spectral_kwargs, verbose=verbose
            )
            if avg_ratio is not None:
                if freq_band:
                    fmin, fmax = freq_band
                    band_mask = (freqs >= fmin) & (freqs <= fmax)
                    band_vals = avg_ratio[band_mask]
                else:
                    band_vals = avg_ratio
                signal_val = np.nanmax(band_vals)
                noise_val = np.nanmin(band_vals)
                snr = np.nanmean(band_vals)
            else:
                signal_val = noise_val = snr = np.nan
        else:
            raise ValueError(f"Unknown SNR method: {method}")

        if method != 'spectral':
            snr = signal_val / noise_val if noise_val != 0 else np.inf

    except Exception as e:
        if verbose:
            print(f"[ERROR] SNR estimation failed: {e}")
        return np.nan, np.nan, np.nan

    # ---- Store metrics ----
    if 'metrics' not in trace.stats:
        trace.stats.metrics = {}
    trace.stats.metrics[f'snr_{method}'] = snr
    trace.stats.metrics[f'signal_level_{method}'] = signal_val
    trace.stats.metrics[f'noise_level_{method}'] = noise_val

    if verbose:
        print(f"[{method}] SNR = {snr:.2f} (signal={signal_val:.2f}, noise={noise_val:.2f})")

    return snr, signal_val, noise_val



def _check_spectral_qc(freqs, ratios, threshold=2.0, min_fraction_pass=0.5):
    """
    Check if a spectral ratio passes quality control.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins.
    ratios : np.ndarray
        Spectral amplitude ratios.
    threshold : float
        Minimum acceptable amplitude ratio.
    min_fraction_pass : float
        Fraction of frequency bins that must exceed threshold.

    Returns
    -------
    passed : bool
        Whether the QC check passed.
    failed_fraction : float
        Fraction of frequencies below threshold.
    """    
    failed = np.sum(ratios < threshold)
    frac_failed = failed / len(ratios)
    return frac_failed <= (1 - min_fraction_pass), frac_failed

def compute_amplitude_ratios(signal_stream, noise_stream, smooth_window=None,
                              verbose=False, average='geometric',
                              qc_threshold=None, qc_fraction=0.5):
    """
    Compute spectral amplitude ratios between signal and noise traces.

    Parameters
    ----------
    signal_stream : obspy.Stream or obspy.Trace
        Signal trace(s). If a single Trace, it will be wrapped in a Stream.
    noise_stream : obspy.Stream or obspy.Trace
        Noise trace(s). If a single Trace, it will be wrapped in a Stream.
    smooth_window : int, optional
        Length of moving average window for smoothing the amplitude ratio.
    verbose : bool, optional
        If True, prints processing information.
    average : str, optional
        Method for computing the average spectral ratio. One of:
        'geometric', 'median', or 'mean'.
    qc_threshold : float, optional
        Minimum acceptable amplitude ratio for QC check.
    qc_fraction : float, optional
        Minimum fraction of frequency bins that must exceed `qc_threshold`
        for the QC to pass.

    Returns
    -------
    avg_freqs : np.ndarray or None
        Frequencies (Hz) corresponding to the average spectral ratio.
    avg_spectral_ratio : np.ndarray or None
        Averaged amplitude ratio spectrum (signal / noise).
    individual_ratios : list of np.ndarray
        List of spectral amplitude ratios for each matching trace pair.
    freqs_list : list of np.ndarray
        List of frequency arrays corresponding to each ratio.
    trace_ids : list of str
        Trace IDs for which amplitude ratios were computed.
    qc_results : dict
        Dictionary of QC results per trace. Each entry contains:
        {'passed': bool, 'failed_fraction': float}
    """
    if isinstance(signal_stream, Trace):
        signal_stream = Stream([signal_stream])
    if isinstance(noise_stream, Trace):
        noise_stream = Stream([noise_stream])

    noise_dict = {tr.id: tr for tr in noise_stream}
    individual_ratios = []
    freqs_list = []
    trace_ids = []
    qc_results = {}

    for sig_tr in signal_stream:
        trace_id = sig_tr.id
        if trace_id not in noise_dict:
            if verbose:
                print(f"Skipping {trace_id}: No matching noise trace.")
            continue

        noise_tr = noise_dict[trace_id]

        max_len = max(len(sig_tr.data), len(noise_tr.data))
        sig_data = np.pad(sig_tr.data, (0, max_len - len(sig_tr.data)), mode='constant')
        noise_data = np.pad(noise_tr.data, (0, max_len - len(noise_tr.data)), mode='constant')

        dt = sig_tr.stats.delta
        N = max_len

        fft_signal = np.fft.fft(sig_data)
        fft_noise = np.fft.fft(noise_data)
        freqs = np.fft.fftfreq(N, d=dt)

        amp_signal = np.abs(fft_signal)
        amp_noise = np.abs(fft_noise)
        amp_noise[amp_noise == 0] = 1e-10  # Avoid division by zero

        amplitude_ratio = amp_signal / amp_noise

        if smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            amplitude_ratio = np.convolve(amplitude_ratio, kernel, mode="same")

        ratio_half = amplitude_ratio[:N//2]
        freq_half = freqs[:N//2]

        individual_ratios.append(ratio_half)
        freqs_list.append(freq_half)
        trace_ids.append(trace_id)

        if qc_threshold is not None:
            passed, frac_failed = _check_spectral_qc(freq_half, ratio_half,
                                                     threshold=qc_threshold,
                                                     min_fraction_pass=qc_fraction)
            qc_results[trace_id] = {'passed': passed, 'failed_fraction': frac_failed}

    if not individual_ratios:
        return None, None, [], [], [], {}

    if average == 'geometric':
        avg_spectral_ratio = np.exp(np.mean(np.log(np.vstack(individual_ratios) + 1e-10), axis=0))
    elif average == 'median':
        avg_spectral_ratio = np.median(np.array(individual_ratios), axis=0)
    else:
        avg_spectral_ratio = np.mean(np.array(individual_ratios), axis=0)

    avg_freqs = freqs_list[0]

    if verbose:
        print(f"Computed amplitude ratios for {len(individual_ratios)} traces.")

    return avg_freqs, avg_spectral_ratio, individual_ratios, freqs_list, trace_ids, qc_results

def plot_amplitude_ratios(avg_freqs, avg_spectral_ratio,
                          individual_ratios=None, freqs_list=None, trace_ids=None,
                          log_scale=False, outfile=None, max_freq=50, threshold=None):
    """
    Plot spectral amplitude ratios (signal / noise), including optional individual traces.

    Parameters
    ----------
    avg_freqs : np.ndarray
        Frequency bins for the averaged spectral ratio.
    avg_spectral_ratio : np.ndarray
        Averaged amplitude ratio (signal / noise) spectrum.
    individual_ratios : list of np.ndarray, optional
        List of individual trace amplitude ratio spectra to overlay (default: None).
    freqs_list : list of np.ndarray, optional
        List of frequency bins corresponding to each entry in `individual_ratios`.
        Must match in length and shape (default: None).
    trace_ids : list of str, optional
        Labels for individual traces to include in the legend (default: index numbers).
    log_scale : bool, optional
        If True, plot the log10 of (amplitude ratio + 1) to better show weak signals (default: False).
    outfile : str, optional
        Path to save the figure instead of displaying it interactively (default: None).
    max_freq : float, optional
        Upper x-axis limit for frequency (Hz) (default: 50).
    threshold : float, optional
        If provided, plot a horizontal line at this amplitude ratio to indicate a QC threshold.

    Returns
    -------
    None
        Displays or saves the plot. Does not return anything.
    """


    plt.figure(figsize=(10, 6))

    if individual_ratios and freqs_list:
        for i, ratio in enumerate(individual_ratios):
            freqs = freqs_list[i]
            label = trace_ids[i] if trace_ids else f"Trace {i}"
            y = np.log10(ratio + 1) if log_scale else ratio
            plt.plot(freqs, y, label=label, alpha=0.5, linewidth=1)

    if avg_spectral_ratio is not None:
        y_avg = np.log10(avg_spectral_ratio + 1) if log_scale else avg_spectral_ratio
        plt.plot(avg_freqs, y_avg, color='black', linewidth=2.5, label='Average')

    if threshold:
        y_thresh = np.log10(threshold + 1) if log_scale else threshold
        plt.axhline(y_thresh, color='red', linestyle='--', label='SNR Threshold')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Amplitude Ratio" if log_scale else "Amplitude Ratio")
    plt.title("Amplitude Spectrum Ratio (Signal/Noise)")
    plt.xlim(0, max_freq)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()

    if outfile:
        print(f'Saving amplitude_ratio plot to {outfile} from {os.getcwd()}')
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


def compute_amplitude_spectra(stream):
    """
    Compute amplitude spectra for all traces in a stream and store results in trace.stats.spectral.

    Parameters
    ----------
    stream : obspy.Stream
        Stream of Trace objects.

    Returns
    -------
    stream : obspy.Stream
        Same stream with each trace tagged with:
        - tr.stats.spectral['freqs']
        - tr.stats.spectral['amplitudes']
    """
    for tr in stream:
        dt = tr.stats.delta
        N = len(tr.data)
        fft_vals = np.fft.fft(tr.data)
        freqs = np.fft.fftfreq(N, d=dt)
        amp = np.abs(fft_vals)
        pos_freqs = freqs[:N//2]
        pos_amps = amp[:N//2]

        if not hasattr(tr.stats, 'spectral'):
            tr.stats.spectral = {}
        tr.stats.spectral['freqs'] = pos_freqs
        tr.stats.spectral['amplitudes'] = pos_amps

    return stream


def plot_amplitude_spectra(stream, max_freq=50, outfile=None):
    """
    Plot amplitude spectra for all traces in a stream, assuming they have spectral data.

    Parameters
    ----------
    stream : obspy.Stream
        Stream of Trace objects. Each trace must have .stats.spectral['freqs'] and ['amplitudes'].
    max_freq : float, optional
        Upper limit of x-axis (Hz). Default is 50 Hz.
    outfile : str, optional
        If provided, save the plot to this file instead of showing it.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for tr in stream:
        try:
            f = tr.stats.spectral['freqs']
            a = tr.stats.spectral['amplitudes']
            plt.plot(f, a, label=tr.id)
        except (AttributeError, KeyError):
            continue

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude Spectrum of Seismic Signals")
    plt.legend()
    plt.grid()
    plt.xlim(0, max_freq)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


def export_trace_metrics_to_dataframe(stream, include_id=True, include_starttime=True):
    """
    Export all trace.stats.metrics and associated metadata into a pandas DataFrame.

    Parameters
    ----------
    stream : obspy.Stream
        Stream with populated trace.stats.metrics dictionaries.
    include_id : bool
        Whether to include trace.id as a column.
    include_starttime : bool
        Whether to include trace.stats.starttime as a column.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of metrics, one row per trace.
    """
    rows = []

    for tr in stream:
        s = tr.stats
        row = {}

        if include_id:
            row['id'] = tr.id
        if include_starttime:
            row['starttime'] = s.starttime

        row['Fs'] = s.sampling_rate
        row['calib'] = getattr(s, 'calib', None)
        row['units'] = getattr(s, 'units', None)
        row['quality'] = getattr(s, 'quality_factor', None)

        if hasattr(s, 'spectrum'):
            for item in ['medianF', 'peakF', 'peakA', 'bw_min', 'bw_max']:
                row[item] = s.spectrum.get(item, None)

        if hasattr(s, 'metrics'):
            m = s.metrics
            for item in [
                'snr', 'signal_level', 'noise_level', 'twin',
                'peakamp', 'peaktime', 'energy', 'RSAM_high', 'RSAM_low',
                'sample_min', 'sample_max', 'sample_mean', 'sample_median',
                'sample_lower_quartile', 'sample_upper_quartile', 'sample_rms',
                'sample_stdev', 'percent_availability', 'num_gaps', 'skewness', 'kurtosis'
            ]:
                row[item] = m.get(item, None)
            for key, value in m.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        row[f"{key}_{subkey}"] = subval

        if 'bandratio' in m:
            for dictitem in m['bandratio']:
                label = 'bandratio_' + "".join(str(dictitem['freqlims'])).replace(', ', '_')
                row[label] = dictitem['RSAM_ratio']

        if 'lon' in s:
            row['lon'] = s['lon']
            row['lat'] = s['lat']
            row['elev'] = s['elev']

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.round({
        'Fs': 2, 'secs': 2, 'quality': 2, 'medianF': 1, 'peakF': 1,
        'bw_max': 1, 'bw_min': 1, 'peaktime': 2, 'twin': 2,
        'skewness': 2, 'kurtosis': 2
    })
    return df

        
def load_trace_metrics_from_dataframe(df, stream, match_on='id'):
    """
    Load metrics from a DataFrame into each trace.stats.metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with one row per trace and metrics as columns.
    stream : obspy.Stream
        Stream of Trace objects to update.
    match_on : str
        Column to match traces by ('id' or 'starttime').

    Returns
    -------
    stream : obspy.Stream
        Stream with updated .stats.metrics per trace.
    """
    for tr in stream:
        key = tr.id if match_on == 'id' else tr.stats.starttime
        row = df[df[match_on] == key]
        if row.empty:
            continue

        tr.stats.metrics = {}

        for col in row.columns:
            if col in ['id', 'starttime']:
                continue
            val = row.iloc[0][col]

            # Optional: split compound keys like 'ssam_A' back into nested dicts
            if '_' in col:
                main_key, sub_key = col.split('_', 1)
                if main_key not in tr.stats.metrics:
                    tr.stats.metrics[main_key] = {}
                tr.stats.metrics[main_key][sub_key] = val
            else:
                tr.stats.metrics[col] = val

    return stream

def save_enhanced_stream(st, enhanced_wavpath, save_pickle=False):
    """
    Save a stream to MiniSEED along with a CSV of enhanced metrics.

    Parameters
    ----------
    st : obspy.Stream
        Stream with .stats.metrics, .stats.spectrum, etc.
    enhanced_wavpath : str
        Path without extension (.mseed, .csv will be added).
    save_pickle : bool
        Whether to also save as ObsPy .pickle format (optional).
    """
    if enhanced_wavpath.endswith('.mseed'):
        enhanced_wavpath = enhanced_wavpath[:-6]

    os.makedirs(os.path.dirname(enhanced_wavpath), exist_ok=True)

    # Save MiniSEED
    st.write(enhanced_wavpath + '.mseed', format='MSEED')

    # Save metrics CSV
    df = export_trace_metrics_to_dataframe(st)
    df.to_csv(enhanced_wavpath + '.csv', index=False)

    # Optional: also save full stream with attributes
    if save_pickle:
        st.write(enhanced_wavpath + '.pickle', format='PICKLE')


def read_enhanced_stream(enhanced_wavpath):
    """
    Read a stream from MiniSEED + CSV metrics and restore enhanced trace.stats.

    Parameters
    ----------
    enhanced_wavpath : str
        Path without extension (.mseed, .csv will be added).

    Returns
    -------
    st : obspy.Stream
        Stream with restored .stats.metrics and other attributes.
    """
    if enhanced_wavpath.endswith('.mseed'):
        enhanced_wavpath = enhanced_wavpath[:-6]

    st = read(enhanced_wavpath + '.mseed')
    df = pd.read_csv(enhanced_wavpath + '.csv')

    # First pass: restore metrics
    st = load_trace_metrics_from_dataframe(df, st)

    # Second pass: restore extra fields not in .metrics
    for tr in st:
        row = df[df['id'] == tr.id]
        if row.empty:
            continue
        row = row.iloc[0]

        s = tr.stats
        s.units = row.get('units', None)
        s.calib = row.get('calib', None)
        s.quality_factor = row.get('quality', None)

        # Restore spectrum if present
        spectrum_keys = ['medianF', 'peakF', 'peakA', 'bw_min', 'bw_max']
        s.spectrum = {k: row[k] for k in spectrum_keys if k in row and not pd.isna(row[k])}

        # Restore bandratio (if any)
        bandratios = []
        for col in row.index:
            if col.startswith('bandratio_') and not pd.isna(row[col]):
                try:
                    freqlims_str = col.replace('bandratio_', '').replace('[', '').replace(']', '')
                    freqlims = [float(x) for x in freqlims_str.split('_')]
                    bandratios.append({'freqlims': freqlims, 'RSAM_ratio': row[col]})
                except Exception as e:
                    print(f"Warning: couldn't parse {col}: {e}")
        if bandratios:
            s.bandratio = bandratios

        for coord in ['lon', 'lat', 'elev']:
            if coord in row:
                s[coord] = row[coord]

    return st



    
def VASR(Eacoustic, Eseismic):
    # From Johnson and Aster 2005
    eta = Eacoustic / Eseismic
    return eta

def attenuation(tr, R, Q=50, c_earth=2500):
    s = tr.stats
    if 'spectrum' in s: 
        peakF = s['spectrum']['peakF']
        exponent = - ((pi) * peakF * R) / (c_earth * Q)
        A = np.exp(exponent)
        return A
    else:
        return None

def Eseismic2magnitude(Eseismic, correction=3.7):
    # after equation 7 in Hanks and Kanamori 1979, where moment is substitute with energy
    # energy in Joules rather than ergs, so correction is 3.7 rather than 10.7
    if isinstance(Eseismic, list): # list of stationEnergy
        mag = [] 
        for thisE in Eseismic:
            mag.append(Eseismic2magnitude(thisE, correction=correction))
    else:
        mag = np.log10(Eseismic)/1.5 - correction
    return mag


def stream_to_event(stream, source_coords, df=None,
                    event_id=None, creation_time=None,
                    event_type=None, mainclass=None, subclass=None):
    """
    Convert a Stream and source metadata into an ObsPy Event object.

    Parameters
    ----------
    stream : obspy.Stream
        Stream of traces with metrics in .stats.metrics
    source_coords : dict
        {'latitude': ..., 'longitude': ..., 'depth': ...} in meters
    df : pandas.DataFrame, optional
        Output from summarize_magnitudes()
    event_id : str, optional
        Resource ID for the Event
    creation_time : str or UTCDateTime, optional
        Event creation time
    event_type : str, optional
        QuakeML-compatible event type (e.g. "earthquake", "volcanic eruption")
    mainclass : str, optional
        High-level classification (e.g., "volcano-seismic", "tectonic")
    subclass : str, optional
        Detailed classification (e.g., "hybrid", "long-period")

    Returns
    -------
    event : obspy.core.event.Event
    """
    event = Event()

    # Resource ID and creation time
    if event_id:
        event.resource_id = ResourceIdentifier(id=event_id)
    if creation_time:
        event.creation_info = {'creation_time': UTCDateTime(creation_time)}

    # Event type
    if event_type:
        event.event_type = event_type

    # Add origin
    origin = Origin(
        latitude=source_coords['latitude'],
        longitude=source_coords['longitude'],
        depth=source_coords.get('depth', 0)
    )
    event.origins.append(origin)

    # Add network-averaged magnitudes from DataFrame
    if df is not None and hasattr(df, 'iloc'):
        net_ml = df.iloc[-1].get('network_mean_ML')
        net_me = df.iloc[-1].get('network_mean_ME')

        if pd.notnull(net_ml):
            mag_ml = Magnitude(
                mag=net_ml,
                magnitude_type="ML",
                origin_id=origin.resource_id
            )
            event.magnitudes.append(mag_ml)

        if pd.notnull(net_me):
            mag_me = Magnitude(
                mag=net_me,
                magnitude_type="Me",
                origin_id=origin.resource_id
            )
            event.magnitudes.append(mag_me)

    # Add station magnitudes
    for tr in stream:
        ml = tr.stats.metrics.get('local_magnitude')
        if ml is None:
            continue

        wid = WaveformStreamID(
            network_code=tr.stats.network,
            station_code=tr.stats.station,
            location_code=tr.stats.location,
            channel_code=tr.stats.channel
        )

        smag = StationMagnitude(
            mag=ml,
            magnitude_type="ML",
            station_magnitude_type="ML",
            waveform_id=wid
        )
        event.station_magnitudes.append(smag)

        if event.magnitudes:
            contrib = StationMagnitudeContribution(
                station_magnitude_id=smag.resource_id,
                weight=1.0
            )
            event.magnitudes[0].station_magnitude_contributions.append(contrib)

    # Add trace-level metrics as JSON comment
    try:
        metrics_dict = {tr.id: tr.stats.metrics for tr in stream if hasattr(tr.stats, 'metrics')}
        comment = Comment(
            text=json.dumps(metrics_dict, indent=2),
            force_resource_id=False
        )
        event.comments.append(comment)
    except Exception:
        pass

    # Add mainclass and subclass if given
    if mainclass or subclass:
        class_comment = {
            "mainclass": mainclass,
            "subclass": subclass
        }
        comment = Comment(
            text=json.dumps(class_comment, indent=2),
            force_resource_id=False
        )
        event.comments.append(comment)

    return event


################################################################
'''
def max_3c(st):
    """ max of a 3-component seismogram """
    N = len(st)/3
    m = []

    if N.is_integer():
        st.detrend()
        for c in range(int(N)):
            y1 = st[c*3+0].data
            y2 = st[c*3+1].data
            y3 = st[c*3+2].data
            y = np.sqrt(np.square(y1) + np.square(y2) + np.square(y3))
            m.append(max(y))
    return m 
'''
def peak_amplitudes(st):   
    """ Peak Ground Motion. Should rename it peakGroundMotion """
    
    seismic1d_list = []
    seismic3d_list = []
    infrasound_list = []
    
    #ls.clean_stream(st, taper_fraction=0.05, freqmin=0.05, causal=True)
               
    # velocity, displacement, acceleration seismograms
    stV = st.select(channel='[ESBH]H?') 
    stD = stV.copy().integrate()
    for tr in stD:
        add_to_trace_history(tr, 'integrated')    
    stA = stV.copy().differentiate()
    for tr in stA:
        add_to_trace_history(tr, 'differentiated') 
     
    # Seismic vector data  
    stZ = stV.select(channel="[ESBH]HZ")
    for tr in stZ:
        thisID = tr.id[:-1]
        st3cV = stV.select(id = '%s[ENZ12RT]' % thisID)
        if len(st3cV)==3:
            st3cD = stD.select(id = '%s[ENZ12RT]' % thisID)
            st3cA = stA.select(id = '%s[ENZ12RT]' % thisID)
            md = ls.max_3c(st3cD)
            mv = ls.max_3c(st3cV)
            ma = ls.max_3c(st3cA)
            d = {'traceID':thisID, 'PGD':md[0], 'PGV':mv[0], 'PGA':ma[0], 'calib':tr.stats.calib, 'units':tr.stats.units}
            seismic3d_list.append(d)              
    seismic3d = pd.DataFrame(seismic3d_list)
    
    # Seismic 1-c data
    peakseismo1cfile = os.path.join(eventdir, 'summary_seismic_1c.csv')
    for c in range(len(stV)):
        md = max(abs(stD[c].data))
        mv = max(abs(stV[c].data))
        ma = max(abs(stA[c].data))  
        d = {'traceID':stV[c].id, 'PGD':md[0], 'PGV':mv[0], 'PGA':ma[0], 'calib':stV[c].stats.calib, 'units':stV[c].stats.units}
        seismic1d_list.append(d)    
    seismic1d = pd.DataFrame(seismic1d_list)        
            
    # Infrasound data
    peakinfrafile = os.path.join(eventdir, 'summary_infrasound.csv')
    stP = st.select(channel="[ESBH]D?")
    stPF = stP.copy().filter('bandpass', freqmin=1.0, freqmax=20.0, corners=2, zerophase=True)    
    for c in range(len(stP)):
        mb = max(abs(stP[c].data))
        mn = max(abs(stPF[c].data)) 
        d = {'traceID':stP[c].id, 'PP':mb[0], 'PPF':mn[0], 'calib':stP[c].stats.calib, 'units':stP[c].stats.units}
        infrasound_list.append(d)  
    infrasound = pd.DataFrame(infrasound_list)    
    
    return (seismic3d, seismic1d, infrasound)

from collections import defaultdict

def max_3c(st):
    """Compute maximum 3-component vector amplitude per station."""
    grouped = defaultdict(list)
    for tr in st:
        key = tr.id[:-1]  # Strip component letter
        grouped[key].append(tr)

    max_vals = []
    for traces in grouped.values():
        if len(traces) == 3:
            traces.sort(key=lambda t: t.stats.channel)  # Optional: sort by N/E/Z
            y1, y2, y3 = traces[0].data, traces[1].data, traces[2].data
            y = np.sqrt(y1**2 + y2**2 + y3**2)
            max_vals.append(np.max(y))

    return max_vals

###########################################################
#####  Helper functions for machine learning workflow #####
def choose_best_traces(st, MAX_TRACES=8, include_seismic=True, include_infrasound=False, include_uncorrected=False):

    priority = np.array([float(tr.stats.quality_factor) for tr in st])      
    for i, tr in enumerate(st):           
        if tr.stats.channel[1]=='H':
            if include_seismic:
                if tr.stats.channel[2] == 'Z':
                    priority[i] *= 2
            else:
                priority[i] = 0
        if tr.stats.channel[1]=='D':
            if include_infrasound:
                priority[i] *= 2 
            else:
                priority[i] = 0
        if not include_uncorrected:
            if 'units' in tr.stats:
                if tr.stats.units == 'Counts':
                    priority[i] = 0
            else:
                priority[i] = 0

    n = np.count_nonzero(priority > 0.0)
    n = min([n, MAX_TRACES])
    j = np.argsort(priority)
    chosen = j[-n:]  
    return chosen        
        
def select_by_index_list(st, chosen):
    st2 = Stream()
    for i, tr in enumerate(st):
        if i in chosen:
            st2.append(tr)
    return st2 

'''
def compute_stationEnergy(val):
    # seismic: eng is sum of velocity trace (in m/s) squared, divided by samples per second
    # infrasound: eng is sum of pressure trace (in Pa) squared, divided by samples per second
    if isinstance(val, Stream):
        stationEnergy =[]
        for tr in val:
            stationEnergy.append(compute_stationEnergy(tr))
    if isinstance(val, Trace):
        tr2 = val.copy()
        tr2.detrend()
        y = tr2.data
        stationEnergy = np.sum(y ** 2)*tr2.stats.delta
    return stationEnergy
'''


def compute_dominant_frequency(disp_st, average=False):
    """
    Compute dominant frequency (Hz) in the time domain for each trace 
    using the ratio of velocity to displacement amplitude.

    Parameters
    ----------
    disp_st : obspy.Stream
        Stream of displacement traces.

    Returns
    -------
    freqs : dict
        Dictionary mapping trace.id to estimated dominant frequency in Hz.
    """

    for disp_tr in disp_st:
        # Detrend displacement
        disp_trace = disp_tr.copy().detrend("linear")

        # Differentiate to get velocity
        vel_trace = disp_trace.copy().differentiate()

        # Estimate f = |v| / (2π * |x|) using max absolute values
        disp = np.abs(disp_trace.data)+1e-20
        vel = np.abs(vel_trace.data)+1e-20
        fdom = vel / (2 * np.pi * disp)
        if average:
            fdom = np.nanmedian(fdom)
        disp_tr.stats.metrics['fdom'] = fdom

