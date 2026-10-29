"""Versioned, non-mutating event features; see docs/event_features.md.

Independent implementation inspired by domain-weighted shape descriptors.
No Malfante/AAA implementation is imported or copied. Existing amplitude and
amplitude-spectrum metrics remain owned by EnhancedStream.ampengfft.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, correlate, correlation_lags, hilbert, periodogram, sosfiltfilt

VERSION = '1.0.0'


@dataclass(frozen=True)
class FeatureConfig:
    lf_band: tuple = (0.8, 4.0)
    hf_band: tuple = (4.0, 20.0)
    filter_order: int = 4
    smoothing_seconds: float = 0.25
    threshold_factor: float = 3.0
    minimum_run_seconds: float = 0.5
    edge_cycles: float = 2.0
    max_lag_seconds: float = 10.0
    cepstrum_min_seconds: float = 0.05
    cepstrum_max_seconds: float = 2.0
    log_floor_relative: float = 1e-12

    def __post_init__(self):
        for band in (self.lf_band, self.hf_band):
            if len(band) != 2 or not np.all(np.isfinite(band)) or not 0 < band[0] < band[1]:
                raise ValueError('Bands require finite 0 < low < high')
        if self.lf_band[1] > self.hf_band[0]:
            raise ValueError('LF and HF bands must not overlap')
        for key in ('smoothing_seconds', 'threshold_factor', 'minimum_run_seconds',
                    'edge_cycles', 'max_lag_seconds', 'cepstrum_min_seconds', 'cepstrum_max_seconds'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + ' must be finite and positive')
        if isinstance(self.filter_order, bool) or int(self.filter_order) != self.filter_order or not 1 <= self.filter_order <= 12:
            raise ValueError('filter_order must be an integer in [1, 12]')
        if self.cepstrum_min_seconds >= self.cepstrum_max_seconds:
            raise ValueError('Invalid cepstrum range')
        if not 0 < self.log_floor_relative < 1:
            raise ValueError('log_floor_relative must lie in (0, 1)')


SHAPE_KEYS = ('centroid', 'spread', 'skewness', 'excess_kurtosis', 'entropy', 'concentration')
TIMING_KEYS = ('peak_time_s', 'energy_t05_s', 'energy_t10_s', 'energy_t50_s',
               'energy_t90_s', 'energy_t95_s', 'duration_90_s', 'duration_80_s',
               'energy_centroid_s', 'energy_spread_s', 'threshold_onset_s',
               'threshold_offset_s', 'threshold_span_s', 'threshold_active_s',
               'envelope_snr', 'attack_per_s', 'decay_per_s')
CROSS_KEYS = ('peak_delay_s', 'centroid_delay_s', 'onset_delay_s', 't50_delay_s',
              'xcorr_lag_s', 'xcorr_coefficient', 'energy_ratio', 'log2_energy_ratio')


def _schema():
    keys = ['time.crest_factor', 'time.zero_crossings_per_s',
            'spectral.power_peak_hz', 'spectral.power_median_hz',
            'spectral.power_rolloff95_hz', 'spectral.flatness',
            'cepstral.peak_quefrency_s', 'cepstral.peak_coefficient']
    for domain in ('time_energy', 'spectral_power', 'cepstral_squared'):
        keys.extend(domain + '.' + key for key in SHAPE_KEYS
                    if domain != 'time_energy' or key not in ('centroid', 'spread'))
    for band in ('full', 'lf', 'hf'):
        keys.extend(band + '.' + key for key in TIMING_KEYS)
    keys.extend('hf_lf.' + key for key in CROSS_KEYS)
    return {key: float('nan') for key in keys}


def _array(data):
    x = np.ma.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError('Expected a one-dimensional waveform')
    return np.asarray(x.filled(np.nan))


def _weights(axis, power):
    """Population central moments of a nonnegative discrete measure."""
    p = power / np.sum(power)
    mean = np.sum(p * axis)
    d = axis - mean
    variance = np.sum(p * d**2)
    spread = np.sqrt(max(0., variance))
    positive = p > 0
    return dict(centroid=float(mean), spread=float(spread),
                skewness=float(np.sum(p*d**3)/spread**3) if spread > 0 else np.nan,
                excess_kurtosis=float(np.sum(p*d**4)/spread**4-3) if spread > 0 else np.nan,
                entropy=float(-np.sum(p[positive]*np.log(p[positive]))/np.log(len(p))) if len(p) > 1 else np.nan,
                concentration=float(np.sum(p*p)))


def _envelope(x, fs, cfg):
    # Odd smoothing length gives a centered kernel without half-sample delay.
    width = max(1, int(round(cfg.smoothing_seconds*fs)))
    width += (width % 2 == 0)
    return uniform_filter1d(np.abs(hilbert(x)), width, mode='reflect')


def _runs(mask, minimum):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, stops) if b-a >= minimum]


def _timing(x, noise, fs, cfg, prefix, values, flags, edge):
    scale = np.max(np.abs(x))
    if scale == 0:
        flags.append(prefix + ':zero_energy')
        return None
    power = (x/scale)**2
    t = np.arange(len(x))/fs
    cumulative = np.cumsum(power)/np.sum(power)
    put = lambda key, value: values.__setitem__(prefix + '.' + key, float(value))
    quantiles = {}
    for percent in (5, 10, 50, 90, 95):
        quantiles[percent] = t[min(np.searchsorted(cumulative, percent/100), len(t)-1)]
        put('energy_t%02d_s' % percent, quantiles[percent])
    put('duration_90_s', quantiles[95]-quantiles[5])
    put('duration_80_s', quantiles[90]-quantiles[10])
    shape = _weights(t, power)
    put('energy_centroid_s', shape['centroid'])
    put('energy_spread_s', shape['spread'])
    env = _envelope(x, fs, cfg)
    peak = int(np.argmax(env))
    put('peak_time_s', t[peak])
    slopes = np.diff(env)*fs/np.max(env)
    put('attack_per_s', np.max(slopes))
    put('decay_per_s', np.min(slopes))
    if peak < edge or peak >= len(x)-edge:
        flags.append(prefix + ':peak_near_edge')
    if noise is not None:
        noise_env = _envelope(noise, fs, cfg)
        baseline = np.sqrt(np.mean(noise_env**2))
        if baseline > 0:
            snr = np.sqrt(np.mean(env**2))/baseline
            put('envelope_snr', snr)
            if snr < cfg.threshold_factor:
                flags.append(prefix + ':low_snr')
            runs = _runs(env >= cfg.threshold_factor*baseline,
                         max(1, int(np.ceil(cfg.minimum_run_seconds*fs))))
            if runs:
                first, last = runs[0][0], runs[-1][1]
                put('threshold_onset_s', first/fs)
                put('threshold_offset_s', last/fs)
                put('threshold_span_s', (last-first)/fs)
                put('threshold_active_s', sum(b-a for a, b in runs)/fs)
                if first < edge or last > len(x)-edge:
                    flags.append(prefix + ':threshold_near_edge')
                if first == 0 or last == len(x):
                    flags.append(prefix + ':duration_censored')
            else:
                flags.append(prefix + ':no_threshold_detection')
        else:
            flags.append(prefix + ':zero_noise')
    return env


def extract_features(data, sampling_rate, *, noise=None, config=None, units='unknown'):
    """Return stable scalar values, QC flags, and provenance without mutation.

    Arrays must be contiguous, uniformly sampled, finite, real 1-D data. Noise
    must have the same sampling rate, units and processing as the event. Bad
    samples cause rejection, never silent interpolation or time compression.
    Invalid configuration raises ValueError; data quality is reported in flags.
    """
    cfg = config or FeatureConfig()
    fs = float(sampling_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('sampling_rate must be finite and positive')
    if np.iscomplexobj(data) or (noise is not None and np.iscomplexobj(noise)):
        raise ValueError('Real waveforms required')
    x = _array(data)
    values, flags = _schema(), []
    result = dict(version=VERSION, values=values, qc_flags=flags,
                  provenance=dict(config=asdict(cfg), sampling_rate_hz=fs,
                                  npts=len(x), window_support_s=len(x)/fs,
                                  units=units, noise_npts=0 if noise is None else len(noise),
                                  demean=True, spectrum='hann_periodogram',
                                  duration_energy='squared_demeaned_waveform'))
    if len(x) < 16 or not np.all(np.isfinite(x)):
        flags.append('invalid_waveform' if not np.all(np.isfinite(x)) else 'too_short')
        return result
    # Normalize before centering to avoid overflow; amplitude-independent output.
    scale = np.max(np.abs(x))
    x = x/scale if scale else x.copy()
    x = x - np.mean(x)
    if not np.any(x):
        flags.append('zero_variance')
        return result
    n = None
    if noise is not None:
        n = _array(noise)
        if len(n) < 16 or not np.all(np.isfinite(n)):
            flags.append('invalid_noise')
            n = None
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                n = n/scale
                n = n-n.mean()
            if not np.all(np.isfinite(n)):
                flags.append('invalid_noise_scale')
                n = None
    else:
        flags.append('noise_unavailable')
    t = np.arange(len(x))/fs
    for key, value in _weights(t, x*x).items():
        if key not in ('centroid', 'spread'):
            values['time_energy.'+key] = value
    values['time.crest_factor'] = float(np.max(np.abs(x))/np.sqrt(np.mean(x*x)))
    nz = x[x != 0]
    values['time.zero_crossings_per_s'] = float(np.count_nonzero(np.diff(np.signbit(nz)))/((len(x)-1)/fs))
    f, psd = periodogram(x, fs, window='hann', detrend=False, scaling='density')
    if np.sum(psd) > 0:
        for key, value in _weights(f, psd).items():
            values['spectral_power.'+key] = value
        cp = np.cumsum(psd)/np.sum(psd)
        values['spectral.power_peak_hz'] = float(f[np.argmax(psd)])
        for key, fraction in (('power_median_hz', .5), ('power_rolloff95_hz', .95)):
            values['spectral.'+key] = float(f[min(np.searchsorted(cp, fraction), len(f)-1)])
        values['spectral.flatness'] = float(np.exp(np.mean(np.log(psd)))/np.mean(psd)) if np.all(psd > 0) else 0.
    else:
        flags.append('spectral:zero_windowed_power')
    # Real cepstrum, natural log amplitude, no one-sided doubling. Remove c[0]
    # from summaries and restrict to the unique positive quefrency half.
    amplitude = np.abs(np.fft.rfft(x*np.hanning(len(x))))
    floor = cfg.log_floor_relative*np.max(amplitude)
    if floor > 0:
        c = np.fft.irfft(np.log(np.maximum(amplitude, floor)), n=len(x))
        q = np.arange(len(x))/fs
        select = (q >= cfg.cepstrum_min_seconds) & (q <= cfg.cepstrum_max_seconds) & (np.arange(len(x)) <= len(x)//2)
        if np.count_nonzero(select) >= 2 and np.sum(c[select]**2) > 0:
            qs, cs = q[select], c[select]
            for key, value in _weights(qs, cs*cs).items():
                values['cepstral_squared.'+key] = value
            k = np.argmax(np.abs(cs))
            values['cepstral.peak_quefrency_s'] = float(qs[k])
            values['cepstral.peak_coefficient'] = float(cs[k])
        else:
            flags.append('cepstral:insufficient_quefrency_bins')
    else:
        flags.append('cepstral:zero_windowed_amplitude')
    if cfg.cepstrum_max_seconds > len(x)/(2*fs):
        flags.append('cepstral:range_truncated')
    edge = max(1, int(np.ceil(cfg.smoothing_seconds*fs)))
    _timing(x, n, fs, cfg, 'full', values, flags, edge)
    filtered, envelopes = {}, {}
    for name, band in (('lf', cfg.lf_band), ('hf', cfg.hf_band)):
        if band[1] >= fs/2:
            flags.append(name + ':invalid_nyquist')
            continue
        edge = max(int(np.ceil(cfg.smoothing_seconds*fs)),
                   int(np.ceil(cfg.edge_cycles*fs/band[0])))
        if len(x) <= 2*edge:
            flags.append(name + ':too_short_for_band')
            continue
        sos = butter(int(cfg.filter_order), band, btype='bandpass', fs=fs, output='sos')
        try:
            y = sosfiltfilt(sos, x)
        except ValueError:
            flags.append(name + ':too_short_for_filter')
            continue
        nn = None
        if n is not None:
            if len(n) <= 2*edge:
                flags.append(name + ':noise_too_short_for_band')
            else:
                try:
                    nn = sosfiltfilt(sos, n)
                except ValueError:
                    flags.append(name + ':noise_too_short_for_filter')
        env = _timing(y, nn, fs, cfg, name, values, flags, edge)
        if env is not None:
            filtered[name], envelopes[name] = y, env
    if len(envelopes) == 2:
        for key, suffix in (('peak', 'peak_time_s'), ('centroid', 'energy_centroid_s'),
                            ('onset', 'threshold_onset_s'), ('t50', 'energy_t50_s')):
            values['hf_lf.'+key+'_delay_s'] = values['lf.'+suffix]-values['hf.'+suffix]
        ratio = np.sum(filtered['hf']**2)/np.sum(filtered['lf']**2)
        values['hf_lf.energy_ratio'] = float(ratio)
        values['hf_lf.log2_energy_ratio'] = float(np.log2(ratio))
        h, l = (envelopes[k]-envelopes[k].mean() for k in ('hf', 'lf'))
        denominator = np.linalg.norm(h)*np.linalg.norm(l)
        if denominator > 0:
            corr = correlate(l, h, mode='full', method='fft')/denominator
            lags = correlation_lags(len(l), len(h))/fs
            allowed = np.abs(lags) <= cfg.max_lag_seconds
            k = np.flatnonzero(allowed)[np.argmax(corr[allowed])]
            values['hf_lf.xcorr_lag_s'] = float(lags[k])
            values['hf_lf.xcorr_coefficient'] = float(corr[k])
            if abs(lags[k]) >= min(cfg.max_lag_seconds, (len(x)-1)/fs):
                flags.append('hf_lf:lag_at_search_boundary')
        else:
            flags.append('hf_lf:constant_envelope')
    # Never expose infinity as a usable ML value.
    for key, value in values.items():
        if np.isinf(value):
            values[key] = np.nan
            flags.append(key + ':numeric_overflow')
    return result


def extract_trace_features(trace, noise_trace=None, *, config=None, attach=False):
    """ObsPy adapter; optional attachment under stats.metrics.features only."""
    if noise_trace is not None:
        if trace.stats.sampling_rate != noise_trace.stats.sampling_rate:
            raise ValueError('Event and noise sampling rates must match')
        if trace.stats.get('units', 'unknown') != noise_trace.stats.get('units', 'unknown'):
            raise ValueError('Event and noise units must match')
    result = extract_features(trace.data, trace.stats.sampling_rate,
                              noise=noise_trace.data if noise_trace is not None else None,
                              config=config, units=trace.stats.get('units', 'unknown'))
    result['provenance']['starttime'] = str(trace.stats.starttime)
    result['provenance']['trace_id'] = trace.id
    result['provenance']['processing'] = list(trace.stats.get('processing', []))
    if noise_trace is not None:
        result['provenance']['noise_starttime'] = str(noise_trace.stats.starttime)
        result['provenance']['noise_processing'] = list(noise_trace.stats.get('processing', []))
    if attach:
        if trace.stats.get('metrics') is None:
            trace.stats.metrics = {}
        trace.stats.metrics['features'] = result
    return result
