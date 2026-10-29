# Event feature extraction, schema 1.0.0

`flovopy.core.features.extract_features(data, sampling_rate, noise=None,
config=FeatureConfig(), units='unknown')` is a pure NumPy/SciPy API. It returns
`version`, a fixed-key `values` dictionary, `qc_flags`, and `provenance`.
Undefined values are NaN, never substituted with zero. Invalid configuration
raises ValueError. Invalid waveform data returns the complete schema with NaNs.
The function does not calibrate, detect events, assign earthquake classes, or
estimate physical radiated energy. Input must already be an event window.

`extract_trace_features(trace, noise_trace=None, config=None, attach=False)` adds
ObsPy metadata. `EnhancedTrace.extract_features(..., attach=True)` and
`EnhancedStream.extract_features(noise_by_id=None, ..., attach=True)` optionally
store results under `stats.metrics.features`; existing metrics are preserved.
The stream method returns an ordered list, including separate records for
repeated channel IDs. Noise selection is explicit by exact trace ID; each noise
trace must match the event sampling rate and units. Callers must also match
instrument response, filtering and gain. Absolute times are starttime plus the
reported relative seconds. Waveform arrays are never modified.

```python
from flovopy.core.features import FeatureConfig, extract_trace_features
result = extract_trace_features(event_trace, noise_trace,
    config=FeatureConfig(lf_band=(0.8, 4), hf_band=(4, 20)))
print(result['values']['hf_lf.peak_delay_s'], result['qc_flags'])
```

## Existing metric ownership and compatibility

EnhancedStream `_td_stats` already supplies sample min/max/mean/median/RMS/std,
abs statistics, skewness, excess kurtosis and sum(y²)/fs. These are not duplicated
in this module. SAM/SEM and amplitude-band ratios remain unchanged.
`spectral_block` supplies amplitude-weighted `meanf`, `peakf`, bandwidth, and its
historical `medianf` (median of frequencies with positive amplitude, not a power
median). Those names and calculations are retained. New `spectral_power` fields
use a defined PSD measure, so they are distinct estimators, not replacements for
legacy `meanf` or `medianf`. Full-band time centroid/spread live only under `full`.

The local FloVoPy revision lacked v6's required `allow_uncalibrated` argument.
An optional argument now enables descriptive counts metrics in `ampengfft`.
Its default is False, preserving rejection of counts/unspecified units. This
branch does not create PGV/PGA/PGD/PAP or claim a calibration.

Pinatubo `compute_reusable_features` defaults to False for existing configs.
When True, event rows gain `features.*` columns and a features QC row. Original
v6 temporal calculations remain unchanged. `config.features.example.json`
enables the new module and disables legacy temporal calculations to avoid
parallel representations of the same concepts. Existing v6 configs still work.
New parameters are under `processing.reusable_features`; unknown parameters
raise an error. The original `config.example.json` is unchanged.

## Definitions

Let x be the input after constant mean removal, t_i=i/fs, and N sample count.
Internal rescaling prevents overflow and does not affect dimensionless features
or timing. No linear trend is removed here. Provenance records the configuration,
N, fs, support N/fs, upstream trace processing and units. Pinatubo also records
its processing configuration, including lowpass filtering before windowing.

For a discrete nonnegative measure w on axis u, p=w/sum(w):

| Suffix | Definition | Units |
|---|---|---|
| centroid | sum(p*u) | axis units |
| spread | sqrt(sum(p*(u-centroid)²)) | axis units |
| skewness | sum(p*(u-centroid)³)/spread³ | dimensionless |
| excess_kurtosis | sum(p*(u-centroid)⁴)/spread⁴ - 3 | dimensionless |
| entropy | -sum(p*ln(p))/ln(number of bins), excluding zero terms | dimensionless |
| concentration | sum(p²) | dimensionless |

Zero-spread standardized moments are undefined (NaN). Moments are population
moments, without finite-sample bias correction. No square roots of skewness or
kurtosis are taken. Entropy and concentration depend on record length/resolution;
compare records with consistent windows and sampling. Domain weights are:

* `time_energy`: x² on t in seconds; centroid/spread appear under `full` instead.
* `spectral_power`: one-sided Hann periodogram density on frequency in Hz;
  SciPy one-sided endpoint normalization is used, with no extra detrending or
  zero padding. Equal frequency bins mean normalized PSD weights are power
  fractions. Includes DC and Nyquist where available.
* `cepstral_squared`: squared real cepstrum on selected positive quefrencies in
  seconds. The real cepstrum is irfft(ln(max(abs(rfft(x*Hann)), floor)), n=N).
  The symmetric Hann window is used here; floor is max amplitude times
  `log_floor_relative` (default 1e-12). Zero quefrency is excluded. Range defaults
  to 0.05–2 s, limited to N/(2fs). This is log-amplitude real cepstrum, not MFCC,
  power cepstrum, or a spectrum of quefrency. These weights are descriptors,
  not physical energy. Quefrency structure may reflect echoes or periodicity.

| Field | Definition |
|---|---|
| time.crest_factor | max(abs(x))/sqrt(mean(x²)) |
| time.zero_crossings_per_s | sign changes after ignoring exact zeros, divided by (N-1)/fs |
| spectral.power_peak_hz | frequency of largest PSD bin; earliest bin wins ties |
| spectral.power_median_hz | first frequency bin whose cumulative PSD fraction reaches 0.50 |
| spectral.power_rolloff95_hz | same at 0.95 |
| spectral.flatness | geometric/arithmetic mean PSD, zero if any PSD bin is zero |
| cepstral.peak_quefrency_s | quefrency of largest absolute selected coefficient |
| cepstral.peak_coefficient | signed coefficient at that quefrency |

### Full, LF and HF timing

`full` uses demeaned x. LF/HF use SciPy Butterworth SOS bandpass and `sosfiltfilt`
(default order 4; forward/backward filtering squares the magnitude response).
Bands must satisfy 0 < low < high < fs/2 and LF must not overlap HF. Filtering
uses reflection padding. Peaks near an edge are reported with a warning, rather
than silently shifted or removed. For best timing use sufficiently long windows;
zero-phase filtering is noncausal and can smear an onset backwards.

Envelope e is abs(Hilbert(y)), smoothed using a centered odd-length boxcar with
reflection edges. Requested smoothing defaults to 0.25 s, rounded to samples
and increased by one if even. Noise uses the identical filtering and envelope
procedure. No smoothing is applied to the waveform energy measure y².

| Suffix (under full, lf, hf) | Definition |
|---|---|
| peak_time_s | first maximum of smoothed envelope, relative to window start |
| energy_t05/10/50/90/95_s | first sample whose cumulative y² reaches the named percentage |
| duration_90_s | energy_t95_s - energy_t05_s |
| duration_80_s | energy_t90_s - energy_t10_s |
| energy_centroid_s, energy_spread_s | y²-weighted time centroid and standard deviation |
| envelope_snr | RMS(smoothed event envelope)/RMS(smoothed noise envelope) |
| threshold_onset_s | start of first sustained run with envelope >= threshold_factor * noise-envelope RMS |
| threshold_offset_s | exclusive end of last sustained run |
| threshold_span_s | last exclusive end minus first start, including internal quiet intervals |
| threshold_active_s | summed lengths of qualifying runs only |
| attack_per_s | maximum adjacent envelope difference * fs / max(envelope) |
| decay_per_s | minimum adjacent envelope difference * fs / max(envelope); usually negative |

Default threshold_factor=3, minimum_run_seconds=0.5. Runs require ceil(duration*fs)
samples. Offset is a sample-support boundary, so it can equal N/fs. These are
window-dependent duration estimates, not an automatic physical event-end pick.
Energy durations include background energy; no noise subtraction is performed.
Persistent tremor or noise can therefore produce long durations. A cumulative
energy duration alone cannot establish whether the event is fully recorded.

### HF/LF comparisons

All delays mean LF minus HF; positive indicates LF later. `peak_delay_s`,
`centroid_delay_s`, `onset_delay_s`, `t50_delay_s` subtract the corresponding
HF measure from LF. `energy_ratio` is sum(HF²)/sum(LF²);
`log2_energy_ratio` is its base-2 logarithm. These are filtered-waveform ratios,
not legacy amplitude-spectrum ratios.

`xcorr_lag_s` maximizes full cross-correlation of mean-removed LF and HF
smoothed envelopes, restricted to +/-max_lag_seconds (default 10).
`xcorr_coefficient` divides that peak by the product of full-record L2 norms;
normalization is global, not overlap-normalized. First maximum wins ties.
Positive lag again means LF later. Search-boundary solutions are flagged.
Neither a positive delay nor a high ratio by itself identifies a hybrid event;
propagation, surface waves, windowing and response can all affect these values.

## QC and reproducibility

* `invalid_waveform`, `too_short`, `zero_variance`: all values NaN. Masked samples,
  NaN and infinity are rejected; gaps are not filled or compressed. Minimum N=16.
* `noise_unavailable`, `invalid_noise`, `invalid_noise_scale`: threshold metrics
  unavailable; energy timing remains useful. Noise must also have at least 16
  finite samples. `zero_noise` suppresses threshold timing and SNR for that band.
* `invalid_nyquist`, `too_short_for_band`, `too_short_for_filter`: band metrics
  unavailable. Band windows require more than twice the edge margin, with margin
  max(smoothing_seconds, edge_cycles/low), default edge_cycles=2. This is a
  conservative heuristic, not a rigorous filter settling-time estimate.
* `noise_too_short_for_band`: band noise/threshold metrics unavailable.
* `peak_near_edge`, `threshold_near_edge`: extrema/detections within the above
  margin (full-band margin uses smoothing_seconds). Treat timing cautiously.
* `duration_censored`: threshold activity touches the actual window boundary.
* `low_snr`: envelope RMS ratio below threshold_factor; values retained, flagged.
* `no_threshold_detection`: no qualifying run; threshold values NaN, not zero.
* `cepstral:range_truncated` or `insufficient_quefrency_bins`: requested range is
  not fully represented or has fewer than two bins/nonzero weight.
* `hf_lf:constant_envelope`, `lag_at_search_boundary`: correlation reliability.
* `numeric_overflow`: affected output converted to NaN.

Each prefix identifies the affected band/field. QC flags must accompany exported
ML features. NaNs should be handled explicitly in downstream training. Clipping
and instrument artifacts are owned by the existing acquisition audit: a finite
array alone cannot establish whether it is clipped. No new clipping classifier
is claimed. Use station/instrument-aware splits to avoid ML leakage.

## Conceptual source and deliberate departures

Reviewed Malfante's AAA `featuresFunctions.py`:
https://github.com/gthompson/SoufriereHillsVolcano/blob/main/catalog/machinelearning/AAA-master/automatic_processing/featuresFunctions.py

The useful ideas are domain-weighted moments, concentration, envelope evolution
and cepstral structure. The implementation here is independent. It does not seek
an arbitrary 34-times-three count or apply identical formulas blindly across
domains. In particular, it uses conventional standardized moments, explicit
axes/units, stable missing values and consistent energy definitions. MFCCs and
arbitrary histogram features are not included without a scientific use case.

## Verification

Run `python -m pytest tests/test_features.py` from the FloVoPy checkout, with the
updated `pinatubo_features_v6` directory beside it for the pipeline integration
test. Tests cover synthetic known-delay HF/LF bursts at multiple rates, a known
0.5-s echo, sinusoidal power metrics, tiny-amplitude/DC invariance, analytic
impulse energy duration/moments, invalid inputs/configuration, noise thresholds,
window censoring, nonmutation, legacy metric preservation and pipeline CSV/JSON.
These tests establish estimator behavior; real-event scientific validation is
still required before using these features as a classification rule.

Additional edge cases: zero windowed spectral power or cepstral amplitude is
flagged with `spectral:zero_windowed_power` or
`cepstral:zero_windowed_amplitude`; corresponding fields remain NaN. Noise shorter
than SOS reflection padding is flagged `noise_too_short_for_filter`.

A real June 3, 1991 Pinatubo waveform smoke test produced 18 window rows from
9 traces with no processing errors. Its five-second noise window fails the
default LF (>5 s) length criterion at 0.8 Hz; LF threshold durations and onset
comparisons are therefore explicitly unavailable. A longer independently
verified pre-event noise interval is needed for those metrics. Do not extend
the initial window into the event just to remove this flag. LF energy durations
and peak delays remain available, with applicable edge/SNR flags. This smoke
test is operational validation, not a labeled classification benchmark.
