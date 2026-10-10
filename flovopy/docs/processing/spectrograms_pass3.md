# FLOVOpy spectrograms — Pass 3: spectral metrics

`flovopy.processing.spectral_metrics` contains numerical measurements derived from
`SpectrogramResult`, independently of plotting and archive scheduling.

```python
from flovopy.processing.spectrograms import compute_spectrogram_result
from flovopy.processing.spectral_metrics import compute_spectral_metrics, band_power

result = compute_spectrogram_result(trace, fft_seconds=2.56, step_seconds=0.64)
metrics = compute_spectral_metrics(result, low_band=(0.5, 4), high_band=(4, 18))
```

## Definitions

- `band_power`: integrated PSD over a frequency interval; units are waveform units squared.
  Uses frequency-bin overlap weights, not an unscaled sum.
- `peak_frequency`: frequency of the largest PSD bin.
- `spectral_centroid`: PSD-weighted mean frequency.
- `spectral_bandwidth`: PSD-weighted standard deviation about the centroid.
- `spectral_ratio`: high/low integrated band power; default `log2` ratio.
- `compute_spectral_metrics`: tabular measurements with UTC timestamps and coverage.

Metrics accept PSD and ASD (ASD is squared internally), but reject legacy magnitude.
Columns containing NaNs return NaN metrics. Frequency grids must be uniformly spaced.
A zero-power spectrum has undefined centroid, bandwidth and peak frequency.

These quantities are **spectral** measurements; `band_power` is not the
legacy `compute_metrics_TFS()` `sam` statistic and is not automatically RSAM.
The old function remains unchanged in `spectrograms.py` for compatibility.

Run: `pytest -q tests/test_spectral_metrics.py`.
