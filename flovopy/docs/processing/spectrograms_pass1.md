# Spectrogram numerical API — Pass 1

This is an **additive** change to `flovopy.processing.spectrograms`. All original functions and classes, including `compute_spectrogram`, `icewebSpectrogram`, and `dailySpectrogram`, remain in place and retain their existing behavior. The new API does not yet replace the legacy implementation.

## Installation

Copy `flovopy/processing/spectrograms.py` into the repository, and add `tests/test_spectrogram_numerical.py` to its tests. Run:

```bash
pytest -q tests/test_spectrogram_numerical.py
```

## Usage

```python
from flovopy.processing.spectrograms import (
    compute_spectrogram_result, compute_stream_spectrograms,
)

result = compute_spectrogram_result(
    trace, fft_seconds=2.56, step_seconds=0.64,
    quantity='psd', align='utc', units='m/s',
)
# result.values.shape == (len(result.frequencies), len(result.times))
# result.times: absolute UTC Unix timestamps at FFT centres
# result.coverage: fraction of finite input samples per FFT window

results_by_id = compute_stream_spectrograms(stream, fft_seconds=2, step_seconds=0.5)
```

### Spectral quantities

- `psd` (default): one-sided power spectral density from `scipy.signal.periodogram`, units `(input_units)²/Hz`.
- `asd`: square root of PSD, units `(input_units)/√Hz`.
- `legacy_magnitude`: square root of `periodogram(..., scaling='spectrum')`. **Not** a byte-for-byte or numerically guaranteed match to the old `mlab.specgram` implementation, especially where zero padding was used.

The engine does not guess physical calibration from channel names. Supply `units` or attach `trace.stats.units` if known.

### Timing, gaps, and reproducibility

FFT starts are aligned to a global UTC sample grid (`align='utc'`), with the first possible FFT start at an integer multiple of the step from the epoch. `align='start'` instead anchors FFT starts to each trace's start. FFT timestamps mark the **centres** of the sample support, not starts.

FFT duration and step must both be representable as whole numbers of samples. For `align='utc'`, trace start must lie on the UTC sample lattice within 1 microsecond. Results from adjacent archive tasks can be merged by selecting FFT centres in their assigned output intervals. Read a halo at least as long as the FFT duration on each side of the task interval, then select by centre. Archive integration is not yet implemented.

Any FFT window containing NaNs, masked samples, or infinities yields a NaN spectral column. `coverage` is still recorded. **The current gap policy always rejects partial FFT windows, even if `min_coverage < 1`**; a partial-window spectral estimator is a separate future feature. No interpolation or zero fill occurs.

The input trace is never modified. Each FFT window is detrended independently, preventing whole-chunk demeaning effects.

### Scope and limitations

- Original IceWeb plotting, spectral metrics, and `dailySpectrogram` are untouched.
- This implementation calculates individual FFT windows in Python. Benchmarking and vectorization are needed before very large archives.
- The function does not merge overlapping or fragmented traces. Input streams with duplicate IDs are rejected.
- Results are in memory only; no Zarr/NetCDF/NPZ persistence or scheduler integration yet.
- The new `legacy_magnitude` option is *not* an exact historical IceWeb reproduction mode.
- Tests are provided but require execution in a local environment with ObsPy installed.
