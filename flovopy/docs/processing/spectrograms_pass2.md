# FLOVOpy spectrograms — Pass 2: plotting

Install `flovopy/plotting/spectrograms.py` and replace `flovopy/processing/spectrograms.py` with the included version (Pass 1 plus UTC timestamp correction and optional `icewebSpectrogram.plot_modern`). No changes to `archive_processor.py` are required.

```python
from flovopy.plotting.spectrograms import plot_waveform_spectrograms
fig = plot_waveform_spectrograms(stream, fft_seconds=2.56, step_seconds=0.64,
                                frequency_limits=(0.5, 20), db=True,
                                color_scale='robust', title='Waveform + PSD')
fig.savefig('spectrogram.png', dpi=200)
```

Precomputed results are accepted as a dictionary keyed by `Trace.id`:

```python
from flovopy.processing.spectrograms import compute_stream_spectrograms
spectra = compute_stream_spectrograms(stream, fft_seconds=2.56, step_seconds=0.64)
fig = plot_waveform_spectrograms(stream, spectra, color_scale='shared')
```

Legacy `icewebSpectrogram.plot()` is unchanged. To opt in to the new plotting engine, call `icewebSpectrogram(stream).plot_modern(...)`.

## Options and conventions

- Waveform/spectrogram pairs share absolute UTC time axes across channels.
- PSD is the default numerical quantity. PSD decibels use `10*log10(PSD/reference)`; ASD and magnitude use `20*log10(value/reference)`.
- `reference=1.0` is a **numerical reference**, not a physical reference standard; explicitly set it when comparing calibrated quantities.
- `color_scale='individual'` scales each channel independently; `'shared'` uses one global range; `'robust'` uses global percentiles (default 5–99%). `clim=(min,max)` overrides these, in displayed units.
- `frequency_scale='log'` excludes the zero-frequency bin.
- Input Streams and spectral arrays are not modified.
- The plotting layer is Matplotlib-only; it does not implement interactive Pensive browsing.

## Limitations

- Duplicate Trace IDs are rejected: merge or select waveform segments first.
- The numerical engine rejects incomplete FFT windows, retaining NaN spectral columns.
- Legacy `plot()` and `dailySpectrogram` remain unchanged and retain their historical limitations.
- `plot_waveform_spectrograms` requires one precomputed result per plotted trace.
- UTC dates are rendered with Matplotlib date coordinates; numerical spectral timestamps remain Unix seconds.

## Tests

```bash
pytest -q tests/test_spectrogram_numerical.py tests/test_spectrogram_plotting.py
```
