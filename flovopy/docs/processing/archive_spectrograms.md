# Continuous numerical spectrogram archives (Pass 4)

`flovopy.processing.archive_spectrograms` uses the source-independent
`archive_processor` scheduler and the Pass 1 numerical spectrogram engine.
It does **not** require IceWeb or a plotting backend.

## Example

```python
from flovopy.processing.archive_processor import SourceSpec, Selectors
from flovopy.processing.archive_spectrograms import (
    SpectrogramArchiveConfig, run, iter_spectrograms,
)

config = SpectrogramArchiveConfig(
    source=SourceSpec('sds', {'root': '/data/KSC/SDS'}),
    output_root='/data/KSC/SPECTRA',
    start='2016-08-01', end='2016-08-03',  # exclusive end
    selectors=Selectors(network='1R', station='BCHH', channel='DH*'),
    fft_seconds=2.56, step_seconds=0.64,
    window_seconds=86400, workers=4,
)
summary = run(config)
for result in iter_spectrograms(config.output_root, summary['run_id'],
                                trace_id='1R.BCHH.10.DHZ'):
    print(result.times.shape, result.values.shape)
```

Sources may be SDS, FDSN, Earthworm, files, or a custom plugin, subject to
`archive_processor`'s source adapter support. This does not imply native
SEISAN database or Antelope Datascope support.

## Ownership, gaps, and storage

- FFT starts follow a global UTC sample grid; FFT **centres** belong to
  half-open task intervals `[start, end)`. A halo of at least `fft_seconds`
  is required on each side, ensuring FFTs crossing task boundaries are available.
- Spectrogram columns are PSD by default, using `scipy.signal.periodogram`.
  Partial or missing FFT windows are rejected as NaN; coverage is retained.
- Nonoverlapping fragments with the same trace ID are merged with masked gaps.
  Overlaps and sampling-rate changes for one ID are rejected.
- Each scheduler task writes compressed `.npz` files with metadata, frequencies,
  UTC times, spectral values, and coverage. The task's `complete.json` marker
  is written only after artifacts exist. Results are restartable.
- `iter_spectrograms` streams the completed shards chronologically, with
  optional trace/time selection. No giant combined spectral cube is built.
- NPZ loading uses `allow_pickle=False`. Shards are not Zarr/NetCDF yet.

## Limitations and cautions

- Spectral units follow the supplied waveform units. No automatic response
  removal is performed. Calibrate waveforms upstream when needed.
- A global UTC FFT grid requires waveform start times to lie on the sample
  lattice. Mixed sample rates are supported across trace IDs, not within one.
- `min_coverage < 1` does **not** enable partial-FFT estimation; partial FFTs
  remain NaN under the current conservative gap policy.
- Checkpoint fingerprints reflect configuration, not changes to source data.
  Increase `algorithm_version` or use a new output root after modifying source
  data or scientific code.
- Time selection is on FFT centres; these estimates use data spanning their
  centres. For a requested interval, some spectral columns depend on data
  outside that interval by design.
- Scheduler task numbering is based on the requested run interval; avoid
  reusing a run output directory for changed intervals without checking the
  checkpoint layout.
- Processing long continuous spectrogram archives can consume substantial
  storage and CPU. Choose FFT and step durations appropriately.

## Tests

```bash
pytest -q tests/test_archive_spectrograms.py \
  tests/test_archive_processor.py \
  tests/test_spectrogram_numerical.py \
  tests/test_spectrogram_plotting.py \
  tests/test_spectral_metrics.py
```
