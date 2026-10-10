# FLOVOpy archive processor

`flovopy.processing.archive_processor` is a **source-independent scheduling layer** for ObsPy waveform streams. It does not replace `archive_metrics.py`; the latter remains the validated SDS-specific RSAM/VSAM/DSAM/VSEM CLI and yearly-output writer.

## Architecture

- **Waveform source:** fetches ObsPy `Stream` objects for a requested UTC interval.
- **Window processor:** transforms each fetched stream into one or more files (SAM metrics now; spectrograms later).
- **Scheduler:** UTC-epoch-aligned consecutive windows, optional halo reads, independent output directories, atomic completion markers, resumability, multiprocessing, error propagation.

Processing intervals are half-open `[start, end)`. The source may read a wider halo interval. **Processors must retain only products owned by `[start, end)`**. This matters for spectral products with overlapping FFT windows.

## Supported sources

| `SourceSpec.kind` | Backend | Notes |
| --- | --- | --- |
| `sds` | `EnhancedSDSClient` | `root`; raw reads, no implicit postprocessing. |
| `fdsn` | `obspy.clients.fdsn.Client` | `base_url`, optional `client_kwargs` and `waveform_kwargs`. Network required. |
| `earthworm` | `obspy.clients.earthworm.Client` | `host`, `port`. Earthworm/Winston protocol support depends on server compatibility. |
| `files` | `obspy.read` | `paths` are file paths or glob patterns; handles readable SEISAN waveform files and supported CSS descriptor files. **Not a SEISAN S-file database index or a native Antelope/Datascope database interface.** |
| `plugin` | user-supplied factory | Recommended for indexed SEISAN event catalogs, Antelope/CSS database queries, and custom Winston integrations. |

`files` currently scans all matching files on every window; it is for small datasets and testing, **not** an efficient large SEISAN or CSS database adapter. Plugins should use native indexes and fetch only requested time ranges.

## Example: SDS to per-window RSAM

```python
from flovopy.processing.archive_processor import (
    ArchiveConfig, SourceSpec, Selectors, run,
)

cfg = ArchiveConfig(
    source=SourceSpec('sds', {'root': '/data/KSC/SDS'}),
    output_root='/data/KSC/PRODUCTS',
    start='2016-08-01', end='2016-08-03',
    selectors=Selectors(network='1R', station='BCHH', channel='DH*'),
    processor='flovopy.processing.archive_processor:SAMWindowProcessor',
    processor_kwargs={
        'products': ['RSAM'],
        'sampling_interval': 60,
        'bands': {'VLP': [0.02, 0.2], 'LP': [0.5, 4], 'VT': [4, 18]},
    },
    window_seconds=86400,
    halo_seconds=300,
    workers=4,
)
print(run(cfg))
```

For `VSAM`, `DSAM`, or `VSEM`, include `stationxml` in `processor_kwargs`. Results are stored as per-window pickle shards, **not** consolidated yearly SAM files. Use `archive_metrics.py` for its established yearly consolidation workflow until a shared consolidation layer is implemented.

## FDSN example

```python
source = SourceSpec('fdsn', {'base_url': 'IRIS'})
```

Specify suitable NSLC selectors; unrestricted wildcard requests may be enormous or rejected by the service.

## Extending to spectrograms

Create an importable processor factory:

```python
class SpectrogramWindowProcessor:
    def __init__(self, ...): ...
    def process(self, stream, context, output_dir):
        # Compute spectra using context.read_start/read_end halo.
        # Keep only spectral bins owned by [context.start, context.end).
        # Write output atomically; return relative filenames.
        return {'artifacts': ['spectrogram.zarr.zip']}
```

The factory path is `my_package.my_module:SpectrogramWindowProcessor`. Each task runs in a separate directory. Artifacts must be **files**, not directories, for current resume checks; choose e.g. NetCDF/HDF5 or a zipped Zarr store.

## Scientific and operational caveats

1. Source waveform `trim()` is inclusive at the sample level. The processor must enforce half-open ownership of derived products.
2. The scheduler **does not merge** waveform fragments, interpolate gaps, or remove responses. The processor makes those choices.
3. `SAMWindowProcessor` reuses `archive_metrics` merging/calibration logic and the revised `sam.py`. Its numeric output needs testing against `archive_metrics.py` on real KSC data.
4. A configuration hash identifies task runs; changing input data at the same source path **does not** invalidate existing checkpoints. Increment `algorithm_version` or use a new output root after data or code changes. The hash also does not include StationXML file content.
5. Processing tasks can be parallel, but there is no shared writer or automatic consolidation. Never run two independent jobs concurrently with identical run IDs/output roots.
6. Failed tasks raise exceptions; completed task markers remain resumable. For transient FDSN/network failures, retry the job after the service recovers.
7. A `files` source with many files will be inefficient because each window reopens every matching file. Implement indexed source plugins for real database workloads.
8. This first release is a Python API, not a CLI. It has no SQLite manifest; JSON completion markers provide task-level restartability.

## Tests

```bash
pytest -q tests/test_archive_processor.py
```
