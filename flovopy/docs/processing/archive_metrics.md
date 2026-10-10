# Computing SAM metrics from an SDS archive

`flovopy.processing.archive_metrics` calculates windowed **RSAM, VSAM, DSAM, and VSEM** across consecutive UTC days in an SDS archive. It uses `EnhancedSDSClient` for waveform access and the classes in `flovopy.processing.sam` for the metrics. This is the initial archive processor implementation; numerical spectrogram processing is planned as a separate engine sharing the scheduling infrastructure.

## Requirements

Run from an environment where FLOVOpy and its dependencies (including ObsPy, NumPy, and pandas) are installed. Use a valid SDS directory and a **separate output directory**. Calibrated metrics require StationXML covering the requested channels and times. Parquet shards additionally require a pandas-compatible Parquet engine.

Verify the command-line options:

```bash
python -m flovopy.processing.archive_metrics --help
```

## Quick start: counts-based RSAM

```bash
python -m flovopy.processing.archive_metrics \
  --sds-root /data/KSC/SDS \
  --output-root /data/KSC/SAM_RSAM \
  --start 2016-08-01 \
  --end 2016-08-03 \
  --network 1R \
  --station BCHH \
  --channel 'DH*' \
  --products RSAM \
  --sampling-interval 60 \
  --bands-preset volcano \
  --workers 2
```

`--end` is **exclusive**: this example covers August 1 and 2 UTC. Quote wildcard selectors to prevent shell expansion. Omitted NSLC selectors default to `*` (all). The location selector `--location '--'` is interpreted as the empty location code.

## Calibrated seismic metrics

```bash
python -m flovopy.processing.archive_metrics \
  --sds-root /data/KSC/SDS \
  --output-root /data/KSC/SAM_CALIBRATED \
  --start 2016-08-01 --end 2016-08-03 \
  --network 1R --station BCHH --channel 'DH*' \
  --products RSAM VSAM DSAM VSEM \
  --stationxml /data/KSC/metadata/KSC.stationxml \
  --pre-filt 0.02 0.05 80 100 \
  --sampling-interval 60 --workers 2
```

**Adjust `--pre-filt` for your instrument response and Nyquist frequency.** The four values above are illustrative, not universally appropriate. The code performs ObsPy `remove_response` to `VEL` for VSAM/VSEM and `DISP` for DSAM. A response-removal error aborts the affected daily task rather than silently relabeling counts. StationXML can be a single file or a directory containing `*.xml` files. RSAM is computed from uncalibrated waveform values (typically counts); it is **not** a pressure metric.

## Product definitions

| Product | Input to SAM class | Interpretation |
| --- | --- | --- |
| `RSAM` | Raw waveform values | Amplitude statistics in input units, usually counts |
| `VSAM` | Response-corrected velocity | Amplitude statistics in m/s |
| `DSAM` | Response-corrected displacement | Amplitude statistics in m |
| `VSEM` | Response-corrected velocity | Integrated squared velocity, m²/s; **not joules** |

The `sam.py` implementation supplies windowed statistics, optional band columns, and `coverage` (fraction of observed samples). Its `rms` amplitude statistic is the standard deviation of signed samples, not RMS of absolute values. All coverage values are retained; this processor does **not** impose a minimum-coverage threshold. Apply `SAM.filter_by_coverage()` later as needed.

## Frequency filtering and bands

The default overall filter is `0.5 18` Hz. Disable it with `--no-filter`, or set another band:

```bash
--filter 0.1 20
```

Choose one built-in frequency-band preset with `--bands-preset`:

| Preset | Named bands (Hz) |
| --- | --- |
| `volcano` | VLP 0.02–0.2; LP 0.5–4; VT 4–18 |
| `storm_seismo` | PRI 0.05–0.10; SEC 0.10–0.35; HI 1–5 |
| `storm_infra` | TC 0.01–0.10; MB 0.15–0.35; TH 1–10 |
| `storm` | PRI 0.02–0.10; SEC 0.10–0.35; HI 1–10 |

Alternatively, supply a JSON object inline or a path to a JSON file:

```bash
--bands '{"VLP":[0.02,0.2],"LP":[0.5,4],"VT":[4,18]}'
```

Use `--bands '{}'` to request no named bands. If neither `--bands` nor `--bands-preset` is supplied, the SAM classes use their own default bands. The filtering corner count defaults to `--corners 4`. A very low frequency requires sufficiently long contiguous valid waveform segments; a one-second measurement window does **not** imply a one-second filter input.

## Windowing, gaps, and daily boundaries

- `--sampling-interval` is the SAM output-window duration in seconds (default **60**, minimum **1**). It must divide 86,400 seconds exactly for daily UTC processing.
- Tasks are divided by UTC day. Each task reads extra waveform data on both sides using `--halo-seconds` (default **300**), computes metrics, then keeps only **fully contained** output windows within the requested interval.
- The output windows are UTC-aligned, with half-open time intervals `[start, end)`; an arbitrary start or end that cuts through a window results in that partial window being omitted.
- SDS fragments are merged with masked gaps (`fill_value=None`), not zeros. `coverage` describes observed input samples, which may differ from usable filtered coverage.
- Filtering near gaps and task boundaries can introduce edge effects. Compare a short run against an uninterrupted waveform before relying on long-period measurements for publication.

## Parallel execution, restart, and outputs

Set `--workers N` (alias `--nprocs`) to process independent UTC days concurrently. Worker exceptions propagate to the caller. Each day writes unique **atomic shards**, followed by a `complete.json` marker; a rerun with the same settings skips completed days by default.

Output layout is approximately:

```text
SAM_RSAM/
  .archive_metrics_owner.json
  config_<hash>.json
  shards/<hash>/YYYY-MM-DD/
    RSAM_<trace-hash>.pkl
    complete.json
  ... yearly SAM files written by SAM.write() ...
```

The actual yearly filenames are determined by `SAM.write()`. Shard formats are `pickle` (default), `csv`, and `parquet` via `--shard-format`. Yearly consolidation uses the existing SAM pickle output format regardless of shard format. Shards are retained after consolidation.

**Restart caveats:** `--no-resume` forces daily tasks to be recalculated; `--no-consolidate` stops after creating daily shards. Different scientific configurations must use **different output roots**: an owner file prevents mixing configurations. The configuration hash deliberately excludes requested start/end dates, worker count, and output root, so extending a date range can reuse completed day shards. However, the current consolidation writes via `SAM.write(overwrite=False)`; inspect existing yearly outputs carefully when rerunning overlapping or extended ranges. There is no `--dry-run` flag.

## CLI reference

| Option | Default | Purpose |
| --- | --- | --- |
| `--sds-root` | required | Input SDS root directory |
| `--output-root` | required | Output SAM directory |
| `--start`, `--end` | required | UTC interval; end exclusive |
| `--products` | `RSAM` | One or more of `RSAM VSAM DSAM VSEM` |
| `--network`, `--station`, `--location`, `--channel` | `*` | SDS/NSLC selectors |
| `--sampling-interval` | `60` | Output-window length (seconds) |
| `--filter LOW HIGH` | `0.5 18` | Overall bandpass |
| `--no-filter` | off | Disable overall bandpass |
| `--bands` | SAM defaults | Inline JSON or JSON file path; `{}` disables bands |
| `--bands-preset` | none | `volcano`, `storm_seismo`, `storm_infra`, `storm` |
| `--corners` | `4` | Filter corners |
| `--stationxml` | none | StationXML file/directory for calibrated products |
| `--pre-filt F1 F2 F3 F4` | none | ObsPy response-removal prefilter |
| `--halo-seconds` | `300` | Extra waveform context at daily boundaries |
| `--workers` | `1` | Parallel daily tasks |
| `--shard-format` | `pickle` | `pickle`, `csv`, or `parquet` |
| `--no-resume` | off | Recompute days despite completion markers |
| `--no-consolidate` | off | Skip yearly SAM consolidation |

Legacy aliases: `--sds_root`, `--sam_root`, `--sampling_interval`, and `--nprocs`.

## Python API

```python
from flovopy.processing.archive_metrics import Config, run

config = Config(
    sds_root='/data/KSC/SDS',
    output_root='/data/KSC/SAM_RSAM',
    start='2016-08-01',
    end='2016-08-03',
    products=('RSAM',),
    network='1R',
    station='BCHH',
    channel='DH*',
    sampling_interval=60.0,
    workers=2,
)
summary = run(config)
print(summary)
```

Call `run(config, consolidate_yearly=False)` to generate shards without yearly consolidation. The return value includes configuration digest, day count, processed/skipped counts, and (when consolidating) trace-ID counts by product.

## Validation and limitations

The initial orchestration tests passed in the FLOVOpy development environment (`pytest -q tests/test_archive_metrics.py`: **4 passed**). This does not yet validate response correction, large multi-station archives, conflicting MiniSEED overlaps, or split-versus-unsplit long-period filtering. A previously identified whole-trace demeaning issue in `VSEM` should be fixed in the installed `sam.py`; verify that fix before computing VSEM. The base SAM implementation may also demean entire read segments, making filtered amplitudes sensitive to halo length and gaps.

For a first real-data trial, use a **small date range, one station, and a disposable output root**. Inspect coverage, timestamps, physical units, and sample values before running across a full archive.
