# Gem: raw data → gemconvert → mapped MiniSEED → SDS + QC

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)


[← Index](README.md) · [Quick start](Quick-Start.md) · [SDS merging](SDS-Merging.md)

## Stages and ownership

```text
Gem raw recordings -> gemconvert -> mseed/, gps/, metadata/
                                         |
                            serial-to-SEED CSV mapping
                                         |
                            stage_gem_mseed() -> staging SDS
                                         |
                            report_gem_deployment() -> QC CSVs
                                         |
                            merge_sds_archives() -> master SDS
```

`gemconvert` is supplied by the upstream **gemlog** project; FLOVOpy invokes it but does not reimplement raw decoding. The mapping is **deployment-specific** and belongs in the project repository, not in `flovopy.ingest.gem`.

## Stage 1: raw Gem conversion

Prepare a workspace with a `raw/` directory:

```text
/data/gem_work/
  raw/            # raw Gem files, laid out as expected by gemconvert
```

Then invoke the installed upstream `gemconvert` executable in that workspace:

```python
from flovopy.ingest.gem import run_gemconvert

mseed_dir = run_gemconvert('/data/gem_work')
print(mseed_dir)  # /data/gem_work/mseed
```

Or:

```bash
python -m flovopy.ingest.cli gem-convert /data/gem_work
```

**Important:** the current wrapper runs `gemconvert` in the workspace; it does not validate raw-file layout beyond checking `raw/`. Rerunning `gemconvert` may overwrite hourly MiniSEED and append logs/metadata; retain original raw files and review the upstream converter's instructions before rerunning.

Expected conversion outputs:

```text
/data/gem_work/
  raw/
  mseed/*.mseed
  gps/*gps*.txt
  metadata/*metadata*.txt
```

## Stage 2: create a serial-to-SEED CSV

Example mapping (illustrative IDs, **not** a KSC inventory):

```csv
serial,id
273,XX.TEST1..HDF
287,XX.TEST2.10.HDF
```

Columns are `serial` and `id` (full `NET.STA.LOC.CHA`, including the empty location field where appropriate). An alternative CSV with `SN,Network,Station,Location,Channel` is supported. A `station_info.txt` lacking `Channel` is **insufficient** for ingestion. The Gem serial is taken from the original MiniSEED station field; it is replaced by the mapped full trace ID.

Mapping checks:

- All observed Gem serials must be mapped in strict mode.
- Any mapping with station code `UNK` is considered **unresolved** and rejected in strict mode.
- The mapping is a static serial→ID dictionary: **time-varying deployment mapping is not implemented**. Separate campaigns or a future time-aware mapping will be needed if a serial is reassigned during the data span.
- Keep the original, historically established KSC assignments in your **KSC project** mapping CSV, not in FLOVOpy source.

## Stage 3: write staging SDS

```python
from flovopy.ingest.gem import stage_gem_mseed

result = stage_gem_mseed(
    '/data/gem_work/mseed',
    '/data/20_archive/Gem/SDS',
    mapping='/project/config/gem_mapping.csv',
    dry_run=True,
)
print(result)

# After checking missing/unresolved serials:
result = stage_gem_mseed(
    '/data/gem_work/mseed',
    '/data/20_archive/Gem/SDS',
    mapping='/project/config/gem_mapping.csv',
)
print(result['counts'], result['errors'])
```

Equivalent CLI:

```bash
python -m flovopy.ingest.cli stage-gem \
    /data/gem_work/mseed /data/20_archive/Gem/SDS \
    --mapping /project/config/gem_mapping.csv --dry-run

python -m flovopy.ingest.cli stage-gem \
    /data/gem_work/mseed /data/20_archive/Gem/SDS \
    --mapping /project/config/gem_mapping.csv
```

`dry_run` reads MiniSEED headers to validate mappings, but does not decode sample arrays. The function catches per-file exceptions and returns them in `errors` rather than necessarily raising at the end. **Check errors before merging.** `strict=False`/`--allow-unmapped` is available but may leave serial numbers as station IDs; it is not recommended for canonical archives.

## Stage 4: generate the two complementary QC reports

```bash
python -m flovopy.ingest.cli gem-report /data/gem_work \
    /data/30_qc/gem_deployment_inventory.csv \
    --daily-csv /data/30_qc/gem_daily_availability.csv \
    --mapping /project/config/gem_mapping.csv \
    --deployment-start 2026-03-10T00:00:00Z \
    --deployment-end 2026-03-15T00:00:00Z
```

The CLI `converted_root` must contain `mseed/`, `gps/`, `metadata/` subdirectories. The deployment inventory is one row per serial/SEED ID, including start/end timestamps, sampling rates, data volume, merged sample coverage, gaps, GPS and recorder health where available. The daily report has one row per UTC day with covered/expected/missing seconds, completeness, gaps and segments. Overlaps are unioned for coverage.

**Coverage semantics:** inventory coverage is measured between the first and last observed samples, **not** as percentage of a planned deployment. Daily completeness uses the explicit deployment interval when supplied; otherwise it uses the observed span (partial boundary days). Supply start **and** end when you want a meaningful deployment-wide denominator. GPS estimates from logs are indicative and are not surveyed coordinates or a replacement for StationXML. Missing or inconsistent telemetry can limit QC accuracy.

Python equivalent:

```python
from flovopy.ingest.gem_report import report_gem_deployment

report = report_gem_deployment(
    '/data/gem_work',
    '/project/config/gem_mapping.csv',
    '/data/30_qc/gem_deployment_inventory.csv',
    daily_csv='/data/30_qc/gem_daily_availability.csv',
    deployment_start='2026-03-10T00:00:00Z',
    deployment_end='2026-03-15T00:00:00Z',
)
print(report)
```

### GPS-only summary

```bash
python -m flovopy.ingest.cli gem-gps /data/gem_work/gps \
    /data/30_qc/gem_gps_coordinates.csv \
    --mapping /project/config/gem_mapping.csv
```

This uses `gemlog.summarize_gps()` with a temporary station-info CSV constructed from the complete mapping.

## Stage 5: merge into master SDS

```python
from flovopy.sds.merge_sds_archives import merge_sds_archives

stage = '/data/20_archive/Gem/SDS'
master = '/data/SDS'
print(merge_sds_archives(stage, master, dry_run=True).as_dict())
# Only after reviewing mapping, QC, and preview:
print(merge_sds_archives(stage, master).as_dict())
```

See [SDS merging](SDS-Merging.md) for transaction tracking and rollback.

## StationXML

The serial-to-NSLC mapping and GPS logs provide inputs for the [deployment metadata table and Nexus workflow](StationXML.md), but do **not** define sensor response/gain or valid epochs by themselves.
