# Configuration, CLI reference and troubleshooting

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)


[← Index](README.md) · [Quick start](Quick-Start.md) · [Centaur](Centaur.md) · [SiliconAudio](SiliconAudio.md) · [Gem](Gem.md)

## Two ways to operate

**Direct-path APIs** (`stage_gem_mseed`, `convert_siliconaudio`, `merge_sds_archives`) are project-independent. **Service-run APIs** use a project YAML and a service-run YAML to resolve source and output directories. The project name, station mapping, instrument serials, and local disk paths belong outside FLOVOpy's reusable source code.

## Minimal project YAML

```yaml
project_root: /data/monitoring
paths:
  master_sds: SDS
  database: database/archive.sqlite
```

The `database` setting is a project-level path setting; the **generic merger's own** transaction database is independently maintained at `SDS/.merge_tracking.sqlite` unless explicitly overridden in its Python API. Do not confuse the two.

A service-run directory named `20260310_service` might contain:

```yaml
service_run:
  id: 20260310_service
  start_date: '2026-03-10'
  end_date: null
paths:
  raw: 00_download
  conversion: 10_conversion
  archive: 20_archive
  qc: 30_qc
sources:
  centaur:
    enabled: true
    input: 00_download/Centaur
    mode: fast
  gem:
    enabled: true
    converted: 10_conversion/Gem/mseed
  silicon_audio:
    enabled: true
    input: 00_download/SiliconAudio
```

The service-run code expects this YAML at `PROJECT_ROOT/20260310_service/service_run.yaml`.

## CLI command reference (current migration)

The following commands are implemented in `flovopy.ingest.cli`:

| Command | Needs `--config`? | Action |
|---|---|---|
| `new-run START_DATE` | Yes | Create service-run directories and YAML |
| `status RUN_ID` | Yes | Report configured source directories |
| `centaur RUN_ID` | Yes | Merge existing Centaur SDS into master |
| `merge-sds SOURCE` | Yes | Merge arbitrary SDS tree into configured master |
| `siliconaudio RUN_ID` | Yes | Convert downloaded station data to staged SDS |
| `gem RUN_ID --mapping CSV` | Yes | Convert Gem MiniSEED to staged SDS |
| `ingest RUN_ID --instrument NAME` | Yes | Execute selected service workflows (see warning below) |
| `rollback TRANSACTION_ID` | Yes | Roll back recorded SDS merge |
| `stage-siliconaudio INPUT OUTPUT --year YEAR` | No | Direct-path SiliconAudio staging |
| `stage-gem INPUT OUTPUT --mapping CSV` | No | Direct-path Gem staging |
| `gem-convert WORKSPACE` | No | Invoke upstream `gemconvert` |
| `gem-report ROOT INVENTORY_CSV --mapping CSV` | No | Gem inventory and daily availability CSVs |
| `gem-gps GPS_DIR OUTPUT_CSV --mapping CSV` | No | Gem GPS summary |

Use `python -m flovopy.ingest.cli --help` and `python -m flovopy.ingest.cli COMMAND --help` for authoritative flags. `--config` is a **top-level option** and must precede the subcommand.

**Important:** `ingest RUN_ID` defaults to **Centaur only**. Selecting Gem through this aggregate command currently does **not** provide a mapping argument; use the dedicated `gem` or `stage-gem` command with `--mapping` instead. Gem and SiliconAudio service commands stage data; Centaur service command merges directly to master. An aggregate `ingest` call should not be assumed to have published all staged sources.

### Combined SiliconAudio module CLI

The newer combined module also provides:

```bash
python -m flovopy.ingest.siliconaudio recover SOURCE_CARD DESTINATION [--station NAME] [--verify]
python -m flovopy.ingest.siliconaudio archive INPUT_DATA_DIR OUTPUT_SDS [--year YEAR]
```

These `recover` and `archive` subcommands are **not yet registered** under `flovopy.ingest.cli` in the inspected migration. Use the module invocation shown above.

## Troubleshooting

| Symptom | Likely explanation / action |
|---|---|
| `No module named flovopy.ingest` | Copy modules into installed FLOVOpy checkout; reinstall editable |
| `recover_sd_card` import fails | Your `siliconaudio.py` is the old conversion-only migration version |
| `No dated subdirectories` | Pass Gecko station's `data/` directory, not station root; check year/layout |
| `Year required` | Converter received `MM/DD/HH` without `year=` |
| `days_skipped` | Strict SiliconAudio mode encountered a bad minute file; review errors |
| Gem `Unmapped serial` | Add a confirmed serial→full SEED ID entry in deployment CSV |
| Gem `Unresolved station` | Mapping uses station `UNK`; reconcile location before staging |
| Gem report has surprising percentages | Check explicit deployment interval and observed-span semantics |
| `No SDS files found` | Check Centaur source root and SDS filename hierarchy |
| CLI `--config` error | Supply project YAML before the subcommand |
| Merge rollback refuses a file | It changed after the merge; investigate before considering force |

## Version and validation notes

These pages describe the inspected FLOVOpy ingestion migration and the **later combined SiliconAudio module**, not a formally released, field-tested FLOVOpy API. Examples have been checked against current function signatures and CLI definitions, but **no real-card recovery, gemconvert run, or end-to-end archive merge was executed** for this documentation task. Validate on a disposable staging archive before applying to your master data.

## Current EPIC and RT130 commands (verified against `cli.py`)

The following commands are **direct-path** operations; they do not require `--config`:

```bash
python -m flovopy.ingest.cli stage-instrument {centaur,pegasus,q330} RAW_ROOT STAGING_SDS [--execute] [--allow-errors]
python -m flovopy.ingest.cli epic-validate SDS_ROOT [--json OUT.json] [--csv OUT.csv] [--no-header-check]
python -m flovopy.ingest.cli epic-bud SDS_ROOT DAYS_ROOT [--execute] [--validate] [--mode symlink|hardlink|copy] [--overwrite-links] [--no-soh]
python -m flovopy.ingest.cli rt130 {explore,convert,logs} WORKSPACE [--parfile parfile.txt] [--executable rt2ms] [--execute]
python -m flovopy.ingest.cli rt130 stage MSEED_ROOT STAGING_SDS [--execute] [--allow-errors]
```

**Defaults differ between commands:** `stage-instrument`, `rt130` and `epic-bud` preview by default and require `--execute` to write. `epic-validate` reads and reports. Project-oriented `merge-sds` and `centaur` require `--config PROJECT.yaml`. For instrument-specific Python APIs, see each linked page.

`stage-instrument` does not support RT130 raw CF cards, Gem raw, SmartSolo DLD, or Gecko SD cards; use their dedicated converters first. [Index](README.md).
