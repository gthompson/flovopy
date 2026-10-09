# SiliconAudio Gecko: SD card → download → SDS

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)


[← Index](README.md) · [Quick start](Quick-Start.md) · [SDS merging](SDS-Merging.md)

The **current combined** `flovopy.ingest.siliconaudio` module contains two independent stages. The older `flovopy_ingest_from_v10.zip` contains a **conversion-only** version; replace it with the newer combined module before using `recover_sd_card()`.

## Stage 1: recover raw SD card files

```text
SD_CARD/
  data/YYYY/MM/DD/HH/*.ms, *.ss
  histogram/*.csv
       ↓ recover_sd_card()
00_download/SiliconAudio/STATION/
  data/YYYY/MM/DD/HH/*.ms, *.ss
  histogram/*.csv
  gecko_recovery.log
```

Recovery accepts either the **card root** (containing `data/`) or its `data/` directory. It scans a strict date/hour hierarchy and validates Gecko filenames, avoids symlinks, records errors, copies via `.partial` temporary files and atomic rename, and can verify SHA256. It does **not** modify the card. A `station` filter is optional; there is no baked-in KSC station.

```python
from flovopy.ingest.siliconaudio import recover_sd_card

recovery = recover_sd_card(
    '/Volumes/GECKO_CARD',
    '/data/20260922_service/00_download/SiliconAudio/STATION',
    station='STATION', verify=True,
    dry_run=True,
)
print(recovery)

# Repeat with dry_run=False after reviewing paths and counts.
```

```bash
python -m flovopy.ingest.siliconaudio recover \
    '/Volumes/GECKO_CARD' \
    '/data/20260922_service/00_download/SiliconAudio/STATION' \
    --station STATION --dry-run

python -m flovopy.ingest.siliconaudio recover \
    '/Volumes/GECKO_CARD' \
    '/data/20260922_service/00_download/SiliconAudio/STATION' \
    --station STATION --verify
```

Options: `--dry-run`, `--verify`, `--overwrite-mismatched`, `--quiet-existing`, `--station`. Default behavior does not overwrite an existing mismatched file. SHA256 verification causes extra reads, which can matter for a failing card. Recovery returns a dictionary with `status` (0 or 1), counters and log path. **Review the log and counters even when status is zero.**

## Stage 2: minute MiniSEED → staging SDS

Pass the downloaded station's **`data/` directory**, not the station root:

```python
from flovopy.ingest.siliconaudio import convert_siliconaudio

source = '/data/20260922_service/00_download/SiliconAudio/STATION/data'
stage = '/data/20260922_service/20_archive/SiliconAudio/SDS'

preview = convert_siliconaudio(source, stage, dry_run=True)
print(preview)

result = convert_siliconaudio(source, stage, strict=True)
print(result['errors'])
```

`convert_to_sds()` is an alias for `convert_siliconaudio()`. The converter accepts `YYYY/MM/DD/HH` directories **without** `year`, or `MM/DD/HH` directories **with** `year=2026`. It reads `*.ms` by default; `.ss` and histogram CSV files are **preserved by recovery but not converted into waveform SDS** by this function.

```bash
python -m flovopy.ingest.siliconaudio archive \
    '/data/20260922_service/00_download/SiliconAudio/STATION/data' \
    '/data/20260922_service/20_archive/SiliconAudio/SDS' \
    --dry-run

python -m flovopy.ingest.siliconaudio archive \
    '/data/20260922_service/00_download/SiliconAudio/STATION/data' \
    '/data/20260922_service/20_archive/SiliconAudio/SDS'
```

The converter merges each day's traces in memory (`method=0`), trims to UTC day boundaries and writes with `EnhancedSDSClient.write_stream()`. It does not rewrite the original minute files. By default, `strict=True` skips writing a day if **any** source minute file failed to read; inspect `errors` and `days_skipped`. `dry_run` counts files but does **not** validate MiniSEED decoding or IDs; do not mistake it for full data validation.

### Existing generic CLI alternative

```bash
python -m flovopy.ingest.cli stage-siliconaudio \
    '/data/00_download/SiliconAudio/STATION/data' \
    '/data/20_archive/SiliconAudio/SDS' \
    --year 2026 --dry-run
```

The generic `stage-siliconaudio` CLI currently **requires `--year` even for a `YYYY/MM/DD` tree**, although the Python converter itself does not. The module-level `python -m flovopy.ingest.siliconaudio` CLI does not have that restriction. The `--allow-partial-days` flag disables strict day skipping.

## Stage 3: merge staged SDS

```python
from flovopy.sds.merge_sds_archives import merge_sds_archives

stage = '/data/20260922_service/20_archive/SiliconAudio/SDS'
master = '/data/SDS'
print(merge_sds_archives(stage, master, dry_run=True).as_dict())
# After review:
print(merge_sds_archives(stage, master).as_dict())
```

See [merging and rollback](SDS-Merging.md).

## Common pitfalls

- Use `.../STATION/data` as converter input, not `.../STATION`.
- Confirm the input file trace IDs are correct: this converter does **not** provide a Gem-like serial→SEED mapping.
- Recovery's source may be a removable, unreliable card; retain downloaded originals and inspect recovery log.
- Don't use `--overwrite-mismatched` casually.
- Do not run the converter with the source directory as, or containing, its output directory.

[Next: Gem →](Gem.md)

## StationXML

Record the Gecko digitizer model/serial, attached sensor, sensitivity, gain, component orientations, coordinates and deployment epochs separately. See [StationXML](StationXML.md).
