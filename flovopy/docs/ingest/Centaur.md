# Centaur: existing SDS to canonical SDS

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)


[← Index](README.md) · [Quick start](Quick-Start.md) · [SDS merging](SDS-Merging.md)

## Scope

Centaur archives in this workflow are **already in SDS layout**. The current `flovopy.ingest.service.ingest_centaur()` does not decode a proprietary Centaur raw format: it discovers existing SDS files and calls the generic FLOVOpy merger. This same operation can ingest any other **valid SDS tree**, not just Centaur.

## Input and output

```text
SOURCE/2026/1R/B01/D/1R.B01.00.DHZ.D.2026.069
                         │
                         ▼
                  MASTER_SDS/2026/...
```

Input can be the root of an SDS tree or an appropriate subtree. Use an isolated source that **does not contain the destination**.

## Preferred direct Python workflow

```python
from flovopy.sds.merge_sds_archives import merge_sds_archives

source = '/data/00_download/Centaur'  # SDS tree
master = '/data/SDS'

preview = merge_sds_archives(source, master, mode='fast', dry_run=True, progress_every=50)
print(preview.as_dict())

# After checking the preview:
result = merge_sds_archives(source, master, mode='fast', progress_every=50)
print(result.as_dict())
```

`fast` trusts valid SDS filenames, copies new files without waveform decoding and performs waveform merges on collisions. `slow` decodes files and derives SDS destinations from trace headers; it is more expensive and useful when filename/header consistency is uncertain. The merger supports `.part` sources and SDS `D` waveform and `S` state-of-health types. Verify the source before deciding which mode to use.

## Configuration-driven CLI

Given a project YAML (`project_root`, `paths.master_sds`) and a service-run YAML with `sources.centaur.input`:

```bash
python -m flovopy.ingest.cli --config /project/config/project.yaml \
    centaur 20260310_service --dry-run --mode fast

python -m flovopy.ingest.cli --config /project/config/project.yaml \
    centaur 20260310_service --mode fast
```

The `centaur` command **merges into the configured master directly**; unlike the Gem and SiliconAudio service commands, it does not first create a staging SDS tree.

You can also merge an arbitrary SDS source via:

```bash
python -m flovopy.ingest.cli --config /project/config/project.yaml \
    merge-sds /data/another_existing_SDS --dry-run
```

## Python service API

```python
from flovopy.ingest.service import ingest_centaur

summary = ingest_centaur(
    '/project/config/project.yaml', '20260310_service',
    source='/data/00_download/Centaur', mode='fast', dry_run=True,
)
```

## Checks and cautions

- Confirm source has recognizable SDS files; service wrapper refuses empty sources.
- Validate trace IDs and station epochs independently; merging is not a StationXML generator.
- Inspect collisions and transaction output. Never confuse an ingest operation with a response correction.
- Preserve original downloaded SDS unchanged.

[Next: SiliconAudio →](SiliconAudio.md)

## EPIC-supplied Centaur (non-SDS downloads)

When the recorder is **not** configured to write SDS, use the Python `stage-instrument centaur` workflow or EPIC `dataselect` as documented in [EPIC](EPIC.md). USF Centaur SDS can be merged directly, but confirm SOH `.S` and `.D` channels. Prepare experiment-wide [Nexus StationXML](StationXML.md) separately.
