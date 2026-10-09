# SDS merging, transaction tracking and rollback

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)


[← Index](README.md) · [Quick start](Quick-Start.md) · [Centaur](Centaur.md) · [SiliconAudio](SiliconAudio.md) · [Gem](Gem.md)

## Ownership

Merging is **generic SDS infrastructure**, implemented in `flovopy.sds.merge_sds_archives`, not in an instrument adapter. All three workflows converge here. The examples assume the version of the FLOVOpy merger used by the current ingestion migration; check your installed function signature if the FLOVOpy checkout has evolved.

## Safest workflow: inspect → dry-run → merge → audit

```python
from flovopy.sds.merge_sds_archives import merge_sds_archives

stage = '/data/20_archive/Gem/SDS'
master = '/data/SDS'

preview = merge_sds_archives(stage, master, mode='fast', dry_run=True,
                             progress_every=50)
print(preview.as_dict())

# Verify station IDs, source counts, collisions, target and free disk space.
summary = merge_sds_archives(stage, master, mode='fast', progress_every=50)
print(summary.as_dict())
```

Or via the **project-configured** CLI (no generic `merge-sds` command without `--config` in the current CLI):

```bash
python -m flovopy.ingest.cli --config /project/config/project.yaml \
    merge-sds /data/20_archive/Gem/SDS --dry-run

python -m flovopy.ingest.cli --config /project/config/project.yaml \
    merge-sds /data/20_archive/Gem/SDS
```

## Fast versus slow

| Mode | Behavior | Use when |
|---|---|---|
| `fast` | Trusts SDS path/filename; copies new files; reads and waveform-merges collisions | Source SDS names and structure are trusted |
| `slow` | Reads MiniSEED files and derives SDS destinations from trace headers | SDS naming/headers may disagree |

Both modes require careful review for scientific validity. `fast` supports valid SDS data types including `D` and `S`; `.part` files are handled by the merger. A malformed or zero-byte source should be investigated, not silently treated as a valid waveform.

## Tracking and backup cache

By default the merger creates a SQLite tracking database **inside the master SDS root** (`.merge_tracking.sqlite`) and uses `.merge_cache/` for necessary backup material. These files are operational state, not waveforms. Preserve them if you may need rollback. Review the returned merge summary and transaction ID; record it in your ingest log.

## Rollback

```python
from flovopy.sds.merge_sds_archives import rollback_merge_session

result = rollback_merge_session(
    'TRANSACTION_ID_FROM_MERGE_SUMMARY',
    db_path='/data/SDS/.merge_tracking.sqlite',
    force=False,
)
print(result)
```

CLI:

```bash
python -m flovopy.ingest.cli --config /project/config/project.yaml \
    rollback TRANSACTION_ID_FROM_MERGE_SUMMARY
```

By default rollback checks that files still match their post-merge size/mtime signatures before removing/restoring them. **Do not use `--force` casually**: subsequent legitimate changes may otherwise be undone. Back up the master archive and transaction database before any high-risk recovery.

## Independent archive checks

- Compare SDS file counts, station/channel IDs and day coverage against staging reports.
- Read representative waveforms and verify start/end times, gaps and sample rates.
- Validate response/StationXML associations separately; an SDS merge does not establish correct instrument sensitivity.
- Confirm unexpected overlaps and `.part` files have been resolved appropriately.
- Never point an instrument converter directly at the canonical SDS unless you deliberately accept bypassing staging and review.
