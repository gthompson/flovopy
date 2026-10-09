# Quick start and archive architecture

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)

## Install

From the FLOVOpy repository root:

```bash
python -m pip install -e .
python -m flovopy.ingest.cli --help
python -m flovopy.ingest.siliconaudio --help
```

The shorthand `flovopy-ingest` works **only if** FLOVOpy's packaging defines the console entry point `flovopy-ingest = "flovopy.ingest.cli:main"`. The examples below use `python -m flovopy.ingest.cli` so they do not depend on that optional installation step.

## Example service layout (not a hard-coded KSC requirement)

```text
PROJECT/
  SVC1/
    00_download/{Centaur,Pegasus,Q330,RT130,SiliconAudio,Gem,SmartSolo}/
    10_conversion/{RT130,SmartSolo,Gem}/
    20_archive/{Pegasus,Q330,RT130,SiliconAudio,Gem,SmartSolo}/SDS/
    30_qc/
  SDS/                 # master, never the first staging destination
  EPIC/DAYS/           # generated BUD-style export
  metadata/            # station tables, responses, Nexus project, StationXML
```

## Pick the correct input path

1. **Already SDS:** validate and use `merge_sds_archives` directly. See [Centaur](Centaur.md).
2. **MiniSEED but not SDS:** stage into a **new empty** SDS directory, inspect and merge. See [Pegasus](Pegasus.md), [Q330](Q330.md), or [generic MiniSEED](MiniSEED.md).
3. **Proprietary raw:** use the vendor/EPIC conversion tool first. See [RT130](RT130.md), [Gem](Gem.md), [SmartSolo](SmartSolo.md) and [SiliconAudio](SiliconAudio.md).

## Merge example (Python, dry run first)

```python
from flovopy.sds.merge_sds_archives import merge_sds_archives
src = '/data/SVC1/20_archive/Pegasus/SDS'
dst = '/data/SDS'
print(merge_sds_archives(src, dst, mode='fast', dry_run=True))
# Only after checking validation reports and the dry-run summary:
print(merge_sds_archives(src, dst, mode='fast', dry_run=False))
```

## Metadata and delivery

After waveform ingest, inventory NSLC identifiers, reconcile instrument histories and prepare experiment-wide [StationXML with Nexus](StationXML.md). For EPIC delivery, generate and check [DAYS/BUD export](EPIC.md); waveform conversion and metadata construction remain independent.
