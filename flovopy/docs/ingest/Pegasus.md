# Nanometrics Pegasus — Harvester to SDS

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)

## 1. Preserve Harvester downloads

Copy the complete Pegasus Harvester download (waveforms, SOH, logs and any automatically generated XML) to an immutable `00_download/Pegasus/` directory. Inspect the actual folder layout; Harvester configuration and software versions can change paths. Harvester-produced StationXML is useful supporting evidence but **not** a substitute for experiment-wide Nexus StationXML for EPIC.

## 2A. Python/ObsPy daily staging

```bash
python -m flovopy.ingest.cli stage-instrument pegasus \
  /data/SVC1/00_download/Pegasus \
  /data/SVC1/20_archive/Pegasus/SDS
# After checking preview and ensuring destination is fresh/empty:
python -m flovopy.ingest.cli stage-instrument pegasus \
  /data/SVC1/00_download/Pegasus \
  /data/SVC1/20_archive/Pegasus/SDS --execute
```

The adapter preserves MiniSEED NSLC headers, indexes by UTC time and batches by day. It does not decode proprietary files, automatically fix incorrect headers, or guarantee that every SOH/log file is included. Compare file counts and channel lists against the original download.

## 2B. EPIC `dataselect` alternative

Use the Pegasus glob matching the **actual** Harvester directory tree; the example in [EPIC](EPIC.md) illustrates common layouts. `dataselect` creates daily EPIC files; to retain SDS as master, stage those MiniSEED files using the [generic ingestor](MiniSEED.md), rather than renaming them to SDS.

## 3. Verify and deliver

Run [EPIC validation](Validation-and-QC.md), inspect SOH and timing, merge staging to master, and generate [BUD/DAYS export](EPIC.md). Prepare [StationXML in Nexus](StationXML.md), including all instrument changes and responses.
