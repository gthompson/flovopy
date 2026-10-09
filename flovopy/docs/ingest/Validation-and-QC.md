# Validation and quality control

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)

## What FLOVOpy currently checks

```bash
python -m flovopy.ingest.cli epic-validate \
  /data/SVC1/20_archive/Pegasus/SDS \
  --json /data/SVC1/30_qc/pegasus.json \
  --csv /data/SVC1/30_qc/pegasus.csv
```

`flovopy.ingest.epic.validate_sds` inspects SDS paths and filenames, reads MiniSEED headers (unless `--no-header-check`), and reports detected inconsistencies and daily coverage-related fields. Inspect the generated files and errors. **This is a preliminary validator, not an EPIC certification.**

## Required additional checks before master merge / EPIC delivery

| Check | Why it matters | Recommended method |
| --- | --- | --- |
| File inventory vs source | Avoid missing channels, days or CF cards | Compare download manifests and staged NSLC/day counts |
| Gaps and overlaps | Detect lost, duplicated or discontinuous samples | ObsPy `Stream.get_gaps()` / EPIC waveform QC; analyze full traces |
| GPS and timing quality | Detect incorrect time tags | Instrument SOH/logs, GPS status, EPIC tools and timing-quality review |
| SOH and auxiliary channels | EPIC may require waveform, mass position, administrative and log channels | Compare source and staged channels, inspect `.S` and `.D` streams |
| Record byte order | EPIC expects big-endian MiniSEED | EPIC `fixhdr` / record-level checks, not merely ObsPy `headonly=True` |
| Header identity and day boundary | Correct SEED identifiers and daily volumes | `epic-validate`, spot-check first/last record |
| Response coverage | Accurate physical units and epoch changes | [StationXML](StationXML.md) and `Inventory.get_response` |
| Duplicate/overlap merge behavior | Avoid silently overwriting nonidentical data | Review [transactional merge](SDS-Merging.md) logs and spot-check waveforms |

`fixhdr` can correct record headers and endian issues according to EPIC procedures; alternatively, ObsPy can read full traces and rewrite MiniSEED. **Header-only ObsPy reads cannot modify samples or headers in place.** Preserve original downloads and log every correction. Be especially careful that converting to a new MiniSEED file does not discard timing-quality information.

## Instrument-specific reports

- **Gem:** [deployment and daily availability reports](Gem.md), plus GPS and battery/temperature metadata.
- **RT130:** review `rt2ms.msg`, per-card `LOGS/` and timing warnings; use `sohviewer` or `logpeek`. [RT130](RT130.md).
- **Pegasus/Centaur:** review Harvester/offload SOH, timing and EPIC SQLX/SOHViewer. [EPIC](EPIC.md).
- **Q330:** inspect baler logs, mass position, GPS/SOH and EPIC QC (`qpeek`, `pql` as appropriate).
- **SmartSolo/SiliconAudio:** validate channel identity, continuity, sampling rate and station inventory before merge.

## Delivery gate

Only export [EPIC DAYS/BUD](EPIC.md) after waveform QC and [StationXML](StationXML.md) epoch checks. A symlink export does not convert endian format or repair headers; use materialized corrected files if needed.
