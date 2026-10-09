# FLOVOpy ingestion — documentation index

This is the **version-controlled Markdown documentation** for `flovopy.ingest`, checked against the modules in `flovopy_ingest_with_rt130.zip` (9 October 2026). Links between pages are relative, so the documentation works in a GitHub repository, a local Markdown viewer, or a GitHub Wiki with compatible filenames. The code is generic; KSC-specific station assignments and service-run configurations belong in the separate project repository.

## Start here

- [Quick start and data flow](Quick-Start.md)
- [Configuration and exact CLI reference](Configuration-and-CLI.md)
- [Generic MiniSEED ingestion](MiniSEED.md)
- [SDS merging, transaction tracking and rollback](SDS-Merging.md)
- [Validation and quality control](Validation-and-QC.md)
- [StationXML and Nexus metadata workflow](StationXML.md)
- [EPIC workflows: dataselect, fixhdr, BUD export and submission](EPIC.md)

## Instrument-specific guides

| Instrument | Original format | Converter / route | Guide |
| --- | --- | --- | --- |
| Nanometrics Centaur | Usually MiniSEED; sometimes native SDS | Direct SDS merge or Python/`dataselect` | [Centaur](Centaur.md) |
| Nanometrics Pegasus | Harvester MiniSEED | Python daily staging or `dataselect` | [Pegasus](Pegasus.md) |
| Quanterra Q330 | Baler/B14/B44 data | Python daily staging or `dataselect` | [Q330](Q330.md) |
| REF TEK RT130 | CF-card REF TEK packets | PASSOFT `rt2ms`, then SDS staging | [RT130](RT130.md) |
| SiliconAudio Gecko | SD card / minute MiniSEED | `recover_sd_card` then daily SDS conversion | [SiliconAudio](SiliconAudio.md) |
| Gem | Gem raw | `gemconvert`, serial mapping CSV, SDS + QC | [Gem](Gem.md) |
| SmartSolo | DLD | SoloLite/Harvester, serial mapping CSV, SDS | [SmartSolo](SmartSolo.md) |
| Any MiniSEED source | MiniSEED | Chronological UTC-day batching | [MiniSEED](MiniSEED.md) |

## Fundamental separation

`00_download` (immutable original data) → optional `10_conversion` → **staging SDS** → validate → tracked **master SDS**. Export EPIC daily `DAYS/` from master SDS as a separate operation. Maintain **StationXML separately**; it describes channel responses, equipment, coordinates, orientations and epochs, not raw waveform conversion.

**Scope warning:** The current code offers useful header and directory checks, but does **not** establish complete EPIC compliance (including record byte order, timing quality and full gap/overlap review). Use EPIC QC tools and consult EPIC before submission. [Details](Validation-and-QC.md).
