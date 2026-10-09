# Generic MiniSEED tree ingestion

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)

The shared engine is `flovopy.ingest.miniseed.stage_miniseed_tree`. It recursively indexes MiniSEED headers by actual trace time, groups files by UTC day, reads and merges each day's data, and writes to a **fresh staging SDS**. It avoids repeatedly merging thousands of short files with an existing daily SDS file.

```python
from flovopy.ingest.miniseed import stage_miniseed_tree
source = '/data/SVC1/10_conversion/mseed'
staging = '/data/SVC1/20_archive/Generic/SDS'
preview = stage_miniseed_tree(source, staging, dry_run=True)
print(preview)
# Review and then execute:
result = stage_miniseed_tree(source, staging, dry_run=False)
print(result)
```

Default discovery patterns are `*.mseed`, `*.miniseed`, `*.ms`; adjust `patterns` for other filenames. For instrument-specific formats, use [EPIC staging](EPIC.md) or [RT130](RT130.md). The generic staging engine **does not** infer or correct NSLC identifiers: fix headers before staging if needed. Do not point it at your master SDS or at a populated staging directory. Large, high-rate days can require substantial RAM; the current implementation does not yet offer memory-bounded chunking.

After staging: [validate](Validation-and-QC.md) → [merge](SDS-Merging.md).
