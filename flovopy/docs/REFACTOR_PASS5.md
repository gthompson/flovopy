# FLOVOpy SEISAN refactor — Pass 5

This is an incremental patch **on top of Pass 4**. It does not replace conversion.py or modify research.mvo/enhanced.

## Install
Copy `seisanio/catalog.py` to `flovopy/seisanio/catalog.py` and `tests/test_seisan_catalog_pass5.py` to repository `tests/`.

## Example
```python
from flovopy.seisanio.catalog import convert_archive_catalog, write_catalog_result
from flovopy.research.mvo.mvosfile import MVOSfile

result = convert_archive_catalog(archive, start, end, parser=MVOSfile, enhanced=True)
write_catalog_result(result, 'out/events.xml', metadata_path='out/events.metadata.jsonl',
                     failures_path='out/events.failures.jsonl')
print(len(result.catalog), len(result.failures))
```

`seisan_to_catalog` and `iter_parsed_sfiles` remain available for compatibility.

## Important compatibility finding
The supplied `MVOSfile.to_enhancedevent()` uses legacy keyword arguments (`obspy_event`, `metrics`, etc.) that the supplied `EnhancedEvent.__init__` does not accept. Pass 5 intentionally does **not** call that method. It uses `EnhancedEvent.wrap(..., meta=EnhancedEventMeta(...))`, preserving the MVO source references and selected historical metadata. A future refactor should repair the MVO method and test against real S-files.

QuakeML stores only standard ObsPy event data; the optional JSONL sidecar stores MVO classifications, AEF rows and provenance. Unsupported metadata types cause an explicit serialization error rather than silently disappearing. Source S-files are not modified. No SQLite migration is included.

## Testing
`pytest -q tests/test_seisan_catalog_pass5.py` then previous Pass 2–4 tests. The code was syntax-compiled here, but runtime tests could not be run because ObsPy is unavailable in this environment.

## Limitations
- Real Nordic S-files are needed to validate the existing Sfile/MVOSfile parsing, including pick/origin/magnitude extraction.
- Enhanced metadata is not included in QuakeML and requires sidecar preservation.
- QuakeML event IDs come from the existing parser; stable source-derived IDs are not imposed.
- Export is not transactional across QuakeML and sidecar files.
