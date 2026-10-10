# Pass 6 — MVO / EnhancedEvent integration

Changes:
- `MVOSfile.to_enhancedevent()` now constructs `EnhancedEventMeta` and calls `EnhancedEvent.wrap()` instead of unsupported legacy constructor keywords.
- `EnhancedEvent.wrap()` attaches transient waveform streams with `object.__setattr__`, avoiding ObsPy `Event` unknown-attribute warnings. The stream is not serialized to QuakeML.
- Catalog conversion prefers a parser’s native `to_enhancedevent()` when available; generic parsers continue to use the Pass 5 fallback.
- `to_enhanced_catalog()` converts an enhanced catalog result to `EnhancedCatalog` without duplicating Event objects.

No source waveform or S-file is modified. No database is introduced.

Install paths relative to repository root: `flovopy/enhanced/event.py`, `flovopy/research/mvo/mvosfile.py`, `flovopy/seisanio/catalog.py`, and `tests/test_seisan_catalog_pass6.py`.

Run `pytest -q tests/test_seisan_conversion_pass2.py tests/test_seisan_conversion_pass3.py tests/test_seisan_conversion_pass4.py tests/test_seisan_catalog_pass5.py tests/test_seisan_catalog_pass6.py`.

Runtime tests require ObsPy and your FLOVOpy environment and have not been executed here.
