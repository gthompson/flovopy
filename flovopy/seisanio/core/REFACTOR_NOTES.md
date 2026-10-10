# FLOVOpy SEISAN refactor (incremental)

- `core/seisanarchive.py`: retains archive discovery, reading, S-file iteration and its `to_sds()` API. The writer is now delegated to `conversion.archive_to_sds`. Added missing `Stream` import.
- `conversion.py`: extracted existing day-grouped SDS converter; adds `seisan_to_sds()` convenience entry point. Reuses `EnhancedSDSClient.write_stream()`; no second SDS implementation.
- `catalog.py`: small ObsPy Catalog/QuakeML adapter accepting either `Sfile` or `MVOSfile` parser. The latter's AEF extras are **not** serialized by QuakeML; use `to_enhancedevent()` sidecars separately.
- `research.mvo` and `enhanced` are **unchanged**. `EnhancedCatalog`/`EnhancedEvent` already own the richer scientific event model, so do not duplicate them in `seisanio`.

## Known limitations / next passes

1. Existing `iter_waveform_files()` discovers files based on start times and may miss a file that starts before the requested interval but overlaps it. Existing preceding-day pattern is also suspicious; test with real files before changing behavior.
2. Existing conversion groups by filename date, not by actual trace timestamps. The SDS client splits traces at day boundaries; this does not imply strict clipping to the requested interval.
3. Existing `fixid=True` code in the generic archive calls MVO logic. This is a compatibility issue to disentangle after representative historical data tests.
4. `to_sds()` currently has no structured failure record for writer exceptions and only optionally returns counts; a future pass should add a per-file result type.
5. Verify conversion and QuakeML round-trips with ObsPy in your flovopy environment; this sandbox does not have ObsPy installed.
6. The historical `run_seisan2sds.py` has extra day-clipping and Datascope behavior; it is intentionally not replaced until parity tests are available.
