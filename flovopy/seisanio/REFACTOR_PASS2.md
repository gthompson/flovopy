# SEISAN refactor — Pass 2

- Corrected WAV discovery: scan actual month directories and include configurable preceding days; old code used a date pattern from the wrong day in the previous directory and excluded every pre-start file.
- Incremental SDS conversion with bounded batches (`batch_files=32` default), rather than an unbounded daily Stream.
- Trim by actual trace timestamps to the requested interval.
- Report failed SDS writes in `failed_files`.
- Refuse incremental `overwrite` to prevent earlier batches destroying existing day-file content.
- Generic conversion still uses `EnhancedSDSClient.write_stream`; MVO-specific fixes remain opt-in.

## Limitations

- SEISAN filename-based discovery does not include SAC/MiniSEED files with arbitrary names.
- The configurable `lookback_days` is a search bound, not a proof that all earlier long recordings are found.
- A successful write is not an end-to-end verification of SDS content.
- Do not enable `fixid` without reviewing its MVO dependencies.
- Test against representative Montserrat recordings before bulk conversion.
