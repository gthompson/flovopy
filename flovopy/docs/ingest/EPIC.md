# EPIC passive-source ingestion — FLOVOpy extension

**Source:** EarthScope EPIC's August 12, 2025 Centaur, Pegasus, Q330 and Nexus processing guides. This is a proposed FLOVOpy patch based on `flovopy_ingest_from_v10.zip`, **not** an EPIC-certified submission pipeline.

## Recommended archive model

1. Preserve the original download unchanged in `00_download/{Centaur,Pegasus,Q330}`.
2. USF Centaur downloads already in SDS: use the existing `flovopy-ingest --config ... centaur RUN_ID` merge path. No `dataselect` needed. **Check the SDS archive for SOH (`*.S`)**.
3. Other Centaur/Pegasus/Q330 downloads: choose either EPIC `dataselect` or the Python staging workflow below. Both should yield station/channel/day volumes; avoid staging into a nonempty directory.
4. Validate the staged SDS; review QC and timing issues; merge using existing tracked `merge_sds_archives` workflow.
5. Export `DAYS/STA/STA.NET.LOC.CHAN.YEAR.DOY` from SDS. Symlinks are convenient for Nexus/local checks; verify with EPIC before using them for `data2passcal` submission. Export copies if required.
6. Maintain experiment-wide StationXML separately using Nexus; update epochs after equipment changes. Pegasus-generated StationXML is **not** a replacement.

## Python/ObsPy path

```bash
flovopy-ingest stage-instrument pegasus /data/SVC1/00_download/Pegasus /data/SVC1/20_archive/Pegasus/SDS
flovopy-ingest stage-instrument pegasus /data/SVC1/00_download/Pegasus /data/SVC1/20_archive/Pegasus/SDS --execute
flovopy-ingest stage-instrument q330 /data/SVC1/00_download/Q330 /data/SVC1/20_archive/Q330/SDS --execute
flovopy-ingest stage-instrument centaur /data/SVC1/00_download/Centaur /data/SVC1/20_archive/Centaur/SDS --execute
flovopy-ingest epic-validate /data/SVC1/20_archive/Pegasus/SDS --json /data/SVC1/30_qc/pegasus.json --csv /data/SVC1/30_qc/pegasus.csv
flovopy-ingest epic-bud /data/master/SDS /data/epic/DAYS --validate
flovopy-ingest epic-bud /data/master/SDS /data/epic/DAYS --validate --execute
```

The Python path uses ObsPy's MiniSEED reader and `EnhancedSDSClient.write_stream()`. It is intentionally conservative: a nonempty output stage is refused. Confirm your local `EnhancedSDSClient` behavior on overlapping traces, day splitting, and SOH before bulk ingestion. **ObsPy `headonly=True` cannot modify headers**: actual corrections require reading and rewriting the affected records, or using `fixhdr`.

## EPIC dataselect path

Run from a service directory with `RAW/` containing the original downloads, adapting the glob to the actual device configuration:

```bash
# Centaur waveform and SOH (example EPIC archive pattern YEAR/JDAY)
dataselect -A DAYS/%s/%s.%n.%l.%c.%Y.%j RAW/*/20??/???/*.miniseed
dataselect -A DAYS/%s/%s.%n.%l.%c.%Y.%j RAW/*/20??/???/soh/*.miniseed
# Pegasus (EPIC example pattern YEAR/NET/STA; confirm before use)
dataselect -A DAYS/%s/%s.%n.%l.%c.%Y.%j RAW/*/20??/??/*/???.D/*
dataselect -A DAYS/%s/%s.%n.%l.%c.%Y.%j RAW/*/*/20??/??/*/???.D/*
# Q330 B14
dataselect -A DAYS/%s/%s.%n.%l.%c.%Y.%j RAW/*.sdr/*
# Q330 B44
dataselect -A DAYS/%s/%s.%n.%l.%c.%Y.%j RAW/STATION*/data*/*
```

If using `dataselect`, ingest the resulting DAYS MiniSEED through an explicitly reviewed converter/stager before merging into master SDS; do not simply rename files to SDS. Check for SOH de-multiplexing, GPS timing, corrupt records, overlaps and gaps. EPIC also requires big-endian data: **this patch does not perform endianness conversion or timing-quality flag editing**. Use EPIC `fixhdr` and its QC tools as appropriate; retain audit records.

## Outstanding work before EPIC submission

- Add a MiniSEED **record-level** validator for byte order, encoding, time quality, and timing flags (ObsPy's `headonly` check is not sufficient).
- Add gap/overlap and GPS/SOH trend reports, not merely per-file header inventory.
- Verify Pegasus SOH glob and Q330 B14/B44 variants against actual offloads; discover files conservatively.
- Verify `EnhancedSDSClient` writes SOH with correct SDS `.S` typing (some SOH is carried in ordinary `.D` streams).
- Run integration tests on real Centaur/Pegasus/Q330 service runs; do not submit unvalidated data.
- Create/update experiment-wide Nexus StationXML with responses and epoch changes; deliver via `data2passcal` only after coordination with EPIC.


### Efficient daily batching (Centaur, Pegasus and Q330)

`stage-instrument` now delegates to `flovopy.ingest.miniseed.stage_miniseed_tree`.
It scans MiniSEED headers into a temporary SQLite day index, reads all
files contributing to each UTC day, sorts by actual sample times, merges
segments in memory, and writes one daily stream to staging SDS. This avoids
merging a succession of small files into the same existing SDS day file.
Q330 `.sdr` files are included in discovery when ObsPy can read them.

The staging SDS directory **must be empty** (also for dry-run). Create a
new staging directory for each service run. After validating it, merge into
the master SDS archive using the tracked `merge_sds_archives` function.
A non-strict run may contain incomplete days; do not promote it automatically.

Example:

```bash
flovopy-ingest stage-instrument pegasus /data/SVC1/00_download/Pegasus \
    /data/SVC1/20_archive/Pegasus/SDS
flovopy-ingest stage-instrument pegasus /data/SVC1/00_download/Pegasus \
    /data/SVC1/20_archive/Pegasus/SDS --execute
```

**Scope:** This adapter is for raw MiniSEED that needs reorganizing. Centaur
recordings already in valid SDS layout can instead be merged directly,
without decoding and rewriting their waveform records. This adapter does not
apply EPIC `fixhdr` corrections or prove EPIC submission compliance.
