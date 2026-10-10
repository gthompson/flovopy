# FLOVOpy SEISAN / MVO developer and user guide

**Status:** October 2026 · FLOVOpy refactor Passes 1–7 · implementation and future-work handover

## 1. Purpose and scope

This guide documents the FLOVOpy library work already undertaken to read SEISAN WAV/REA archives, convert waveforms to SDS, convert Nordic S-file event metadata to QuakeML, preserve Montserrat-specific information, and audit conversions. It also identifies what has **not** yet been validated and describes how the eventual **Montserrat Seismic Data Modernization Project** should build on these modules. The modernization project itself—historical inventory, relational database, comprehensive trace-ID mapping, complete archive migration—is **not** implemented here.

**Validation baseline reported by the developer:** Passes 2–6: **17 tests passed**. Pass 7 was delivered but no Pass 7 test result has been reported. The passing tests predominantly exercise interfaces and synthetic inputs; they do **not** establish scientific correctness on historical MVO recordings.

### 1.1 Data domains

- **Waveforms:** `WAV/<DB>/YYYY/MM/` files, potentially native SEISAN, SAC, or MiniSEED; continuous and triggered/event databases may be separate.
- **Event records:** `REA/<DB>/YYYY/MM/` Nordic-format S-files, usually one per event, with links to waveform files.
- **MVO extensions:** local event classifications, historical DSN/ASN associations, AEF amplitude/spectral data, instrument/station/trace-ID conventions, and timing/sampling-rate corrections.
- **Modern products:** SDS waveform dayfiles, ObsPy Event/Catalog and QuakeML, EnhancedEvent/EnhancedCatalog objects, StationXML metadata, and later SAM/event metrics.

## 2. Package responsibilities and dependencies

| Package/module | Responsibility | What not to put there |
|---|---|---|
| `flovopy.seisanio.core.seisanarchive.SeisanArchive` | Discover WAV/REA files, read waveform files, iterate events, backwards-compatible `to_sds()` | MVO-only correction rules; relational DB |
| `flovopy.seisanio.core.sfile.Sfile` | Generic Nordic S-file parsing | MVO-only fields |
| `flovopy.seisanio.core.wavfile.Wavfile` | SEISAN waveform file representation | SDS storage |
| `flovopy.seisanio.core.aeffile.AEFfile` | Generic AEF parsing | MVO-specific interpretation where it differs |
| `flovopy.seisanio.conversion` | Batch SEISAN→SDS workflow, clipping, write failure handling, optional transforms, audit/verification | Hard-coded Montserrat assumptions |
| `flovopy.seisanio.validation` | Advisory StationXML channel/epoch/response checks | Automatic ID guessing |
| `flovopy.seisanio.catalog` | S-file→ObsPy Catalog/QuakeML and JSONL enhanced metadata | Database schema or waveforms in QuakeML |
| `flovopy.seisanio.integration_audit` | Read-only event structural round-trip and parser audits | Full numerical scientific validation |
| `flovopy.seisanio.mvo_transform` | Optional adapter invoking existing MVO correction rules | Generic format parsing |
| `flovopy.research.mvo` | MVOSfile, MVO AEF handling, waveform IDs, MVO archive behavior | General-purpose archive conversion |
| `flovopy.enhanced` | EnhancedEvent, EnhancedCatalog, EnhancedTrace/Stream, EnhancedSDSClient | SEISAN-specific filename discovery |
| `flovopy.processing` | Continuous SAM, spectrograms, spectral/event metrics | Historical file migration logic |

**Import direction:** generic `seisanio` does not implicitly import `research.mvo` for normal conversion. The opt-in `mvo_transform` adapter imports the MVO correction code when called. `seisanio` delegates SDS writes to `EnhancedSDSClient`; event conversion uses ObsPy plus optional EnhancedEvent wrappers.

## 3. Installed refactor passes and tests

| Pass | Main deliverable | Tests / status |
|---|---|---|
| 1 | Initial generic conversion/catalog separation | No separate reported test baseline |
| 2 | Lookback discovery, bounded batching, actual-time clipping, safe write mode | 2 passed after correcting mock signature |
| 3 | Optional MVO transformation, trace ID/time/sample-rate change audit | 4 cumulative passed |
| 4 | StationXML advisory validation, JSONL manifest, optional SDS read-back | 8 cumulative passed |
| 5 | QuakeML conversion, enhanced metadata/failure sidecars | 13 cumulative passed; original ObsPy warning |
| 6 | MVOSfile→EnhancedEvent compatibility, stream handling, EnhancedCatalog | 17 cumulative passed |
| 7 | Read-only event/QuakeML integration audit and optional real-S-file test | **Not yet reported run** |

The code samples below describe the **combined installed state after Passes 1–7**, not an isolated ZIP. Later ZIPs contain only changed files; install them in sequence. Preserve existing package directory layout (`flovopy/seisanio`, `flovopy/research/mvo`, `flovopy/enhanced`).

## 4. SEISAN archive discovery and reading

```python
from obspy import UTCDateTime
from flovopy.seisanio.core.seisanarchive import SeisanArchive

archive = SeisanArchive(
    '/path/to/SEISAN',
    db_cont='DSNC_',   # example; use actual archive DB name
    db_event='MVOE_',  # example; use actual archive DB name
)
t0 = UTCDateTime('2001-01-01')
t1 = UTCDateTime('2001-01-03')

for path in archive.iter_waveform_files(t0, t1, db='DSNC_'):
    stream = archive.read_waveform_file(path)
    print(path, len(stream))

for sfile_path in archive.iter_sfiles(t0, t1, db='MVOE_', mainclass='*'):
    print(sfile_path)
```

`SeisanArchive` also provides `iter_events()` and `iter_event_waveforms()` for S-file-driven workflows. Check their actual signatures before using options beyond those shown. The filename-based discovery is a *candidate selector*, not an authoritative waveform-time index: actual sample times are read and clipped later. The lookback is finite; recordings starting more than the configured lookback before the query may be missed. Nonstandard SAC/MiniSEED filenames may not be discoverable with the current SEISAN filename parser.

### Continuous versus triggered archives

Treat them as separate source collections even if both are ultimately converted into SDS. Do **not** automatically merge event-triggered and continuous records into a single canonical SDS tree until overlap, timing, and precedence policies are defined. Preserve the original REA↔WAV associations outside the SDS file naming convention.

## 5. Waveform conversion to SDS

### 5.1 Generic conversion

```python
from flovopy.seisanio.conversion import archive_to_sds

summary = archive_to_sds(
    archive,
    t0, t1,
    '/path/to/output/SDS',
    db='DSNC_',
    write_mode='merge',
    batch_files=32,
    lookback_days=1,
    return_counts=True,
    verbose=True,
)
print(summary)
```

Alternatively, the convenience function constructs the generic archive:

```python
from flovopy.seisanio.conversion import seisan_to_sds

summary = seisan_to_sds(
    '/path/to/SEISAN', '/path/to/output/SDS', t0, t1,
    db_cont='DSNC_', return_counts=True,
)
```

**Result when `return_counts=True`:** `files_read`, `files_failed`, `traces_read`, `days_written`, `sds_files_written`, `failed_files`. `files_read` counts source files that yielded retained samples; it is not necessarily the number of files discovered. These are processing counts, not a validated completeness inventory.

**Behavior:** reads each candidate, optionally transforms, clips to `[starttime, endtime)`, batches traces, calls `EnhancedSDSClient.write_stream()`, and reports failures. Source files are not changed. `write_mode='merge'` is the normal mode. `write_mode='overwrite'` is explicitly rejected by the converter for incremental migration; `fail` is also accepted. The writer's optional preprocessing is controlled by `write_preprocess` (default `True`). For scientific preservation, review the exact writer preprocessing configuration before using it for canonical raw archives.

### 5.2 MVO historical corrections (opt-in)

```python
from flovopy.seisanio.mvo_transform import correct_mvo_stream

changes_by_file = []
summary = archive_to_sds(
    archive, t0, t1, '/path/to/output/SDS',
    db='DSNC_',
    waveform_transform=correct_mvo_stream,
    transform_audit=lambda path, changes: changes_by_file.append((path, changes)),
    manifest_path='/path/to/logs/seisan_conversion.jsonl',
    return_counts=True,
)
```

`correct_mvo_stream()` copies the stream and calls `flovopy.research.mvo.mvo_ids.fix_trace_mvo()` on each trace. That historical function can change **NSLC, start time, and sampling rate**. The converter's audit captures before/after ID, start time, and sampling rate for changed traces. It does not yet capture a named correction-rule ID, confidence/evidence, or all data transformations. Avoid combining the archive reader's `fixid=True` with `waveform_transform=correct_mvo_stream` unless you have confirmed that applying both is intentional and idempotent: corrections may otherwise be duplicated or absent from the transform audit.

The transform interface expects a Stream (or in-place mutation returning `None`) and currently requires the **same number and order of traces** so before/after records can be paired. It is not suitable for splitting/merging/reordering traces without further provenance design.

### 5.3 StationXML checks and provenance

```python
from obspy import read_inventory
inventory = read_inventory('/path/to/stations.xml')

summary = archive_to_sds(
    archive, t0, t1, '/path/to/output/SDS',
    db='DSNC_', inventory=inventory,
    manifest_path='/path/to/logs/conversion.jsonl',
    verify_writes=True,  # slow; use on small batches
    return_counts=True,
)
```

`validate_stream_inventory(stream, inventory)` checks each trace's corrected NSLC against StationXML at trace **start and end**, including response lookup. Statuses are `valid`, `missing_channel_or_epoch`, and `invalid_response`. Validation is **advisory**: absence of historical metadata does not reject otherwise readable waveforms. A `valid` result is not proof of correct calibration or the correct historical station assignment.

The append-only JSONL manifest contains source path, corrections, StationXML validation, trace summaries, status (`written`, `verified`, `write_failed`, or `read_or_transform_failed`), and SDS output paths when available. A successful write is not equivalent to a scientifically accepted canonical mapping. The current manifest is a processing log, not a deduplicated source-file inventory or transaction-safe relational provenance store.

`verify_writes=True` reads the writer-reported output paths and compares expected samples at expected timestamps. This can be expensive and is not a comprehensive archive-level conflict or completeness audit. In particular, read-back verification may need special care when writer preprocessing, rate harmonization, or cross-day splitting alters the exact representation.

### 5.4 Conversion safety checklist

1. Work on a small time range and a separate output SDS tree first.
2. Retain immutable source WAV/REA/AEF and station metadata.
3. Decide whether to apply MVO corrections; document `fixid` versus transform usage.
4. Inspect `write_preprocess`, merge policy, sample-rate harmonization, and gap filling.
5. Enable JSONL provenance; inspect every failure and advisory validation result.
6. Read SDS output back using `EnhancedSDSClient`; compare trace IDs, times, sample counts, and selected sample values.
7. Do not interpret `files_failed == 0` as proof that every source file was discovered.

## 6. Nordic S-file and QuakeML conversion

```python
from flovopy.seisanio.catalog import (
    convert_archive_catalog, write_catalog_result, to_enhanced_catalog,
)
from flovopy.research.mvo.mvosfile import MVOSfile

result = convert_archive_catalog(
    archive, t0, t1,
    db='MVOE_',
    parser=MVOSfile,
    mainclass='*',
    enhanced=True,
    strict=False,
)
print(len(result.catalog.events), len(result.failures))

write_catalog_result(
    result,
    '/path/to/output/events.xml',
    metadata_path='/path/to/output/events.metadata.jsonl',
    failures_path='/path/to/output/events.failures.jsonl',
)

enhanced_catalog = to_enhanced_catalog(result)
```

`convert_archive_catalog()` returns `CatalogConversionResult` with `catalog` (ObsPy Catalog), `records` (EnhancedEvent instances when requested), `source_paths`, and `failures`. The default parser is generic `Sfile`; pass `MVOSfile` explicitly for Montserrat. `strict=True` raises the first parsing error; default `False` records failures and continues. The older `seisan_to_catalog()`, `iter_parsed_sfiles()`, and `write_quakeml()` entry points remain available.

**QuakeML stores standard event fields** (subject to actual parser support): event identifiers, picks, origins, magnitudes, arrivals, etc. **MVO-specific data are not automatically preserved in standard QuakeML.** The optional JSONL metadata sidecar serializes `EnhancedEvent.meta`, including available source references, analyst/classification data, and AEF rows. `metadata_path` requires enhanced records for every successfully converted event; it does not invent missing metadata. Keep S-files and sidecars together for auditability.

### 6.1 EnhancedEvent and EnhancedCatalog

`MVOSfile.to_enhancedevent(stream=None)` now constructs `EnhancedEventMeta` and uses `EnhancedEvent.wrap(self.eventobj, meta=meta, stream=stream)`. This is the preferred bridge from the MVO parser to the enhanced scientific-event representation. Waveform samples live in the transient enhanced-event stream, **not** in QuakeML. `to_enhanced_catalog(result)` creates an `EnhancedCatalog` from enhanced records. Pass 6 tests cover object identity between its events and records, but not a complete event-processing workflow.

Do not add arbitrary `.stream` attributes to plain ObsPy Event objects; this previously generated an ObsPy warning. `EnhancedEvent.wrap` was changed to attach transient waveform state without triggering that warning.

## 7. Integration auditing (Pass 7)

```python
from flovopy.seisanio.integration_audit import audit_sfiles, write_audit_json
from flovopy.research.mvo.mvosfile import MVOSfile

report = audit_sfiles(
    ['/path/to/REA/MVOE_/2001/01/EXAMPLE.SFILE'],
    parser=MVOSfile,
    enhanced=True,
)
write_audit_json(report, '/path/to/output/sfile_audit.json')
print(report['quakeml_roundtrip'])
```

`audit_catalog_roundtrip(catalog)` writes and reloads temporary QuakeML, comparing **event IDs, pick IDs, origin IDs, magnitude IDs, arrival counts, and event type**. It does **not** compare pick times, phase names, location coordinates, depth, magnitudes, uncertainties, arrival associations, comments, or AEF values numerically. The `audit_sfiles()` report includes parsed events, failures, enhanced metadata, and a structural round-trip result. A structural pass is necessary but insufficient for scientific acceptance.

Test commands (from repository root, adjust paths to your checkout):

```bash
pytest -q \
  tests/test_seisan_conversion_pass2.py \
  tests/test_seisan_conversion_pass3.py \
  tests/test_seisan_conversion_pass4.py \
  tests/test_seisan_catalog_pass5.py \
  tests/test_seisan_catalog_pass6.py

pytest -q tests/test_seisan_integration_pass7.py -rs

FLOVOPY_MVO_SFILE=/absolute/path/to/real/S-file \
  pytest -q tests/test_seisan_integration_pass7.py -rs
```

Reported baseline is **17 passing tests** for Passes 2–6. Pass 7's real-data test is conditional and is not yet a demonstrated pass. Running unqualified `pytest -q` in this repository previously hit unrelated legacy `research/tests` and `stationmetadata/test_module.py` collection errors; do not interpret those as SEISAN conversion regressions. Use targeted tests until the repository's test collection is cleaned up.

## 8. MVO package: targeted future verification

Do not rewrite the historical parsers without representative fixtures. Build a small **read-only, version-controlled fixture set** (where data permissions allow), with expected values manually checked against original source records.

| Component | Specific tests needed | Acceptance criterion |
|---|---|---|
| `mvosfile.MVOSfile` | Early/late S-file variants, multiple waveform references, analyst/class/subclass, missing optional records | Parsed values match S-file columns and known MVO conventions |
| MVO `aeffile.AEFfile` | Standalone versus embedded AEF; frequency bins, units, trigger/averaging windows, malformed/missing rows | Numerical values, bins, units and provenance preserved |
| `mvo_ids` | Station aliases, short/long-period channels, Y2K fixes, sample-rate correction, repeated correction | Before/after trace headers match documented ground truth; corrections idempotent where intended |
| `MVOSeisanArchive` | Continuous/event file iteration and native SEISAN/SAC/MiniSEED reading | No omitted or duplicated candidates in known test windows |
| `MVOSfile.to_enhancedevent` | Meta and waveform associations | Event fields, source paths and AEF rows match original; no ObsPy warnings |
| `EnhancedEvent` | Compute actual per-trace metrics on calibrated and uncalibrated waveforms | Units and numerical results validated against reference calculations |
| `EnhancedCatalog` | Multiple events, repeated imports, serialization boundaries | Stable event associations and no unexpected loss of metadata |
| `EnhancedSDSClient` | Cross-midnight traces, overlaps/conflicts, gaps, duplicate samples, mixed rates | Explicit merge policy; exact or documented transformed read-back |
| StationXML | Historical channel epochs, response availability and conflicts | Missing response distinguished from invalid identity; no guessed calibration |
| QuakeML | Picks, phases, origin coordinates/depth, magnitudes, preferred solutions | Numeric and semantic round-trip comparison, not merely object counts |

**Suggested fixture matrix:** at least one event with location/magnitude/picks, one with embedded AEF, one with external AEF, one with DSN+ASN waveform associations, one early legacy filename/trace-ID case, one Y2K/sampling-rate anomaly, one missing waveform reference, and one event whose waveform overlaps continuous data. Include a native SEISAN waveform and examples of SAC/MiniSEED where present.

### 8.1 Known technical limitations / review points

- **Discovery:** filename lookback is heuristic; nonstandard filenames and very long recordings need header-based indexing.
- **Corrections:** transform audit tracks changed header triples, not exact rule provenance or mapping confidence. Avoid unreviewed bulk corrections.
- **Writer behavior:** batching and `merge` do not by themselves resolve scientifically conflicting samples; writer preprocessing must be understood before defining canonical SDS.
- **Manifest:** append-only source-file logs are not a database, and batch-level write failures can affect multiple sources.
- **Catalog:** MVO metadata in sidecars must be joined back to event IDs; stable IDs across reparses are not yet established.
- **QuakeML:** existing round-trip audit is structural; numerical correctness is untested.
- **Station metadata:** `valid` at endpoints is not full instrument history validation.
- **Tests:** synthetic tests pass; real historical waveform/S-file/AEF integration is outstanding.

## 9. FLOVOpy boundary versus Montserrat modernization project

**Keep in FLOVOpy:** generic SEISAN readers, MVO adapters, trace correction functions, SDS and QuakeML conversion, validation, enhanced scientific objects, SAM/spectral/event metric algorithms, and reusable interfaces for callbacks and provenance.

**Build in the separate Montserrat project:** complete inventory and database; migration scheduling/checkpointing; human-reviewed canonical identity maps; station/channel epoch reconstruction; archive-wide deduplication; event-to-waveform-to-SDS associations; reprocessing and product dependency tracking; project-specific deployment, QA dashboards and publication datasets.

Avoid embedding a project-specific SQLite schema into `seisanio` now. The future database should consume stable FLOVOpy outputs (source file metadata, corrected traces, conversion manifests, catalog/sidecars, and metric records) via a documented adapter.

## 10. Proposed modernization data model (future design, not implemented)

**Level 0 — immutable sources:** `source_archive`, `source_file` (path, size, checksum, type, original DB), `source_trace` (original NSLC, times, sample rate), `sfile`, `aef_record`, `source_event_waveform_link`.

**Level 1 — canonical metadata and products:** `trace_mapping_rule` (version, effective interval, evidence, confidence), `trace_mapping_application` (original/corrected headers), `station_epoch`, `channel_epoch`, `response_source`, `conversion_run`, `conversion_output` (SDS path and trace/day span), `event`, `pick`, `arrival`, `origin`, `magnitude`, `event_classification`, `event_source_link`. Export SDS, StationXML and QuakeML as durable interoperable products.

**Level 2 — derived measurements:** `metric_definition` (units, bands, statistic), `processing_run` (code/config/version), `continuous_metric_window`, `event_trace_metric`, `historical_aef_measurement`, `spectrogram_product` (external file path), `quality_flag`. For high-volume time series or arrays, use external Parquet/NPZ/Zarr or similar storage indexed from the relational database, rather than one SQLite row per sample.

**Level 3 — interpretations:** `asl_run`, `asl_observation`, `asl_solution`, `asl_residual`, `relocation_run`, `relocated_origin`, `classification_run`, `ml_dataset`, `ml_model`, `ml_prediction`. Preserve multiple competing solutions and their input versions rather than overwriting old results.

### 10.1 Essential identity and provenance rules

1. **Stable source identity:** derive an internal file/event key from archive ID and relative source path; do not rely solely on mutable origin time or auto-generated ObsPy IDs.
2. **Time-bounded mapping:** historical original NSLC → canonical SEED-compliant NSLC may depend on date, sensor and sample rate. Never assume one global string replacement.
3. **Distinct corrections:** ID correction, timestamp correction, sample-rate correction, instrument-response application and signal processing are separate operations with separate provenance.
4. **Explicit associations:** S-file ↔ original WAV file ↔ original trace ↔ corrected trace ↔ SDS segment ↔ StationXML epoch ↔ derived metric.
5. **Versioned science:** changing a station response or mapping rule should identify which derived metrics/ASL results need recalculation.
6. **Uncertainty allowed:** preserve unknown responses and ambiguous station IDs as unresolved rather than fabricating certainty.
7. **Continuous and event coexistence:** establish overlap/deduplication policy before publishing one consolidated SDS archive.

## 11. Suggested staged future plan

**Stage A — establish historical regression corpus (next FLOVOpy task).** Gather 8–15 representative S-files, AEF files and WAV files plus manually checked expected values. Run Pass 7; extend to numeric picks/origins/magnitudes/AEF validation. Fix only observed parser defects.

**Stage B — harden FLOVOpy conversion.** Support header-based discovery for SEISAN/SAC/MiniSEED, verify ID correction rules and writer transformations, produce stable machine-readable conversion records, test cross-day overlaps and failures. Maintain a small CLI thin wrapper; keep science in library functions.

**Stage C — design Montserrat database.** Inventory every file without waveform decoding first; assign source identities and checksums; ingest original trace headers and S-file/AEF associations. Define versioned correction tables and channel epochs.

**Stage D — canonicalize and migrate.** Human-review ambiguous mappings, reconstruct StationXML, convert continuous/event data with explicit precedence, compare source and SDS coverage, migrate events to QuakeML plus MVO metadata, and produce audit reports.

**Stage E — scientific measurements.** Run continuous SAM/DR and spectral archives plus EnhancedEvent event metrics with calibrated units and quality flags. Preserve historical AEF values alongside new measurements; link every observation to a processing run and source trace.

**Stage F — research applications.** Build reusable ASL observation sets and inversion results; event relocation and ML feature/label datasets, all versioned and queryable. IceWeb can later consume canonical waveforms and metrics without owning migration logic.

## 12. Handover checklist

- [x] Generic waveform conversion with bounded batches and clipping implemented.
- [x] Optional MVO trace correction adapter implemented.
- [x] Advisory StationXML validation and append-only JSONL manifest implemented.
- [x] Catalog/QuakeML and enhanced metadata sidecar implemented.
- [x] EnhancedEvent/EnhancedCatalog compatibility tests passing.
- [x] **17** cumulative tests for Passes 2–6 reported passing.
- [ ] Pass 7 tests run and reported, including a real historical S-file.
- [ ] Native SEISAN/SAC/MiniSEED historical fixture coverage.
- [ ] Numeric validation of AEF, picks, origins, magnitudes and event metrics.
- [ ] Full trace-ID mapping review against historical station metadata.
- [ ] Archive-level completeness, deduplication, and conflict policy.
- [ ] Montserrat database and migration orchestration (future separate project).

### Source basis

This guide was checked against the uploaded original `seisanio.zip`, `mvo.zip`, `enhanced.zip`, and refactor bundles `flovopy_seisan_refactor_pass2.zip` through `pass7.zip`, including their Python module signatures and delivered test files. It distinguishes tested interface behavior from future project design. Paths and DB names in examples are illustrative, not verified against the actual Montserrat filesystem.
