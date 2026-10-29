# FloVoPy Post-Merge Cleanup

Date: 2026-10-29

## Background

The `feature/sds-seisan-modernization` branch was merged into
`main` following verification that:

- `main` was an ancestor of the feature branch.
- No commits on `main` were missing from the feature branch.
- A fast-forward merge was possible.
- The working tree was clean.
- Production tests: 109 passed, 1 failed, 1 skipped.

The modernization involved 71 changed files, including major
refactoring of SDS, SEISAN, SAM, spectrogram, and event-feature
processing.

## Priority 1: Resolve failing production test

Test:
flovopy/tests/test_features.py::test_pipeline_opt_in_and_legacy

Problem:
The test expects an external file:

pinatubo_features_v6/pipeline.py

This file was not found.

Actions:
- Determine whether the experimental Pinatubo pipeline exists elsewhere.
- Decide whether to integrate it into FloVoPy.
- Alternatively, make the test skip when the optional dependency is absent.
- Verify that the Malfante-inspired features work independently.
- Rerun the production test suite.

## Priority 2: Resolve legacy research test failures

Five tests fail during collection:

- test_asl_event.py
- test_fdsn2sds.py
- test_fdsn2sds2sam.py
- test_run_iceweb.py
- test_sds2sam.py

Problems:
- Missing obsolete.asl module.
- Imports referencing flovopy.wrappers, which does not exist.

Actions:
- Determine which tests cover functionality still supported.
- Update tests to use the modernized processing APIs.
- Archive or remove genuinely obsolete tests.
- Consider configuring pytest to run flovopy/tests by default.
- Establish a clean full-suite baseline.

## Priority 3: Audit deleted legacy wrappers

Deleted modules include:

- compute_sam_metrics.py
- fdsn2sds2sam.py
- reduce_metrics.py
- run_fdsn2sds.py
- run_iceweb.py
- run_sds2rsam.py
- sds2sam.py

Actions:
- Determine which workflows were replaced by archive_processor.py,
  archive_metrics.py, archive_spectrograms.py, and related modules.
- Check whether FDSN download functionality remains available.
- Check whether IceWeb processing functionality remains available.
- Check whether SDS-to-SAM workflows remain supported.
- Restore or replace useful functionality where necessary.
- Audit pipeline_runner.py and its reduce_metrics import.

## Priority 4: Fix pyproject.toml entry points

Current entry points include:

run-iceweb
run-asl
run-sds2rsam
run-sds2disp
run-fdsn2sds
run-seisan2sds
flovopy-ingest

Several reference flovopy.wrappers, which appears to be absent.

Actions:
- Verify every configured entry point.
- Correct module paths for supported commands.
- Remove commands that have been deliberately retired.
- Test commands following an editable installation.

## Priority 5: Review legacy backup files

Files:

- flovopy/enhanced/stream_old.py
- flovopy/processing/sam_old.py
- flovopy/processing/spectrograms_old.py

No textual references were found using git grep.

Actions:
- Confirm no dynamic imports or external dependencies.
- Compare with modern implementations.
- Remove unnecessary backup files after verification.
- Rely on Git history for preservation.

## Priority 6: Functional regression testing

Perform real-data smoke tests for:

- Reading SDS waveform archives.
- Reading SEISAN waveform databases.
- Converting SEISAN to SDS.
- Computing RSAM and VSAM metrics.
- EnhancedStream event processing.
- Spectrogram generation.
- Event-feature extraction.
- Pinatubo 1991 waveform analysis.

## Priority 7: General validation

Run:

python -m pytest flovopy/tests -q
python -m compileall -q flovopy
git diff --check HEAD~1 HEAD

Note: The final command checks only the most recent commit.
For a complete modernization diff, compare against the pre-merge
main commit, 209bfcc.

Review packaging, documentation, imports, and API compatibility.

## Current status

The modernization has been integrated into main, but the above
issues remain unresolved.

The observed test failures have not established a regression in
core processing functionality. Full regression testing remains
outstanding.
