# Pass 7 — Integration audit harness

This pass **does not change production parsers or converters**. It adds a read-only
integration audit and tests for QuakeML event IDs, picks, origins, magnitudes,
and arrival counts, along with source-file and enhanced metadata capture.

Install `seisanio/integration_audit.py` at `flovopy/seisanio/integration_audit.py`
and the test under your repository's `tests/` directory.

Run:

```bash
pytest -q tests/test_seisan_integration_pass7.py
```

The real-MVO test skips until you supply a representative S-file:

```bash
FLOVOPY_MVO_SFILE=/path/to/REA/MVOE_/2001/01/example.S200101 \
  pytest -q tests/test_seisan_integration_pass7.py -rs
```

This is **not yet a complete historical end-to-end test**: no representative
Montserrat S-file/AEF/waveform trio was supplied, and no SDS archive is
written by this harness. It does not assert equivalence of original AEF
measurements, corrected trace IDs, or waveform samples. Those checks require
known-good fixtures and explicit expected values, not fabricated examples.

A failed QuakeML audit flags discrepancies for investigation. The sidecar
retains additional EnhancedEvent metadata separately from QuakeML.
