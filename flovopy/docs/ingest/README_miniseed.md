# FLOVOpy generic MiniSEED ingestor addition

Copy `flovopy/ingest/miniseed.py` into your FLOVOpy checkout and `docs/ingest/Generic-MiniSEED.md` into the documentation directory. No changes to the existing instrument-specific modules are required. Run using `python -m flovopy.ingest.miniseed`.

The generic ingestor accepts any directory tree of MiniSEED, groups by actual UTC day, writes staging SDS via `EnhancedSDSClient`, and merges into a master SDS archive via the existing FLOVOpy transactional merger. See the guide for details and limitations.
