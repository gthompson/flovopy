#!/usr/bin/env python3
"""Backward-compatible launcher for the FLOVOpy EPIC SDS-to-BUD exporter."""
from flovopy.ingest.sds_to_bud import main
if __name__ == '__main__':
    raise SystemExit(main())
