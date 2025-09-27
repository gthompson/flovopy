#!/usr/bin/env bash
set -euo pipefail

PY=/opt/anaconda3/envs/flovopy_plus/bin/python   # or just: python
SCRIPT="$HOME/Developer/flovopy/flovopy/dem/channel_finder.py"

DEM="$HOME/Dropbox/PROFESSIONAL/DATA/WadgeDEMs/conversions/DEM_1999_WGS84_rotated.tif"
OUTDIR="$HOME/Dropbox/BRIEFCASE/SSADenver/channel_finder"

$PY "$SCRIPT" \
  --dem "$DEM" \
  --outdir "$OUTDIR" \
  --prep \
  --fa-percentile 96 \
  --min-cells 700 \
  --top-n 100 \
  --min-len-m 500 \
  --flip horizontal \
