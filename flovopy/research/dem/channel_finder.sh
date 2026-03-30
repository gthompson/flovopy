#!/usr/bin/env bash
set -euo pipefail

PY=/opt/anaconda3/envs/flovopy_plus/bin/python   # or just: python
SCRIPT="$HOME/Developer/flovopy/flovopy/dem/channel_finder2.py"

#DEM="$HOME/Dropbox/PROFESSIONAL/DATA/WadgeDEMs/conversions/DEM_1999_WGS84_rotated.tif"
#DEM="$HOME/Dropbox/BRIEFCASE/SSADenver/wgs84_s0.4_3_clean_shifted.tif"
DEM="$HOME/Dropbox/MONTSERRAT_DEM_WGS84_MASTER.tif"
OUTDIR="$HOME/Dropbox/BRIEFCASE/SSADenver/channel_finder"

$PY "$SCRIPT" \
  --dem "$DEM" \
  --outdir "$OUTDIR" \
  --prep \
  --fa-percentile 96 \
  --min-cells 700 \
  --top-n 100 \
  --min-len-m 500 \
#  --flip horizontal \
