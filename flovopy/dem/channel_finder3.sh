PY=/opt/anaconda3/envs/flovopy_plus/bin/python
SCRIPT="$HOME/Developer/flovopy/flovopy/dem/channel_finder3.py"

DEM="$HOME/Dropbox/MONTSERRAT_DEM_WGS84_MASTER.tif"
OUTDIR="$HOME/Dropbox/BRIEFCASE/SSADenver/channel_finder5"

$PY "$SCRIPT" \
  --dem "$DEM" \
  --outdir "$OUTDIR" \
  --prep \
  --auto-cut \
  --sea-level 0 \
  --fa-percentile 96 \
  --min-cells 700 \
  --top-n 100 \
  --min-len-m 500