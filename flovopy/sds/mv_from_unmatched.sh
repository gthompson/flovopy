#!/bin/bash
# move_unmatched.sh
# Usage:
#   ./move_unmatched.sh <SDS_PATH> <YEAR> <NET> <STA> [DOY|DOY1-DOY2] [--dry-run]
#   ./move_unmatched.sh <SDS_PATH> <YEAR> <NET.STA> [DOY|DOY1-DOY2]   [--dry-run]
# Examples:
#   ./move_unmatched.sh /data/SDS 2016 1R BCHH --dry-run
#   ./move_unmatched.sh /data/SDS 2016 AM.R37BE 084-086
#   ./move_unmatched.sh /data/SDS 2016 1R BCHH 084

set -euo pipefail

usage() {
  echo "Usage:"
  echo "  $0 <SDS_PATH> <YEAR> <NET> <STA> [DOY|DOY1-DOY2] [--dry-run]"
  echo "  $0 <SDS_PATH> <YEAR> <NET.STA> [DOY|DOY1-DOY2]   [--dry-run]"
  exit 1
}

# ---- peel off trailing --dry-run if present ----
DRYRUN=""
if [ "${@: -1}" = "--dry-run" ]; then
  DRYRUN="--dry-run"
  set -- "${@:1:$(($#-1))}"
  echo "### DRY-RUN MODE ###"
fi

# ---- positional args ----
if [ $# -lt 3 ] || [ $# -gt 5 ]; then usage; fi

SDS_PATH="$1"
YEAR="$2"

# NET/STA (either split or combined)
NET=""; STA=""
if [ $# -ge 4 ] && [[ "$3" != *.* ]]; then
  NET="$3"; STA="$4"
  shift 4
else
  IFS='.' read -r NET STA <<< "$3"
  shift 3
fi
[ -z "${NET:-}" ] && usage
[ -z "${STA:-}" ] && usage

# Optional DOY/DOY-range
DOY_SPEC="${1:-}"
if [ -n "$DOY_SPEC" ] && [[ "$DOY_SPEC" =~ ^[0-9]{3}(-[0-9]{3})?$ ]]; then
  shift 1
else
  DOY_SPEC=""
fi

# sanity on leftovers
[ $# -gt 0 ] && usage

SRC="${SDS_PATH}/unmatched/${YEAR}/${NET}/${STA}/"
DST="${SDS_PATH}/${YEAR}/${NET}/${STA}/"

if [ ! -d "$SRC" ]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi

command -v rsync >/dev/null 2>&1 || { echo "rsync not found"; exit 1; }

# ensure destination dir exists (best-effort)
mkdir -p "$DST" 2>/dev/null || true

# decide if we can remove sources afterwards
ALLOW_REMOVE=0
if [ -w "$DST" ] && [ -z "$DRYRUN" ]; then
  ALLOW_REMOVE=1
fi

RSYNC_OPTS="-av --ignore-existing"
[ -n "$DRYRUN" ] && RSYNC_OPTS="$RSYNC_OPTS --dry-run"
[ $ALLOW_REMOVE -eq 1 ] && RSYNC_OPTS="$RSYNC_OPTS --remove-source-files"

# If DOY specified, build include/exclude rules so only matching days move
RSYNC_FILTERS=""
if [ -n "$DOY_SPEC" ]; then
  # expand to a space-separated list of DOYs
  if [[ "$DOY_SPEC" =~ ^([0-9]{3})-([0-9]{3})$ ]]; then
    START=${BASH_REMATCH[1]}
    END=${BASH_REMATCH[2]}
    if (( 10#$END < 10#$START )); then
      echo "Invalid DOY range: $DOY_SPEC" >&2; exit 1
    fi
    DOYS=()
    for ((d=10#$START; d<=10#$END; d++)); do
      DOYS+=($(printf "%03d" "$d"))
    done
  else
    DOYS=("$DOY_SPEC")
  fi

  # include dirs, include only files for selected days, exclude the rest
  RSYNC_FILTERS+=" --prune-empty-dirs --include='*/'"
  for D in "${DOYS[@]}"; do
    RSYNC_FILTERS+=" --include='*.D.${YEAR}.${D}'"
  done
  RSYNC_FILTERS+=" --exclude='*'"
fi

echo "Source:      $SRC"
echo "Destination: $DST"
[ -n "$DOY_SPEC" ] && echo "DOY filter:  $DOY_SPEC"
if [ -n "$DRYRUN" ]; then
  echo "Mode:        DRY RUN (no changes)"
elif [ $ALLOW_REMOVE -eq 1 ]; then
  echo "Destination writable: yes (source files removed after copy)"
else
  echo "Destination writable: no (source files NOT removed)"
fi

# run rsync (quote filters via eval to preserve single-quoted patterns)
if [ -n "$RSYNC_FILTERS" ]; then
  eval rsync $RSYNC_OPTS $RSYNC_FILTERS \""$SRC"\" \""$DST"\"
else
  rsync $RSYNC_OPTS "$SRC" "$DST"
fi
RS=$?

if [ $RS -ne 0 ]; then
  echo "rsync failed with status $RS; no source deletion performed." >&2
  exit $RS
fi

# clean up empty source dirs only if we actually removed source files
if [ $ALLOW_REMOVE -eq 1 ]; then
  find "$SRC" -type d -empty -delete
fi
