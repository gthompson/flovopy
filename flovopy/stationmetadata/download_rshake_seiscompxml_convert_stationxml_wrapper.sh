#!/usr/bin/env bash
# Wrapper to set SeisComP environment and run the Raspberry Shake metadata download/conversion script

# === CONFIGURE THESE ===
SEISCOMP_ROOT="/home/sysop/seiscomp"   # Path where SeisComP is installed
PYTHON_SCRIPT="download_rshake_seiscompxml_convert_stationxml.py"
PYTHON_BIN="$(command -v python3)"     # Change to your desired python executable

# === SET ENVIRONMENT ===
export SEISCOMP_ROOT
export PATH="$SEISCOMP_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$SEISCOMP_ROOT/lib:$LD_LIBRARY_PATH"

# Optional: sanity check that fdsnxml2inv is available
if ! command -v fdsnxml2inv >/dev/null 2>&1; then
    echo "[ERROR] fdsnxml2inv not found in PATH after setting SEISCOMP_ROOT=$SEISCOMP_ROOT"
    echo "        Check your SeisComP installation."
    exit 1
fi

# === RUN PYTHON SCRIPT ===
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

echo "[INFO] Running $PYTHON_SCRIPT ..."
"$PYTHON_BIN" "$PYTHON_SCRIPT" "$@"
