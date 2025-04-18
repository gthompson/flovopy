#!/bin/bash

# Set up log directory
LOGDIR="./logs"
mkdir -p "$LOGDIR"

conda activate flovopy-env
SEISAN_TOP="/data/SEISAN_DB"
DB="MVOE_"
SQLDB="${HOME}/public_html/seiscomp_like.sqlite"
PATH=${PATH}:${HOME}/Developer/flovopy/flovopy/wrappers/MVOE_conversion_pipeline


echo "[INFO] Starting extended SeisComP DB pipeline"

# --- Step 1: Create schema
echo "[INFO] Running b10_create_extseiscompdb.py..."
python b10_create_extseiscompdb.py > "$LOGDIR/b10_create_extseiscompdb.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed during schema creation. Check $LOGDIR/b10_create_extseiscompdb.log"
    exit 1
fi
echo "[INFO] Schema created successfully."

# --- Step 2: Index cleaned MiniSEED files
echo "[INFO] Running b11_cleaned2wfdisc.py..."
python b11_cleaned2wfdisc.py > "$LOGDIR/b11_cleaned2wfdisc.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed during MiniSEED indexing. Check $LOGDIR/b11_cleaned2wfdisc.log"
    exit 1
fi
echo "[INFO] Cleaned waveform indexing completed."

# --- Step 3: Load QML + JSON metadata
echo "[INFO] Running b12_qmljson2extseiscompdb.py..."
python b12_qmljson2extseiscompdb.py > "$LOGDIR/b12_qmljson2extseiscompdb.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed during QML/JSON import. Check $LOGDIR/b12_qmljson2extseiscompdb.log"
    exit 1
fi
echo "[INFO] Event metadata ingestion completed."

echo "[SUCCESS] Extended SeisComP DB pipeline finished."
