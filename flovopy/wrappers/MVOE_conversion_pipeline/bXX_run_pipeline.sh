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

: <<'COMMENT'
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

# --- Step 4: fix time strings that do not have 'Z' at end
echo "[INFO] Running b13_fix_time_strings.py..."
python b13_fix_time_strings.py > "$LOGDIR/b13_fix_time_strings.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed during time string fixing. Check $LOGDIR/b13_fix_time_strings.log"
    exit 1
fi
echo "[INFO] Time strings fixed."

# --- Step 5: link events and waveforms
echo "[INFO] Running b14_link_events_waveforms.py..."
python b14_link_events_waveforms.py > "$LOGDIR/b14_link_events_waveforms.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed during linking events and waveforms. Check $LOGDIR/b14_link_events_waveforms.log"
    exit 1
fi
echo "[INFO] Events and waveforms linked."


# --- Step 6: make dfile non-unique in event_classifications file, so each event can have multiple classifications
echo "[INFO] Running b15_make_dfile_nonunique.py..."
python b15_make_dfile_nonunique.py > "$LOGDIR/b15_make_dfile_nonunique.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed during making dfile non-unique. Check $LOGDIR/b15_make_dfile_nonunique.log"
    exit 1
fi
echo "[INFO] Made dfile non-unique."

# --- Step 7: study regionals so we can estimate station corrections
echo "[INFO] Running b16_study_regionals.py..."
python b16_study_regionals.py > "$LOGDIR/b16_study_regionals.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed while processing regionals. Check $LOGDIR/b16_study_regionals.log"
    exit 1
fi
echo "[INFO] Processed regionals."

# --- Step 8: estimate station corrections
echo "[INFO] Running b17_compute_station_corrections.py..."
python b17_compute_station_corrections.py > "$LOGDIR/b17_compute_station_corrections.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed while estimating station corrections. Check $LOGDIR/b17_compute_station_corrections.log"
    exit 1
fi
echo "[INFO] Estimated station corrections."
COMMENT

# --- Step 9: event waveform processing
echo "[INFO] Running b18_waveform_processing.py..."
python b18_waveform_processing.py > "$LOGDIR/b18_waveform_processing.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed while processing waveforms. Check $LOGDIR/b18_waveform_processing.log"
    exit 1
fi
echo "[INFO] Processed all waveforms."

echo "[SUCCESS] Extended SeisComP DB pipeline finished."
