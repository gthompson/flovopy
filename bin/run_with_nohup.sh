#!/bin/bash

# Usage: ./run_with_nohup.sh /path/to/your_script.py

SCRIPT_PATH="$1"
if [ -z "$SCRIPT_PATH" ]; then
  echo "❌ Please provide a Python script path."
  exit 1
fi

SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$HOME/work"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

echo "🚀 Starting $SCRIPT_NAME with nohup..."
echo "📄 Log file: $LOG_FILE"

while true; do
  nohup python "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
  PID=$!

  echo "👷 Running as PID $PID..."

  # Wait for the process to finish
  wait $PID
  EXIT_CODE=$?

  # Check for successful termination
  if tail -n 5 "$LOG_FILE" | grep -q "OK"; then
    echo "$(date): ✅ Script finished successfully (OK found in log)."
    break
  else
    echo "$(date): ⚠️ Script may have crashed or exited early (no OK found). Restarting in 60 seconds..."
    sleep 60
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.log"
    echo "🆕 New log file: $LOG_FILE"
  fi
A
A
A
A
done
