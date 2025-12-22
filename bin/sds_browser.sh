#!/usr/bin/env bash
#
# Launch the flovopy SDS browser
#
# Works when:
# - flovopy is installed (pip / pip -e)
# - OR flovopy source tree is present and activated via PYTHONPATH
#

set -euo pipefail

# Optional: allow user to override python executable
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Resolve script directory (robust to symlinks)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

# If flovopy is not installed, try to infer project root
if ! $PYTHON_BIN - <<EOF >/dev/null 2>&1
import flovopy
EOF
then
  # Assume bin/ lives inside flovopy repo
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
fi

# Launch browser
exec "$PYTHON_BIN" -m flovopy.sds.browser "$@"
