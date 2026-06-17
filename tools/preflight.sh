#!/usr/bin/env bash
# tools/preflight.sh — Pre-flight deployment gate (shell wrapper)
#
# Activates the project venv and delegates to tools/preflight.py.
# All flags are forwarded verbatim:
#
#   ./tools/preflight.sh                 # default run, warnings allowed
#   ./tools/preflight.sh --strict        # any warning = failure (CI gate)
#   ./tools/preflight.sh --no-color      # plain text (redirect-safe)
#   ./tools/preflight.sh --config path   # alternate pipeline.yaml
#
# Exit codes:
#   0  All required checks passed
#   1  One or more checks failed (or any warning with --strict)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "[preflight] ERROR: venv not found at $VENV" >&2
    echo "[preflight] Create it: python3.8 -m venv --system-site-packages .venv" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

exec python "$SCRIPT_DIR/preflight.py" "$@"
