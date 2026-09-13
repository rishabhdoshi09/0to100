#!/usr/bin/env bash
# macOS production entrypoint: verify/mount external runtime, then hand off to
# the canonical complete QuantTerm launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_STORAGE_PREFLIGHT_REQUIRED=1
bash "$ROOT/scripts/quantterm_storage_preflight.sh"

exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"
