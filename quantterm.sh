#!/usr/bin/env bash
# Tiny exec wrapper. All orchestration lives in scripts/run_quantterm_complete.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"
