#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  exec "$PY" -m product.launchd_control start "$@"
fi
exec "$PY" -m product.host_install start "$@"
