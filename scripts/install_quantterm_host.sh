#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

case "${QT_RUNTIME_ROOT_REQUIRE_EXISTING:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    exec "$PY" -m product.host_install_existing "$@"
    ;;
  *)
    exec "$PY" -m product.host_install install "$@"
    ;;
esac
