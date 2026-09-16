#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

# macOS production uses the external-APFS runtime. A direct invocation of this
# generic wrapper must therefore fail closed by default on Darwin: if the volume
# is absent, the permissive installer could otherwise create a new internal
# /Volumes/... path and fork durable state. Linux keeps permissive first-install
# behavior because setup_server.sh legitimately initializes a new local runtime.
if [[ -n "${QT_RUNTIME_ROOT_REQUIRE_EXISTING+x}" ]]; then
  REQUIRE_EXISTING="$QT_RUNTIME_ROOT_REQUIRE_EXISTING"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  REQUIRE_EXISTING=1
else
  REQUIRE_EXISTING=0
fi

case "$REQUIRE_EXISTING" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      exec "$PY" -m product.host_install_existing_v2 "$@"
    fi
    exec "$PY" -m product.host_install_existing "$@"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      echo "[install] WARNING: permissive macOS bootstrap explicitly requested." >&2
      echo "[install] A missing runtime may be created; use deploy/setup_mac.sh for production." >&2
    fi
    exec "$PY" -m product.host_install install "$@"
    ;;
  *)
    echo "Invalid QT_RUNTIME_ROOT_REQUIRE_EXISTING=$REQUIRE_EXISTING" >&2
    exit 64
    ;;
esac
