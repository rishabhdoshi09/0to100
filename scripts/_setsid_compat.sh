# QuantTerm process-session compatibility.
# Linux launchers use util-linux `setsid` so Vite owns a killable process group.
# macOS does not ship that command. Python's os.setsid() is available on both.
# Source this file from the canonical launchers so `bash scripts/run_quantterm_complete.sh`
# starts the desk without a macOS-only wrapper.
if ! command -v setsid >/dev/null 2>&1; then
  setsid() {
    local py
    py="$(command -v python3 || command -v python || true)"
    if [[ -z "$py" ]]; then
      echo "[STACK] setsid is unavailable and no python interpreter was found to start a session." >&2
      return 127
    fi
    exec "$py" - "$@" <<'PY'
import os
import sys

if len(sys.argv) < 2:
    raise SystemExit("setsid compatibility launcher requires a command")
command = sys.argv[1]
if command == "npm":
    command = os.environ.get("QT_NPM_BIN") or command
argv = [command, *sys.argv[2:]]
os.setsid()
os.execvpe(command, argv, os.environ)
PY
  }
  export -f setsid
fi
