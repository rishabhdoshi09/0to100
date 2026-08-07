# Shared helpers for QuantTerm local stack scripts (macOS + Linux).
# Sourced by run_quantterm.sh and run_quantterm_complete.sh — not executed directly.
#
# IMPORTANT: Never capture stack_start_or_reuse_uvicorn via $(...). Backgrounding a
# process inside command substitution starts it in a subshell that exits immediately
# and kills the child (SIGHUP) — that surfaces as "exited during startup".
# Callers must use STACK_UVICORN_PID / STACK_UVICORN_RC after a direct call.

STACK_UVICORN_PID=""
STACK_UVICORN_RC=0

stack_health_ok() {
  local url="$1"
  local timeout="${2:-5}"
  curl -fsS --max-time "$timeout" "$url" >/dev/null 2>&1
}

stack_pids_on_port() {
  local port="$1"
  lsof -ti ":${port}" 2>/dev/null || true
}

stack_port_listening() {
  local port="$1"
  [[ -n "$(stack_pids_on_port "$port")" ]]
}

stack_free_port() {
  local port="$1"
  local label="${2:-service}"
  local pids
  pids="$(stack_pids_on_port "$port")"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "[STACK] Port ${port} still in use by ${label} (pid(s): ${pids}) — stopping stale process(es)…" >&2
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(stack_pids_on_port "$port")"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    sleep 0.5
  fi
}

# Stop listeners on one or more ports (reaps npm/vite/uvicorn orphans by bind port).
stack_stop_ports() {
  local port
  for port in "$@"; do
    stack_free_port "$port" "port-${port}"
  done
}

# Start uvicorn or reuse an already-healthy listener on the same port.
# Sets STACK_UVICORN_PID (empty when reusing) and STACK_UVICORN_RC:
#   0 = started a new process (STACK_UVICORN_PID set)
#   2 = reused an existing healthy API (STACK_UVICORN_PID empty)
# Return code matches STACK_UVICORN_RC.
stack_start_or_reuse_uvicorn() {
  local port="$1"
  local app="$2"
  local health_url="$3"
  local label="$4"

  STACK_UVICORN_PID=""
  STACK_UVICORN_RC=0

  if stack_health_ok "$health_url"; then
    echo "[STACK] ${label} already healthy at ${health_url} — reusing existing process." >&2
    STACK_UVICORN_RC=2
    return 2
  fi

  stack_free_port "$port" "$label"

  if stack_health_ok "$health_url"; then
    echo "[STACK] ${label} became healthy after freeing port ${port} — reusing." >&2
    STACK_UVICORN_RC=2
    return 2
  fi

  echo "[STACK] Starting ${label} at http://127.0.0.1:${port} (${app})…" >&2
  # Must run in the caller's shell (not inside $(...)) so the child survives.
  python -u -m uvicorn "${app}" --host 127.0.0.1 --port "${port}" &
  STACK_UVICORN_PID=$!
  STACK_UVICORN_RC=0
  return 0
}

stack_wait_for_health() {
  local health_url="$1"
  local pid="${2:-}"
  local label="${3:-API}"
  local attempts="${4:-240}"

  for _ in $(seq 1 "$attempts"); do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[STACK] ${label} exited during startup. Review errors above." >&2
      return 1
    fi
    if stack_health_ok "$health_url"; then
      return 0
    fi
    sleep 0.5
  done
  echo "[STACK] ${label} did not become ready at ${health_url} within $((attempts / 2))s." >&2
  return 1
}
