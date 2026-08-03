#!/usr/bin/env bash
# Low-CPU QuantTerm for older Macs (e.g. MacBook Air Early 2015).
# Same terminal + sniper + Telegram; far less background work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=apply_low_power_env.sh
source "$ROOT/scripts/apply_low_power_env.sh"

exec bash "$ROOT/scripts/run_quantterm_complete.sh"
