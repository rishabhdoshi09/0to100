#!/usr/bin/env bash
# Low-CPU QuantTerm for older Macs — SAME services as complete (including
# market-scan bootstrap that feeds autopilot). Only background CPU is reduced.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=apply_low_power_env.sh
source "$ROOT/scripts/apply_low_power_env.sh"

exec bash "$ROOT/scripts/run_quantterm_complete.sh"
