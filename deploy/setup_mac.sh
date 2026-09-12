#!/usr/bin/env bash
# Backward-compatible Mac installer entrypoint.
#
# The canonical host deployment is product.host_install, which owns the
# persistent runtime root, exact-SHA pinning, launchd supervision, bounded
# restart policy, power/memory preflight, and PAPER/SHADOW safety checks.
# Keep this historical command working, but route it through the canonical
# installer so an operator cannot accidentally install the obsolete two-agent
# launchd topology.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[QuantTerm] deploy/setup_mac.sh is a compatibility wrapper."
echo "[QuantTerm] Using canonical persistent host installer: scripts/install_quantterm_host.sh"
exec bash "$ROOT/scripts/install_quantterm_host.sh" "$@"
