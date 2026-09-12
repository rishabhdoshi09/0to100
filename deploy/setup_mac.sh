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

# Clean up agents created by the pre-host-supervisor installer before handing
# control to the canonical installer. These are migration identifiers only; this
# script never recreates their plists. The content probes make cleanup narrowly
# target the old QuantTerm files rather than unrelated user launch agents.
LEGACY_UI="$HOME/Library/LaunchAgents/com.quantterm.ui.plist"
LEGACY_AUTONOMY="$HOME/Library/LaunchAgents/com.quantterm.autonomy.plist"
for legacy in "$LEGACY_UI" "$LEGACY_AUTONOMY"; do
  [[ -f "$legacy" ]] || continue
  if grep -Eq 'run_quantterm_complete\.sh|QT_NONINTERACTIVE|<string>autonomy</string>' "$legacy" 2>/dev/null; then
    launchctl bootout "gui/$(id -u)" "$legacy" 2>/dev/null || launchctl unload "$legacy" 2>/dev/null || true
    rm -f "$legacy"
    echo "[QuantTerm] Removed obsolete launchd agent: $legacy"
  fi
done

echo "[QuantTerm] deploy/setup_mac.sh is a compatibility wrapper."
echo "[QuantTerm] Using canonical persistent host installer: scripts/install_quantterm_host.sh"
exec bash "$ROOT/scripts/install_quantterm_host.sh" "$@"
