#!/usr/bin/env bash
# Fail-closed storage gate for the macOS QuantTerm service.
#
# The production runtime may live on an APFS sparsebundle stored on an external
# disk.  No QuantTerm process may start until that exact storage chain is
# present, mounted, verified and writable.  In particular, this script never
# creates the canonical runtime path: a missing external volume must remain a
# hard failure instead of silently becoming a fresh local runtime.
set -euo pipefail

PREFIX="[STORAGE PREFLIGHT]"
fail() {
  echo "$PREFIX FAIL: $*" >&2
  exit 78
}
info() {
  echo "$PREFIX $*"
}

OS_NAME="${QT_PREFLIGHT_UNAME:-$(uname -s)}"
REQUIRED="${QT_STORAGE_PREFLIGHT_REQUIRED:-0}"

if [[ "$OS_NAME" != "Darwin" ]]; then
  if [[ "$REQUIRED" == "1" ]]; then
    fail "external APFS preflight is required but this host is $OS_NAME, not Darwin"
  fi
  info "non-macOS host; external APFS preflight not required"
  exit 0
fi

EXTERNAL_VOLUME="${QT_STORAGE_EXTERNAL_VOLUME:-/Volumes/Expansion}"
SPARSEBUNDLE="${QT_STORAGE_BUNDLE:-$EXTERNAL_VOLUME/QuantTermStorage.sparsebundle}"
STORAGE_MOUNT="${QT_STORAGE_MOUNT:-/Volumes/QuantTermStorage}"
EXPECTED_RUNTIME="${QT_STORAGE_RUNTIME:-$STORAGE_MOUNT/QuantTerm/runtime}"
CANONICAL_RUNTIME="${QT_RUNTIME_LINK:-$HOME/Library/Application Support/QuantTerm/runtime}"

[[ -d "$EXTERNAL_VOLUME" ]] || fail "external volume is not mounted: $EXTERNAL_VOLUME"
[[ -d "$SPARSEBUNDLE" ]] || fail "APFS sparsebundle is missing: $SPARSEBUNDLE"

# The canonical path must already be a symlink.  Do not mkdir it, repair it, or
# replace it here: doing so while the external disk is absent can split durable
# state between the internal and external disks.
[[ -L "$CANONICAL_RUNTIME" ]] || fail "canonical runtime is not a symlink: $CANONICAL_RUNTIME"
LINK_TARGET="$(readlink "$CANONICAL_RUNTIME")"
[[ "$LINK_TARGET" == "$EXPECTED_RUNTIME" ]] || fail \
  "runtime symlink target mismatch: expected '$EXPECTED_RUNTIME', got '$LINK_TARGET'"

if ! diskutil info "$STORAGE_MOUNT" >/dev/null 2>&1; then
  info "mounting APFS sparsebundle: $SPARSEBUNDLE"
  hdiutil attach -nobrowse "$SPARSEBUNDLE" >/dev/null || fail "could not attach sparsebundle"
fi

DISK_INFO="$(diskutil info "$STORAGE_MOUNT" 2>/dev/null)" || fail \
  "storage volume did not appear at $STORAGE_MOUNT"
MOUNT_POINT="$(printf '%s\n' "$DISK_INFO" | awk -F: '/^[[:space:]]*Mount Point:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
FS_PERSONALITY="$(printf '%s\n' "$DISK_INFO" | awk -F: '/^[[:space:]]*File System Personality:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"

[[ "$MOUNT_POINT" == "$STORAGE_MOUNT" ]] || fail \
  "unexpected mount point: expected '$STORAGE_MOUNT', got '${MOUNT_POINT:-unknown}'"
[[ "$FS_PERSONALITY" == "APFS" ]] || fail \
  "runtime volume must be APFS, got '${FS_PERSONALITY:-unknown}'"
[[ -d "$EXPECTED_RUNTIME" ]] || fail "expected runtime directory is missing: $EXPECTED_RUNTIME"
[[ -r "$EXPECTED_RUNTIME" ]] || fail "runtime directory is not readable: $EXPECTED_RUNTIME"
[[ -w "$EXPECTED_RUNTIME" ]] || fail "runtime directory is not writable: $EXPECTED_RUNTIME"

# Resolve through both paths only after the mount is verified.  This proves the
# canonical application-support path and the expected external directory are
# the same directory, without depending on the `realpath` utility (not present
# on older macOS releases such as Catalina).
RESOLVED_CANONICAL="$(cd "$CANONICAL_RUNTIME" && pwd -P)" || fail \
  "cannot resolve canonical runtime: $CANONICAL_RUNTIME"
RESOLVED_EXPECTED="$(cd "$EXPECTED_RUNTIME" && pwd -P)" || fail \
  "cannot resolve external runtime: $EXPECTED_RUNTIME"
[[ "$RESOLVED_CANONICAL" == "$RESOLVED_EXPECTED" ]] || fail \
  "canonical runtime resolves somewhere unexpected"

PROBE="$EXPECTED_RUNTIME/.quantterm-storage-preflight.$$"
( umask 077; : > "$PROBE" ) || fail "runtime write probe failed"
rm -f "$PROBE" || fail "runtime write probe cleanup failed"

info "PASS: APFS runtime verified at $EXPECTED_RUNTIME"
