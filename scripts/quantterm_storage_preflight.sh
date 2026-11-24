#!/usr/bin/env bash
# Fail-closed storage gate for the macOS QuantTerm service.
#
# The production runtime may live on an APFS sparsebundle stored on an external
# disk. No QuantTerm process may start until that exact storage chain is
# present, mounted, verified and writable. In particular, this script never
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
ATTACH_IF_NEEDED="${QT_STORAGE_PREFLIGHT_ATTACH:-1}"

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

# The canonical path must already be a symlink. Do not mkdir it, repair it, or
# replace it here: doing so while the external disk is absent can split durable
# state between the internal and external disks.
[[ -L "$CANONICAL_RUNTIME" ]] || fail "canonical runtime is not a symlink: $CANONICAL_RUNTIME"
LINK_TARGET="$(readlink "$CANONICAL_RUNTIME")"
[[ "$LINK_TARGET" == "$EXPECTED_RUNTIME" ]] || fail \
  "runtime symlink target mismatch: expected '$EXPECTED_RUNTIME', got '$LINK_TARGET'"

# OPT-IN, OFF BY DEFAULT (QT_STORAGE_PREFLIGHT_STALE_DETACH=1, the owner's decision): clear a
# STALE attachment of the CONFIGURED image. After a sleep the USB disk re-enumerates and the
# image's helper is left holding revoked descriptors: the image shows as attached, its volume can
# never mount, and re-attaching fails forever (19, 21, 22 and 23 Nov). Only that exact image, only
# when none of its devices is mounted: first a plain mount of its own volume; only if that fails,
# a plain detach of its device (NEVER -force), then the normal attach below. Every command is
# bounded so a dead device cannot hang the gate.
bounded() { perl -e 'alarm shift; exec @ARGV' "$@"; }
if [[ "${QT_STORAGE_PREFLIGHT_STALE_DETACH:-0}" == "1" && "$ATTACH_IF_NEEDED" == "1" ]] \
   && ! diskutil info "$STORAGE_MOUNT" >/dev/null 2>&1; then
  STALE="$(bounded 60 hdiutil info -plist 2>/dev/null | "${QT_STORAGE_PREFLIGHT_PYTHON:-python3}" -c '
import os, plistlib, sys
expected = os.path.realpath(sys.argv[1])
try:
    images = plistlib.loads(sys.stdin.buffer.read()).get("images") or []
except Exception:
    sys.exit(0)
mine = [im for im in images if os.path.realpath(str(im.get("image-path") or "")) == expected]
if len(mine) != 1:
    sys.exit(0)                      # none, or ambiguous: never act
ents = mine[0].get("system-entities") or []
if any(e.get("mount-point") for e in ents):
    sys.exit(0)                      # something of it is mounted: not stale, never detach
devs = [str(e.get("dev-entry") or "") for e in ents]
top = sorted((d for d in devs if d.startswith("/dev/disk") and "s" not in d[len("/dev/disk"):]), key=len)
vol = [str(e.get("dev-entry")) for e in ents if e.get("content-hint") == "41504653-0000-11AA-AA11-00306543ECAC"]
if top:
    print(top[0], vol[0] if len(vol) == 1 else "")
' "$SPARSEBUNDLE")" || STALE=""
  if [[ -n "$STALE" ]]; then
    read -r STALE_DEV STALE_VOL <<<"$STALE"
    info "configured image is attached but not mounted ($STALE_DEV): trying to mount its own volume"
    if [[ -n "${STALE_VOL:-}" ]] && bounded 60 diskutil mount "$STALE_VOL" >/dev/null 2>&1 \
       && diskutil info "$STORAGE_MOUNT" >/dev/null 2>&1; then
      info "mounted the configured image's volume $STALE_VOL"
    else
      info "its volume cannot be mounted: detaching the stale attachment $STALE_DEV (never -force)"
      DETACH_OUTPUT="$(bounded 60 hdiutil detach "$STALE_DEV" 2>&1)" \
        || fail "the stale attachment $STALE_DEV could not be detached: $(printf '%s\n' "$DETACH_OUTPUT" | tail -n 1)"
      sleep 2
    fi
  fi
fi

if ! diskutil info "$STORAGE_MOUNT" >/dev/null 2>&1; then
  if [[ "$ATTACH_IF_NEEDED" != "1" ]]; then
    fail "runtime volume is not mounted at $STORAGE_MOUNT"
  fi
  info "mounting APFS sparsebundle: $SPARSEBUNDLE"

  # Preserve the real DiskImages error. Older code redirected hdiutil stdout and
  # then collapsed every failure into "could not attach sparsebundle", which
  # made an already-attached image, a busy device, and a genuine image failure
  # indistinguishable. Also accept the race where hdiutil returns non-zero but
  # Disk Arbitration finishes mounting the expected volume immediately after.
  ATTACH_OUTPUT=""
  ATTACH_RC=0
  ATTACH_OUTPUT="$(hdiutil attach -nobrowse "$SPARSEBUNDLE" 2>&1)" || ATTACH_RC=$?
  if [[ "$ATTACH_RC" -ne 0 ]]; then
    sleep 1
    if ! diskutil info "$STORAGE_MOUNT" >/dev/null 2>&1; then
      ATTACH_DETAIL="$(printf '%s\n' "$ATTACH_OUTPUT" | tail -n 1)"
      [[ -n "$ATTACH_DETAIL" ]] || ATTACH_DETAIL="hdiutil attach exited rc=$ATTACH_RC"
      fail "could not attach sparsebundle (rc=$ATTACH_RC): $ATTACH_DETAIL"
    fi
    info "attach command returned rc=$ATTACH_RC but expected runtime volume is mounted; continuing"
  fi
fi

DISK_INFO="$(diskutil info "$STORAGE_MOUNT" 2>/dev/null)" || fail \
  "storage volume did not appear at $STORAGE_MOUNT"
MOUNT_POINT="$(printf '%s\n' "$DISK_INFO" | awk -F: '/^[[:space:]]*Mount Point:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
FS_PERSONALITY="$(printf '%s\n' "$DISK_INFO" | awk -F: '/^[[:space:]]*File System Personality:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"

[[ "$MOUNT_POINT" == "$STORAGE_MOUNT" ]] || fail \
  "unexpected mount point: expected '$STORAGE_MOUNT', got '${MOUNT_POINT:-unknown}'"
[[ "$FS_PERSONALITY" == "APFS" ]] || fail \
  "runtime volume must be APFS, got '${FS_PERSONALITY:-unknown}'"

# A mount path and APFS label alone do not identify the backing image.
# On a shared APFS volume name, Finder/Disk Arbitration may mount a restored
# clone from a different sparsebundle at the expected mount point. Fail closed
# unless DiskImages itself binds that mount point to the configured image.
# Apple's hdiutil -plist is machine-readable and avoids fragile text parsing.
PREFLIGHT_PYTHON="${QT_STORAGE_PREFLIGHT_PYTHON:-python3}"
command -v "$PREFLIGHT_PYTHON" >/dev/null 2>&1 || fail \
  "Python is required to verify sparsebundle mount identity"
IMAGE_INFO="$(hdiutil info -plist 2>/dev/null)" || fail \
  "could not inspect attached sparsebundle identities"
IMAGE_BINDING="$(
  printf '%s' "$IMAGE_INFO" | "$PREFLIGHT_PYTHON" -c '
import os
import plistlib
import sys

expected = os.path.realpath(sys.argv[1])
mount = os.path.realpath(sys.argv[2])
try:
    images = plistlib.loads(sys.stdin.buffer.read()).get("images") or []
    matched = []
    for item in images:
        entities = item.get("system-entities") or []
        if any(
            os.path.realpath(ent["mount-point"]) == mount
            for ent in entities
            if ent.get("mount-point")
        ):
            matched.append(os.path.realpath(str(item.get("image-path") or "")))
    if len(matched) != 1 or matched[0] != expected:
        raise ValueError("mounted APFS volume is not from configured sparsebundle")
    print("verified")
except Exception as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(78)
' "$SPARSEBUNDLE" "$STORAGE_MOUNT"
)" || fail "mounted APFS volume backing image identity is missing or mismatched"
[[ "$IMAGE_BINDING" == "verified" ]] || fail "unexpected sparsebundle binding result"
[[ -d "$EXPECTED_RUNTIME" ]] || fail "expected runtime directory is missing: $EXPECTED_RUNTIME"
[[ -r "$EXPECTED_RUNTIME" ]] || fail "runtime directory is not readable: $EXPECTED_RUNTIME"
[[ -w "$EXPECTED_RUNTIME" ]] || fail "runtime directory is not writable: $EXPECTED_RUNTIME"

# Resolve through both paths only after the mount is verified. This proves the
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
