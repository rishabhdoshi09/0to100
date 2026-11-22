#!/usr/bin/env bash
# Final, VERIFIED pass of a runtime backup, run with the desk STOPPED.
#
#   bash scripts/runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file> \
#        [--baseline <manifest.sha256>] [--verify-only]
#
# <stamp_file> was touched BEFORE the bulk pass started (runbook stage 1).
#
# Delta (always, unless --verify-only):
#   1. rsync -a --delete catches new, removed and resized files;
#   2. every file whose data OR inode changed since the stamp is copied AGAIN,
#      unconditionally (-I). The inode-change time (ctime) is the key: a copy
#      that preserves an old mtime still gets a new ctime, and utime cannot set
#      ctime (GPT R8, cycle 6);
#   3. each of those is compared byte for byte (cmp) with its source.
#
# Baseline (--baseline, Cycle 7): an UNCHANGED source file is never re-read by
# 1-3, so a backup copy corrupted after the bulk pass (same size, same mtime)
# would pass. The baseline is the stage-1 manifest of the backup ('sha256  path',
# verified against the source when it was written). Every baseline entry that
# is NOT in the delta is re-hashed in the backup and must match. That re-reads
# the whole backup; --verify-only runs exactly this check (no copy, no delete)
# so it can be done BEFORE the maintenance window, leaving only the delta inside.
#
# Refuses (exit 2): identical or nested source/backup paths, a backup on the
# same filesystem as the source, a missing stamp or baseline.
# Prints FINAL_PASS_OK (or BASELINE_OK for --verify-only), else exits 1.
set -euo pipefail

usage="usage: runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file> [--baseline <manifest>] [--verify-only]"
SRC=${1:?$usage}
DST=${2:?$usage}
STAMP=${3:?$usage}
shift 3
BASELINE=""
VERIFY_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline) BASELINE=${2:?$usage}; shift 2;;
        --verify-only) VERIFY_ONLY=1; shift;;
        *) echo "$usage" >&2; exit 2;;
    esac
done
[ -d "$SRC" ] || { echo "refusing: no source directory $SRC" >&2; exit 2; }
[ -d "$DST" ] || { echo "refusing: no backup directory $DST" >&2; exit 2; }
[ -f "$STAMP" ] || { echo "refusing: no bulk-pass stamp $STAMP" >&2; exit 2; }
[ -z "$BASELINE" ] || [ -f "$BASELINE" ] || { echo "refusing: no baseline manifest $BASELINE" >&2; exit 2; }
[ "$VERIFY_ONLY" -eq 0 ] || [ -n "$BASELINE" ] || { echo "refusing: --verify-only needs --baseline" >&2; exit 2; }

SRC=$(cd -P "$SRC" && pwd)
DST=$(cd -P "$DST" && pwd)
# absolute: the delta search runs from inside the source (a relative stamp or
# baseline path would silently resolve against the wrong directory)
STAMP="$(cd -P "$(dirname "$STAMP")" && pwd)/$(basename "$STAMP")"
[ -z "$BASELINE" ] || BASELINE="$(cd -P "$(dirname "$BASELINE")" && pwd)/$(basename "$BASELINE")"
case "$DST/" in "$SRC/"*) echo "refusing: the backup $DST is the source or inside it" >&2; exit 2;; esac
case "$SRC/" in "$DST/"*) echo "refusing: the source $SRC is inside the backup" >&2; exit 2;; esac
if [ "${QT_FINAL_PASS_ALLOW_SAME_FS:-0}" != "1" ]; then
    sdev=$(df -P "$SRC" | awk 'NR==2 {print $1}')
    ddev=$(df -P "$DST" | awk 'NR==2 {print $1}')
    if [ "$sdev" = "$ddev" ]; then
        echo "refusing: the backup is on the same filesystem as the source ($sdev)" >&2
        exit 2
    fi
fi

LIST=$(mktemp "${TMPDIR:-/tmp}/qt-final-pass.XXXXXX")     # portable: BSD and GNU mktemp
CHECK=$(mktemp "${TMPDIR:-/tmp}/qt-final-base.XXXXXX")
PATHS=$(mktemp "${TMPDIR:-/tmp}/qt-final-paths.XXXXXX")
trap 'rm -f "$LIST" "$CHECK" "$PATHS"' EXIT
(cd "$SRC" && find . -type f \( -newer "$STAMP" -o -cnewer "$STAMP" \) -print0) > "$LIST"
n=$(tr -cd '\0' < "$LIST" | wc -c | tr -d ' ')

if [ "$VERIFY_ONLY" -eq 0 ]; then
    rsync -a --delete "$SRC/" "$DST/"
    if [ "$n" -gt 0 ]; then
        rsync -a -I --from0 --files-from="$LIST" "$SRC/" "$DST/"
    fi
    bad=0
    while IFS= read -r -d '' f; do
        if ! cmp -s "$SRC/$f" "$DST/$f"; then
            echo "MISMATCH $f" >&2
            bad=$((bad + 1))
        fi
    done < "$LIST"
    if [ "$bad" -ne 0 ]; then
        echo "FINAL_PASS_FAILED mismatches=$bad" >&2
        exit 1
    fi
fi

if [ -z "$BASELINE" ]; then
    echo "FINAL_PASS_OK rechecked=$n NO_BASELINE (unchanged files were not re-verified)"
    exit 0
fi
# Baseline: every standard 'sha256  path' entry not in the delta, re-hashed in the backup.
tr '\0' '\n' < "$LIST" | sed 's|^\./||' > "$PATHS"
awk -v pf="$PATHS" 'BEGIN { while ((getline l < pf) > 0) skip[l] = 1 }
     length($1) == 64 && $1 ~ /^[0-9a-f]+$/ && substr($0, 65, 2) == "  " {
         p = substr($0, 67); if (!(p in skip)) print }' "$BASELINE" > "$CHECK"
# (the skip list is read with getline: with NR==FNR an EMPTY skip list would
#  swallow the whole baseline)
# (no {64} interval: BSD awk on macOS does not support it)
gone=0
if [ "$VERIFY_ONLY" -eq 0 ]; then
    # a file the desk deleted since the bulk pass was mirrored by --delete: its
    # baseline entry no longer applies (verify-only keeps it: the backup still has it)
    KEPT=$(mktemp "${TMPDIR:-/tmp}/qt-final-kept.XXXXXX")
    while IFS= read -r line; do
        if [ -e "$SRC/${line:66}" ]; then printf '%s\n' "$line"; else gone=$((gone + 1)); fi
    done < "$CHECK" > "$KEPT"
    mv "$KEPT" "$CHECK"
fi
m=$(wc -l < "$CHECK" | tr -d ' ')
total=$(grep -c '' "$BASELINE" || true)
if [ "$m" -eq 0 ] && [ "$total" -gt 0 ] && [ $((n + gone)) -lt "$total" ]; then
    echo "refusing: the baseline has no usable 'sha256  path' entries" >&2
    exit 2
fi
bad=0
if [ "$m" -gt 0 ]; then
    result=$(cd "$DST" && shasum -a 256 -c "$CHECK" 2>&1 || true)
    bad=$(printf '%s\n' "$result" | grep -v -c ': OK$' || true)
    if [ "$bad" -ne 0 ]; then
        printf '%s\n' "$result" | grep -v ': OK$' | head -20 | sed 's/^/BASELINE_MISMATCH /' >&2
    fi
fi
if [ "$bad" -ne 0 ]; then
    echo "FINAL_PASS_FAILED baseline_mismatches=$bad" >&2
    exit 1
fi
if [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "BASELINE_OK verified=$m"
else
    echo "FINAL_PASS_OK rechecked=$n baseline_verified=$m deleted_since_bulk=$gone"
fi
