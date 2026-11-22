#!/usr/bin/env bash
# Final, VERIFIED pass of a runtime backup, run with the desk STOPPED.
#
#   bash scripts/runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file>
#
# <stamp_file> was touched BEFORE the bulk pass started (runbook stage 1). The
# bulk pass ran while the desk was live, so a file written after the stamp may
# have been copied mid-write, and rsync's size+mtime quick check (whole seconds
# on macOS rsync 2.6.9) can skip a same-size rewrite in the same second (GPT
# QT-07, cycle 6). So:
#   1. an ordinary rsync -a --delete pass catches new, removed and resized files;
#   2. every file modified since the stamp is copied AGAIN, unconditionally (-I);
#   3. each of those is compared byte for byte (cmp) with its source.
# A file untouched since the stamp was copied whole by the bulk pass, which
# started after the stamp. Prints FINAL_PASS_OK and the count, or exits non-zero.
set -euo pipefail

SRC=${1:?usage: runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file>}
DST=${2:?usage: runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file>}
STAMP=${3:?usage: runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file>}
SRC=${SRC%/}
DST=${DST%/}
[ -d "$SRC" ] || { echo "no source directory $SRC" >&2; exit 2; }
[ -d "$DST" ] || { echo "no backup directory $DST" >&2; exit 2; }
[ -f "$STAMP" ] || { echo "no bulk-pass stamp $STAMP" >&2; exit 2; }

rsync -a --delete "$SRC/" "$DST/"

LIST=$(mktemp -t qt-final-pass)
trap 'rm -f "$LIST"' EXIT
(cd "$SRC" && find . -type f -newer "$STAMP" -print0) > "$LIST"
n=$(tr -cd '\0' < "$LIST" | wc -c | tr -d ' ')
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
echo "FINAL_PASS_OK rechecked=$n"
