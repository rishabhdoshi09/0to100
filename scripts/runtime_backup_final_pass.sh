#!/usr/bin/env bash
# Final, VERIFIED pass of a runtime backup, run with the desk STOPPED.
#
#   bash scripts/runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file> \
#        [--baseline <manifest.sha256>] [--verify-only] [--exclude <relpath>]...
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
# verified against the source when it was written). The verifier then requires
# EVERY regular file in the backup to be covered: compared with its source in
# the delta, or matching its baseline entry. --verify-only checks every baseline
# entry (no copy, no delete), so the whole-backup re-read can be done BEFORE the
# maintenance window, leaving only the delta inside it.
#
# Exclusions (--exclude, repeatable; Stage 1 of 22 Nov 2026): a literal path relative
# to <source_dir> (a file, or a directory and everything under it) that the backup
# deliberately does not hold. It is neither copied nor deleted by rsync, never in the
# delta, and not required to be covered by the baseline. Without this, a path the
# stage-1 backup excluded (two static 10 GB September logs) would be copied INSIDE
# the maintenance window, or fail coverage if present. Wildcards, '.', '..', empty
# components, a leading '/' and newlines are refused: the path is matched literally.
#
# Refuses (exit 2): any path containing a newline (arguments or files: the
# baseline is line-based), identical or nested source/backup paths, a backup on
# the same filesystem as the source, a missing stamp, a missing, empty or
# malformed baseline. Prints FINAL_PASS_OK (BASELINE_OK for --verify-only),
# else exits 1. The baseline verifier needs python3 (QT_PYTHON overrides).
set -euo pipefail

usage="usage: runtime_backup_final_pass.sh <source_dir> <backup_dir> <stamp_file> [--baseline <manifest>] [--verify-only] [--exclude <relpath>]..."
SRC=${1:?$usage}
DST=${2:?$usage}
STAMP=${3:?$usage}
shift 3
BASELINE=""
VERIFY_ONLY=0
EXCLUDES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline) BASELINE=${2:?$usage}; shift 2;;
        --verify-only) VERIFY_ONLY=1; shift;;
        --exclude) [ $# -ge 2 ] || { echo "$usage" >&2; exit 2; }; EXCLUDES+=("$2"); shift 2;;
        *) echo "$usage" >&2; exit 2;;
    esac
done
NL=$'\n'
RSYNC_EXCL=()
FIND_EXCL=()
for ex in ${EXCLUDES[@]+"${EXCLUDES[@]}"}; do
    case "$ex" in
        ""|/*|*"$NL"*|*'*'*|*'?'*|*'['*|*'\'*)
            echo "refusing: --exclude takes a literal relative path (no wildcard, newline or leading /): $ex" >&2; exit 2;;
    esac
    ex=${ex%/}                                  # one trailing slash names a directory
    case "/$ex/" in
        *"//"*|*"/./"*|*"/../"*)
            echo "refusing: --exclude path has an empty, '.' or '..' component: $ex" >&2; exit 2;;
    esac
    RSYNC_EXCL+=("--exclude=/$ex")
    FIND_EXCL+=(! -path "./$ex" ! -path "./$ex/*")
done
for arg in "$SRC" "$DST" "$STAMP" "$BASELINE"; do
    case "$arg" in *"$NL"*)
        # command substitution strips trailing newlines: 'source\n' would
        # silently become its sibling 'source' (GPT C7-03, cycle 7)
        echo "refusing: a path argument contains a newline" >&2; exit 2;;
    esac
done
[ -d "$SRC" ] || { echo "refusing: no source directory $SRC" >&2; exit 2; }
[ -d "$DST" ] || { echo "refusing: no backup directory $DST" >&2; exit 2; }
[ -f "$STAMP" ] || { echo "refusing: no bulk-pass stamp $STAMP" >&2; exit 2; }
[ -z "$BASELINE" ] || [ -f "$BASELINE" ] || { echo "refusing: no baseline manifest $BASELINE" >&2; exit 2; }
[ "$VERIFY_ONLY" -eq 0 ] || [ -n "$BASELINE" ] || { echo "refusing: --verify-only needs --baseline" >&2; exit 2; }
if [ "$SRC" -ef "$DST" ]; then echo "refusing: the source and the backup are the same directory" >&2; exit 2; fi

# Physical paths, with a sentinel: $(...) strips EVERY trailing newline, so a
# symlink to 'source\n' would otherwise resolve to its sibling 'source'
# (GPT C7V-04, cycle 7). The resolved paths are then refused if they hold one.
phys_dir() {
    local out
    out=$(cd -P -- "$1" && pwd -P && printf x) || return 1
    out=${out%x}
    printf '%s' "${out%"$NL"}"
}
SRC=$(phys_dir "$SRC"; printf x) || exit 2; SRC=${SRC%x}
DST=$(phys_dir "$DST"; printf x) || exit 2; DST=${DST%x}
sdir=$(phys_dir "$(dirname -- "$STAMP")"; printf x) || exit 2; sdir=${sdir%x}
STAMP="$sdir/$(basename -- "$STAMP")"
if [ -n "$BASELINE" ]; then
    bdir=$(phys_dir "$(dirname -- "$BASELINE")"; printf x) || exit 2; bdir=${bdir%x}
    BASELINE="$bdir/$(basename -- "$BASELINE")"
fi
for resolved in "$SRC" "$DST" "$STAMP" "$BASELINE"; do
    case "$resolved" in *"$NL"*)
        echo "refusing: a path resolves to a name containing a newline" >&2; exit 2;;
    esac
done
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
PY=${QT_PYTHON:-python3}
if [ -n "$BASELINE" ] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "refusing: the baseline verifier needs python3 (set QT_PYTHON)" >&2; exit 2
fi

LIST=$(mktemp "${TMPDIR:-/tmp}/qt-final-pass.XXXXXX")     # portable: BSD and GNU mktemp
trap 'rm -f "$LIST"' EXIT
(cd "$SRC" && find . -type f \( -newer "$STAMP" -o -cnewer "$STAMP" \) ${FIND_EXCL[@]+"${FIND_EXCL[@]}"} -print0) > "$LIST"
n=$(tr -cd '\0' < "$LIST" | wc -c | tr -d ' ')
if [ "$(tr -cd '\n' < "$LIST" | wc -c | tr -d ' ')" -ne 0 ]; then
    echo "refusing: a file changed since the stamp has a newline in its name; the baseline cannot represent it" >&2
    exit 2
fi

if [ "$VERIFY_ONLY" -eq 0 ]; then
    rsync -a --delete ${RSYNC_EXCL[@]+"${RSYNC_EXCL[@]}"} "$SRC/" "$DST/"
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
    echo "FINAL_PASS_OK rechecked=$n NO_BASELINE (unchanged files were not re-verified) excluded=${#EXCLUDES[@]}"
    exit 0
fi

# Baseline and coverage, on exact path bytes (Cycle 7).
"$PY" -I - "$SRC" "$DST" "$BASELINE" "$LIST" "$VERIFY_ONLY" ${EXCLUDES[@]+"${EXCLUDES[@]}"} <<'PYEOF'
import hashlib
import os
import sys

src, dst, base, lst, verify_only = sys.argv[1:6]
src, dst = os.fsencode(src), os.fsencode(dst)
verify_only = verify_only == "1"
excluded = [os.fsencode(e.rstrip("/")) for e in sys.argv[6:]]


def is_excluded(rel):
    return any(rel == e or rel.startswith(e + b"/") for e in excluded)
HEX = set(b"0123456789abcdef")


def refuse(msg):
    print(f"refusing: {msg}", file=sys.stderr)
    sys.exit(2)


delta = set()
for p in open(lst, "rb").read().split(b"\0"):
    if p:
        delta.add(p[2:] if p.startswith(b"./") else p)

entries, malformed, symlinks = {}, 0, 0
for raw in open(base, "rb").read().split(b"\n"):
    if not raw.strip():
        continue
    if raw.startswith(b"symlink:"):
        symlinks += 1                       # recreated by rsync -a; not a content entry
        continue
    h, sep, path = raw[:64], raw[64:66], raw[66:]
    if len(raw) > 66 and sep == b"  " and set(h) <= HEX:
        entries[path] = h.decode()
    else:
        malformed += 1
if malformed:
    refuse(f"the baseline has {malformed} malformed line(s)")


def regular_files(root):
    out = []
    for d, _subs, files in os.walk(root):
        for f in files:
            full = os.path.join(d, f)
            if os.path.isfile(full) and not os.path.islink(full):
                rel = os.path.relpath(full, root)
            if not is_excluded(rel):
                out.append(rel)
    return out


present = regular_files(dst)
if not entries and present:
    refuse("the baseline has no entries but the backup holds files")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_regular(root, rel):
    """'' when root/rel is a regular file reached through no symlink and inside
    root; else why not (GPT C7V-03, cycle 7: a backup file replaced by a symlink
    to the source would otherwise verify while holding nothing)."""
    import stat as S
    parts = rel.split(b"/")
    if rel.startswith(b"/") or any(p in (b"", b".", b"..") for p in parts):
        return "the path escapes the backup root"
    cur = root
    for i, part in enumerate(parts):
        cur = os.path.join(cur, part)
        try:
            st = os.lstat(cur)
        except OSError:
            return "missing from the backup"
        if S.S_ISLNK(st.st_mode):
            return "a symlink in the backup path"
        if i == len(parts) - 1 and not S.S_ISREG(st.st_mode):
            return "not a regular file in the backup"
    return ""


verified, bad, gone = set(), [], 0
for path, want in entries.items():
    if is_excluded(path):
        continue                            # deliberately not held by the backup
    if not verify_only:
        if path in delta:
            continue                        # compared with its source above
        if not os.path.lexists(os.path.join(src, path)):
            gone += 1                       # deleted since the bulk pass; --delete mirrored it
            continue
    target = os.path.join(dst, path)
    why = safe_regular(dst, path)
    if why:
        bad.append((path, why))
        continue
    try:
        got = sha(target)
    except OSError:
        bad.append((path, "unreadable in the backup"))
        continue
    if got == want:
        verified.add(path)
    else:
        bad.append((path, "differs from the baseline"))

bad_paths = {p for p, _ in bad}
uncovered = [p for p in present if p not in verified and p not in bad_paths
             and (verify_only or p not in delta)]
for path, why in bad[:20]:
    print(f"BASELINE_MISMATCH {os.fsdecode(path)}: {why}", file=sys.stderr)
for path in uncovered[:20]:
    print(f"UNCOVERED {os.fsdecode(path)}: neither in the delta nor verified against the baseline",
          file=sys.stderr)
if bad or uncovered:
    print(f"FINAL_PASS_FAILED baseline_mismatches={len(bad)} uncovered={len(uncovered)}", file=sys.stderr)
    sys.exit(1)
if verify_only:
    print(f"BASELINE_OK verified={len(verified)} files={len(present)} excluded={len(excluded)}")
else:
    print(f"FINAL_PASS_OK rechecked={len(delta)} baseline_verified={len(verified)} "
          f"deleted_since_bulk={gone} files={len(present)} excluded={len(excluded)}")
PYEOF
