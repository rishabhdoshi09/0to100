#!/usr/bin/env bash
# Independent, READ-ONLY GPT cross-review of a commit range (OpenAI Codex CLI,
# signed in with ChatGPT). A development checkpoint for substantial changes:
# GPT is a critical reviewer, never an approval authority, and every finding
# must be verified by a deterministic test before it is acted on.
#
#   bash scripts/gpt_crossreview.sh <base_sha> <head_sha> [extra_scope.md]
#
# GPT reads a frozen detached checkout of <head_sha> in a read-only sandbox, so
# edits made while it runs cannot shift its line numbers. Nothing is sent but
# the prompt; the sandbox forbids writes, network and process control. Output:
# the findings path (and the log beside it). See
# docs/audits/2026-11-20-gpt-crossreview.md for the protocol.
set -euo pipefail

BASE=${1:?usage: gpt_crossreview.sh <base_sha> <head_sha> [extra_scope.md]}
HEAD_REF=${2:?usage: gpt_crossreview.sh <base_sha> <head_sha> [extra_scope.md]}
SCOPE=${3:-}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
command -v codex >/dev/null || { echo "codex CLI not found (npm install -g @openai/codex; codex login)" >&2; exit 2; }
HEAD_SHA=$(git rev-parse --verify "${HEAD_REF}^{commit}")
git rev-parse --verify "${BASE}^{commit}" >/dev/null

OUT=${QT_REVIEW_OUT:-$(mktemp -d -t qt-gptreview.XXXXXX)}
mkdir -p "$OUT"
git worktree add --detach "$OUT/snap" "$HEAD_SHA" >/dev/null
trap 'git -C "$ROOT" worktree remove --force "$OUT/snap" >/dev/null 2>&1 || true' EXIT

git log --format='%h %ad %s' --date=short "$BASE..$HEAD_SHA" > "$OUT/commits.txt"
git diff --stat "$BASE" "$HEAD_SHA" > "$OUT/stat.txt"
git diff "$BASE" "$HEAD_SHA" > "$OUT/diff.patch"
sed -e "s#{{BASE}}#$BASE#g" -e "s#{{HEAD}}#$HEAD_SHA#g" -e "s#{{BUNDLE}}#$OUT#g" \
    docs/review/gpt_crossreview_prompt.md > "$OUT/prompt.md"
if [ -n "$SCOPE" ]; then printf '\n' >> "$OUT/prompt.md"; cat "$SCOPE" >> "$OUT/prompt.md"; fi

codex exec -C "$OUT/snap" -s read-only -c model_reasoning_effort='"high"' --color never \
    -o "$OUT/findings.md" - < "$OUT/prompt.md" > "$OUT/log.txt" 2>&1
echo "$OUT/findings.md"
