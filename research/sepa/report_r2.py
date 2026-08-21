"""Write SEPA-001R2 markdown reports from a payload. Research only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_OUT = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R2"
_ROOT = Path(__file__).resolve().parents[2]


def _var_row(vid: str, stats: dict[str, Any]) -> str:
    ci = stats.get("block_ci") or {}
    if isinstance(ci, dict) and "ci_lower" in ci:
        ci_s = f"[{ci.get('ci_lower'):+.3f}, {ci.get('ci_upper'):+.3f}]" if ci.get("ci_lower") is not None else "—"
    else:
        ci_s = "—"
    dep = (stats.get("deployment") or {}).get("label") or ""
    return (
        f"| {vid} | {stats.get('unique_setups', '—')} | {stats.get('n', 0)} | "
        f"{stats.get('expectancy_r')} | {stats.get('profit_factor')} | {stats.get('win_rate')} | "
        f"{stats.get('avg_win')} | {stats.get('avg_loss')} | {stats.get('max_dd_r')} | "
        f"{stats.get('pct_1r')} | {stats.get('fail_break_pct')} | {ci_s} | "
        f"{stats.get('statistical_verdict')} / {dep} |"
    )


def write_results(payload: dict[str, Any]) -> Path:
    sample = payload.get("sample") or {}
    lines = [
        "# SEPA-001R2 Results",
        "",
        f"**Eligibility:** `{payload.get('eligibility_version')}`  ",
        f"**VCP:** `{payload.get('vcp_version')}`  ",
        f"**Pivot:** `{payload.get('pivot_version')}`  ",
        f"**Data:** {payload.get('data_source')}  ",
        f"**Eval:** {sample.get('evaluation_start')} → {sample.get('evaluation_end')} "
        f"(warmup {sample.get('warmup_sessions')} sessions)  ",
        f"**Universe:** as-of investable, `top_n={sample.get('top_n')}`  ",
        f"**Unique setups:** {sample.get('unique_setups')} "
        f"(left-censored unique {sample.get('left_censored_unique')})  ",
        "",
        "SEPA-001 and SEPA-001R files are immutable. Layer 1 = signal quality. "
        "A–D/G are not portfolio CAGR. Harness PROMOTE is never a deployment label.",
        "",
        "## Main comparison",
        "",
        "| Variant | Unique setups | Trades | E[R] | PF | Win % | Avg Win | Avg Loss | Max DD R | +1R % | Fail-break % | CI | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for vid, stats in (payload.get("variants") or {}).items():
        lines.append(_var_row(vid, stats))
    lines += [
        "",
        "## Funnel (snapshot counts)",
        "",
        "| Stage | Count |",
        "|---|---|",
    ]
    funnel = payload.get("funnel_snapshots") or {}
    for k, v in funnel.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Yearly universe (mean as-of size)",
        "",
        "| Year | Mean candidates | Mean investable | As-of points |",
        "|---|---|---|---|",
    ]
    for y, rec in sorted((payload.get("yearly_universe") or {}).items()):
        lines.append(
            f"| {y} | {rec.get('mean_candidates')} | {rec.get('mean_investable')} | {rec.get('as_of_points')} |"
        )
    lines += [
        "",
        "## RS buckets (ungated 20d forward %, as-of universe)",
        "",
        "| Bucket | n | Mean 20d % | Median 20d % |",
        "|---|---|---|---|",
    ]
    for k, rec in sorted((payload.get("rs_buckets") or {}).items()):
        lines.append(f"| {k} | {rec.get('n')} | {rec.get('mean_fwd_20d')} | {rec.get('median_fwd_20d')} |")
    lines += ["", "## Sample warnings", ""]
    integ = payload.get("integrity") or {}
    lines.append(f"- PIT class: `{integ.get('overall')}`")
    ca = payload.get("ca_audit") or {}
    lines.append(f"- CA complete: `{ca.get('ca_complete')}` quarantine n={payload.get('quarantine_n')}")
    cov = payload.get("coverage") or {}
    lines.append(f"- Coverage: {cov.get('first')} → {cov.get('last')} ({cov.get('n_symbols')} symbols)")
    text = "\n".join(lines) + "\n"
    path = _OUT / "SEPA_001R2_RESULTS.md"
    path.write_text(text)
    (_ROOT / "SEPA_001R2_RESULTS.md").write_text(text)
    return path


def write_funnel(payload: dict[str, Any]) -> Path:
    funnel = payload.get("funnel_snapshots") or {}
    inv = max(1, int(funnel.get("investable") or 1))
    lines = [
        "# SEPA-001R2 funnel",
        "",
        "Snapshot counts (one row per symbol×as-of). Unique-setup counts are separate.",
        "",
        "| Stage | Count | % of investable snapshots |",
        "|---|---|---|",
    ]
    for k, v in funnel.items():
        lines.append(f"| {k} | {v} | {100.0 * int(v) / inv:.4f}% |")
    lines += [
        "",
        f"Unique setups: {(payload.get('sample') or {}).get('unique_setups')}",
        f"Left-censored unique: {(payload.get('sample') or {}).get('left_censored_unique')}",
        "",
        "Why specific-entry trades are rare is the drop from `vcp_confirmed` / "
        "`pivot_defined` into `entry_ready`, then `extended_missed` / `gap_through` / "
        "`stop_too_wide` / `left_censored`. Core F fills only `valid_fill`.",
        "",
    ]
    path = _OUT / "SEPA_001R2_FUNNEL.md"
    path.write_text("\n".join(lines))
    return path


def write_all(payload: dict[str, Any]) -> dict[str, Path]:
    _OUT.mkdir(parents=True, exist_ok=True)
    return {
        "results": write_results(payload),
        "funnel": write_funnel(payload),
    }
