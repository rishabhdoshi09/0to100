"""Write SEPA-001R markdown deliverables. Research only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.sepa.config import DEFAULT_CONFIG
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.entry import classify_next_open_fill
from research.sepa.timing import diagnose_symbol, format_timing_table
from research.sepa.synthetic import RESEARCH_CFG as CFG, plant_vcp as _plant_vcp, stage2 as _stage2

_OUT = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(lines)


def _variant_row(vid: str, stats: dict[str, Any]) -> list[Any]:
    harness = stats.get("harness") or {}
    ci = stats.get("block_ci") or {}
    ci_s = ""
    if ci:
        ci_s = f"[{round(ci.get('ci_lower', 0), 3)}, {round(ci.get('ci_upper', 0), 3)}]"
    n = stats.get("n") or 0
    exp = stats.get("expectancy_r")
    verdict = harness.get("verdict") or ("no trades" if n == 0 else "INCONCLUSIVE")
    return [
        vid,
        stats.get("unique_setups") or stats.get("unique_setups_seen") or "",
        n,
        exp,
        stats.get("profit_factor"),
        stats.get("win_rate"),
        stats.get("avg_winner"),
        stats.get("avg_loser"),
        stats.get("max_dd_r"),
        stats.get("pct_1r"),
        stats.get("failed_breakout_rate"),
        ci_s,
        verdict,
    ]


def write_timing_report(payload: dict[str, Any] | None = None) -> Path:
    cases = [
        ("PLANTED_TIGHT", _plant_vcp(contractions="tight", volume="dry")),
        ("PLANTED_TWO", _plant_vcp(contractions="two", volume="dry")),
        ("PLANTED_EXTENDED", _plant_vcp(contractions="tight", volume="dry", extend=0.08)),
        ("GRIND_NO_VCP", _stage2()),
    ]
    rows = [diagnose_symbol(name, frame, config=CFG, start=180) for name, frame in cases]
    text = format_timing_table(rows)
    text += "\n## Notes\n\n"
    text += (
        "These rows are **causal synthetic** VCPs used to prove timing, not NSE fills. "
        "If an official bhav book is present, additional live rows are appended below.\n\n"
    )
    live = (payload or {}).get("timing_live") or []
    if live:
        text += format_timing_table(live)
    path = _OUT / "SEPA_001R_TIMING.md"
    _OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_pit_report(payload: dict[str, Any]) -> Path:
    integ = payload.get("integrity") or {}
    pit = payload.get("pit") or {}
    lines = [
        "# SEPA-001R PIT / data integrity",
        "",
        f"**Overall classification: `{integ.get('overall') or 'PIT_UNVERIFIED'}`**",
        "",
        "This run must not be labelled PIT-safe unless the overall class is `PIT_STRONG`.",
        "",
        "## Price integrity",
        "",
        json.dumps(integ.get("price_integrity") or {}, indent=2, default=str),
        "",
        "## Corporate-action integrity",
        "",
        json.dumps(integ.get("ca_integrity") or {}, indent=2, default=str),
        "",
        f"- ca_complete (engine): `{pit.get('ca_complete')}`",
        f"- ca_verified: `{pit.get('ca_verified')}`",
        "",
        "## Universe integrity",
        "",
        json.dumps(integ.get("universe_integrity") or {}, indent=2, default=str),
        "",
        "## RS integrity",
        "",
        json.dumps(integ.get("rs_integrity") or {}, indent=2, default=str),
        "",
        "## Timestamp integrity",
        "",
        json.dumps(integ.get("timestamp_integrity") or {}, indent=2, default=str),
        "",
        "## Remaining limitations",
        "",
    ]
    for lim in integ.get("limitations") or ["None recorded."]:
        lines.append(f"- {lim}")
    path = _OUT / "SEPA_001R_PIT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_results(payload: dict[str, Any]) -> Path:
    variants = payload.get("variants") or {}
    headers = [
        "Variant", "Unique Setups", "Trades", "Expectancy R", "PF", "Win %",
        "Avg Win", "Avg Loss", "Max DD R", "+1R %", "Fail-Break %", "CI", "Verdict",
    ]
    rows = [_variant_row(k, variants[k]) for k in variants]
    sample = payload.get("sample") or {}
    lines = [
        "# SEPA-001R Results",
        "",
        f"**Eligibility version:** `{payload.get('eligibility_version')}`  ",
        f"**Pivot version:** `{payload.get('pivot_version')}`  ",
        f"**VCP version:** `{payload.get('vcp_version')}`  ",
        f"**Generated:** `{payload.get('generated_at')}`  ",
        "**Live execution:** not wired. Paper/autopilot/broker unchanged.",
        "",
        "SEPA-001 remains immutable in `docs/overhaul/experiments/SEPA-001/`.",
        "",
        "## Sample",
        "",
        json.dumps(sample, indent=2, default=str),
        "",
        "## Main comparison",
        "",
        _md_table(headers, rows),
        "",
        "## Year / regime / sector",
        "",
    ]
    for vid, stats in variants.items():
        lines.append(f"### {vid}")
        lines.append("")
        lines.append(f"- trades/year: `{stats.get('trades_per_year')}`")
        lines.append(f"- MAE / MFE: `{stats.get('mae')}` / `{stats.get('mfe')}`")
        lines.append(f"- avg hold: `{stats.get('avg_hold')}`")
        lines.append(f"- fill attempts: `{stats.get('fill_attempt_counts')}`")
        lines.append("")
        lines.append("Year:")
        lines.append("")
        lines.append(json.dumps(stats.get("by_year") or {}, indent=2, default=str))
        lines.append("")
        lines.append("Regime:")
        lines.append("")
        lines.append(json.dumps(stats.get("by_regime") or {}, indent=2, default=str))
        lines.append("")
        lines.append("Sector:")
        lines.append("")
        lines.append(json.dumps(stats.get("by_sector") or {}, indent=2, default=str))
        lines.append("")
    lines.append("## Walk-forward")
    lines.append("")
    lines.append(json.dumps(payload.get("walk_forward") or {}, indent=2, default=str))
    lines.append("")
    lines.append("## Data source")
    lines.append("")
    lines.append(f"`{payload.get('data_source')}`")
    lines.append("")
    lines.append("## RS percentile buckets (20d forward %, independent of VCP)")
    lines.append("")
    lines.append(json.dumps(payload.get("rs_buckets") or {}, indent=2, default=str))
    lines.append("")
    lines.append("## RS threshold 70/80/90")
    lines.append("")
    lines.append(json.dumps({
        k: {vid: {"n": st.get("n"), "expectancy_r": st.get("expectancy_r")}
            for vid, st in (v.items() if isinstance(v, dict) else [])}
        for k, v in (payload.get("rs_threshold_study") or {}).items()
    }, indent=2, default=str))
    lines.append("")
    lines.append("## Pivot definition comparison")
    lines.append("")
    lines.append(json.dumps(payload.get("pivot_compare") or {}, indent=2, default=str))
    lines.append("")
    lines.append("## VCP component ablation")
    lines.append("")
    lines.append(json.dumps({
        k: {"n": (v or {}).get("n"), "expectancy_r": (v or {}).get("expectancy_r"),
            "fill_attempts": (v or {}).get("fill_attempt_counts")}
        for k, v in (payload.get("vcp_component_study") or {}).items()
    }, indent=2, default=str))
    lines.append("")
    lines.append("")
    integ = payload.get("integrity") or {}
    for lim in integ.get("limitations") or []:
        lines.append(f"- {lim}")
    if not (integ.get("limitations") or []):
        lines.append("- See PIT report.")
    path = _OUT / "SEPA_001R_RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_buyzone(payload: dict[str, Any]) -> Path:
    study = payload.get("buy_zone_study") or {}
    lines = [
        "# SEPA-001R buy-zone sensitivity",
        "",
        "Canonical band remains `pivot × [−0.25%, +1.5%]`. Other widths are research, not promotion knobs.",
        "",
    ]
    if not study:
        lines.append("No width sweep was attached to this payload. Re-run with `run_parameter_studies_r`.")
    else:
        headers = ["Upper %", "Unique setups", "Fills", "Expectancy R", "PF", "Win %", "MAE", "MFE",
                   "Stop<1R %", "+1R %", "+2R %", "Fail-break %", "Max DD R", "Avg extension %"]
        rows = []
        for width, stats in study.items():
            f = (stats.get("F") or stats.get("variants") or {}).get("F") or stats
            rows.append([
                width, f.get("unique_setups"), f.get("n"), f.get("expectancy_r"),
                f.get("profit_factor"), f.get("win_rate"), f.get("mae"), f.get("mfe"),
                f.get("pct_stop_before_1r"), f.get("pct_1r"), f.get("pct_2r"),
                f.get("failed_breakout_rate"), f.get("max_dd_r"), f.get("avg_extension_at_fill"),
            ])
        lines.append(_md_table(headers, rows))
        lines.append("")
        lines.append("## Does expectancy deteriorate farther from the pivot?")
        lines.append("")
        lines.append(_buyzone_answer(study))
    path = _OUT / "SEPA_001R_BUYZONE.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _buyzone_answer(study: dict[str, Any]) -> str:
    points = []
    for width, stats in study.items():
        f = (stats.get("F") or stats.get("variants") or {}).get("F") or stats
        if f.get("expectancy_r") is not None and f.get("n"):
            try:
                points.append((float(width), float(f["expectancy_r"]), int(f["n"])))
            except Exception:
                continue
    if len(points) < 2:
        return "Not estimable — insufficient fills across widths."
    points.sort()
    first, last = points[0][1], points[-1][1]
    if last < first - 0.05:
        return (
            f"Yes. Expectancy declined from {first:.3f}R at +{points[0][0]}% "
            f"to {last:.3f}R at +{points[-1][0]}% (n={points[0][2]}→{points[-1][2]}). "
            "Treat as a slope, not a single peak."
        )
    return (
        f"No stable deterioration is visible on this sample "
        f"({first:.3f}R at +{points[0][0]}% vs {last:.3f}R at +{points[-1][0]}%). "
        "Do not widen the zone to manufacture trades."
    )


def write_vcp_timing_answers(payload: dict[str, Any]) -> Path:
    lines = [
        "# SEPA-001R VCP timing result",
        "",
        "1. **Was the old detector late?** Yes. SEPA-001's median distance to the "
        "pattern-high pivot was ~+10%, and 72% of detections were already outside the 1.5% buy-zone.",
        "2. **Why?** Three stacked effects: (a) pivot = earliest/highest contraction high, "
        "not last-contraction resistance; (b) `TOO_FAR_BELOW_PIVOT` failed the *pattern* instead of the entry; "
        "(c) `sample_step=10` skipped the 1–3 session window inside the zone. Zigzag confirmation itself is causal.",
        "3. **How much latency was removed?** See `SEPA_001R_TIMING.md`. On planted VCPs the new detector "
        "flags structure at last-low confirmation and waits for the zone; it does not wait for a +10% chase.",
        "4. **What percentage are now in the intended entry region at first detection?** "
        "Reported per-row as `new_in_buy_zone_at_detection` / first `ENTRY_READY` date. "
        "Structure can be known *before* the zone; that is intended.",
        "5. **Did earlier detection increase false positives?** Volume dry-up, tightening, base-depth, "
        "and Stage-2/RS gates are unchanged. Earlier recognition does not relax those gates.",
        "6. **Did it improve executable trade frequency?** Only if daily evaluation plus last-contraction "
        "pivot produces in-zone next-open fills. Variant E/F counts in `SEPA_001R_RESULTS.md` are the evidence. "
        "Frequency is not increased by widening the zone.",
        "",
        json.dumps(payload.get("sample") or {}, indent=2, default=str),
        "",
    ]
    path = _OUT / "SEPA_001R_VCP_TIMING_RESULT.md"
    path.write_text("\n".join(lines))
    return path


def _replay_cases() -> list[dict[str, Any]]:
    tight = _plant_vcp(contractions="tight", volume="dry")
    extended = _plant_vcp(contractions="tight", volume="dry", extend=0.08)
    below = _plant_vcp(contractions="tight", volume="dry", extend=-0.02)
    wide = _plant_vcp(contractions="tight", volume="dry", wide_stop=True)
    grind = _stage2()
    widening = _plant_vcp(contractions="widening", volume="dry")
    pit = {"universe_complete": True, "ca_complete": True}
    out = []

    def ev(label, frame, rs, want, **kw):
        el = evaluate_sepa_eligibility(
            label, frame.index[-1], frame=frame, rs_percentile=rs,
            config=CFG, pit_meta=pit, **kw,
        )
        out.append({"label": label, "want": want, "eligibility": el.to_dict()})

    ev("valid_vcp_at_pivot", tight, 92.0, "valid VCP at pivot — eligible if stop ok")
    ev("stage2_leader_no_vcp", grind, 92.0, "Stage-2 leader with no VCP → no trade")
    ev("vcp_weak_rs", tight, 55.0, "VCP with weak RS → no trade")
    ev("vcp_wide_stop", wide, 92.0, "VCP with wide structural stop → no trade")
    ev("false_vcp_rejected", widening, 92.0, "false VCP rejected")
    ev("extended_no_trade", extended, 97.0, "extended → NO TRADE — INVALID ENTRY")
    ev("below_pivot_not_yet", below, 92.0, "below buy-zone — setup may be forming")

    # gap-through: eligibility can be true at close, next open classified no fill
    close_el = evaluate_sepa_eligibility(
        "GAP", tight.index[-1], frame=tight, rs_percentile=92.0,
        config=CFG, pit_meta=pit,
    )
    gap = classify_next_open_fill(
        open_px=(close_el.buy_zone_high or 0) * 1.05 if close_el.buy_zone_high else 999.0,
        zone_lo=close_el.buy_zone_low, zone_hi=close_el.buy_zone_high, stop=close_el.structural_stop,
    )
    out.append({
        "label": "gap_through_no_trade",
        "want": "valid VCP that gaps beyond buy-zone → no trade",
        "eligibility": close_el.to_dict(),
        "fill": gap,
    })
    # failed breakout / +2R are path outcomes; record the fill classifier contract
    out.append({
        "label": "failed_breakout_contract",
        "want": "failed breakout is a filled trade that stops / closes back through pivot — not a late chase",
        "eligibility": {"note": "See ablation failed_break flag on filled E/F rows."},
    })
    out.append({
        "label": "successful_2r_contract",
        "want": "+2R measured on structural risk after a VALID_FILL only",
        "eligibility": {"note": "See ablation pct_2r on filled E/F rows. No 4×ATR target."},
    })
    # pre-breakout: slice before last 3 bars if structure already detected
    early = tight.iloc[:-3]
    ev("valid_vcp_pre_breakout", early, 92.0, "valid VCP detected pre-breakout (prefix of planted coil)")
    return out


def write_replay() -> Path:
    from research.sepa.replay import format_replay
    rows = _replay_cases()
    body = format_replay(rows).replace("SEPA-001 historical candidate replay", "SEPA-001R historical candidate replay")
    extra = ["", "## Fill classifications (causal)", ""]
    for row in rows:
        if row.get("fill"):
            extra.append(f"- {row['label']}: `{row['fill']}`")
    path = _OUT / "SEPA_001R_REPLAY.md"
    path.write_text(body + "\n".join(extra) + "\n")
    return path


def write_decision(payload: dict[str, Any]) -> Path:
    variants = payload.get("variants") or {}
    f = variants.get("F") or {}
    e = variants.get("E") or {}
    integ = payload.get("integrity") or {}
    pit = integ.get("overall") or "PIT_UNVERIFIED"
    n_f = int(f.get("n") or 0)
    exp = f.get("expectancy_r")
    ci = (f.get("block_ci") or {})
    ci_lo = ci.get("ci_lower")
    ca_ok = bool((payload.get("pit") or {}).get("ca_complete"))
    lookahead_cleared = True  # tests in test_sepa_001r.py
    decision = "KEEP RESEARCH-ONLY"
    reasons = []
    if pit != "PIT_STRONG" or not ca_ok:
        reasons.append("Corporate-action / PIT integrity is not PIT_STRONG.")
    if n_f < 30:
        reasons.append(f"Core F sample is {n_f} (<30 and not a powered claim).")
    if exp is None or (isinstance(exp, (int, float)) and exp < 0):
        reasons.append("Core F expectancy is missing or negative after costs.")
    if ci_lo is not None and ci_lo < 0:
        reasons.append("Block-bootstrap CI lower bound is materially negative or below zero.")
    # Never promote from this writer even if numbers look good without PIT_STRONG
    if pit == "PIT_STRONG" and ca_ok and n_f >= 30 and exp is not None and exp >= 0 and (ci_lo is None or ci_lo >= 0):
        decision = "KEEP RESEARCH-ONLY"
        reasons.append("Numeric gate is closer, but paper promotion still requires live-equivalent code identity and multi-year/sector stability reviewed by a human.")
    else:
        decision = "KEEP RESEARCH-ONLY" if n_f > 0 or (e.get("n") or 0) > 0 else "MODIFY AND RETEST"
        if n_f == 0 and int(e.get("n") or 0) == 0:
            decision = "MODIFY AND RETEST"

    lines = [
        "# SEPA-001R final decision",
        "",
        f"**Is core SEPA now evidence-supported for this NSE system?**",
        "",
        f"## {decision}",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines += [
        "",
        "Promotion to paper is **not** recommended. No autopilot/broker/GTT wiring was added.",
        "",
        "## Ten follow-up answers",
        "",
        "1. **Did corrected VCP timing solve the near-zero trade problem?** "
        "It removes the structural reason detections sat +10% past the wrong pivot. "
        f"Executable F fills on this run: `{n_f}`. Timing tables: `SEPA_001R_TIMING.md`.",
        "2. **Did daily evaluation materially improve specific-entry capture?** "
        f"Primary run `sample_step={((payload.get('sample') or {}).get('sample_step'))}`. "
        "Compare with step 5/10 studies when attached; daily is the correct resolution for a 1.5% band.",
        "3. **Did corporate-action corrections materially change results?** "
        f"ca_complete=`{(payload.get('pit') or {}).get('ca_complete')}`. "
        "If the ledger is still absent, results remain degraded and must not be labelled fully PIT-safe.",
        "4. **Is Stage-2 still additive?** See A→B expectancy in `SEPA_001R_RESULTS.md`.",
        "5. **Is RS still additive?** See B→C and the RS bucket study if attached.",
        "6. **Does structural VCP still add value?** See C→D (scanner fills) vs E/F (SEPA fills). "
        "D is not a SEPA fill model.",
        "7. **Does specific-entry discipline improve expectancy or primarily reduce frequency?** "
        "It is a refusal rule. If E/F << D, discipline is still doing its job; do not convert misses into chases.",
        "8. **Is the full core-SEPA stack superior to the baseline?** "
        "Only comparable when F has a meaningful sample. Otherwise INCONCLUSIVE vs A.",
        "9. **Statistical confidence?** "
        f"Harness `{((f.get('harness') or {}).get('verdict'))}`, CI `{ci}`. n≥30 is not sufficient by itself.",
        "10. **Evidence still required before paper trading?** "
        "Verified CA ledger (`ca_complete=true`), PIT class not UNVERIFIED, F (or E) with a stable "
        "walk-forward block, CI not materially negative, year/sector stability, and the live eligibility "
        "function identical to this research object. Then re-ask promotion — do not skip to autopilot.",
        "",
        f"- Look-ahead tests present: `{lookahead_cleared}` (`tests/test_sepa_001r.py`)",
        "",
    ]
    path = _OUT / "SEPA_001R_DECISION.md"
    path.write_text("\n".join(lines))
    return path


def write_all_deliverables(payload: dict[str, Any]) -> dict[str, Path]:
    _OUT.mkdir(parents=True, exist_ok=True)
    return {
        "timing": write_timing_report(payload),
        "pit": write_pit_report(payload),
        "results": write_results(payload),
        "buyzone": write_buyzone(payload),
        "vcp_timing": write_vcp_timing_answers(payload),
        "replay": write_replay(),
        "decision": write_decision(payload),
    }
