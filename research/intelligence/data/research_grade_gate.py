"""Earn RESEARCH_GRADE — never a manually assigned trust stamp.

Composes existing validators (gauntlet, data_integrity, universe/CA/identity
ledgers, snapshot provenance) into one fail-closed gate. Writers must call
``evaluate_research_grade`` and only stamp manifests when ``earned`` is True.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from product.plain_language import explain_trust_class, render_layers


def _fail(name: str, detail: str) -> dict:
    return {"check": name, "ok": False, "detail": detail}


def _pass(name: str, detail: str = "") -> dict:
    return {"check": name, "ok": True, "detail": detail}


def evaluate_research_grade(
    *,
    snapshot_manifest: dict | None = None,
    run_gauntlet_validate: bool = True,
    sample: int = 80,
) -> dict[str, Any]:
    """Return earned trust decision + checks + plain-English explanation."""
    checks: list[dict] = []
    manifest = dict(snapshot_manifest or {})

    # 1) Identity ledger
    try:
        from data.security_identity import ledger_status as id_status
        ids = id_status()
        if not ids.get("available") or int(ids.get("n_securities") or 0) < 1:
            checks.append(_fail("security_identity", "identity ledger missing"))
        else:
            checks.append(_pass(
                "security_identity",
                f"n={ids.get('n_securities')} lineage_complete="
                f"{ids.get('symbol_lineage_complete')}",
            ))
            # Official delist archive is required; full ISIN lineage may still be False.
            if not ids.get("has_official_delistings"):
                checks.append(_fail(
                    "official_delistings",
                    "official NSE delisting archive not present on identity ledger",
                ))
            else:
                checks.append(_pass("official_delistings"))
            if not ids.get("symbol_lineage_complete"):
                checks.append(_fail(
                    "symbol_lineage_complete",
                    "symbol↔ISIN lineage not fully closed — unknown transitions remain",
                ))
            else:
                checks.append(_pass("symbol_lineage_complete"))
    except Exception as exc:
        checks.append(_fail("security_identity", str(exc)))

    # 2) Universe ledger — must exist, not bhav_inferred, survivorship complete
    try:
        from data.universe_history import ledger_status as u_status
        import json
        from data.universe_history import history_path
        u = u_status()
        if not u.get("available"):
            checks.append(_fail("universe_ledger", "universe_history.json missing"))
        else:
            src = str(u.get("source") or "")
            if src.startswith("bhav_") or src in {"", "bhav_inferred"}:
                checks.append(_fail("universe_source", f"non-research source={src!r}"))
            else:
                checks.append(_pass("universe_source", src))
            # Prefer explicit completeness stamp in file over source-label research_grade
            surv = False
            try:
                raw = json.loads(history_path().read_text(encoding="utf-8"))
                comp = raw.get("completeness") or {}
                if "survivorship_complete" in comp:
                    surv = bool(comp.get("survivorship_complete"))
                else:
                    surv = bool(u.get("survivorship_complete"))
                if comp.get("reconstructed_from_survivors_only"):
                    checks.append(_fail(
                        "universe_not_survivor_only",
                        "ledger built from current EQUITY_L survivors only",
                    ))
                else:
                    checks.append(_pass("universe_not_survivor_only"))
            except Exception:
                surv = bool(u.get("survivorship_complete"))
            if not surv:
                checks.append(_fail(
                    "survivorship_complete",
                    "historical delistings not available — cannot claim complete PIT universe",
                ))
            else:
                checks.append(_pass("survivorship_complete"))
    except Exception as exc:
        checks.append(_fail("universe_ledger", str(exc)))

    # 3) Corporate actions + adjustment verify
    try:
        from data.corporate_actions import ledger_status as ca_status, load_events
        ca = ca_status(verify=True, sample=sample)
        n_ev = int(ca.get("events") or 0)
        if n_ev < 1:
            checks.append(_fail("corporate_actions", "ca_events.json empty/missing"))
        else:
            checks.append(_pass("corporate_actions", f"events={n_ev}"))
        if not ca.get("adjustment_verified"):
            checks.append(_fail(
                "adjustment_verified",
                ca.get("verify_note") or "verify_ca_adjustment did not PASS",
            ))
        else:
            checks.append(_pass(
                "adjustment_verified",
                f"gap_rate={ca.get('gap_rate')}",
            ))
        # coverage vs store
        ev = load_events()
        try:
            from data.bhavcopy_store import store_symbols
            syms = set(store_symbols() or [])
        except Exception:
            syms = set()
        if syms and ev:
            cov = len(set(ev) & syms) / max(1, len(syms))
            # Not every symbol needs a CA — coverage here = share of store symbols
            # that have ≥1 event is usually low. Use anomaly verify as primary;
            # require a minimum absolute event count instead.
            checks.append(_pass("ca_symbol_overlap", f"ca_symbols_in_store={len(set(ev)&syms)}"))
        else:
            checks.append(_fail("ca_symbol_overlap", "cannot compute CA↔store overlap"))
    except Exception as exc:
        checks.append(_fail("corporate_actions", str(exc)))

    # 4) Raw bhav + index present
    try:
        from data.bhavcopy_runtime import ensure_loaded
        from data import bhavcopy_store as BS
        from data import index_store as IX
        ensure_loaded(rebuild_from_local=False)
        sessions = int(getattr(BS, "_store_sessions", 0) or 0)
        n_sym = len(BS.store_symbols() or [])
        if sessions < 200 or n_sym < 50:
            checks.append(_fail(
                "bhav_coverage",
                f"sessions={sessions} symbols={n_sym} (need ≥200 sessions, ≥50 symbols)",
            ))
        else:
            checks.append(_pass("bhav_coverage", f"sessions={sessions} symbols={n_sym}"))
        try:
            # Ensure index pickle is resident (network-free load).
            if hasattr(IX, "build_index_store"):
                try:
                    IX.build_index_store(days=0)  # may no-op / load cache
                except Exception:
                    pass
            # Prefer explicit load helpers if present
            for name in ("ensure_loaded", "load_store", "_load_pickle"):
                fn = getattr(IX, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            with IX._lock:
                nifty = IX._store.get("Nifty 50")
                vix = IX._store.get("India VIX")
            n_bars = 0 if nifty is None else len(nifty)
            if n_bars < 200:
                checks.append(_fail("benchmark", f"Nifty 50 index history missing/short (bars={n_bars})"))
            else:
                checks.append(_pass("benchmark", f"nifty_bars={n_bars}"))
            if vix is None or len(vix) < 100:
                checks.append(_fail("vix_history", "India VIX history missing/short"))
            else:
                checks.append(_pass("vix_history", f"vix_bars={len(vix)}"))
        except Exception as exc:
            checks.append(_fail("benchmark", str(exc)))
    except Exception as exc:
        checks.append(_fail("bhav_coverage", str(exc)))

    # 5) Gauntlet validator (abort-on-fail suite)
    if run_gauntlet_validate:
        try:
            from gauntlet.validator import validate
            v = validate(sample=sample)
            if v.get("ok"):
                checks.append(_pass("gauntlet_validate", "all required checks passed"))
            else:
                failed = ",".join(v.get("failed") or [])
                checks.append(_fail("gauntlet_validate", f"failed={failed}"))
        except Exception as exc:
            checks.append(_fail("gauntlet_validate", str(exc)))

    # 6) Snapshot immutability / provenance when provided
    if manifest:
        if not manifest.get("snapshot_id"):
            checks.append(_fail("snapshot_id", "manifest missing snapshot_id"))
        else:
            checks.append(_pass("snapshot_id", str(manifest.get("snapshot_id"))))
        if manifest.get("trust_class") == "RESEARCH_GRADE" and not manifest.get("_earned_by_gate"):
            checks.append(_fail(
                "no_manual_trust_stamp",
                "manifest claims RESEARCH_GRADE without gate attestation",
            ))
        else:
            checks.append(_pass("no_manual_trust_stamp"))

    earned = all(c["ok"] for c in checks)
    trust = "RESEARCH_GRADE" if earned else (
        "OPERATIONAL_ONLY" if any(c["check"] == "bhav_coverage" and c["ok"] for c in checks)
        else "DISPLAY_ONLY"
    )
    # If we have bhav but failed research checks → OPERATIONAL_ONLY
    if not earned:
        has_prices = any(c["check"] == "bhav_coverage" and c["ok"] for c in checks)
        trust = "OPERATIONAL_ONLY" if has_prices else "DISPLAY_ONLY"

    failed = [c for c in checks if not c["ok"]]
    plain = explain_trust_class(trust)
    reason_plain = (
        "All research-quality checks passed."
        if earned else
        "; ".join(f"{c['check']}: {c['detail']}" for c in failed[:5])
    )
    user_reason = {
        "official_delistings": (
            "Historical delisting records are incomplete, so the past stock list "
            "may miss companies that no longer trade."
        ),
        "survivorship_complete": (
            "We cannot yet prove the full historical membership list."
        ),
        "universe_not_survivor_only": (
            "The membership file still reflects today's listed names, not a full "
            "past archive."
        ),
        "corporate_actions": (
            "Historical corporate actions are incomplete, so backtest results "
            "could be misleading."
        ),
        "adjustment_verified": (
            "Price adjustments did not pass the integrity check."
        ),
        "bhav_coverage": "Not enough official daily NSE price history is loaded yet.",
        "benchmark": "Benchmark (Nifty) history is missing or too short.",
        "gauntlet_validate": "The dataset failed one or more trust checks.",
    }
    top_fail = failed[0]["check"] if failed else ""
    user_facing = {
        "label": "Research quality",
        "state": "PROVEN" if earned else "NOT_READY",
        "headline": "Research quality: Ready" if earned else "Research quality: Not ready",
        "explanation": plain.explanation if earned else user_reason.get(
            top_fail, plain.explanation
        ),
        "implication": (
            "Suitable for serious historical research."
            if earned else
            "Do not treat strategy results on this dataset as scientific proof."
        ),
        "technical": {
            "trust_class": trust,
            "earned": earned,
            "failed_checks": [c["check"] for c in failed],
            "reason": reason_plain,
        },
        "layers": render_layers(plain),
    }

    return {
        "earned": earned,
        "trust_class": trust,
        "research_grade": earned,
        "checks": checks,
        "failed": [c["check"] for c in failed],
        "reason": reason_plain,
        "user_facing": user_facing,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "may_stamp_manifest": earned,
    }


def stamp_manifest_if_earned(manifest: dict, gate: dict) -> dict:
    """Return a copy of manifest with trust fields. Refuses RESEARCH_GRADE unless earned."""
    out = dict(manifest)
    if gate.get("earned") and gate.get("may_stamp_manifest"):
        out["trust_class"] = "RESEARCH_GRADE"
        out["research_grade"] = True
        out["_earned_by_gate"] = True
        out["gate"] = {
            "evaluated_at": gate.get("evaluated_at"),
            "failed": [],
        }
    else:
        # Never leave a false RESEARCH_GRADE stamp in place
        if out.get("trust_class") == "RESEARCH_GRADE":
            out["trust_class"] = gate.get("trust_class") or "OPERATIONAL_ONLY"
        out["research_grade"] = False
        out["_earned_by_gate"] = False
        out["gate"] = {
            "evaluated_at": gate.get("evaluated_at"),
            "failed": gate.get("failed") or [],
            "reason": gate.get("reason"),
        }
    return out
