"""
Deterministic, network-free tests for the Strategy Studio (autonomous discovery +
human-in-the-loop approval). No network, no wall-clock dependence (except provenance
timestamps we never assert on), no Streamlit runtime, no order path.

Synthetic fixtures verify SOFTWARE behaviour only — never presented as market evidence.
"""
from __future__ import annotations

import inspect

import pytest

from research.strategy_studio import spec as S
from research.strategy_studio import discovery as D
from research.strategy_studio import review as R
from research.strategy_studio import tweak as T
from research.strategy_studio import approval as A
from research.strategy_studio import wizard as W


def _ev(**kw):
    base = dict(n_trades=40, n_symbols=20, gross_expectancy_R=0.4, net_expectancy_R=0.3,
                cost_drag_R=0.1, p_value=0.02, max_drawdown=0.2, turnover=1.5,
                max_symbol_share=0.1, regime_consistency=0.7, sector_consistency=0.6,
                verdict="PROMOTE", is_synthetic=False)
    base.update(kw)
    return D.EvidenceReport(**base)


def _cand(seed=1):
    return D.generate(D.DiscoveryBudget(seed=seed, max_search_attempts=30))[0]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Discovery
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscovery:
    def test_reproducible_candidate_generation(self):
        a = D.generate(D.DiscoveryBudget(seed=5, max_search_attempts=30))
        b = D.generate(D.DiscoveryBudget(seed=5, max_search_attempts=30))
        assert [x.config_hash() for x in a] == [x.config_hash() for x in b] and a

    def test_search_budget_enforced(self):
        small = D.generate(D.DiscoveryBudget(seed=1, max_search_attempts=5))
        assert len(small) <= 5

    def test_attempts_are_all_logged(self):
        reg = D.AttemptRegistry()
        for c in D.generate(D.DiscoveryBudget(seed=1, max_search_attempts=22)):
            reg.record(c, S.UNDER_REVIEW)
        assert len(reg) >= 1 and len(reg.all()) == len(reg)

    def test_family_diversity(self):
        cands = D.generate(D.DiscoveryBudget(seed=3, max_search_attempts=44))
        assert len({c.family for c in cands}) >= 6      # not one family repeated

    def test_leakage_and_impossible_entry_rejected(self):
        leak = S.StrategySpec("STR-L", "leak", 1, "h", "breakout",
                              entry_rules=("require:breakout_pivot",),
                              feature_defs=("breakout_pivot",),
                              thresholds={"entry_timing": "same_bar_close"})
        reasons = D.structural_reasons(leak, D.DiscoveryBudget())
        assert any("leakage" in r for r in reasons)
        assert any("impossible entry" in r for r in reasons)

    def test_unsupported_pit_input_rejected(self):
        # a fundamentals block with no PIT fundamentals available
        s = S.StrategySpec("STR-F", "f", 1, "h", "quality_plus_momentum",
                           feature_defs=("earnings_growth",),
                           thresholds={"entry_timing": "next_bar_open"})
        r = D.structural_reasons(s, D.DiscoveryBudget(), has_fundamentals_pit=False)
        assert any("point-in-time" in x for x in r)

    def test_cost_greater_than_edge_rejected(self):
        ev = _ev(gross_expectancy_R=0.2, net_expectancy_R=-0.05, cost_drag_R=0.25)
        assert any("costs" in r for r in D.evidence_reasons(ev, D.DiscoveryBudget()))

    def test_minimum_sample_rejected(self):
        ev = _ev(n_trades=5)
        assert any("too few trades" in r for r in D.evidence_reasons(ev, D.DiscoveryBudget()))

    def test_concentration_rejected(self):
        ev = _ev(max_symbol_share=0.9)
        assert any("concentrated" in r for r in D.evidence_reasons(ev, D.DiscoveryBudget()))

    def test_untouched_test_isolation(self):
        t = D.UntouchedTestSet()
        assert t.evaluate_once(lambda: 42) == 42
        with pytest.raises(RuntimeError):
            t.evaluate_once(lambda: 1)                    # second touch forbidden

    def test_multiple_testing_burden_rises_with_attempts(self):
        few = D.evidence_burden(1)
        many = D.evidence_burden(50000)
        assert many > few                                 # 50k attempts → stronger burden
        mt = D.apply_multiple_testing([0.01, 0.2], n_total_attempts=50000)
        assert mt["family_size"] == 50000 and "bh" in mt

    def test_simpler_baseline_prefers_simpler_when_similar(self):
        strat = _ev(net_expectancy_R=0.31); base = _ev(net_expectancy_R=0.30)
        cmp = D.simpler_baseline_comparison(strat, base, 8, 2)
        assert cmp["prefer_simpler"] is True              # within noise → prefer simpler
        strat2 = _ev(net_expectancy_R=0.50)
        cmp2 = D.simpler_baseline_comparison(strat2, base, 8, 2)
        assert cmp2["complexity_justified"] is True

    def test_discovery_unavailable_without_data(self):
        assert D.data_ready(None) is False
        assert D.data_ready({"color": "red", "can_run": False}) is False
        assert D.data_ready({"color": "green", "can_run": True}) is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. Versioning & tweaking
# ══════════════════════════════════════════════════════════════════════════════

class TestVersioning:
    def test_material_tweak_creates_new_version(self):
        c = _cand()
        diff = T.parse_nl("reduce the maximum stop to 5%", c)
        new = T.apply_diff(c, diff)
        assert new.version == c.version + 1 and new.parent_id == c.strategy_id

    def test_old_version_unchanged_and_hash_changes(self):
        c = _cand()
        old_hash = c.config_hash()
        new = T.apply_diff(c, T.parse_nl("hold winners longer", c))
        assert c.config_hash() == old_hash                # parent untouched (frozen)
        assert new.config_hash() != old_hash

    def test_display_change_does_not_alter_evidence_identity(self):
        c = _cand()
        renamed = T.apply_display_change(c, name="Nicer name", hypothesis="reworded")
        assert renamed.config_hash() == c.config_hash()   # evidence identity preserved
        assert renamed.version == c.version

    def test_nl_maps_to_explicit_diff(self):
        d = T.parse_nl("remove the sector filter", _cand())
        assert d["status"] == "ready" and d["field"] == "sector_conditions"
        assert "old" in d and "proposed" in d and d["new_version_required"]

    def test_unsafe_request_rejected(self):
        d = T.parse_nl("make it always win with no losses", _cand())
        assert d["status"] == "rejected"

    def test_ambiguous_request_needs_clarification(self):
        d = T.parse_nl("do something clever", _cand())
        assert d["status"] == "needs_clarification"

    def test_tweak_preview_requires_new_version(self):
        c = _cand()
        prev = T.tweak_impact_preview(c, T.parse_nl("reduce the maximum stop to 5%", c))
        assert prev["new_version_required"] is True
        assert "run a new historical test" in prev["required_action"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. User approval + paper activation
# ══════════════════════════════════════════════════════════════════════════════

class TestApproval:
    _GREEN = {"color": "green", "can_run": True, "reasons": []}

    def _approve(self, actor="user", ev=None, readiness=None, state=S.AWAITING_USER_APPROVAL):
        return A.approve_for_paper(
            _cand(), ev or _ev(), readiness or self._GREEN, actor=actor, approver="u",
            current_state=state, max_allocation=100000, max_open_risk_pct=5,
            max_trades_per_day=3, review_date="2026-09-01")

    def test_research_code_cannot_self_approve(self):
        with pytest.raises(S.LifecycleError):
            self._approve(actor="system")

    def test_user_approval_makes_immutable_paper_only_record(self):
        rec, state = self._approve()
        assert state == S.APPROVED_FOR_PAPER
        assert rec.allowed_mode == "PAPER"
        with pytest.raises(Exception):
            rec.allowed_mode = "LIVE"                     # frozen/immutable

    def test_red_gate_prevents_approval(self):
        with pytest.raises(A.ApprovalRefused):
            self._approve(readiness={"color": "red", "can_run": False})

    def test_synthetic_evidence_cannot_be_approved(self):
        with pytest.raises(A.ApprovalRefused):
            self._approve(ev=_ev(is_synthetic=True))

    def test_non_survivor_verdict_cannot_be_approved(self):
        with pytest.raises(A.ApprovalRefused):
            self._approve(ev=_ev(verdict="INCONCLUSIVE"))

    def test_paper_activation_requires_separate_confirmation(self):
        c = _cand()
        rec, _ = A.approve_for_paper(c, _ev(), self._GREEN, actor="user", approver="u",
                                     current_state=S.AWAITING_USER_APPROVAL,
                                     max_allocation=100000, max_open_risk_pct=5,
                                     max_trades_per_day=3, review_date="2026-09-01")
        with pytest.raises(A.ApprovalRefused):
            A.activate_paper(rec, c, actor="user", confirmed=False)   # approval alone insufficient
        act = A.activate_paper(rec, c, actor="user", confirmed=True)
        assert act.mode == "PAPER"

    def test_old_approval_does_not_transfer_to_tweaked_version(self):
        c = _cand()
        rec, _ = A.approve_for_paper(c, _ev(), self._GREEN, actor="user", approver="u",
                                     current_state=S.AWAITING_USER_APPROVAL,
                                     max_allocation=100000, max_open_risk_pct=5,
                                     max_trades_per_day=3, review_date="2026-09-01")
        tweaked = T.apply_diff(c, T.parse_nl("reduce the maximum stop to 5%", c))
        assert A.approval_valid_for(rec, c) is True
        assert A.approval_valid_for(rec, tweaked) is False
        with pytest.raises(A.ApprovalRefused):
            A.activate_paper(rec, tweaked, actor="user", confirmed=True)

    def test_no_live_mode_anywhere_in_approval(self):
        src = inspect.getsource(A)
        # the only mode issued is PAPER; no live activation path exists
        assert '"LIVE"' not in src.replace('!= "LIVE"', "")  # only guards, never grants
        assert "allowed_mode=\"PAPER\"" in src.replace(" ", "") or 'allowed_mode="PAPER"' in src


# ══════════════════════════════════════════════════════════════════════════════
# 4. Explainability
# ══════════════════════════════════════════════════════════════════════════════

class TestExplainability:
    def _cm(self, ev=None, trades=None):
        return R.convince_me(_cand(), ev or _ev(), data_status={"color": "green", "can_run": True},
                             n_attempts=1000, n_rejected=990,
                             trades=trades or [{"net_R": 1.0}, {"net_R": -1.0}],
                             dataset_period="2019-2024")

    def test_review_has_defence_and_prosecution(self):
        cm = self._cm()
        assert cm["why_it_may_work"] and cm["what_could_go_wrong"]

    def test_representative_trades_include_wins_and_losses(self):
        cm = self._cm(trades=[{"net_R": 2.0}, {"net_R": -0.5}, {"net_R": 0.1}])
        assert cm["example_trades"]["wins"] and cm["example_trades"]["losses"]
        assert cm["example_trades"]["cherry_picked"] is False

    def test_attempt_count_disclosed(self):
        cm = self._cm()
        assert cm["why_this_is_not_luck"]["total_strategies_attempted"] == 1000

    def test_limitations_displayed(self):
        cm = self._cm()
        assert "evidence_limitations" in cm["what_could_go_wrong"]

    def test_confidence_not_collapsed_into_one_number(self):
        cm = self._cm()
        conf = cm["confidences"]
        assert {"evidence", "prediction", "data", "stability", "execution"} <= set(conf)

    def test_synthetic_results_labelled_non_evidence(self):
        cm = self._cm(ev=_ev(is_synthetic=True))
        assert cm["synthetic_labelled_non_evidence"] is True
        assert cm["evidence_summary"]["is_market_evidence"] is False
        assert cm["system_recommendation"] == R.RECO_UNSUITABLE

    def test_recommendation_never_over_promises(self):
        for txt in ("guaranteed 20% returns", "safe profit assured"):
            assert R.sanitize_language(txt).startswith("[removed")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Comparison & wizard
# ══════════════════════════════════════════════════════════════════════════════

class TestComparisonAndWizard:
    def test_comparison_never_auto_selects_and_flags_noise(self):
        a = {"name": "A", "spec": _cand(), "ev": _ev(net_expectancy_R=0.30)}
        b = {"name": "B", "spec": _cand(2), "ev": _ev(net_expectancy_R=0.31)}
        cmp = R.compare([a, b])
        assert cmp["auto_selected"] is None
        assert cmp["difference_is_meaningful"] is False

    def test_wizard_builds_a_spec_subject_to_same_standards(self):
        spec = W.wizard_to_spec({"behaviour": "a breakout from a base", "universe": "nifty200",
                                 "entry": "breakout above resistance", "stop": "structural_stop",
                                 "exit": "trend_exit", "regime": "only strong markets",
                                 "risk": "medium"})
        assert spec.family == "breakout" and spec.generation_method == "user_wizard"
        # a manual spec still faces structural validity like any other
        assert D.structural_reasons(spec, D.DiscoveryBudget()) == []


# ══════════════════════════════════════════════════════════════════════════════
# 6. Isolation & regression
# ══════════════════════════════════════════════════════════════════════════════

class TestIsolationAndRegression:
    def test_no_module_imports_an_order_path(self):
        import research.strategy_studio as pkg
        from research.strategy_studio import (spec, grammar, discovery, review, tweak,
                                              approval, wizard)
        for mod in (pkg, spec, grammar, discovery, review, tweak, approval, wizard):
            src = inspect.getsource(mod)
            code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
            for pat in ("import execution", "from execution", "import alerts", "from alerts",
                        ".place_trade(", "place_trade(", "kite_client", "GTT", ".arm("):
                assert pat not in code, f"{mod.__name__} references {pat}"

    def test_ui_page_has_no_order_actions(self):
        from ui import strategy_studio_page as page
        src = inspect.getsource(page)
        code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
        for pat in ("place_trade", ".arm(", "GTT", "zerodha", "kite_client",
                    "telegram_actions", "consider("):
            assert pat not in code, f"strategy_studio_page references {pat}"
        # there must be NO live-approval control on the page
        assert "Approve for Live" not in code and "approve_for_live" not in code

    def test_paper_autopilot_and_live_lock_unchanged(self, monkeypatch):
        import execution.autopilot as ap
        monkeypatch.delenv("QT_LIVE_ENABLED", raising=False)
        assert ap._live_enabled() is False

    def test_telegram_order_path_still_paper_only(self):
        import alerts.telegram_actions as ta
        src = inspect.getsource(ta)
        assert src.count("place_trade(") == 1 and "paper=True" in src

    def test_generation_is_wall_clock_independent(self):
        # same seed → identical specs regardless of when the test runs
        a = D.generate(D.DiscoveryBudget(seed=9, max_search_attempts=15))
        b = D.generate(D.DiscoveryBudget(seed=9, max_search_attempts=15))
        assert [x.as_dict() for x in a] == [x.as_dict() for x in b]
