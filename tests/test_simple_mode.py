"""
Deterministic, network-free tests for QuantTerm Simple Mode.

Simple Mode is PRESENTATION ONLY. These tests prove it: the plain-language logic is
correct, a green "Ready" is never shown on missing/stale data, every safety and
isolation guarantee from earlier milestones still holds, and nothing in the Simple
layer can reach an order path. No wall-clock, no network, no streamlit runtime needed
(the pure logic lives in core.simple_language; the Streamlit layer is checked by source
inspection).
"""
from __future__ import annotations

import inspect

import core.simple_language as S


# ══════════════════════════════════════════════════════════════════════════════
# 1. Simple is the default; depth is presentation only
# ══════════════════════════════════════════════════════════════════════════════

class TestModeDefaults:
    def test_simple_is_the_new_user_default(self):
        assert S.DEFAULT_MODE == S.SIMPLE
        assert S.is_simple(None) is True          # unset → Simple
        assert S.is_simple("") is True
        assert S.is_simple("anything_unknown") is True

    def test_advanced_is_opt_in_only(self):
        assert S.is_simple(S.ADVANCED) is False
        assert S.is_simple(S.SIMPLE) is True

    def test_depth_flag_carries_no_trading_authority(self):
        # the depth constants are plain strings — no execution/config coupling
        assert isinstance(S.SIMPLE, str) and isinstance(S.ADVANCED, str)
        # the pure module imports nothing that can trade
        src = inspect.getsource(S)
        for pat in ("import execution", "from execution", "place_trade", "arm(",
                    "import streamlit", "requests", "kite"):
            assert pat not in src, f"core.simple_language references {pat}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Home status — Simple hides detail, never risk
# ══════════════════════════════════════════════════════════════════════════════

class TestHomeStatus:
    def test_missing_data_is_never_ready(self):
        h = S.home_status({"mode": "PAPER", "data_ok": False, "market_open": True,
                           "autopilot_armed": True})
        assert h["headline"] == "DATA_MISSING"
        assert h["headline_status"]["tone"] == "bad"

    def test_stale_data_is_never_ready(self):
        h = S.home_status({"mode": "PAPER", "data_ok": True, "data_stale": True,
                           "market_open": True})
        assert h["headline"] == "STALE_DATA"

    def test_safety_stop_is_visible_in_home(self):
        h = S.home_status({"mode": "PAPER", "data_ok": True, "market_open": True,
                           "safety_stop": True})
        assert h["headline"] == "DAILY_SAFETY_STOP"
        assert "safety stop" in h["answers"]["safety_stop"].lower()

    def test_trade_limit_is_visible_in_home(self):
        h = S.home_status({"mode": "PAPER", "data_ok": True, "market_open": True,
                           "trades_allowed": 3, "trades_used": 3})
        assert "3 of 3" in h["answers"]["trades"]
        assert "not happening" in h["answers"]["trading_allowed"].lower()

    def test_healthy_paper_is_ready_only_when_everything_clears(self):
        h = S.home_status({"mode": "PAPER", "data_ok": True, "market_open": True,
                           "autopilot_armed": True, "trades_allowed": 4, "trades_used": 0})
        assert h["headline"] == "PAPER_PRACTICE_ACTIVE"

    def test_home_always_states_live_is_locked(self):
        h = S.home_status({"mode": "PAPER", "data_ok": True, "market_open": True})
        assert "locked" in h["answers"]["live"].lower()

    def test_next_best_action_prioritises_data_then_safety(self):
        assert "historical market data" in S.next_best_action({"data_ok": False})
        assert "safety stop" in S.next_best_action(
            {"data_ok": True, "safety_stop": True}).lower()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Decisions & research verdicts explained correctly
# ══════════════════════════════════════════════════════════════════════════════

class TestExplanations:
    def test_disabled_reason_is_always_given(self):
        # a rejected setup always exposes a plain reason + next step (no dead ends)
        for code in ("OVEREXTENDED_CHASE", "STRUCTURAL_RISK_TOO_HIGH",
                     "UNCONFIRMED_BREAKOUT", "DAILY_TRADE_LIMIT", "DAILY_SAFETY_STOP",
                     "NO_FILL", "DATA_UNAVAILABLE"):
            d = S.decision_for(code)
            assert d["decision"] and d["main_reason"] and d["next_step"]

    def test_unknown_code_falls_back_honestly(self):
        d = S.decision_for("SOME_NEW_CODE")
        assert d["decision"] == "Skipped"
        assert "exact technical reason" in d["next_step"].lower()  # points to Advanced

    def test_verdicts_are_explained_without_overclaiming(self):
        assert "not a promise" in S.verdict_meaning("PASS")["plain"].lower()
        assert "did not meet" in S.verdict_meaning("FAIL")["plain"].lower()
        assert "not enough" in S.verdict_meaning("INCONCLUSIVE")["plain"].lower()
        assert "not installed" in S.verdict_meaning("DATA_UNAVAILABLE")["plain"].lower()

    def test_pass_is_never_described_as_guaranteed_profit(self):
        for v in S.VERDICTS.values():
            assert "guarantee" not in v["plain"].lower() or "not" in v["plain"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Data-unavailable honest experience
# ══════════════════════════════════════════════════════════════════════════════

class TestDataUnavailable:
    def test_panel_is_honest_and_actionable(self):
        p = S.data_unavailable_panel(operator_step="Install the NSE bhavcopy data.")
        assert p["current_status"] == "INCONCLUSIVE — DATA UNAVAILABLE"
        assert "not installed" in p["what_happened"].lower()
        assert "cannot honestly" in p["what_it_means"].lower()
        assert p["what_still_works"] and p["what_to_do_next"]

    def test_missing_is_never_treated_as_zero_in_words(self):
        # the "what not to do" list explicitly warns against missing==zero
        joined = " ".join(x["dont"] + x["because"] for x in S.WHAT_NOT_TO_DO).lower()
        assert "missing data means zero" in joined or "missing is unknown" in joined


# ══════════════════════════════════════════════════════════════════════════════
# 5. Behaviour matrix matches real permissions (no real-order path in Simple)
# ══════════════════════════════════════════════════════════════════════════════

class TestBehaviourMatrix:
    def test_no_row_ever_offers_a_real_order(self):
        assert S.any_row_allows_real_order() is False

    def test_research_rows_block_orders(self):
        for name in ("research_data_available", "research_data_unavailable"):
            row = S.matrix_row(name)
            assert "place any order" in " ".join(row["blocked_actions"])

    def test_live_row_blocks_arming_and_real_orders(self):
        row = S.matrix_row("live_migration_lock")
        blocked = " ".join(row["blocked_actions"]).lower()
        assert "arm live" in blocked and "real order" in blocked
        assert "environment variable alone" in row["explanation"].lower()

    def test_telegram_row_is_paper_only(self):
        row = S.matrix_row("telegram_paper_action")
        blocked = " ".join(row["blocked_actions"]).lower()
        assert "real order" in blocked and "ever" in blocked   # paper-only, never live
        assert "only ever" in row["explanation"].lower()

    def test_every_row_has_explanation_and_next(self):
        for row in S.BEHAVIOUR_MATRIX:
            assert row["explanation"] and row["next_action"]
            assert row["available_actions"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# 6. Safety-sensitive confirmations are SPECIFIC, not generic
# ══════════════════════════════════════════════════════════════════════════════

class TestSafetyConfirmations:
    def test_confirmation_is_specific_and_states_live_impact(self):
        c = S.safety_confirmation("the daily PAPER trade limit", 3, 5,
                                  "This permits up to two additional paper trades per "
                                  "NSE trading day.", "PAPER", "Change it back to 3")
        assert "from 3 to 5" in c["message"]
        assert "does NOT enable LIVE trading" in c["message"]
        assert c["message"].lower().count("are you sure") == 0   # never generic

    def test_live_affecting_change_does_not_falsely_reassure(self):
        c = S.safety_confirmation("live allocation", 0, 100000, "This would size live "
                                  "orders.", "LIVE", "Set back to 0")
        assert "does NOT enable LIVE" not in c["message"]  # honest for a LIVE change


# ══════════════════════════════════════════════════════════════════════════════
# 7. Contextual help + content completeness
# ══════════════════════════════════════════════════════════════════════════════

class TestContentCompleteness:
    def test_every_major_page_has_five_question_help(self):
        for page in ("home", "opportunities", "positions", "research_lab",
                     "safety", "data_health", "help"):
            q = S.page_help(page)
            assert all(q[k] for k in ("what_is_this", "why_it_matters",
                                      "what_should_i_do", "what_will_happen", "what_next"))

    def test_onboarding_is_at_most_seven_steps(self):
        assert 1 <= len(S.ONBOARDING_STEPS) <= 7
        assert any("cannot promise" in s["title"].lower() for s in S.ONBOARDING_STEPS)

    def test_glossary_covers_the_scary_words(self):
        for term in ("Circuit breaker", "Expectancy", "Drawdown", "Migration lock",
                     "DATA_UNAVAILABLE", "PASS", "Structural stop"):
            assert S.define(term)

    def test_user_action_guide_marks_real_money_false_everywhere_in_simple_flow(self):
        # every documented beginner action is non-live (real_money False)
        assert all(a["real_money"] is False for a in S.USER_ACTION_GUIDE)
        # and each entry is complete
        for a in S.USER_ACTION_GUIDE:
            assert a["action"] and a["checks"] and a["verify"]

    def test_success_is_defined_as_process_not_profit(self):
        assert "not necessarily a profitable day" in S.GOOD_DAY
        assert len(S.SUCCESS_CHECKLIST) >= 8


# ══════════════════════════════════════════════════════════════════════════════
# 8. Fictional walkthrough touches nothing live
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkthrough:
    def test_walkthrough_is_fictional(self):
        assert S.WALKTHROUGH_FICTIONAL is True
        assert len(S.WALKTHROUGH_STEPS) >= 10

    def test_walkthrough_covers_the_required_lessons(self):
        text = " ".join(s["title"] + s["body"] for s in S.WALKTHROUGH_STEPS).lower()
        for lesson in ("qualified", "skipped", "after the signal", "stop",
                       "maximum", "no fill", "winning", "losing", "safety stop",
                       "no valid", "inconclusive"):
            assert lesson in text, f"walkthrough missing lesson: {lesson}"


# ══════════════════════════════════════════════════════════════════════════════
# 9. Execution isolation — the Simple layer cannot trade or bypass a control
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionIsolation:
    def test_simple_ui_layer_imports_no_order_path(self):
        import ui.simple_mode as sm
        src = inspect.getsource(sm)
        code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
        for pat in (".place_trade(", "place_trade(", ".arm(", "ap.arm", "consider(",
                    "import alerts", "from alerts", "kite_client", "GTT"):
            assert pat not in code, f"ui.simple_mode references {pat}"

    def test_simple_layer_only_reads_autopilot_status(self):
        # the only execution touchpoint is the pure read get_status()
        import ui.simple_mode as sm
        src = inspect.getsource(sm)
        assert "get_status()" in src
        assert "set_config" not in src and "set_status" not in src

    def test_paper_autopilot_and_live_lock_unchanged(self, monkeypatch):
        import execution.autopilot as ap
        monkeypatch.delenv("QT_LIVE_ENABLED", raising=False)
        assert ap._live_enabled() is False        # LIVE still migration-locked

    def test_telegram_order_path_still_paper_only(self):
        import alerts.telegram_actions as ta
        src = inspect.getsource(ta)
        assert src.count("place_trade(") == 1 and "paper=True" in src

    def test_exp006_research_still_execution_isolated(self):
        import inspect as _i
        from research.momentum_breakout import runner as R
        src = _i.getsource(R)
        code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
        for pat in ("import execution", "from execution", ".place_trade("):
            assert pat not in code


# ══════════════════════════════════════════════════════════════════════════════
# 10. App wiring — the beginner routes exist (no broken primary routes)
# ══════════════════════════════════════════════════════════════════════════════

class TestAppWiring:
    def test_app_registers_the_simple_pages(self):
        src = open("app.py").read()
        for page in ("Simple Home", "Getting Started", "Practice Walkthrough",
                     "User Guide"):
            assert f'"{page}"' in src, f"app.py does not route {page}"

    def test_app_defaults_new_user_to_getting_started(self):
        src = open("app.py").read()
        assert "_qt_onboarded" in src and "Getting Started" in src

    def test_app_uses_the_simple_layer_not_a_duplicate(self):
        src = open("app.py").read()
        assert "from ui import simple_mode" in src


# ══════════════════════════════════════════════════════════════════════════════
# 11. Usability acceptance — scripted "a new user can..." scenarios
# ══════════════════════════════════════════════════════════════════════════════

class TestUsabilityAcceptance:
    def test_new_user_can_tell_if_real_money_is_at_risk(self):
        # in every mode the plain text tells them about real money
        for mode in ("RESEARCH", "PAPER", "LIVE"):
            assert S.mode_meaning(mode)["money"]

    def test_new_user_can_recognise_no_trade_as_valid(self):
        assert "correct" in S.decision_for("NO_VALID_SETUP")["risk"].lower() or \
               "correct" in S.matrix_row("no_valid_setup")["explanation"].lower()

    def test_new_user_can_respond_to_data_unavailable(self):
        d = S.decision_for("DATA_UNAVAILABLE")
        assert "install" in d["next_step"].lower() or "data health" in d["next_step"].lower()

    def test_learning_cards_explain_key_confusions(self):
        for key in ("no_trade_is_good", "paper_vs_live", "gap_stop",
                    "high_score_can_fail", "missing_data_blocks_verdict"):
            assert S.LEARNING_CARDS[key]["title"] and S.LEARNING_CARDS[key]["body"]
