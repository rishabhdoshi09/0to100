"""Product-level acceptance tests for the retail projection."""
from product.projection import ProductInputs, TERMINOLOGY, build_product_state


def test_market_closed_home_is_useful_not_blank():
    state = build_product_state(ProductInputs(
        kite_connected=True, active_snapshot_id="abc", data_ready=True,
        paper_auto_enabled=True, worker_running=True, market_open=False,
    ))
    assert "Market closed" in state.headline
    assert state.primary_action == "Run Backtest"
    assert "Find Momentum Stocks" in state.useful_actions


def test_no_data_state_displays_guided_setup():
    state = build_product_state(ProductInputs(kite_connected=True))
    assert state.primary_key == "update_data"
    assert len(state.setup_steps) == 5
    assert any(step.label == "Download market history" for step in state.setup_steps)


def test_zerodha_login_is_first_action_when_disconnected():
    state = build_product_state(ProductInputs())
    assert state.primary_action == "Connect Zerodha"
    assert state.primary_key == "connect"


def test_data_ready_but_paper_off_explains_next_step():
    state = build_product_state(ProductInputs(
        kite_connected=True, active_snapshot_id="abc", data_ready=True,
        paper_auto_enabled=False,
    ))
    assert state.primary_key == "paper"
    assert "paper trading is off" in state.headline.lower()


def test_running_open_market_requires_no_user_trade_approval():
    state = build_product_state(ProductInputs(
        market_open=True, kite_connected=True, active_snapshot_id="abc",
        data_ready=True, paper_auto_enabled=True, worker_running=True,
    ))
    assert state.primary_key == "none"
    assert "Nothing required" in state.primary_action


def test_plain_language_terminology_map():
    assert TERMINOLOGY["snapshot"] == "Saved market data"
    assert TERMINOLOGY["risk_governor"] == "Safety checks"
    assert TERMINOLOGY["no_eligible_intent"] == "No safe trade found"


def test_market_closed_actions_include_backtest_and_fno():
    state = build_product_state(ProductInputs(
        kite_connected=True, active_snapshot_id="abc", data_ready=True,
        paper_auto_enabled=True,
    ))
    assert "Run Backtest" in state.useful_actions
    assert "See F&O Momentum" in state.useful_actions


def test_error_is_surfaced_not_silently_hidden():
    state = build_product_state(ProductInputs(
        kite_connected=True, last_error="snapshot is stale",
    ))
    assert "snapshot is stale" in state.attention
