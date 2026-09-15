"""Every route to a real broker mutation must refuse while the interlock is locked.

Section 9 of the release contract warns that this has regressed before: an
adapter was verified while a second execution route could still reach
trade_executor or the Kite APIs directly. Checking one entry point is therefore
not enough -- this walks every route that can reach an order, including the
legacy compatibility handle and the raw-SDK escape hatches.

A route is safe if it either raises, or provably degrades to paper. Returning
without raising is NOT on its own evidence of safety, so each executor probe
asserts the returned mode.
"""
from __future__ import annotations

import pytest

from product.live_execution_interlock import get_live_execution_state


def test_interlock_reports_locked_and_unauthorized():
    state = get_live_execution_state()
    assert state.locked is True
    assert state.verified is True
    assert state.authorized is False


@pytest.mark.parametrize("call", [
    pytest.param(lambda k: k.cancel_order("1"), id="client_cancel_order"),
    pytest.param(lambda k: k._kite.place_order(tradingsymbol="X"), id="legacy_place_order"),
    pytest.param(lambda k: k._kite.cancel_order(order_id="1"), id="legacy_cancel_order"),
    pytest.param(lambda k: k._kite.place_gtt(trigger_type="x"), id="legacy_place_gtt"),
    pytest.param(lambda k: k._kite.modify_order(order_id="1"), id="legacy_modify_order"),
])
def test_every_mutation_route_is_refused(call):
    from data.kite_client import KiteClient

    with pytest.raises(Exception) as excinfo:
        call(KiteClient())
    # a TypeError from a bad signature would not prove the guard fired
    assert "LiveExecutionBlocked" in type(excinfo.value).__name__ or \
        "not allowed" in str(excinfo.value).lower() or \
        "blocked" in str(excinfo.value).lower(), \
        f"route did not refuse for a live-execution reason: {excinfo.value!r}"


@pytest.mark.parametrize("attr", ["_raw", "__raw", "_GuardedKiteProxy__raw"])
def test_raw_sdk_handle_is_not_reachable(attr):
    """The compatibility proxy must not leak the unguarded KiteConnect object."""
    from data.kite_client import KiteClient

    with pytest.raises(AttributeError):
        getattr(KiteClient()._kite, attr)


@pytest.mark.parametrize("paper_flag", [False, True])
def test_place_trade_cannot_send_a_live_order(paper_flag):
    """paper=False must still degrade to paper while live execution is locked.

    This probe asserts the RESULT rather than the absence of an exception: the
    executor legitimately returns success here, and only the mode proves that
    no live order was sent.
    """
    from execution.trade_executor import place_trade

    result = place_trade(symbol="TESTFIXB", qty=1, entry_type="MARKET",
                         entry_price=100.0, stop=95.0, target=110.0,
                         paper=paper_flag)
    assert result.get("mode") == "PAPER", f"live order path reached: {result!r}"
    assert not result.get("order_id"), "a broker order id was produced"
    if not paper_flag:
        assert result.get("fallback") is True, "live request must record the downgrade"


def test_unsafe_legacy_override_cannot_defeat_the_interlock(monkeypatch):
    """Worst case: the emergency legacy override is ON.

    QT_ENABLE_UNSAFE_LEGACY_LIVE exists as a compatibility escape hatch and is
    false by default, but a defence that only works while nobody sets the
    override is not a defence. The interlock must be the final authority, so
    even with the override enabled no live order may leave the process.
    """
    monkeypatch.setenv("QT_ENABLE_UNSAFE_LEGACY_LIVE", "1")
    from execution.trade_executor import legacy_live_enabled, place_trade

    assert legacy_live_enabled() is True, "precondition: the override is genuinely on"
    result = place_trade(symbol="TESTFIXB", qty=1, entry_type="MARKET",
                         entry_price=100.0, stop=95.0, target=110.0, paper=False)
    assert result.get("mode") == "PAPER", f"live order reached: {result!r}"
    assert not result.get("order_id")


@pytest.mark.parametrize("var", [
    "QT_ALLOW_LIVE", "LIVE_TRADING_ENABLED", "QUANTTERM_LIVE", "QT_LIVE_EXECUTION",
])
def test_no_stray_env_var_enables_live_execution(monkeypatch, var):
    """Only the one explicitly-named unsafe override exists; nothing else flips live."""
    monkeypatch.setenv(var, "1")
    from execution.trade_executor import legacy_live_enabled

    assert legacy_live_enabled() is False, f"{var} unexpectedly enabled live execution"
