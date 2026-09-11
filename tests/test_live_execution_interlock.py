from __future__ import annotations

import pytest

from data.kite_client import KiteClient
from execution.fo_executor import FnOExecutor
from product.execution_adapter import LiveExecutionAdapter
from product.live_execution_interlock import (
    LiveExecutionBlocked,
    get_live_execution_state,
)
from product.startup_check import _aggregate_operational, _lane, _live_lock_readiness


class FakeKite:
    GTT_TYPE_OCO = "two-leg"

    def __init__(self) -> None:
        self.mutations: list[str] = []
        self.reads: list[str] = []

    def _mutation(self, name: str, *args, **kwargs):
        self.mutations.append(name)
        return "SHOULD_NOT_BE_REACHED"

    def place_order(self, *args, **kwargs):
        return self._mutation("place_order", *args, **kwargs)

    def modify_order(self, *args, **kwargs):
        return self._mutation("modify_order", *args, **kwargs)

    def cancel_order(self, *args, **kwargs):
        return self._mutation("cancel_order", *args, **kwargs)

    def place_gtt(self, *args, **kwargs):
        return self._mutation("place_gtt", *args, **kwargs)

    def modify_gtt(self, *args, **kwargs):
        return self._mutation("modify_gtt", *args, **kwargs)

    def delete_gtt(self, *args, **kwargs):
        return self._mutation("delete_gtt", *args, **kwargs)

    def convert_position(self, *args, **kwargs):
        return self._mutation("convert_position", *args, **kwargs)

    def orders(self):
        self.reads.append("orders")
        return []


def _client(fake: FakeKite) -> KiteClient:
    client = object.__new__(KiteClient)
    client._kite = fake
    client._access_token = "test-token"
    client._api_key = "test-key"
    client._api_secret = "test-secret"
    return client


def test_environment_flags_cannot_authorize_live(monkeypatch):
    monkeypatch.setenv("QT_LIVE_ENABLED", "1")
    monkeypatch.setenv("QT_ENABLE_UNSAFE_LEGACY_LIVE", "1")
    state = get_live_execution_state()
    assert state.verified is True
    assert state.locked is True
    assert state.authorized is False
    assert state.status == "LOCKED"


def test_kite_place_order_is_blocked_before_sdk_call():
    fake = FakeKite()
    client = _client(fake)

    with pytest.raises(LiveExecutionBlocked):
        client.place_order("INFY", "BUY", 1)

    assert fake.mutations == []


@pytest.mark.parametrize(
    "method",
    [
        "place_order",
        "modify_order",
        "cancel_order",
        "place_gtt",
        "modify_gtt",
        "delete_gtt",
        "convert_position",
    ],
)
def test_raw_broker_mutations_share_the_same_fail_closed_boundary(method):
    fake = FakeKite()
    client = _client(fake)

    with pytest.raises(LiveExecutionBlocked):
        getattr(client.raw, method)()

    assert fake.mutations == []


def test_raw_read_only_methods_remain_available():
    fake = FakeKite()
    client = _client(fake)

    assert client.raw.orders() == []
    assert fake.reads == ["orders"]
    assert fake.mutations == []


def test_fno_direct_raw_place_order_bypass_is_blocked(monkeypatch):
    fake = FakeKite()
    executor = FnOExecutor(_client(fake))
    monkeypatch.setattr(
        executor,
        "get_front_month_future",
        lambda _symbol: {
            "tradingsymbol": "INFY26SEPFUT",
            "instrument_token": 123,
            "expiry": None,
            "lot_size": 1,
        },
    )
    monkeypatch.setattr(
        executor,
        "check_margin",
        lambda **_kwargs: (True, 1.0, 1000.0),
    )

    result = executor.place_futures_order("INFY", "BUY", 1)

    assert result["status"] == "error"
    assert "blocked" in result["reason"].lower()
    assert fake.mutations == []


def test_live_adapter_uses_canonical_interlock():
    with pytest.raises(LiveExecutionBlocked):
        LiveExecutionAdapter().submit(object())


def test_startup_reports_verified_locked_state():
    locked, verified, detail, payload = _live_lock_readiness()
    assert locked is True
    assert verified is True
    assert payload["locked"] is True
    assert payload["authorized"] is False
    assert detail


def test_startup_verification_failure_is_not_a_green_lock(monkeypatch):
    import product.live_execution_interlock as interlock

    def broken_state():
        raise RuntimeError("verification unavailable")

    monkeypatch.setattr(interlock, "get_live_execution_state", broken_state)
    locked, verified, detail, payload = _live_lock_readiness()

    assert locked is True  # physical policy remains fail-closed
    assert verified is False  # but readiness cannot claim positive evidence
    assert payload == {}
    assert "could not be verified" in detail.lower()

    operational = _aggregate_operational(
        [_lane("LIVE MONEY", "UNVERIFIED", detail, required=True)],
        live_locked=locked,
        live_lock_verified=verified,
    )
    assert operational["ready"] is False
    assert operational["status"] == "FAILED"
    assert "LIVE MONEY" in operational["blockers"]
