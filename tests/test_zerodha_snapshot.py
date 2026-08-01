from __future__ import annotations

from datetime import datetime, timezone

from execution.reconciliation.zerodha_snapshot import capture_zerodha_snapshot


class FakeKite:
    def __init__(self):
        self.calls = []

    def orders(self):
        self.calls.append("orders")
        return [
            {
                "order_id": "broker-order-1",
                "status": "COMPLETE",
                "tradingsymbol": "AAA",
                "transaction_type": "BUY",
                "quantity": 10,
                "filled_quantity": 10,
                "average_price": 101.5,
                "tag": "submit-1",
                "exchange_update_timestamp": "2026-08-01 10:00:00",
            }
        ]

    def trades(self):
        self.calls.append("trades")
        return [
            {
                "trade_id": "trade-1",
                "order_id": "broker-order-1",
                "tradingsymbol": "AAA",
                "transaction_type": "BUY",
                "quantity": 10,
                "average_price": 101.5,
                "fill_timestamp": "2026-08-01 10:00:01",
            }
        ]

    def positions(self):
        self.calls.append("positions")
        return {
            "net": [
                {
                    "tradingsymbol": "AAA",
                    "quantity": 10,
                    "average_price": 101.5,
                    "product": "CNC",
                }
            ],
            "day": [],
        }

    def margins(self):
        self.calls.append("margins")
        return {
            "equity": {
                "available": {"cash": 50_000, "live_balance": 55_000},
                "net": 60_000,
            }
        }

    def get_gtts(self):
        self.calls.append("get_gtts")
        return [
            {
                "id": 9001,
                "status": "active",
                "metadata": {"order_id": "oms-1"},
                "condition": {
                    "tradingsymbol": "AAA",
                    "trigger_values": [90, 120],
                },
                "orders": [
                    {
                        "transaction_type": "SELL",
                        "quantity": 10,
                        # This is the stop-limit price, not the protection trigger.
                        "price": 89.5,
                        "order_id": "stop-order-1",
                    },
                    {
                        "transaction_type": "SELL",
                        "quantity": 10,
                        "price": 120,
                        "order_id": "target-order-1",
                    },
                ],
                "updated_at": "2026-08-01T10:00:02+05:30",
            }
        ]

    def place_order(self, **kwargs):
        raise AssertionError("read-only snapshot adapter attempted order placement")


class Wrapper:
    def __init__(self, raw):
        self.raw = raw


def test_complete_snapshot_maps_all_read_only_lanes():
    client = FakeKite()
    observed_at = datetime(2026, 8, 1, 4, 30, tzinfo=timezone.utc)

    bundle = capture_zerodha_snapshot(client, observed_at=observed_at)

    assert bundle.complete is True
    assert bundle.account.complete is True
    assert bundle.protections_complete is True
    assert bundle.errors == ()
    assert client.calls == ["orders", "trades", "positions", "margins", "get_gtts"]

    order = bundle.account.orders[0]
    assert order.broker_order_id == "broker-order-1"
    assert order.client_order_ref == "submit-1"
    assert order.submission_token == "submit-1"
    assert order.filled_quantity == 10

    trade = bundle.account.trades[0]
    assert trade.trade_id == "trade-1"
    assert trade.price == 101.5

    position = bundle.account.positions[0]
    assert position.quantity == 10
    assert position.protected_quantity == 0
    assert bundle.account.cash == 50_000
    assert bundle.account.available_margin == 60_000

    protection = bundle.protections[0]
    assert protection.broker_protection_id == "9001"
    assert protection.order_id == "oms-1"
    assert protection.active is True
    assert protection.quantity == 10
    assert protection.stop_price == 90
    assert protection.target_price == 120
    assert protection.stop_reference == "stop-order-1"
    assert protection.target_reference == "target-order-1"


def test_wrapper_raw_client_is_supported_and_snapshot_id_is_deterministic():
    observed = "2026-08-01T04:30:00+00:00"

    first = capture_zerodha_snapshot(Wrapper(FakeKite()), observed_at=observed)
    second = capture_zerodha_snapshot(Wrapper(FakeKite()), observed_at=observed)

    assert first.account.snapshot_id == second.account.snapshot_id
    assert first.as_dict() == second.as_dict()


def test_failed_lane_is_incomplete_not_empty_authority():
    class PositionsUnavailable(FakeKite):
        def positions(self):
            self.calls.append("positions")
            raise RuntimeError("positions endpoint unavailable")

    bundle = capture_zerodha_snapshot(
        PositionsUnavailable(),
        observed_at="2026-08-01T04:30:00+00:00",
    )

    assert bundle.account.orders_complete is True
    assert bundle.account.trades_complete is True
    assert bundle.account.positions_complete is False
    assert bundle.account.account_complete is True
    assert bundle.account.complete is False
    assert bundle.account.positions == ()
    assert any(error.startswith("positions:RuntimeError") for error in bundle.errors)
    assert any(error.startswith("positions:RuntimeError") for error in bundle.account.errors)


def test_malformed_response_is_explicitly_incomplete():
    class MalformedOrders(FakeKite):
        def orders(self):
            self.calls.append("orders")
            return {"not": "a list"}

    bundle = capture_zerodha_snapshot(
        MalformedOrders(),
        observed_at="2026-08-01T04:30:00+00:00",
    )

    assert bundle.account.orders_complete is False
    assert bundle.account.orders == ()
    assert any("orders:MALFORMED_RESPONSE" in error for error in bundle.errors)


def test_gtt_failure_does_not_falsify_account_completeness():
    class GttUnavailable(FakeKite):
        def get_gtts(self):
            self.calls.append("get_gtts")
            raise TimeoutError("gtt endpoint timed out")

    bundle = capture_zerodha_snapshot(
        GttUnavailable(),
        observed_at="2026-08-01T04:30:00+00:00",
    )

    assert bundle.account.complete is True
    assert bundle.protections_complete is False
    assert bundle.protections == ()
    assert bundle.complete is False
    assert any(error.startswith("gtts:TimeoutError") for error in bundle.errors)
    assert not any(error.startswith("gtts:") for error in bundle.account.errors)


def test_missing_endpoint_is_reported_without_fallback_mutation():
    class MissingGtt:
        def orders(self):
            return []

        def trades(self):
            return []

        def positions(self):
            return {"net": [], "day": []}

        def margins(self):
            return {"equity": {"available": {"cash": 1_000}, "net": 1_000}}

    bundle = capture_zerodha_snapshot(
        MissingGtt(),
        observed_at="2026-08-01T04:30:00+00:00",
    )

    assert bundle.account.complete is True
    assert bundle.protections_complete is False
    assert "gtts:ENDPOINT_UNAVAILABLE:get_gtts" in bundle.errors
