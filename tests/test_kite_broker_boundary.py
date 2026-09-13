from __future__ import annotations

import ast
from pathlib import Path

import pytest

import data.kite_client as kite_client_module
from data.kite_client import KiteClient, _BROKER_READS, _GuardedKiteProxy
from product.live_execution_interlock import LiveExecutionBlocked


class _FakeKite:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.CONSTANT = "safe"

    def quote(self, *args, **kwargs):
        self.calls.append(("quote", args, kwargs))
        return {"ok": True}

    def place_order(self, *args, **kwargs):
        self.calls.append(("place_order", args, kwargs))
        return "should-never-be-reached"

    def future_capital_mutation(self, *args, **kwargs):
        self.calls.append(("future_capital_mutation", args, kwargs))
        return "should-never-be-reached"


def test_guarded_proxy_allows_known_read_only_calls() -> None:
    raw = _FakeKite()
    proxy = _GuardedKiteProxy(raw)  # type: ignore[arg-type]

    assert proxy.quote(["NSE:INFY"]) == {"ok": True}
    assert raw.calls == [("quote", (["NSE:INFY"],), {})]
    assert proxy.CONSTANT == "safe"


def test_guarded_proxy_blocks_known_mutation_before_sdk_call() -> None:
    raw = _FakeKite()
    proxy = _GuardedKiteProxy(raw)  # type: ignore[arg-type]

    with pytest.raises(LiveExecutionBlocked):
        proxy.place_order(tradingsymbol="INFY")

    assert raw.calls == []


def test_guarded_proxy_blocks_unknown_future_callable_by_default() -> None:
    raw = _FakeKite()
    proxy = _GuardedKiteProxy(raw)  # type: ignore[arg-type]

    with pytest.raises(LiveExecutionBlocked):
        proxy.future_capital_mutation("anything")

    assert raw.calls == []


def test_guarded_proxy_refuses_attribute_assignment() -> None:
    raw = _FakeKite()
    proxy = _GuardedKiteProxy(raw)  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        proxy.place_order = lambda **_: "bypass"  # type: ignore[method-assign]


def test_guarded_proxy_hides_backing_sdk() -> None:
    raw = _FakeKite()
    proxy = _GuardedKiteProxy(raw)  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        _ = proxy._raw  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = proxy._GuardedKiteProxy__raw  # type: ignore[attr-defined]


def test_legacy_private_kite_handle_is_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy ``client._kite`` callers must no longer receive the raw SDK."""
    raw = _FakeKite()
    monkeypatch.setattr(kite_client_module, "KiteConnect", lambda **_: raw)

    client = KiteClient(api_key="test", access_token="", api_secret="")

    assert client._kite.quote(["NSE:INFY"]) == {"ok": True}
    assert raw.calls == [("quote", (["NSE:INFY"],), {})]

    with pytest.raises(LiveExecutionBlocked):
        client._kite.place_order(tradingsymbol="INFY")
    assert raw.calls == [("quote", (["NSE:INFY"],), {})]

    with pytest.raises(AttributeError):
        client._kite = raw  # type: ignore[misc]
    with pytest.raises(AttributeError):
        _ = client._KiteClient__sdk  # type: ignore[attr-defined]


def test_read_only_allowlist_contains_no_obvious_mutator_names() -> None:
    mutation_tokens = {
        "place",
        "modify",
        "cancel",
        "delete",
        "exit",
        "invalidate",
        "renew",
        "generate_session",
        "set_access_token",
        "order",  # handled below for calculator/read exceptions
        "sip",
    }

    explicit_safe_order_reads = {
        "orders",
        "order_history",
        "order_trades",
        "mf_orders",
        "order_margins",
        "basket_order_margins",
    }
    explicit_safe_sip_reads = {"mf_sips"}

    suspicious: list[str] = []
    for name in sorted(_BROKER_READS):
        lowered = name.lower()
        for token in mutation_tokens:
            if token not in lowered:
                continue
            if token == "order" and name in explicit_safe_order_reads:
                continue
            if token == "sip" and name in explicit_safe_sip_reads:
                continue
            suspicious.append(name)
            break

    assert suspicious == [], f"broker read allowlist contains mutation-like methods: {suspicious}"


def test_no_production_module_accesses_backing_kite_sdk_or_constructs_parallel_client() -> None:
    """Only data.kite_client may own the actual KiteConnect object.

    ``KiteClient._kite`` is intentionally retained as a guarded compatibility
    property for older callers, so field-name scanning would produce false
    positives.  The actual privileged handles are name-mangled and forbidden
    everywhere else, while parallel KiteConnect construction remains banned.
    """
    root = Path(__file__).resolve().parents[1]
    allowed = (root / "data" / "kite_client.py").resolve()
    violations: list[str] = []

    ignored_parts = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "tests",
    }
    forbidden_sdk_attrs = {"_KiteClient__sdk", "_GuardedKiteProxy__raw"}

    for path in root.rglob("*.py"):
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.resolve() == allowed:
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_sdk_attrs:
                violations.append(
                    f"{path.relative_to(root)}:{node.lineno}: backing Kite SDK access"
                )
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "KiteConnect":
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: parallel KiteConnect constructor")
                elif isinstance(func, ast.Attribute) and func.attr == "KiteConnect":
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: parallel KiteConnect constructor")

    assert violations == [], "unguarded broker SDK escape path(s):\n" + "\n".join(violations)
