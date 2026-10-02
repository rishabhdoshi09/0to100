from __future__ import annotations

import product.due_diligence.acquire as acquire
import product.pit_backfill as PB


class _FakeResponse:
    def __init__(self, status_code=503, payload=None):
        self.status_code = status_code
        self.content = b""
        self._payload = {} if payload is None else payload
        self.closed = False

    def json(self):
        return self._payload

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self):
        self.closed = False
        self.responses = []

    def get(self, *args, **kwargs):
        response = _FakeResponse()
        self.responses.append(response)
        return response

    def close(self):
        self.closed = True


def _empty_harvest(*args, **kwargs):
    return {
        "parsed": 0,
        "unverified": 0,
        "attempted": 0,
        "acquired": 0,
        "deduped": 0,
        "unavailable": 0,
    }


def test_backfill_symbol_closes_only_session_it_owns(monkeypatch):
    owned = _FakeSession()
    monkeypatch.setattr(acquire, "_nse_session", lambda: owned)
    monkeypatch.setattr(PB, "LANES", ())
    monkeypatch.setattr(PB, "harvest_symbol", _empty_harvest)

    PB.backfill_symbol("INFY")
    assert owned.closed is True

    external = _FakeSession()
    PB.backfill_symbol("INFY", session=external)
    assert external.closed is False


def test_backfill_batch_closes_shared_session(monkeypatch, tmp_path):
    shared = _FakeSession()
    monkeypatch.setattr(acquire, "_nse_session", lambda: shared)
    monkeypatch.setattr(
        PB,
        "backfill_symbol",
        lambda symbol, **kwargs: {
            "symbol": symbol,
            "attempted": 0,
            "acquired": 1,
            "parsed": 0,
            "failed": 0,
            "skipped": 0,
            "reasons": {},
        },
    )

    report = PB.backfill(
        symbols=["INFY"],
        stage=PB.STAGE_CANDIDATES,
        resume=False,
        sleep_s=0,
        state_path=tmp_path / "state.json",
    )

    assert report["acquired"] == 1
    assert shared.closed is True


def test_structured_backfill_closes_direct_responses_and_owned_session(monkeypatch):
    owned = _FakeSession()
    monkeypatch.setattr(acquire, "_nse_session", lambda: owned)

    PB.backfill_structured_financials("INFY", sleep_s=0)

    assert len(owned.responses) == 2
    assert all(response.closed for response in owned.responses)
    assert owned.closed is True


def test_structured_backfill_preserves_caller_session_ownership():
    external = _FakeSession()

    PB.backfill_structured_financials("INFY", session=external, sleep_s=0)

    assert len(external.responses) == 2
    assert all(response.closed for response in external.responses)
    assert external.closed is False
