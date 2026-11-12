"""broker_live() is polled by every Home/dashboard read. Uncached, each call was
a live Zerodha profile() request with a fresh TLS handshake (CA load holds the
GIL 0.3-0.5 s on the owner's Mac) and the API starved during market hours."""
from __future__ import annotations

import product.readiness as R


def _setup(monkeypatch, clock, key):
    calls = []
    monkeypatch.setattr(R, "_BROKER_LIVE_CACHE", {"key": None, "at": 0.0, "value": None})
    monkeypatch.setattr(R._time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(R, "_broker_session_key", lambda: key[0])

    def probe():
        calls.append(key[0])
        return {"ready": key[0] == "fresh", "status": "SESSION_VALID" if key[0] == "fresh" else "SESSION_EXPIRED"}

    monkeypatch.setattr(R, "_probe_broker_live", probe)
    return calls


def test_valid_session_is_probed_once_per_ttl(monkeypatch):
    clock, key = [100.0], ["fresh"]
    calls = _setup(monkeypatch, clock, key)
    for _ in range(10):
        assert R.broker_live()["ready"] is True
    assert len(calls) == 1
    clock[0] += R._BROKER_LIVE_TTL_OK_S + 1
    R.broker_live()
    assert len(calls) == 2


def test_failure_is_rechecked_sooner_and_a_login_bypasses_the_cache(monkeypatch):
    clock, key = [100.0], ["expired"]
    calls = _setup(monkeypatch, clock, key)
    assert R.broker_live()["ready"] is False
    R.broker_live()
    assert calls == ["expired"]
    key[0] = "fresh"                                  # daily login rotated the token
    assert R.broker_live()["ready"] is True
    assert calls == ["expired", "fresh"]
    key[0] = "expired"
    clock[0] += R._BROKER_LIVE_TTL_FAIL_S + 1
    R.broker_live()
    assert calls[-1] == "expired"


def test_returned_dict_is_a_copy(monkeypatch):
    clock, key = [100.0], ["fresh"]
    _setup(monkeypatch, clock, key)
    R.broker_live()["ready"] = "mutated"
    assert R.broker_live()["ready"] is True
