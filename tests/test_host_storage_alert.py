"""The owner hears about a runtime-storage outage (owner's Mac, 16-18 Nov 2026).

The USB runtime drive was missing for about two days; every launchd start
failed the storage preflight and nobody was told.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from product import host_storage_alert as SA


@dataclass
class _Result:
    delivered: bool = True

    def to_dict(self):
        return {"delivered": self.delivered}


class _Sender:
    def __init__(self, delivered: bool = True):
        self.messages: list[str] = []
        self.delivered = delivered

    def __call__(self, message: str):
        self.messages.append(message)
        return _Result(self.delivered)


@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / "internal" / "storage_alert.json"
    monkeypatch.setenv("QT_STORAGE_ALERT_STATE", str(path))
    return path


def test_disabled_without_the_external_storage_contract(monkeypatch):
    monkeypatch.delenv("QT_STORAGE_ALERT_STATE", raising=False)
    monkeypatch.delenv("QT_STORAGE_PREFLIGHT_REQUIRED", raising=False)
    sender = _Sender()
    assert SA.note_storage_down("x", sender=sender) == "DISABLED"
    assert SA.note_storage_recovered(sender=sender) == "DISABLED"
    assert sender.messages == []


def test_a_brief_wake_glitch_is_not_news(state):
    t = [1000.0]
    sender = _Sender()
    assert SA.note_storage_down("Errno 5", sender=sender, now=lambda: t[0]) == "RECORDED"
    t[0] += 30
    assert SA.note_storage_down("Errno 5", sender=sender, now=lambda: t[0]) == "WAITING"
    assert SA.note_storage_recovered(sender=sender, now=lambda: t[0]) == "CLEARED"
    assert sender.messages == []
    assert not state.exists()


def test_a_lasting_outage_alerts_once_then_reminds_then_announces_recovery(state):
    t = [1000.0]
    sender = _Sender()
    reason = "external volume is not mounted: /Volumes/Expansion"
    SA.note_storage_down(reason, sender=sender, now=lambda: t[0])
    t[0] += SA.GRACE_S
    assert SA.note_storage_down(reason, sender=sender, now=lambda: t[0]) == "ALERTED"
    assert "PAUSED" in sender.messages[0] and "/Volumes/Expansion" in sender.messages[0]
    # launchd retries every few seconds: no message storm.
    for _ in range(50):
        t[0] += 10
        assert SA.note_storage_down(reason, sender=sender, now=lambda: t[0]) == "WAITING"
    assert len(sender.messages) == 1
    t[0] += SA.REMIND_S
    assert SA.note_storage_down(reason, sender=sender, now=lambda: t[0]) == "REMINDED"
    assert "STILL PAUSED" in sender.messages[1]
    t[0] += 60
    assert SA.note_storage_recovered(sender=sender, now=lambda: t[0]) == "ANNOUNCED"
    assert "back after 3h" in sender.messages[2]
    assert not state.exists()


def test_an_undelivered_alert_is_retried_soon_not_after_the_reminder_gap(state):
    t = [1000.0]
    offline = _Sender(delivered=False)
    SA.note_storage_down("gone", sender=offline, now=lambda: t[0])
    t[0] += SA.GRACE_S
    assert SA.note_storage_down("gone", sender=offline, now=lambda: t[0]) == "ERROR"
    t[0] += 30
    assert SA.note_storage_down("gone", sender=offline, now=lambda: t[0]) == "WAITING"
    online = _Sender()
    t[0] += SA.RETRY_S
    assert SA.note_storage_down("gone", sender=online, now=lambda: t[0]) == "ALERTED"
    assert "PAUSED" in online.messages[0] and "STILL" not in online.messages[0]


def test_alerting_never_raises(state, monkeypatch):
    def broken(_message):
        raise RuntimeError("telegram exploded")
    SA.note_storage_down("gone", sender=broken, now=lambda: 1000.0)
    assert SA.note_storage_down("gone", sender=broken, now=lambda: 1000.0 + SA.GRACE_S) == "ERROR"
    state.write_text("{not json", encoding="utf-8")
    assert SA.note_storage_recovered(sender=broken) in {"NONE", "CLEARED", "ERROR"}


def test_startup_preflight_failure_is_noted_before_the_process_exits(state, monkeypatch):
    from product import host_entrypoint as HE

    def fail():
        raise RuntimeError("macOS storage preflight failed with rc=78: [STORAGE PREFLIGHT] FAIL: "
                           "external volume is not mounted: /Volumes/Expansion")
    monkeypatch.setattr(HE, "trim_inherited_log", lambda fd: False)
    monkeypatch.setattr(HE, "load_env_file", lambda path: [])
    monkeypatch.setattr(HE, "prepare_runtime_storage_for_startup", fail)
    with pytest.raises(RuntimeError, match="not mounted"):
        HE.main()
    assert state.exists()
    assert "not mounted" in state.read_text(encoding="utf-8")
