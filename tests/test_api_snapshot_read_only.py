"""Read-only dashboard must not manufacture external-volume snapshot readiness."""
from __future__ import annotations

import hashlib
import json

import terminal_api


def _write_valid_snapshot(root, *, sid="a1b2c3d4e5f60708"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "ACTIVE").write_text(json.dumps({"snapshot_id": sid}), encoding="utf-8")
    child = root / sid
    child.mkdir()
    (child / "bars_equity.csv").write_text("symbol,date,open,high,low,close,volume,series\n",
                                            encoding="utf-8")
    manifest = {
        "snapshot_id": sid, "last_trading_date": "2026-10-09",
        "source": "kite_authoritative",
    }
    checksum = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    manifest["manifest_checksum"] = checksum
    (child / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return child


def test_unmounted_snapshot_read_does_not_create_missing_storage(tmp_path, monkeypatch):
    missing_volume = tmp_path / "QuantTermStorage" / "QuantTerm" / "runtime"
    monkeypatch.setattr(terminal_api, "logs_dir", lambda: missing_volume)
    result = terminal_api._snapshot_payload()
    assert result["ready"] is False
    assert not missing_volume.exists()


def test_valid_active_snapshot_is_readable_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_api, "logs_dir", lambda: tmp_path)
    root = tmp_path / "snapshots"
    _write_valid_snapshot(root)
    result = terminal_api._snapshot_payload()
    assert result["ready"] is True
    assert result["snapshot_id"] == "a1b2c3d4e5f60708"
    assert result["latest_date"] == "2026-10-09"


def test_traversal_or_corrupt_manifest_never_claims_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_api, "logs_dir", lambda: tmp_path)
    root = tmp_path / "snapshots"
    child = _write_valid_snapshot(root)
    (root / "ACTIVE").write_text('{"snapshot_id":"../../outside"}', encoding="utf-8")
    assert terminal_api._snapshot_payload()["ready"] is False

    (root / "ACTIVE").write_text('{"snapshot_id":"a1b2c3d4e5f60708"}', encoding="utf-8")
    manifest_file = child / "manifest.json"
    bad = json.loads(manifest_file.read_text(encoding="utf-8"))
    bad["last_trading_date"] = "2026-10-10"  # Manifest checksum no longer matches.
    manifest_file.write_text(json.dumps(bad), encoding="utf-8")
    assert terminal_api._snapshot_payload()["ready"] is False


def test_missing_equity_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_api, "logs_dir", lambda: tmp_path)
    child = _write_valid_snapshot(tmp_path / "snapshots")
    (child / "bars_equity.csv").unlink()
    assert terminal_api._snapshot_payload()["ready"] is False


def test_health_build_sha_is_exact_and_never_exposes_arbitrary_env(monkeypatch):
    verified = "234e3e30e228e515a38bee994436c0af3dad1b85"
    monkeypatch.setenv("QT_BUILD_SHA", verified.upper())
    assert terminal_api.health()["build_sha"] == verified

    # A malformed installation marker must not leak strings/paths/secrets.
    monkeypatch.setenv("QT_BUILD_SHA", "/private/Users/person/.env?token=secret")
    assert terminal_api.health()["build_sha"] == ""

    # A truncated commit prefix is not proof of the reviewed exact head.
    monkeypatch.setenv("QT_BUILD_SHA", verified[:12])
    assert terminal_api.health()["build_sha"] == ""
