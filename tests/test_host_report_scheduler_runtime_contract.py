from __future__ import annotations

import pytest

from core import runtime_paths as RP
import product.host_entrypoint as HE


def test_scheduler_can_initialize_ephemeral_dev_state(tmp_path, monkeypatch):
    monkeypatch.setenv(RP.ENV_VAR, str(tmp_path))
    monkeypatch.delenv(RP.REQUIRE_EXISTING_ENV, raising=False)

    HE._write_scheduler_status({"state": "TEST"})

    assert (tmp_path / HE.REPORT_SCHEDULER_REL).is_file()


def test_scheduler_never_recreates_missing_installed_state(tmp_path, monkeypatch):
    monkeypatch.setenv(RP.ENV_VAR, str(tmp_path))
    monkeypatch.setenv(RP.REQUIRE_EXISTING_ENV, "1")

    with pytest.raises(RuntimeError, match="runtime state directory is unavailable"):
        HE._write_scheduler_status({"state": "SHOULD_NOT_EXIST"})

    assert not (tmp_path / "state").exists()
