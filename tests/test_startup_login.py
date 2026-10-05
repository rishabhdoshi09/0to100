from types import SimpleNamespace
import subprocess
import sys

import pytest

from product import startup_login as login


@pytest.fixture(autouse=True)
def interactive(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    for key in ("QT_NONINTERACTIVE", "QT_NO_BROWSER", "QT_NO_AUTO_LOGIN"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("status,expected_calls", [(0, 1), (1, 2), (2, 1), (3, 1)])
def test_only_confirmed_missing_or_expired_auth_starts_existing_login(status, expected_calls):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=status if len(calls) == 1 else 0)

    login.offer_login(runner=run)
    assert len(calls) == expected_calls
    assert calls[0][1]["timeout"] == 20
    if status == 1:
        assert calls[1][0] == [sys.executable, str(login.ROOT / "main.py"), "login"]
        assert "capture_output" not in calls[1][1]  # retains interactive stdin/stdout


@pytest.mark.parametrize("flag", ["QT_NONINTERACTIVE", "QT_NO_BROWSER", "QT_NO_AUTO_LOGIN"])
def test_suppression_flags_never_probe_or_prompt(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    login.offer_login(runner=lambda *a, **kw: pytest.fail("must not launch"))


def test_headless_worker_never_probes_or_prompts(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    login.offer_login(runner=lambda *a, **kw: pytest.fail("must not launch"))


def test_provider_timeout_does_not_launch_login(capsys):
    def run(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 20)
    login.offer_login(runner=run)
    assert "timed out; desk continues" in capsys.readouterr().out


def test_login_failure_keeps_console_alive(capsys):
    login.offer_login(runner=lambda *a, **kw: SimpleNamespace(returncode=1))
    assert "Login was not completed; desk continues" in capsys.readouterr().out


@pytest.mark.parametrize("status,expected", [
    ("SESSION_VALID", 0), ("TOKEN_MISSING", 1), ("SESSION_EXPIRED", 1),
    ("PROVIDER_UNAVAILABLE", 3), ("CONFIG_INVALID", 3),
])
def test_probe_classification(monkeypatch, status, expected):
    monkeypatch.setitem(sys.modules, "data.kite_client", SimpleNamespace(_fresh_env=lambda k: "configured"))
    monkeypatch.setitem(sys.modules, "research.autonomy.auth", SimpleNamespace(
        TOKEN_MISSING="TOKEN_MISSING", SESSION_EXPIRED="SESSION_EXPIRED",
        probe_auth=lambda: SimpleNamespace(status=status, valid=status == "SESSION_VALID")))
    assert login.probe_status() == expected


def test_missing_credentials_do_not_probe_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "data.kite_client", SimpleNamespace(_fresh_env=lambda k: ""))
    monkeypatch.setitem(sys.modules, "research.autonomy.auth", SimpleNamespace(
        TOKEN_MISSING="TOKEN_MISSING", SESSION_EXPIRED="SESSION_EXPIRED",
        probe_auth=lambda: pytest.fail("must not contact provider")))
    assert login.probe_status() == 2


def test_both_launcher_paths_offer_login_only_after_readiness():
    text = (login.ROOT / "scripts/run_quantterm_complete.sh").read_text()
    host, manual = text.split('if [[ "$(uname -s)" == "Darwin"', 1)
    assert host.index('READY · $summary') < host.index('-m product.startup_login')
    assert manual.index('if wait_for_desk; then') < manual.index('-m product.startup_login')
    assert 'QT_HOST_ENV_FILE="$installed_env_file"' in host
