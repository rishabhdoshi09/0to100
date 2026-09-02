"""The complete stack is one terminal and one command, including the market scan."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_complete_script_starts_every_local_service_in_one_process_tree():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    desk = (ROOT / "scripts" / "run_desk.sh").read_text(encoding="utf-8")

    assert complete.startswith("#!/usr/bin/env bash")
    assert "run_quantterm.sh" in complete
    assert "python main.py login" in complete
    assert "report_api:app" in complete
    assert "terminal_product_api:app" in inner
    assert "npm run dev" in inner
    assert "python -u main.py autonomy" in inner
    assert "scripts/local_stack.py scan" in inner
    assert "curl" not in complete
    assert "curl" not in inner
    assert "Do not start a second terminal" in complete
    assert 'exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"' in desk
    wrapper = (ROOT / "quantterm.sh").read_text(encoding="utf-8")
    assert 'exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"' in wrapper
    assert "print_startup_summary" in complete
    assert "maybe_open_home_browser" in complete
    assert "HOME_OPENED" in complete
    assert complete.count("HOME_OPENED") >= 2


def test_complete_script_always_stops_old_stack_then_starts_everything():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "python scripts/local_stack.py stop --ports 5173,8765,8766" in complete
    assert "python scripts/local_stack.py stop --ports 5173,8765" in inner
    assert "One command, one terminal" in complete
    assert "scripts/local_stack.py scan" in inner
    assert "run_quantterm_complete.sh --restart" not in inner
    assert 'url_ok "http://127.0.0.1:8766/health"' in complete
    assert complete.count('url_ok "http://127.0.0.1:8766/health"') == 1
    assert 'alive "$REPORT_PID"' in complete
    assert "wait_for_api" in inner
    boot = inner.split("start_api || true", 1)[1].split('while [[ "$STOP" != "1" ]]', 1)[0]
    assert boot.index("wait_for_api") < boot.index("start_frontend")
    assert boot.index("start_frontend") < boot.index("kick_scan")
    assert "npm run dev" in inner
    assert "Use --restart" not in inner
    assert "Use --restart" not in complete
    assert "python3 -m venv venv" in complete
    assert "pip install -r requirements.txt" in complete
    assert "npm install" in complete
    assert "Missing venv. Create the QuantTerm Python environment first." not in complete


def test_deploy_services_own_the_complete_stack():
    server = (ROOT / "deploy" / "setup_server.sh").read_text(encoding="utf-8")
    mac = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    unit = (ROOT / "deploy" / "quantterm-ui.service").read_text(encoding="utf-8")
    for blob, name in ((server, "setup_server.sh"), (mac, "setup_mac.sh"), (unit, "quantterm-ui.service")):
        assert "run_quantterm_complete.sh" in blob, name
        assert "QT_NONINTERACTIVE" in blob, name
    assert "report_api:app" in (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    assert 'BRANCH="${QT_BRANCH:-overhaul/evidence-lab}"' not in server
    assert "overhaul/evidence-lab" not in server


def test_canonical_docs_name_one_product_launcher():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    overview = (ROOT / "docs" / "SYSTEM_OVERVIEW.md").read_text(encoding="utf-8")
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    for blob, name in (
        (readme, "README.md"),
        (claude, "CLAUDE.md"),
        (overview, "docs/SYSTEM_OVERVIEW.md"),
        (app_py, "app.py"),
    ):
        assert "run_quantterm_complete.sh" in blob, name
        assert "Branch of record" not in blob, name
    compact = " ".join(readme.split())
    assert "Streamlit UI" not in readme
    assert "Vite/React desk" in readme
    assert "not the current product path" in compact
    # The start snippet must be the complete launcher, not run_desk.sh as the product command.
    start = readme.split("Canonical product path", 1)[1][:900]
    assert "bash scripts/run_quantterm_complete.sh" in start
    assert "bash scripts/run_desk.sh" not in start.split("compatibility wrapper")[0]


def test_how_to_docs_do_not_checkout_historical_branch():
    for rel in (
        "docs/ALWAYS_ON.md",
        "docs/ORACLE_SETUP.md",
        "docs/autonomy/DEPLOYMENT.md",
        "CLAUDE.md",
        "README.md",
        "deploy/setup_server.sh",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "git checkout overhaul/evidence-lab" not in text, rel
        assert "git pull origin overhaul/evidence-lab" not in text, rel


def test_fresh_server_clone_pins_accepted_product_branch():
    """A brand-new Oracle/VPS clone must not land on GitHub's historical default."""
    import re

    accepted = "cursor/live-terminal-contract-858e"
    always_on = (ROOT / "docs" / "ALWAYS_ON.md").read_text(encoding="utf-8")
    oracle = (ROOT / "docs" / "ORACLE_SETUP.md").read_text(encoding="utf-8")
    setup = (ROOT / "deploy" / "setup_server.sh").read_text(encoding="utf-8")
    clone_pin = f"git clone --branch {accepted}"
    assert clone_pin in always_on
    assert clone_pin in oracle
    assert f'git clone --branch {accepted} "$REPO_URL"' in setup
    # Obsolete single-unit restart: setup_server.sh removes quantterm.service.
    assert not re.search(
        r"systemctl restart quantterm(?:\.service)?(?:\s|$)",
        oracle,
    )
    assert "sudo systemctl restart quantterm-ui.service quantterm-autonomy.service" in oracle


def test_issue92_dod_verifier_is_checked_in():
    path = ROOT / "scripts" / "verify_issue92_dod.py"
    src = path.read_text(encoding="utf-8")
    assert "RUN_SCAN_NOW" in src
    assert "REFRESH_MARKET_REPORT_NOW" in src
    assert "due-diligence" in src
    assert "docs/issue92_live_dod_proof.md" in src
    assert "rev-parse" in src or "git_sha" in src
    assert '== "SUCCEEDED"' in src
    assert "switch --detach" in src
    assert "will not certify SHA" in src
    assert "required_named" in src or "piotroski_f" in src
