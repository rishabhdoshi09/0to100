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
    assert "api.app:app" in inner
    assert "npm --prefix" in inner and "run dev" in inner
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
    desk_fn = complete.split("wait_for_desk()", 1)[1].split("if wait_for_desk", 1)[0]
    assert 'url_ok "http://127.0.0.1:8765/api/health"' in desk_fn
    assert 'url_ok "http://127.0.0.1:5173/"' in desk_fn
    assert desk_fn.index("8765/api/health") < desk_fn.index("5173/")
    assert "i < 90" in desk_fn
    assert "i < 120" in desk_fn


def test_report_watchdog_requires_health_not_just_an_open_port():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")

    adopt = complete.split("adopt_report() {", 1)[1].split("\n}", 1)[0]
    assert 'url_ok "http://127.0.0.1:8766/health"' in adopt
    assert "|| port_open 8766" not in adopt

    watch = complete.split('while [[ "$STOP" != "1" ]]', 1)[1]
    assert "Owned report API is listening/alive but unhealthy; restarting." in watch
    assert 'stop_pid "$REPORT_PID" "unhealthy report API"' in watch
    assert "Port 8766 is owned externally but /health is failing" in watch


def test_complete_script_always_stops_old_stack_then_starts_everything():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "machine-lock-path" in complete
    assert "machine-lock-path" in inner
    assert "$ROOT/logs/stack/quantterm.supervisor.lock" not in complete
    assert "QT_MACHINE_OWNER" in complete
    assert "QT_MACHINE_OWNER" in inner
    assert "python scripts/local_stack.py stop --ports 5173,8765,8766" in complete
    assert "try_machine_lock()" in complete
    assert "try-fd-lock --fd 200" in complete
    assert "try-fd-lock --fd 201" in inner
    assert "if flock -n 200; then" not in complete
    assert "if flock -n 201; then" not in inner
    assert complete.index("try_machine_lock") < complete.index(
        "python scripts/local_stack.py stop --ports 5173,8765,8766"
    )
    assert "will not stop :5173/:8765/:8766" in complete
    assert "write-owner" in complete
    assert "The desk is serving" in complete
    assert "python scripts/local_stack.py stop --ports 5173,8765" in inner
    assert inner.index("QT_MACHINE_OWNER") < inner.index(
        "python scripts/local_stack.py stop --ports 5173,8765"
    )
    assert "One command, one terminal" in complete
    assert "scripts/local_stack.py scan" in inner
    assert "run_quantterm_complete.sh --restart" not in inner
    assert 'url_ok "http://127.0.0.1:8766/health"' in complete
    assert complete.count('url_ok "http://127.0.0.1:8766/health"') >= 1
    assert 'alive "$REPORT_PID"' in complete
    assert "wait_for_api" in inner
    boot = inner.split("start_api || true", 1)[1].split('while [[ "$STOP" != "1" ]]', 1)[0]
    assert boot.index("wait_for_api") < boot.index("start_frontend")
    assert boot.index("start_frontend") < boot.index("kick_scan")
    assert "npm --prefix" in inner and "run dev" in inner
    assert "vite.log" in inner
    assert "vite.log" in complete
    assert "Use --restart" not in inner
    assert "Use --restart" not in complete
    assert "python3 -m venv venv" in complete
    assert "pip install -r requirements.txt" in complete
    assert "npm install" in complete
    assert "Missing venv. Create the QuantTerm Python environment first." not in complete


def test_deploy_services_use_one_canonical_host_owner():
    server = (ROOT / "deploy" / "setup_server.sh").read_text(encoding="utf-8")
    mac = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    installer = (ROOT / "scripts" / "install_quantterm_host.sh").read_text(encoding="utf-8")
    host = (ROOT / "product" / "host_install.py").read_text(encoding="utf-8")

    assert "install_quantterm_host.sh" in server
    assert "install_quantterm_host.sh" in mac
    assert "--manager systemd" in server
    assert "--manager launchd" in mac
    assert "product.host_install" in installer
    assert "product.host_install_existing" in installer
    assert "product.host_entrypoint" in host
    assert "EXPECTED_CHILDREN" in host

    # Historical split-service templates must stay retired; setup scripts only
    # mention their names while explicitly disabling/removing old installations.
    assert not (ROOT / "deploy" / "quantterm-ui.service").exists()
    assert not (ROOT / "deploy" / "quantterm-autonomy.service").exists()
    assert "disable --now \"$legacy\"" in server
    assert "com.quantterm.ui" in mac and "com.quantterm.autonomy" in mac
    assert 'BRANCH="${QT_BRANCH:-claude/build-ai-trading-system-miHHd}"' in server
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
        assert "cursor/live-terminal-contract-858e" not in text, rel


def test_fresh_server_clone_pins_production_branch_and_canonical_service():
    """A brand-new VPS clone must land on the production branch and one host service."""
    production = "claude/build-ai-trading-system-miHHd"
    always_on = (ROOT / "docs" / "ALWAYS_ON.md").read_text(encoding="utf-8")
    oracle = (ROOT / "docs" / "ORACLE_SETUP.md").read_text(encoding="utf-8")
    setup = (ROOT / "deploy" / "setup_server.sh").read_text(encoding="utf-8")
    clone_pin = f"git clone --branch {production}"

    assert clone_pin in always_on
    assert clone_pin in oracle
    assert f'BRANCH="${{QT_BRANCH:-{production}}}"' in setup
    assert 'git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"' in setup
    assert "cursor/live-terminal-contract-858e" not in always_on
    assert "cursor/live-terminal-contract-858e" not in oracle
    assert "quantterm-ui.service" not in oracle
    assert "quantterm-autonomy.service" not in oracle
    assert "scripts/quantterm_restart.sh" in oracle
    assert "scripts/quantterm_status.sh" in oracle


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
