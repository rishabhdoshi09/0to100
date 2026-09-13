"""Regression coverage for the Catalina macOS process-session launcher."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mac_wrapper_supplies_setsid_without_extra_system_dependency():
    src = (ROOT / "scripts" / "run_quantterm_mac.sh").read_text(encoding="utf-8")
    assert "if ! command -v setsid" in src
    assert "setsid()" in src
    assert "export -f setsid" in src
    assert "exec python - \"$@\"" in src
    assert "os.setsid()" in src
    assert "os.execvp(sys.argv[1], sys.argv[1:])" in src


def test_mac_launchagent_enters_through_compatibility_wrapper():
    setup = (ROOT / "deploy" / "setup_mac.sh").read_text(encoding="utf-8")
    assert "scripts/run_quantterm_mac.sh" in setup
    assert "scripts/run_quantterm_complete.sh" not in setup.split("cat > \"$APP_PLIST\"", 1)[1].split("PLIST", 1)[0]
