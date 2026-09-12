from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_MAC = ROOT / "deploy" / "setup_mac.sh"


def test_legacy_mac_setup_routes_to_canonical_host_installer() -> None:
    text = SETUP_MAC.read_text(encoding="utf-8")
    assert "scripts/install_quantterm_host.sh" in text
    assert "exec bash" in text


def test_legacy_mac_setup_does_not_recreate_obsolete_launchd_topology() -> None:
    text = SETUP_MAC.read_text(encoding="utf-8")
    assert "com.quantterm.ui" not in text
    assert "com.quantterm.autonomy" not in text
    assert "KeepAlive" not in text
    assert "pmset -a sleep 0" not in text
