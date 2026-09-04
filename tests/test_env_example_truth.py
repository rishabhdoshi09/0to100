from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env_example() -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        rows[key.strip()] = value.strip()
    return rows


def test_optional_secrets_are_empty_in_sample_config() -> None:
    env = _env_example()
    for key in (
        "KITE_API_KEY",
        "KITE_API_SECRET",
        "KITE_ACCESS_TOKEN",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "MARKETAUX_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ):
        assert env.get(key) == "", f"{key} must be blank, not a truthy placeholder"


def test_local_sample_config_does_not_force_noninteractive_login() -> None:
    env = _env_example()
    assert env.get("QT_NONINTERACTIVE") == "0"
