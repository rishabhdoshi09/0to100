"""
Strategy-level kill switches.

Enabled/disabled state persisted to data/strategy_enabled.json.
Used by trade_engine before calling any strategy's generate_signal().

Usage
-----
from monitoring.strategy_manager import StrategyManager

sm = StrategyManager()
sm.is_enabled("lgbm")       # True / False
sm.enable("lgbm")
sm.disable("xgboost")
sm.status()                 # {"lgbm": True, "xgboost": False, ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from logger import get_logger

log = get_logger(__name__)

_STATE_FILE = Path("data/strategy_enabled.json")

_DEFAULTS: Dict[str, bool] = {
    "lgbm":           True,
    "xgboost":        True,
    "multi_horizon":  True,
    "ensemble":       True,
}


class StrategyManager:
    """Persistent strategy on/off toggle."""

    def __init__(self) -> None:
        self._state: Dict[str, bool] = self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def is_enabled(self, strategy: str) -> bool:
        return self._state.get(strategy, True)

    def enable(self, strategy: str) -> None:
        self._set(strategy, True)
        log.info("strategy_enabled", strategy=strategy)
        self._telegram_alert(f"✅ Strategy ENABLED: {strategy}")

    def disable(self, strategy: str) -> None:
        self._set(strategy, False)
        log.warning("strategy_disabled", strategy=strategy)
        self._telegram_alert(f"🔴 Strategy DISABLED: {strategy}")

    def status(self) -> Dict[str, bool]:
        return dict(self._state)

    # ── Persistence ────────────────────────────────────────────────────────

    def _set(self, strategy: str, enabled: bool) -> None:
        self._state[strategy] = enabled
        self._save()

    def _load(self) -> Dict[str, bool]:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                # Merge with defaults so new strategies get default True
                merged = {**_DEFAULTS, **data}
                return merged
            except Exception as exc:
                log.warning("strategy_state_load_failed", error=str(exc))
        return dict(_DEFAULTS)

    def _save(self) -> None:
        try:
            _STATE_FILE.write_text(json.dumps(self._state, indent=2))
        except Exception as exc:
            log.warning("strategy_state_save_failed", error=str(exc))

    # ── Alerts ─────────────────────────────────────────────────────────────

    @staticmethod
    def _telegram_alert(message: str) -> None:
        try:
            from config import settings
            import requests
            token = settings.telegram_bot_token
            chat_id = settings.telegram_chat_id
            if not token or not chat_id:
                return
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=10,
            )
        except Exception:
            pass
