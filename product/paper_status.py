"""Read-only projection of the persisted intelligence PaperBook."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PaperStatus:
    enabled: bool = True
    supervisor_running: bool = False
    capital: float = 100_000.0
    equity: float = 100_000.0
    open_positions: tuple = ()
    closed_trades: tuple = ()
    refusals: tuple = ()
    open_risk: float = 0.0
    risk_per_trade_pct: float = 0.01
    max_positions: int = 5
    last_error: str = ""
    last_cycle: dict = field(default_factory=dict)


def _json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def read_paper_status(*, repo_root=None, autonomy_root=None) -> PaperStatus:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    from research.autonomy import default_root
    auto = Path(autonomy_root) if autonomy_root else default_root()
    book = _json(root / "logs" / "intelligence" / "intel_book.json", {})
    config = _json(root / "logs" / "intelligence" / "paper_config.json", {})
    status = _json(auto / "status.json", {})
    owner = dict(status.get("owner_state", {}))
    opens = tuple(book.get("open", []) or [])
    closed = tuple(book.get("closed", []) or [])
    capital = float(book.get("capital", config.get("starting_capital", 100_000.0)) or 100_000.0)
    curve = list(book.get("equity_curve", []) or [])
    equity = float(curve[-1]) if curve else capital + float(book.get("realized_pnl", 0.0) or 0.0)
    open_risk = sum(float(p.get("risk_amount", 0.0) or 0.0) for p in opens)
    return PaperStatus(
        enabled=bool(owner.get("paper_auto_enabled", config.get("enabled", True))),
        supervisor_running=bool(status.get("heartbeat_ist")), capital=capital, equity=equity,
        open_positions=opens, closed_trades=closed, refusals=tuple(book.get("refusals", []) or []),
        open_risk=open_risk, last_error=str(status.get("explanation", "")),
        last_cycle=dict(status.get("last_cycle", {}) or {}),
    )
