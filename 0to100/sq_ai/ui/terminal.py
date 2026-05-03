"""Textual TUI – polls FastAPI every 2 s.

Layout (single screen):
┌────────────────────────────────────────────────────────────────────────┐
│ HEADER (clock, market hours, next-cycle countdown)                     │
├──────────────────────────────────┬─────────────────────────────────────┤
│ Portfolio summary (DataTable)    │ Open positions (DataTable)          │
├──────────────────────────────────┼─────────────────────────────────────┤
│ Latest signals  (RichLog)        │ Claude reasoning (RichLog)          │
├──────────────────────────────────┴─────────────────────────────────────┤
│ Alerts                                                                 │
├────────────────────────────────────────────────────────────────────────┤
│ FOOTER                                                                 │
└────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
from datetime import datetime, time as dt_time
from typing import Any

import httpx
import pytz
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, RichLog, Static


IST = pytz.timezone(os.environ.get("SQ_TIMEZONE", "Asia/Kolkata"))
HOST = os.environ.get("SQ_API_HOST", "127.0.0.1")
PORT = os.environ.get("SQ_API_PORT", "8000")
BASE_URL = f"http://{HOST}:{PORT}"


def market_hours_label(now: datetime | None = None) -> str:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return "[red]CLOSED (weekend)[/red]"
    if dt_time(9, 15) <= now.time() <= dt_time(15, 30):
        return "[green]OPEN[/green]"
    return "[yellow]CLOSED[/yellow]"


class StatusBar(Static):
    def update_status(self, snap: dict[str, Any]) -> None:
        eq = snap.get("equity", 0.0)
        cash = snap.get("cash", 0.0)
        expo = snap.get("exposure_pct", 0.0)
        dpnl = snap.get("daily_pnl_pct", 0.0)
        hours = market_hours_label()
        self.update(
            f"  Market: {hours}    "
            f"Equity: ₹{eq:,.0f}    Cash: ₹{cash:,.0f}    "
            f"Exposure: {expo:.1f}%    Day P&L: "
            f"{'[green]' if dpnl >= 0 else '[red]'}{dpnl:+.2f}%[/]"
        )


class CockpitApp(App):
    CSS = """
    Screen { background: #0d1117; }
    StatusBar { padding: 1; background: #161b22; color: #c9d1d9; height: 3; }
    DataTable { height: 1fr; }
    RichLog   { height: 1fr; border: solid #30363d; }
    .col      { padding: 0 1; }
    #alerts   { height: 6; border: solid #f85149; color: #f85149; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]

    last_snap: reactive[dict] = reactive({})

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar(id="status")
        with Horizontal():
            with Vertical(classes="col"):
                yield Static("[b]Open Positions[/b]")
                yield DataTable(id="positions", zebra_stripes=True)
            with Vertical(classes="col"):
                yield Static("[b]Latest Signals[/b]")
                yield DataTable(id="signals", zebra_stripes=True)
        with Horizontal():
            with Vertical(classes="col"):
                yield Static("[b]Cycle Log[/b]")
                yield RichLog(id="cycle_log", highlight=True, markup=True, wrap=True)
            with Vertical(classes="col"):
                yield Static("[b]Claude Reasoning[/b]")
                yield RichLog(id="claude_log", highlight=True, markup=True, wrap=True)
        yield RichLog(id="alerts", highlight=True, markup=True, wrap=True)
        yield Footer()

    # ----------------------------------------------------------------- mount
    def on_mount(self) -> None:
        pos_t = self.query_one("#positions", DataTable)
        pos_t.add_columns("Sym", "Qty", "Entry", "Stop", "Target")
        sig_t = self.query_one("#signals", DataTable)
        sig_t.add_columns("Time", "Sym", "Action", "Conf", "Regime")
        self.set_interval(2.0, self.refresh_data)
        self.run_worker(self._initial(), exclusive=True)

    async def _initial(self) -> None:
        await self.refresh_data()

    # ----------------------------------------------------------------- poll
    async def refresh_data(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=2.0) as cl:
                snap_r = await cl.get(f"{BASE_URL}/api/portfolio")
                pos_r = await cl.get(f"{BASE_URL}/api/positions")
                sig_r = await cl.get(f"{BASE_URL}/api/signals/latest?limit=10")
                cyc_r = await cl.get(f"{BASE_URL}/api/cycle/last")
        except Exception as exc:
            self.query_one("#alerts", RichLog).write(f"[red]API error: {exc}[/red]")
            return

        snap = snap_r.json() if snap_r.status_code == 200 else {}
        self.query_one("#status", StatusBar).update_status(snap)

        # positions
        pos_t = self.query_one("#positions", DataTable)
        pos_t.clear()
        for p in pos_r.json() if pos_r.status_code == 200 else []:
            pos_t.add_row(
                p["symbol"], str(p["qty"]),
                f"{p['entry_price']:.2f}",
                f"{p['stop_initial']:.2f}",
                f"{p['target']:.2f}",
            )

        # signals
        sig_t = self.query_one("#signals", DataTable)
        sig_t.clear()
        for s in sig_r.json() if sig_r.status_code == 200 else []:
            sig_t.add_row(
                s["timestamp"][11:19], s["symbol"], s["action"],
                f"{s['confidence']:.2f}", str(s["regime"]),
            )

        # cycle
        cyc = cyc_r.json() if cyc_r.status_code == 200 else {}
        if cyc:
            self.query_one("#cycle_log", RichLog).clear()
            self.query_one("#cycle_log", RichLog).write(
                f"[cyan]{cyc.get('timestamp', '?')}[/cyan]  "
                f"market={cyc.get('market_hours', '?')}  "
                f"used_claude={cyc.get('used_claude', '?')}\n"
                f"events={len(cyc.get('events', []))}  "
                f"decisions={len(cyc.get('decisions', []))}"
            )
            self.query_one("#claude_log", RichLog).clear()
            for d in cyc.get("decisions", []):
                self.query_one("#claude_log", RichLog).write(
                    f"[b]{d.get('symbol')}[/b] [{d.get('action')}] "
                    f"conf={d.get('confidence', 0):.2f}  "
                    f"{d.get('reasoning', '')}"
                )
            for e in cyc.get("events", []):
                if e.get("reason") in ("stop", "target", "time"):
                    self.query_one("#alerts", RichLog).write(
                        f"[yellow]EXIT {e.get('symbol')} ({e['reason']}) "
                        f"@ ₹{e.get('price', 0):.2f}  pnl=₹{e.get('pnl', 0):.0f}[/yellow]"
                    )

    async def action_refresh(self) -> None:
        await self.refresh_data()


def main() -> None:
    CockpitApp().run()


if __name__ == "__main__":
    main()
