"""Idempotent background service for automated news refresh."""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from news.curator import NewsCurator


class NewsCuratorService:
    def __init__(
        self,
        curator: NewsCurator | None = None,
        *,
        market_interval_seconds: int = 300,
        off_market_interval_seconds: int = 1200,
    ) -> None:
        self.curator = curator or NewsCurator()
        self.market_interval_seconds = max(60, int(market_interval_seconds))
        self.off_market_interval_seconds = max(300, int(off_market_interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._refresh_lock = threading.Lock()
        self.last_report: dict = {}
        self.last_error = ""
        self.last_refresh_at = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="quantterm-news-curator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def refresh_now(self) -> dict:
        if not self._refresh_lock.acquire(blocking=False):
            return self.last_report or {"status": "REFRESH_ALREADY_RUNNING"}
        try:
            report = self.curator.refresh()
            self.last_report = report.as_dict()
            self.last_error = ""
            self.last_refresh_at = datetime.now(timezone.utc).isoformat()
            return self.last_report
        except Exception as exc:
            self.last_error = str(exc)
            self.last_refresh_at = datetime.now(timezone.utc).isoformat()
            self.last_report = {"status": "ERROR", "error": self.last_error}
            return self.last_report
        finally:
            self._refresh_lock.release()

    def _market_open(self) -> bool:
        try:
            from core.market_session import in_market_open
            return bool(in_market_open())
        except Exception:
            return False

    def _worker(self) -> None:
        while not self._stop.is_set():
            self.refresh_now()
            interval = (
                self.market_interval_seconds
                if self._market_open()
                else self.off_market_interval_seconds
            )
            self._stop.wait(interval)

    def status(self) -> dict:
        return {
            "running": self.running,
            "last_refresh_at": self.last_refresh_at,
            "last_error": self.last_error,
            "last_report": self.last_report,
        }


_SERVICE: NewsCuratorService | None = None
_LOCK = threading.Lock()


def get_news_curator_service() -> NewsCuratorService:
    global _SERVICE
    if _SERVICE is None:
        with _LOCK:
            if _SERVICE is None:
                _SERVICE = NewsCuratorService()
    return _SERVICE
