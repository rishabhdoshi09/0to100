"""Single-flight, read-only HTTP dashboard cache for slow macOS installations.

The market scan, paper execution, risk interlocks, and other write authorities
are NEVER driven by or gated on this cache. Old reads are labelled and their
operational capabilities are set to unavailable rather than treated as live.
"""
from __future__ import annotations

from copy import deepcopy
import threading
import time
from typing import Any, Callable


class DashboardSnapshotCache:
    def __init__(
        self,
        loader: Callable[[], dict[str, Any]],
        cold_payload: Callable[[], dict[str, Any]],
        *,
        ttl_seconds: float = 45.0,
        failure_retry_seconds: float = 10.0,
    ) -> None:
        self._loader = loader
        self._cold_payload = cold_payload
        self.ttl_seconds = ttl_seconds
        self.failure_retry_seconds = failure_retry_seconds
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] | None = None
        self._completed_at: float | None = None
        self._refreshing = False
        self._next_retry_at = 0.0
        self._last_error = ""

    def _refresh(self) -> None:
        snapshot = None
        failure = ""
        try:
            result = self._loader()
            if not isinstance(result, dict) or result.get("error"):
                failure = "Dashboard builder reported a degraded read"
            else:
                snapshot = result
        except Exception as exc:
            # No potentially sensitive payload is printed into status logs.
            failure = f"Dashboard builder failed: {type(exc).__name__}"
        finally:
            now = time.monotonic()
            with self._lock:
                if snapshot is not None:
                    self._snapshot = snapshot
                    self._completed_at = now
                    self._last_error = ""
                    self._next_retry_at = 0.0
                else:
                    self._last_error = failure or "Dashboard builder returned no snapshot"
                    self._next_retry_at = now + self.failure_retry_seconds
                self._refreshing = False

    @staticmethod
    def _fail_closed_stale(payload: dict[str, Any]) -> None:
        """Keep historical cards but NEVER imply old execution/session truth."""
        for key in ("scan", "long_term", "news"):
            lane = payload.get(key)
            if isinstance(lane, dict):
                lane["available"] = False

        paper = payload.get("paper")
        if isinstance(paper, dict):
            paper.update(available=False, enabled=False, supervisor_running=False)

        autonomy = payload.get("autonomy")
        if isinstance(autonomy, dict):
            autonomy.update(
                available=False,
                running=False,
                process_running=False,
                new_paper_entries=False,
                existing_exits=False,
                research_enabled=False,
                new_entry_capability="blocked",
                existing_exit_capability="blocked",
                research_capability="blocked",
                state="UNKNOWN",
                plain_state="Dashboard status is stale; inspect live operator endpoints.",
            )
            autonomy["broker"] = {
                "state": "UNKNOWN", "ready": False,
                "reason": "Read-only dashboard snapshot is stale",
            }
            autonomy["live_feed"] = {}

        operations = payload.get("operations")
        if isinstance(operations, dict):
            operations["available"] = False
            operations["running"] = False
            operations["active"] = []
            operations["active_lanes"] = {}

        fno = payload.get("fno")
        if isinstance(fno, dict):
            fno["available"] = False
            directional = fno.get("directional")
            if isinstance(directional, dict):
                directional.update(
                    available=False, status="STALE", candidates=[],
                    code="DASHBOARD_STATUS_STALE",
                )
            desk = fno.get("desk")
            if isinstance(desk, dict):
                desk.update(
                    status="BLOCKED", reason="DASHBOARD_STATUS_STALE",
                    candidates=[], candidate_count=0, paper_available=False,
                )

        data = payload.get("data")
        if isinstance(data, dict):
            data["ready"] = False
            blockers = list(data.get("blockers") or [])
            blockers.append("Dashboard snapshot is stale; live operational readiness is unverified.")
            data["blockers"] = blockers
        payload["conviction"] = []
        payload["daily_wrap"] = []

    def read(self) -> dict[str, Any]:
        """Return immediately, spawning at most one background refresh.

        An overloaded or abandoned browser request never multiplies costly
        SQLite/pickle reads. Without a valid completed snapshot, serve only a
        fail-closed cold state. Do not fall back to a blocking read.
        """
        now = time.monotonic()
        with self._lock:
            snapshot = self._snapshot
            completed = self._completed_at
            age = max(0.0, now - completed) if completed is not None else None
            expired = age is None or age >= self.ttl_seconds
            launch = expired and not self._refreshing and now >= self._next_retry_at
            if launch:
                self._refreshing = True
            refreshing = self._refreshing
            last_error = self._last_error

        if launch:
            try:
                threading.Thread(
                    target=self._refresh,
                    name="quantterm-dashboard-read-snapshot",
                    daemon=True,
                ).start()
            except RuntimeError:
                with self._lock:
                    self._refreshing = False
                    self._next_retry_at = time.monotonic() + self.failure_retry_seconds
                    self._last_error = "Dashboard refresh thread could not start"
                refreshing = False
                last_error = "Dashboard refresh thread could not start"

        if snapshot is None:
            result = self._cold_payload()
            status = "DEGRADED" if last_error else "BOOTSTRAPPING"
            # The cold fallback must not imply the data is fresh.
            self._fail_closed_stale(result)
        else:
            result = deepcopy(snapshot)
            status = "STALE" if expired else "FRESH"
            if expired:
                self._fail_closed_stale(result)

        result["dashboard_cache"] = {
            "status": status,
            "snapshot_generated_at": (
                snapshot.get("generated_at", "") if snapshot is not None else ""
            ),
            "age_seconds": round(age, 1) if age is not None else None,
            "refreshing": refreshing,
            "retry_after_seconds": self.failure_retry_seconds if last_error else 0,
            "error": last_error,
            "read_only": True,
        }
        return result
