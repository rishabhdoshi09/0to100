"""Bounded single-flight projection cache for the read-only terminal desk.

A heavy /api/dashboard builder must never accumulate a new build every time a
browser polls or times out. No trading decisions or market writes flow through
this cache. A cached projection that outlives its short freshness window is
explicitly degraded and cannot authorize recommendations.
"""
from __future__ import annotations

from copy import deepcopy
import threading
import time
from typing import Any, Callable


class DeskReadSnapshot:
    def __init__(self, *, fresh_s: float = 15.0, stale_cap_s: float = 120.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._fresh_s = fresh_s
        self._stale_cap_s = stale_cap_s
        self._clock = clock
        self._lock = threading.Lock()
        self._payload: dict[str, Any] | None = None
        self._saved_at = 0.0
        self._building = False

    def _project(self, payload: dict[str, Any], *, age_s: float, stale: bool) -> dict[str, Any]:
        # Never edit the owned cached dictionary; other readers can run concurrently.
        result = deepcopy(payload)
        if stale:
            # Preserve prior research context, but explicitly remove new-entry
            # authority, including the canonical cash scan provenance check.
            data = result.get("data")
            if isinstance(data, dict):
                data["ready"] = False
                data["history_current"] = False
                data["blockers"] = list(data.get("blockers") or []) + [
                    "Desk projection is stale; wait for a new verified read before considering entries."
                ]
            scan = result.get("scan")
            if isinstance(scan, dict):
                provenance = dict(scan.get("provenance") or {})
                provenance["data_current"] = False
                scan["provenance"] = provenance
            market = result.get("market")
            if isinstance(market, dict):
                market["available"] = False
                market["health"] = "Unavailable"
                market["trade_stance"] = "Prior snapshot only; current market stance is unverified."
            result["conviction"] = []
            fno = result.get("fno")
            if isinstance(fno, dict):
                fno["available"] = False
        result["response_meta"] = {
            "source": "READ_ONLY_DESK_PROJECTION",
            "status": "REFRESHING" if stale else "CURRENT",
            "stale": stale,
            "age_s": round(age_s, 2),
        }
        return result

    def _run_build(self, build: Callable[[], dict[str, Any]]) -> None:
        payload: dict[str, Any] | None = None
        try:
            result = build()
            if isinstance(result, dict):
                payload = result
        except Exception as exc:
            # Never log exception strings here: provider exceptions can contain secrets.
            print(f"[DASHBOARD] snapshot refresh failed type={type(exc).__name__}", flush=True)
        finally:
            with self._lock:
                if payload is not None:
                    self._payload = payload
                    self._saved_at = self._clock()
                self._building = False

    def read(self, build: Callable[[], dict[str, Any]],
             fallback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            cached = self._payload
            age = max(0.0, now - self._saved_at) if cached is not None else 0.0
            if cached is not None and age <= self._fresh_s:
                return self._project(cached, age_s=age, stale=False)
            if not self._building:
                self._building = True
                try:
                    threading.Thread(
                        target=self._run_build,
                        args=(build,),
                        daemon=True,
                        name="qt-desk-read-refresh",
                    ).start()
                except Exception:
                    self._building = False
                    raise
            if cached is not None and age <= self._stale_cap_s:
                return self._project(cached, age_s=age, stale=True)

        # No trustworthy snapshot available. Return a clearly blocked desk
        # shell immediately instead of accumulating API_STATUS_BUSY requests.
        result = fallback()
        result["response_meta"] = {
            "source": "READ_ONLY_DESK_PROJECTION",
            "status": "WARMING",
            "stale": True,
            "age_s": None,
        }
        return result
