"""Compute-on-change, serve-from-cache for read-only desk projections.

A terminal screen must answer in well under a second. On the owner's Mac
(2026-11-16, measured) several desk screens rebuilt their whole projection on
every request: the decision board took 15-35 s (it re-enriched and re-froze
every card on each GET), the Evolution Lab 18-20 s, the learning dashboard
5-7 s. Their inputs change only when a scan lands (~15 min) or evidence
settles (EOD). This cache serves the last build at once and rebuilds when the
inputs change:

* ``identity()`` is a cheap fingerprint of the inputs (file stats, a regime
  label). A changed identity, or an entry older than ``fresh_s``, triggers ONE
  background rebuild; readers keep getting the previous build meanwhile.
* The first build for a key is synchronous and single-flight: concurrent
  first callers wait for the same build instead of starting their own.
* ``watch()`` keeps a projection precomputed: a daemon thread rebuilds as soon
  as the identity changes, so a viewer almost never pays for a build.
* Every response carries ``projection_meta`` (HIT/MISS/STALE_REFRESHING, age,
  build time, input identity). Stale is labelled, never silent.

Only for read-only display projections. Nothing that authorises an order or
a paper entry may be served from here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable

Build = Callable[[], Any]
Identity = Callable[[], Any]


def file_identity(*paths) -> tuple:
    """(path, mtime_ns, size) for each path; missing files are part of the identity."""
    out = []
    for path in paths:
        try:
            st = os.stat(path)
            out.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(path), None, None))
    return tuple(out)


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class _Entry:
    __slots__ = ("value", "identity", "built_at", "build_s", "building", "error", "done")

    def __init__(self) -> None:
        self.value: Any = None
        self.identity: str | None = None
        self.built_at = 0.0
        self.build_s = 0.0
        self.building = False
        self.error = ""
        self.done = threading.Event()


class ProjectionCache:
    def __init__(self, name: str, *, fresh_s: float = 60.0, max_entries: int = 32,
                 clock: Callable[[], float] = time.monotonic,
                 wait_first_s: float = 120.0) -> None:
        self.name = name
        self.fresh_s = float(fresh_s)
        self.max_entries = int(max_entries)
        self.clock = clock
        self.wait_first_s = float(wait_first_s)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Hashable, _Entry]" = OrderedDict()

    # ── public ────────────────────────────────────────────────────────────────
    def get(self, key: Hashable, build: Build, *, identity: Identity | None = None) -> Any:
        ident = self._identity(identity)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry
                self._evict()
            self._entries.move_to_end(key)
            have = entry.value is not None
            if not have:
                first_builder = not entry.building
                if first_builder:
                    entry.building = True
                    entry.done.clear()
            else:
                age = self.clock() - entry.built_at
                stale = entry.identity != ident or age > self.fresh_s
                if stale and not entry.building:
                    entry.building = True
                    entry.done.clear()
                    self._spawn(key, entry, build, ident)
                status = "STALE_REFRESHING" if stale else "HIT"
                return self._serve(entry, status, ident)
        if first_builder:
            self._run(entry, build, ident)
        else:
            entry.done.wait(self.wait_first_s)
        with self._lock:
            if entry.value is None:
                raise RuntimeError(entry.error or f"{self.name} projection unavailable")
            return self._serve(entry, "MISS", ident)

    def invalidate(self, key: Hashable | None = None) -> None:
        """Drop one key (or everything): the next read rebuilds synchronously."""
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)

    def watch(self, key: Hashable, build: Build, *, identity: Identity,
              interval_s: float = 20.0, stop: threading.Event | None = None) -> threading.Thread:
        """Keep ``key`` precomputed: rebuild in the background whenever inputs change."""
        stop = stop or threading.Event()

        def loop() -> None:
            while not stop.is_set():
                try:
                    self.get(key, build, identity=identity)
                except Exception as exc:  # never kill the watcher
                    print(f"[PROJECTION] {self.name} warm failed type={type(exc).__name__}", flush=True)
                stop.wait(max(1.0, float(interval_s)))

        thread = threading.Thread(target=loop, name=f"projection-{self.name}", daemon=True)
        thread.start()
        return thread

    # ── internals ─────────────────────────────────────────────────────────────
    def _identity(self, identity: Identity | None) -> str:
        if identity is None:
            return ""
        try:
            return _fingerprint(identity())
        except Exception:
            return f"unreadable:{time.time():.0f}"

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest_key, oldest = next(iter(self._entries.items()))
            if oldest.building:
                break
            self._entries.pop(oldest_key, None)

    def _spawn(self, key, entry: _Entry, build: Build, ident: str) -> None:
        threading.Thread(target=self._run, args=(entry, build, ident),
                         name=f"projection-{self.name}-refresh", daemon=True).start()

    def _run(self, entry: _Entry, build: Build, ident: str) -> None:
        started = self.clock()
        try:
            value = build()
            with self._lock:
                entry.value = value
                entry.identity = ident
                entry.built_at = self.clock()
                entry.build_s = entry.built_at - started
                entry.error = ""
        except Exception as exc:
            with self._lock:
                entry.error = f"{type(exc).__name__}: {exc}"[:300]
            print(f"[PROJECTION] {self.name} build failed type={type(exc).__name__}", flush=True)
        finally:
            with self._lock:
                entry.building = False
            entry.done.set()

    def _serve(self, entry: _Entry, status: str, ident: str) -> Any:
        value = copy.deepcopy(entry.value)
        if isinstance(value, dict):
            value["projection_meta"] = {
                "cache": self.name,
                "status": status,
                "stale": status == "STALE_REFRESHING",
                "age_s": round(max(0.0, self.clock() - entry.built_at), 2),
                "build_s": round(entry.build_s, 2),
                "inputs_changed": bool(entry.identity is not None and entry.identity != ident),
                "refresh_error": entry.error or "",
            }
        return value
