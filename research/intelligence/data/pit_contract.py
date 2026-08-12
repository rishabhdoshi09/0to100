"""
📌 Unified Point-in-Time Access Facade (Phase A / A1)

Thin research contract over the *existing* canonical sources:

  Snapshot / SnapshotStore / SnapshotBarProvider
  data_state readiness + evidence tiers
  universe_history / point_in_time_universe
  corporate_actions ledger
  pit_valuations ledger

Invariants
----------
* Does NOT copy bars into a new store.
* Does NOT reach the network or live quote waterfall.
* Does NOT fabricate missing fundamentals / membership / CA rows.
* Never returns market observations dated after the requested ``as_of`` /
  ``through`` timestamp when a Snapshot is bound.
* Missing PIT support is an explicit status (``INCOMPLETE`` / ``NOT_PIT_SAFE`` /
  ``BLOCKED``), never a silent “use today’s survivors” success.

Production runtime continues to use Snapshot / SnapshotBarProvider directly;
this facade is an additional research access path, not a replacement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from research.intelligence import data_state as DS

# Domains the facade understands. Unknown domains → BLOCKED.
DOMAIN_BARS = "bars"
DOMAIN_BENCHMARK = "benchmark"
DOMAIN_UNIVERSE = "universe"
DOMAIN_CORPORATE_ACTIONS = "corporate_actions"
DOMAIN_VALUATIONS = "valuations"
DOMAIN_FUNDAMENTALS = "fundamentals"
DOMAIN_EVENTS = "events"
DOMAIN_SECTORS = "sectors"

DOMAINS = (
    DOMAIN_BARS,
    DOMAIN_BENCHMARK,
    DOMAIN_UNIVERSE,
    DOMAIN_CORPORATE_ACTIONS,
    DOMAIN_VALUATIONS,
    DOMAIN_FUNDAMENTALS,
    DOMAIN_EVENTS,
    DOMAIN_SECTORS,
)


@dataclass(frozen=True)
class PitReadResult:
    """One explicit PIT read outcome — never a bare untyped payload."""

    status: str
    domain: str
    as_of: str | None = None
    data: Any = None
    reasons: tuple[str, ...] = ()
    snapshot_id: str = ""
    source: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when research may consume ``data`` (with reasons still attached)."""
        return self.status in (DS.READY, DS.DEGRADED, DS.STALE) and self.data is not None

    @property
    def pit_safe(self) -> bool:
        return self.status not in (DS.NOT_PIT_SAFE, DS.BLOCKED, DS.INCOMPLETE)


def _norm_date(value) -> str:
    if value is None:
        raise ValueError("timestamp required")
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp required")
    # Accept ISO dates and synthetic fixture dates (d000) unchanged.
    if "T" in text:
        text = text.split("T", 1)[0]
    return text


class PitContract:
    """Bound, read-only PIT accessor. Prefer ``from_snapshot`` / ``from_store``."""

    def __init__(
        self,
        snapshot=None,
        *,
        universe_history_path=None,
        ca_events_path=None,
        valuations_path=None,
        fundamentals_path=None,
        events_path=None,
        allow_network: bool = False,
    ):
        if allow_network:
            # Explicitly refused: frozen historical research must not open live paths.
            raise ValueError(
                "PitContract refuses allow_network=True — "
                "frozen historical research may not open live fetch paths"
            )
        self._snapshot = snapshot
        self._universe_history_path = universe_history_path
        self._ca_events_path = ca_events_path
        self._valuations_path = valuations_path
        self._fundamentals_path = fundamentals_path
        self._events_path = events_path

    # ── factories ────────────────────────────────────────────────────────────
    @classmethod
    def from_snapshot(cls, snapshot, **kwargs) -> "PitContract":
        if snapshot is None:
            raise ValueError("snapshot is required")
        return cls(snapshot, **kwargs)

    @classmethod
    def from_store(cls, store, snapshot_id: str, **kwargs) -> "PitContract":
        return cls.from_snapshot(store.open_snapshot(snapshot_id), **kwargs)

    @classmethod
    def from_active(cls, store, **kwargs) -> "PitContract | None":
        snap = store.open_active()
        return cls.from_snapshot(snap, **kwargs) if snap is not None else None

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def snapshot_id(self) -> str:
        return getattr(self._snapshot, "snapshot_id", "") or ""

    # ── public contract ──────────────────────────────────────────────────────
    def history(
        self,
        domain: str,
        *,
        symbol: str | None = None,
        through: str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> PitReadResult:
        """Return observations available at or before ``through`` (inclusive)."""
        return self.as_of(
            domain,
            when=through,
            symbol=symbol,
            name=name,
            mode="history",
            **kwargs,
        )

    def latest(
        self,
        domain: str,
        *,
        symbol: str | None = None,
        as_of: str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> PitReadResult:
        """Latest legitimate observation as of ``as_of`` (defaults to snapshot last date)."""
        when = as_of
        if when is None:
            if self._snapshot is None:
                return self._blocked(domain, None, "no snapshot bound and as_of not provided")
            when = self._snapshot.latest_available_date()
            if not when:
                return self._incomplete(domain, None, "snapshot has no last_trading_date")
        result = self.as_of(domain, when=when, symbol=symbol, name=name, mode="latest", **kwargs)
        if not result.usable:
            return result
        data = result.data
        if domain in (DOMAIN_BARS, DOMAIN_BENCHMARK) and isinstance(data, list) and data:
            data = data[-1]
        return PitReadResult(
            status=result.status,
            domain=result.domain,
            as_of=result.as_of,
            data=data,
            reasons=result.reasons,
            snapshot_id=result.snapshot_id,
            source=result.source,
            meta=dict(result.meta),
        )

    def as_of(
        self,
        domain: str,
        *,
        when,
        symbol: str | None = None,
        name: str | None = None,
        mode: str = "as_of",
        universe_source: str = "snapshot",
        **_ignored,
    ) -> PitReadResult:
        """Point-in-time read for ``domain`` at timestamp ``when``."""
        try:
            as_of = _norm_date(when)
        except ValueError as exc:
            return self._blocked(domain, None, str(exc))

        if domain not in DOMAINS:
            return self._blocked(domain, as_of, f"unknown domain '{domain}'")

        if domain == DOMAIN_BARS:
            return self._bars(as_of, symbol)
        if domain == DOMAIN_BENCHMARK:
            return self._benchmark(as_of, name)
        if domain == DOMAIN_UNIVERSE:
            return self._universe(as_of, source=universe_source)
        if domain == DOMAIN_CORPORATE_ACTIONS:
            return self._corporate_actions(as_of, symbol)
        if domain == DOMAIN_VALUATIONS:
            return self._valuations(as_of, symbol)
        if domain == DOMAIN_FUNDAMENTALS:
            return self._fundamentals(as_of, symbol)
        if domain == DOMAIN_EVENTS:
            return self._events(as_of, symbol)
        if domain == DOMAIN_SECTORS:
            return PitReadResult(
                status=DS.NOT_PIT_SAFE,
                domain=domain,
                as_of=as_of,
                data=None,
                reasons=(
                    "NSE sector membership is not historically dated in QuantTerm; "
                    "static sector maps must not contaminate PIT research",
                ),
                snapshot_id=self.snapshot_id,
                source="sector_heat_static",
            )
        return self._blocked(domain, as_of, "unhandled domain")

    def coverage(self, *, as_of: str | None = None, spec=None) -> PitReadResult:
        """Honest coverage / readiness for research claims (delegates to data_state)."""
        when = None
        if as_of is not None:
            try:
                when = _norm_date(as_of)
            except ValueError as exc:
                return self._blocked("coverage", None, str(exc))

        if self._snapshot is None:
            return PitReadResult(
                status=DS.BLOCKED,
                domain="coverage",
                as_of=when,
                data=None,
                reasons=("no snapshot bound — cannot assess market-data coverage",),
            )

        health = dict(self._snapshot.health())
        latest = self._snapshot.latest_available_date()
        reasons: list[str] = []
        status = DS.READY

        if when is not None and latest is not None and when > latest:
            return PitReadResult(
                status=DS.BLOCKED,
                domain="coverage",
                as_of=when,
                data={"health": health, "tier": DS.OPERATIONAL_ONLY},
                reasons=(f"as_of {when} is beyond snapshot last date {latest}",),
                snapshot_id=self.snapshot_id,
                source="snapshot",
                meta={"latest_available_date": latest},
            )

        # Ledger honesty (does not invent rows)
        univ = self._universe_ledger_status()
        ca = self._ca_ledger_status()
        val = self._valuations_ledger_status()
        fund = self._fundamentals_ledger_status()
        ev = self._events_ledger_status()

        if not univ.get("available"):
            health["has_universe_history"] = False
            reasons.append("universe history ledger missing")
        else:
            health["has_universe_history"] = bool(
                health.get("has_universe_history", True) and univ.get("available")
            )
            if not univ.get("research_grade", False):
                reasons.append("universe history not research_grade")
                status = DS.DEGRADED

        if not ca.get("available") or not ca.get("research_grade"):
            reasons.append("corporate-action ledger incomplete or absent")
            if "corporate_action_coverage" not in health:
                health["corporate_action_coverage"] = 0.0
            status = DS.DEGRADED if status == DS.READY else status

        if not val.get("available"):
            reasons.append("PIT valuation ledger absent")
        if not fund.get("available"):
            reasons.append("PIT fundamentals ledger absent or empty")
        if not ev.get("available"):
            reasons.append("PIT events ledger absent or empty")

        # Evidence tier from existing classifier — never upgrades on assumption
        if when is not None and latest is not None:
            health.setdefault("freshness_days", 0.0 if latest >= when else 999.0)
        tier, tier_reasons = DS.classify_tier(health)
        reasons.extend(tier_reasons)

        if tier == DS.OPERATIONAL_ONLY:
            status = DS.INCOMPLETE
        elif tier == DS.LIMITED_RESEARCH:
            status = DS.DEGRADED if status == DS.READY else status
        elif not health.get("has_prices"):
            status = DS.INCOMPLETE

        fund_domain = (
            DS.READY if fund.get("available") and fund.get("research_grade")
            else (DS.INCOMPLETE if fund.get("available") else DS.NOT_PIT_SAFE)
        )
        # Without a publication-dated ledger, fundamentals remain NOT_PIT_SAFE
        # (operational screener caches must never look READY).
        if not fund.get("available"):
            fund_domain = DS.NOT_PIT_SAFE
        ev_domain = (
            DS.READY if ev.get("available") and ev.get("research_grade")
            else (DS.INCOMPLETE if ev.get("available") else DS.INCOMPLETE)
        )

        cov = {
            "health": health,
            "tier": tier,
            "forward_eligible": DS.forward_eligible(tier),
            "ledgers": {
                "universe_history": univ,
                "corporate_actions": ca,
                "valuations": val,
                "fundamentals": fund,
                "events": ev,
            },
            "domains": {
                DOMAIN_BARS: DS.READY if health.get("has_prices") else DS.INCOMPLETE,
                DOMAIN_BENCHMARK: (
                    DS.READY if health.get("has_benchmark") else DS.INCOMPLETE
                ),
                DOMAIN_UNIVERSE: (
                    DS.READY if univ.get("available") and univ.get("research_grade")
                    else (DS.DEGRADED if univ.get("available") else DS.NOT_PIT_SAFE)
                ),
                DOMAIN_CORPORATE_ACTIONS: (
                    DS.READY if ca.get("research_grade") else DS.INCOMPLETE
                ),
                DOMAIN_VALUATIONS: (
                    DS.READY if val.get("available") else DS.INCOMPLETE
                ),
                DOMAIN_FUNDAMENTALS: fund_domain,
                DOMAIN_EVENTS: ev_domain,
                DOMAIN_SECTORS: DS.NOT_PIT_SAFE,
            },
        }
        if spec is not None:
            cov["strategy"] = self._snapshot.coverage_for(spec)

        return PitReadResult(
            status=status,
            domain="coverage",
            as_of=when or latest,
            data=cov,
            reasons=tuple(reasons),
            snapshot_id=self.snapshot_id,
            source="snapshot+ledgers",
            meta={"latest_available_date": latest},
        )

    # ── domain handlers ──────────────────────────────────────────────────────
    def _require_snapshot(self, domain: str, as_of: str) -> PitReadResult | None:
        if self._snapshot is None:
            return self._blocked(domain, as_of, "no snapshot bound")
        latest = self._snapshot.latest_available_date()
        if latest is not None and as_of > latest:
            return PitReadResult(
                status=DS.BLOCKED,
                domain=domain,
                as_of=as_of,
                data=None,
                reasons=(
                    f"refusing future as_of {as_of} beyond snapshot last date {latest}",
                ),
                snapshot_id=self.snapshot_id,
                source="snapshot",
                meta={"latest_available_date": latest},
            )
        return None

    def _bars(self, as_of: str, symbol: str | None) -> PitReadResult:
        blocked = self._require_snapshot(DOMAIN_BARS, as_of)
        if blocked is not None:
            return blocked
        if not symbol:
            return self._blocked(DOMAIN_BARS, as_of, "symbol is required for bars")
        bars = self._snapshot.bars(symbol, through=as_of)
        if not bars:
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_BARS,
                as_of=as_of,
                data=[],
                reasons=(f"no bars for {symbol.upper()} at or before {as_of}",),
                snapshot_id=self.snapshot_id,
                source="snapshot",
            )
        # Defence in depth: never leak a future bar even if Snapshot changed.
        leaked = [b for b in bars if getattr(b, "date", "") > as_of]
        if leaked:
            return PitReadResult(
                status=DS.BLOCKED,
                domain=DOMAIN_BARS,
                as_of=as_of,
                data=None,
                reasons=("snapshot returned bars after as_of — refusing payload",),
                snapshot_id=self.snapshot_id,
                source="snapshot",
            )
        return PitReadResult(
            status=DS.READY,
            domain=DOMAIN_BARS,
            as_of=as_of,
            data=bars,
            reasons=(),
            snapshot_id=self.snapshot_id,
            source="snapshot",
            meta={"symbol": symbol.upper(), "n": len(bars)},
        )

    def _benchmark(self, as_of: str, name: str | None) -> PitReadResult:
        blocked = self._require_snapshot(DOMAIN_BENCHMARK, as_of)
        if blocked is not None:
            return blocked
        if not self._snapshot.has_benchmark():
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_BENCHMARK,
                as_of=as_of,
                data=[],
                reasons=("snapshot has no benchmark series",),
                snapshot_id=self.snapshot_id,
                source="snapshot",
            )
        bars = self._snapshot.benchmark(through=as_of, name=name)
        leaked = [b for b in bars if getattr(b, "date", "") > as_of]
        if leaked:
            return self._blocked(
                DOMAIN_BENCHMARK, as_of, "snapshot returned benchmark bars after as_of"
            )
        return PitReadResult(
            status=DS.READY,
            domain=DOMAIN_BENCHMARK,
            as_of=as_of,
            data=bars,
            reasons=(),
            snapshot_id=self.snapshot_id,
            source="snapshot",
            meta={"n": len(bars), "name": name},
        )

    def _universe(self, as_of: str, *, source: str = "snapshot") -> PitReadResult:
        src = (source or "snapshot").lower()
        if src == "snapshot":
            blocked = self._require_snapshot(DOMAIN_UNIVERSE, as_of)
            if blocked is not None:
                return blocked
            symbols = list(self._snapshot.universe(as_of))
            health = self._snapshot.health()
            status = DS.READY
            reasons: list[str] = []
            # Bar-contemporaneous membership is PIT for *this snapshot*, but if the
            # snapshot itself lacks universe-history attestation, research claims
            # about survivorship remain degraded.
            if not health.get("has_universe_history", False):
                status = DS.DEGRADED
                reasons.append(
                    "snapshot lacks has_universe_history attestation — "
                    "bar-contemporaneous membership only"
                )
            return PitReadResult(
                status=status,
                domain=DOMAIN_UNIVERSE,
                as_of=as_of,
                data=symbols,
                reasons=tuple(reasons),
                snapshot_id=self.snapshot_id,
                source="snapshot.universe",
                meta={"n": len(symbols)},
            )

        if src != "ledger":
            return self._blocked(
                DOMAIN_UNIVERSE, as_of, f"unknown universe_source '{source}'"
            )

        # Ledger path — never present today's survivors as READY.
        from data.universe_history import history_path, ledger_status

        path = (
            self._universe_history_path
            if self._universe_history_path is not None
            else history_path()
        )
        status_info = ledger_status(path)
        if not status_info.get("available"):
            return PitReadResult(
                status=DS.NOT_PIT_SAFE,
                domain=DOMAIN_UNIVERSE,
                as_of=as_of,
                data=None,
                reasons=(
                    "universe history ledger missing — refusing survivorship-biased "
                    "fallback (no fake membership)",
                ),
                snapshot_id=self.snapshot_id,
                source="universe_history",
                meta=status_info,
            )

        from data.nse_universe import point_in_time_universe

        pit = point_in_time_universe(as_of, path=path)
        if not pit.get("survivorship_complete"):
            return PitReadResult(
                status=DS.NOT_PIT_SAFE,
                domain=DOMAIN_UNIVERSE,
                as_of=as_of,
                data=None,
                reasons=(
                    pit.get("note")
                    or "survivorship_complete=False — refusing biased membership",
                ),
                snapshot_id=self.snapshot_id,
                source="universe_history",
                meta=pit,
            )

        status = DS.READY if pit.get("research_grade") else DS.DEGRADED
        reasons = ()
        if not pit.get("research_grade"):
            reasons = ("universe ledger present but not research_grade",)
        return PitReadResult(
            status=status,
            domain=DOMAIN_UNIVERSE,
            as_of=as_of,
            data=list(pit.get("symbols") or []),
            reasons=reasons,
            snapshot_id=self.snapshot_id,
            source=str(pit.get("source") or "universe_history"),
            meta={"n": len(pit.get("symbols") or []), "research_grade": pit.get("research_grade")},
        )

    def _corporate_actions(self, as_of: str, symbol: str | None) -> PitReadResult:
        from data.corporate_actions import ledger_status, load_events

        path = self._ca_events_path
        info = ledger_status(path)
        if not info.get("available") or not info.get("research_grade"):
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_CORPORATE_ACTIONS,
                as_of=as_of,
                data=None,
                reasons=(
                    "corporate-action ledger absent or empty — "
                    "QuantTerm never invents CA events from gaps",
                ),
                snapshot_id=self.snapshot_id,
                source="corporate_actions",
                meta=info,
            )

        events = load_events(path)
        if symbol:
            sym = symbol.upper()
            rows = list(events.get(sym) or [])
        else:
            rows = []
            for sym, evs in events.items():
                for ev in evs:
                    item = dict(ev) if isinstance(ev, dict) else {"raw": ev}
                    item.setdefault("symbol", sym)
                    rows.append(item)

        # Keep only events whose ex-date is known and <= as_of when dated.
        kept = []
        for ev in rows:
            if not isinstance(ev, dict):
                kept.append(ev)
                continue
            ex = ev.get("ex_date") or ev.get("date") or ev.get("exDate")
            if ex is None:
                kept.append(ev)  # undated retained but flagged in meta
                continue
            try:
                if _norm_date(ex) <= as_of:
                    kept.append(ev)
            except ValueError:
                kept.append(ev)

        return PitReadResult(
            status=DS.READY,
            domain=DOMAIN_CORPORATE_ACTIONS,
            as_of=as_of,
            data=kept,
            reasons=(),
            snapshot_id=self.snapshot_id,
            source="corporate_actions",
            meta={"n": len(kept), "symbol": (symbol or "").upper() or None},
        )

    def _valuations(self, as_of: str, symbol: str | None) -> PitReadResult:
        from data.pit_valuations import get_valuation, ledger_status

        path = self._valuations_path
        info = ledger_status(path)
        if not info.get("available"):
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_VALUATIONS,
                as_of=as_of,
                data=None,
                reasons=("PIT valuation ledger absent — refusing current fundamentals",),
                snapshot_id=self.snapshot_id,
                source="pit_valuations",
                meta=info,
            )
        if not symbol:
            return self._blocked(DOMAIN_VALUATIONS, as_of, "symbol is required for valuations")

        row = get_valuation(symbol, as_of, path=path)
        if row is None:
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_VALUATIONS,
                as_of=as_of,
                data=None,
                reasons=(f"no valuation for {symbol.upper()} with available_ts <= {as_of}",),
                snapshot_id=self.snapshot_id,
                source="pit_valuations",
            )
        # Defence: available_ts must not exceed as_of.
        avail = _norm_date(row.get("available_ts"))
        if avail > as_of:
            return PitReadResult(
                status=DS.BLOCKED,
                domain=DOMAIN_VALUATIONS,
                as_of=as_of,
                data=None,
                reasons=("valuation available_ts exceeded as_of — refusing look-ahead",),
                snapshot_id=self.snapshot_id,
                source="pit_valuations",
            )
        return PitReadResult(
            status=DS.READY,
            domain=DOMAIN_VALUATIONS,
            as_of=as_of,
            data=row,
            reasons=(),
            snapshot_id=self.snapshot_id,
            source="pit_valuations",
            meta={"available_ts": avail, "symbol": symbol.upper()},
        )

    def _fundamentals(self, as_of: str, symbol: str | None) -> PitReadResult:
        from data.pit_fundamentals import get_fundamentals, ledger_status

        info = ledger_status(self._fundamentals_path)
        if not info.get("available"):
            # Preserve hard wall: operational caches are never a silent fallback.
            return PitReadResult(
                status=DS.NOT_PIT_SAFE,
                domain=DOMAIN_FUNDAMENTALS,
                as_of=as_of,
                data=None,
                reasons=(
                    "fundamentals caches are as-of-now; no publication-dated ledger "
                    "is bound through PitContract",
                ),
                snapshot_id=self.snapshot_id,
                source="fundamentals_cache",
                meta=info,
            )
        if not symbol:
            return self._blocked(DOMAIN_FUNDAMENTALS, as_of, "symbol is required for fundamentals")
        row = get_fundamentals(symbol, as_of, path=self._fundamentals_path)
        if row is None:
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_FUNDAMENTALS,
                as_of=as_of,
                data=None,
                reasons=(f"no fundamentals for {symbol.upper()} with available_at <= {as_of}",),
                snapshot_id=self.snapshot_id,
                source="pit_fundamentals",
            )
        avail = _norm_date(row.get("available_at"))
        if avail > as_of:
            return PitReadResult(
                status=DS.BLOCKED,
                domain=DOMAIN_FUNDAMENTALS,
                as_of=as_of,
                data=None,
                reasons=("fundamentals available_at exceeded as_of — refusing look-ahead",),
                snapshot_id=self.snapshot_id,
                source="pit_fundamentals",
            )
        return PitReadResult(
            status=DS.READY,
            domain=DOMAIN_FUNDAMENTALS,
            as_of=as_of,
            data=row,
            reasons=(),
            snapshot_id=self.snapshot_id,
            source="pit_fundamentals",
            meta={"available_at": avail, "symbol": symbol.upper()},
        )

    def _events(self, as_of: str, symbol: str | None) -> PitReadResult:
        from data.pit_events import get_events, ledger_status

        info = ledger_status(self._events_path)
        if not info.get("available"):
            return PitReadResult(
                status=DS.INCOMPLETE,
                domain=DOMAIN_EVENTS,
                as_of=as_of,
                data=None,
                reasons=("PIT events ledger absent — refusing undated announcement caches",),
                snapshot_id=self.snapshot_id,
                source="pit_events",
                meta=info,
            )
        rows = get_events(symbol, as_of, path=self._events_path)
        return PitReadResult(
            status=DS.READY,
            domain=DOMAIN_EVENTS,
            as_of=as_of,
            data=rows,
            reasons=(),
            snapshot_id=self.snapshot_id,
            source="pit_events",
            meta={"n": len(rows), "symbol": (symbol or "").upper() or None},
        )

    # ── ledger probes ────────────────────────────────────────────────────────
    def _universe_ledger_status(self) -> dict:
        try:
            from data.universe_history import history_path, ledger_status

            path = (
                self._universe_history_path
                if self._universe_history_path is not None
                else history_path()
            )
            return ledger_status(path)
        except Exception as exc:
            return {"available": False, "note": str(exc)}

    def _ca_ledger_status(self) -> dict:
        try:
            from data.corporate_actions import ledger_status

            return ledger_status(self._ca_events_path)
        except Exception as exc:
            return {"available": False, "note": str(exc)}

    def _valuations_ledger_status(self) -> dict:
        try:
            from data.pit_valuations import ledger_status

            return ledger_status(self._valuations_path)
        except Exception as exc:
            return {"available": False, "note": str(exc)}

    def _fundamentals_ledger_status(self) -> dict:
        try:
            from data.pit_fundamentals import ledger_status

            return ledger_status(self._fundamentals_path)
        except Exception as exc:
            return {"available": False, "note": str(exc)}

    def _events_ledger_status(self) -> dict:
        try:
            from data.pit_events import ledger_status

            return ledger_status(self._events_path)
        except Exception as exc:
            return {"available": False, "note": str(exc)}

    # ── status helpers ───────────────────────────────────────────────────────
    def _blocked(self, domain: str, as_of: str | None, reason: str) -> PitReadResult:
        return PitReadResult(
            status=DS.BLOCKED,
            domain=domain,
            as_of=as_of,
            data=None,
            reasons=(reason,),
            snapshot_id=self.snapshot_id,
        )

    def _incomplete(self, domain: str, as_of: str | None, reason: str) -> PitReadResult:
        return PitReadResult(
            status=DS.INCOMPLETE,
            domain=domain,
            as_of=as_of,
            data=None,
            reasons=(reason,),
            snapshot_id=self.snapshot_id,
        )
