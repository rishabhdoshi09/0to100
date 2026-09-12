"""One contract for getting data, so a source failure is not a product failure.

Before this, every dataset invented its own ladder. The universe had three
tiers, corporate actions had three, quotes had three, and each of them logged
"Tier 2 failed" and moved on without recording why or what it eventually used.
Two consequences, both bad:

* A parser that stopped matching the page produced zero rows, and zero rows
  read as "there is nothing today". The desk cannot tell a quiet market from a
  broken scraper, so it believes the broken scraper.
* Nothing downstream could say which source an artifact came from, how old it
  was, or how far down the ladder the system had to go to get it. Fallback data
  looked exactly like primary data.

So an acquisition here answers three questions and records all three:

    what did we get, where did it come from, and what did we try first?

The failure classification is the part that matters most. UNREACHABLE, EMPTY
and PARSER_CHANGED are different facts with different responses — retry the
first, believe the second, alert on the third — and collapsing them into a bare
exception is how a silent scraper failure becomes a silent trading decision.

Access boundaries are not a tier. A source marked ``requires_bypass`` is never
attempted: publicly reachable pages are fair game, credentials, CAPTCHAs and
paywalls are not, and that is a property of the source rather than a judgement
made per call.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1


class SourceTier(IntEnum):
    """Preference order. Lower is more authoritative."""

    OFFICIAL_API = 1
    OFFICIAL_FILE = 2          # exchange/regulator CSV, ZIP, JSON, XML download
    OFFICIAL_PAGE = 3          # exchange/regulator/company web page
    BROKER = 4                 # Zerodha and friends
    ALTERNATE_AUTHORITATIVE = 5
    REPUTABLE_PUBLIC = 6
    PUBLIC_SCRAPE = 7          # legitimately accessible public web content
    LAST_KNOWN_GOOD = 8        # what we already had on disk


# ── outcomes ───────────────────────────────────────────────────────────────
OK = "OK"
UNREACHABLE = "UNREACHABLE"
TIMEOUT = "TIMEOUT"
HTTP_ERROR = "HTTP_ERROR"
EMPTY = "EMPTY"
SCHEMA_INVALID = "SCHEMA_INVALID"
PARSER_CHANGED = "PARSER_CHANGED"
STALE = "STALE"
NOT_PRESENT = "NOT_PRESENT"      # a local file or cache that simply is not there
REFUSED_BYPASS = "REFUSED_BYPASS"
ERROR = "ERROR"

#: Outcomes that mean "this source is fine, there is genuinely nothing here".
#: Deliberately small: almost nothing qualifies, because "nothing today" is a
#: claim about the world and most failures are claims about our code.
BENIGN_EMPTY: frozenset[str] = frozenset()

# ── acquisition states ─────────────────────────────────────────────────────
ACQUIRED = "ACQUIRED"
LAST_KNOWN_GOOD_STATE = "LAST_KNOWN_GOOD"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
SOURCE_CONFLICT = "SOURCE_CONFLICT"


class BypassRefused(RuntimeError):
    """A source was declared to need an access-control bypass. We do not."""


@dataclass(frozen=True)
class Validation:
    """The verdict on one fetched payload. HTTP 200 is not a verdict."""

    ok: bool
    outcome: str = OK
    record_count: int | None = None
    detail: str = ""

    @classmethod
    def good(cls, record_count: int | None = None, detail: str = "") -> "Validation":
        return cls(True, OK, record_count, detail)

    @classmethod
    def bad(cls, outcome: str, detail: str = "",
            record_count: int | None = None) -> "Validation":
        return cls(False, outcome, record_count, detail)


@dataclass(frozen=True)
class Source:
    """One way to get a dataset.

    ``fetch`` returns the payload or raises. ``validate`` decides whether the
    payload is usable and, when it is not, says which kind of not-usable it is.
    A source without a validator is accepted on non-empty truthiness alone,
    which is weak — prefer giving one.
    """

    name: str
    tier: SourceTier
    fetch: Callable[[], Any]
    validate: Callable[[Any], Validation] | None = None
    #: True when reaching this source would require getting past
    #: authentication, a CAPTCHA, a paywall or another access control. Such a
    #: source is recorded as REFUSED_BYPASS and never called.
    requires_bypass: bool = False
    identifier: str = ""       # URL, path, endpoint — whatever names it
    note: str = ""


@dataclass(frozen=True)
class Attempt:
    source: str
    tier: int
    outcome: str
    detail: str = ""
    record_count: int | None = None
    duration_ms: int = 0
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionResult:
    dataset: str
    state: str
    value: Any = None
    source: str = ""
    tier: int | None = None
    fallback_level: int = 0        # 0 = the first source worked
    record_count: int | None = None
    content_hash: str = ""
    attempts: tuple[Attempt, ...] = ()
    fetched_at: str = ""
    effective_at: str = ""
    conflict: dict[str, Any] | None = None
    parser_version: str = ""

    @property
    def ok(self) -> bool:
        return self.state in (ACQUIRED, LAST_KNOWN_GOOD_STATE)

    @property
    def is_primary(self) -> bool:
        """Whether this came from the most preferred source we were given."""
        return self.ok and self.fallback_level == 0

    @property
    def parser_changed(self) -> bool:
        """Any source whose page or feed no longer matches its parser.

        Worth surfacing even on a successful acquisition: a lower tier covered
        for it this time, and nobody will notice the rot otherwise.
        """
        return any(a.outcome == PARSER_CHANGED for a in self.attempts)

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "state": self.state,
            "source": self.source,
            "tier": self.tier,
            "fallback_level": self.fallback_level,
            "record_count": self.record_count,
            "content_hash": self.content_hash,
            "fetched_at": self.fetched_at,
            "effective_at": self.effective_at,
            "parser_version": self.parser_version,
            "parser_changed_somewhere": self.parser_changed,
            "conflict": self.conflict,
            "attempts": [a.to_dict() for a in self.attempts],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: Any) -> str:
    try:
        blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(value).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _classify(exc: BaseException) -> tuple[str, str]:
    """Turn an exception into one of the outcomes above.

    Deliberately structural rather than string-matching where possible: the
    difference between "we could not reach it" and "we reached it and could not
    read it" changes what the operator should do.
    """
    name = type(exc).__name__
    text = str(exc)[:300]
    if isinstance(exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError)):
        return NOT_PRESENT, text
    if name in ("Timeout", "ConnectTimeout", "ReadTimeout", "socket.timeout"):
        return TIMEOUT, text
    if name in ("ConnectionError", "NewConnectionError", "URLError", "gaierror",
                "NetworkAccessDenied"):
        return UNREACHABLE, text
    if name in ("HTTPError", "TooManyRedirects"):
        return HTTP_ERROR, text
    if name in ("ParserError", "EmptyDataError", "XMLSyntaxError", "JSONDecodeError"):
        return PARSER_CHANGED, text
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return TIMEOUT, text
    if "connection" in lowered or "resolve" in lowered or "unreachable" in lowered:
        return UNREACHABLE, text
    return ERROR, f"{name}: {text}"


def _default_validate(payload: Any) -> Validation:
    if payload is None:
        return Validation.bad(EMPTY, "source returned nothing")
    try:
        count = len(payload)  # type: ignore[arg-type]
    except TypeError:
        return Validation.good()
    if count == 0:
        return Validation.bad(EMPTY, "source returned zero records")
    return Validation.good(record_count=count)


def provenance_path(dataset: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(dataset))
    return logs_path("provenance", f"{safe}.json")


def provenance_log_path(dataset: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(dataset))
    return logs_path("provenance", f"{safe}.jsonl")


def write_provenance(result: AcquisitionResult) -> Path | None:
    """Persist what we used and what we tried. Never fails an acquisition."""
    try:
        payload = result.provenance()
        latest = provenance_path(result.dataset)
        latest.parent.mkdir(parents=True, exist_ok=True)
        tmp = latest.with_suffix(latest.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, latest)
        with provenance_log_path(result.dataset).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        return latest
    except Exception:
        return None


def read_provenance(dataset: str) -> dict[str, Any] | None:
    try:
        return json.loads(provenance_path(dataset).read_text(encoding="utf-8"))
    except Exception:
        return None


def acquire(
    dataset: str,
    sources: Sequence[Source],
    *,
    corroborate: Callable[[Any, Any], bool] | None = None,
    persist: bool = True,
    parser_version: str = "",
    effective_at: str = "",
) -> AcquisitionResult:
    """Walk the ladder until something validates. Record every step.

    Sources are tried in the order given; they are NOT reordered by tier,
    because the caller knows things the tier number does not (a cached broker
    file may beat a slow official endpoint during market hours). Tier is
    recorded so the provenance says how authoritative the answer was.

    ``corroborate`` cross-checks the winner against the next usable source. If
    they materially disagree the result is SOURCE_CONFLICT and carries both:
    picking whichever value lets the workflow continue is how a wrong price
    becomes a position.
    """
    attempts: list[Attempt] = []
    winner: tuple[Source, Any, Validation] | None = None

    for index, source in enumerate(sources):
        started = time.monotonic()
        if source.requires_bypass:
            attempts.append(Attempt(
                source=source.name, tier=int(source.tier), outcome=REFUSED_BYPASS,
                detail="source requires getting past an access control; not attempted",
                at=_now(),
            ))
            continue
        try:
            payload = source.fetch()
        except BaseException as exc:  # noqa: BLE001 - classified, then recorded
            outcome, detail = _classify(exc)
            attempts.append(Attempt(
                source=source.name, tier=int(source.tier), outcome=outcome,
                detail=detail, duration_ms=int((time.monotonic() - started) * 1000),
                at=_now(),
            ))
            continue

        validator = source.validate or _default_validate
        try:
            verdict = validator(payload)
        except BaseException as exc:  # noqa: BLE001 - a validator that throws is a parser problem
            verdict = Validation.bad(PARSER_CHANGED, f"validator raised: {type(exc).__name__}")

        attempts.append(Attempt(
            source=source.name, tier=int(source.tier),
            outcome=verdict.outcome if not verdict.ok else OK,
            detail=verdict.detail, record_count=verdict.record_count,
            duration_ms=int((time.monotonic() - started) * 1000), at=_now(),
        ))
        if verdict.ok and winner is None:
            winner = (source, payload, verdict)
            if corroborate is None:
                break
            # Keep walking one more usable source to cross-check the winner.
            continue
        if winner is not None and verdict.ok and corroborate is not None:
            agreed = False
            try:
                agreed = bool(corroborate(winner[1], payload))
            except Exception:
                agreed = False
            if not agreed:
                result = AcquisitionResult(
                    dataset=dataset, state=SOURCE_CONFLICT, value=None,
                    source=winner[0].name, tier=int(winner[0].tier),
                    fallback_level=0, attempts=tuple(attempts), fetched_at=_now(),
                    effective_at=effective_at, parser_version=parser_version,
                    conflict={
                        "a": winner[0].name, "b": source.name,
                        "detail": "two trusted sources disagree materially",
                    },
                )
                if persist:
                    write_provenance(result)
                return result
            break

    if winner is None:
        result = AcquisitionResult(
            dataset=dataset, state=DATA_UNAVAILABLE, value=None,
            attempts=tuple(attempts), fetched_at=_now(),
            effective_at=effective_at, parser_version=parser_version,
        )
        if persist:
            write_provenance(result)
        return result

    source, payload, verdict = winner
    level = next(i for i, s in enumerate(sources) if s is source)
    state = LAST_KNOWN_GOOD_STATE if source.tier == SourceTier.LAST_KNOWN_GOOD else ACQUIRED
    result = AcquisitionResult(
        dataset=dataset, state=state, value=payload, source=source.name,
        tier=int(source.tier), fallback_level=level,
        record_count=verdict.record_count, content_hash=_hash(payload),
        attempts=tuple(attempts), fetched_at=_now(), effective_at=effective_at,
        parser_version=parser_version,
    )
    if persist:
        write_provenance(result)
    return result
