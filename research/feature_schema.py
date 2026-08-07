"""
🧬 Feature Schema — the ONE canonical definition of every feature, versioned.

The Research OS reached a transition point: Phase 1 built the scientific *process*
(can we trust this edge? is it drifting? did we overfit?). This is the first piece
of the scientific *memory* — the shift from thinking in TRADES to thinking in
OBSERVATIONS. Every observation (a scan, a trade, a rejection, a near-miss) gets
ONE immutable feature vector, and every research subsystem consumes exactly this
representation instead of re-deriving RSI/ATR/regime from scratch in five places.

This module owns three of the Feature Platform's four responsibilities (the
fourth, snapshotting, lives in `feature_store`):

  1. CANONICAL COMPUTATION — every feature is defined once here: name, dtype,
     valid range, description. No duplication, no drift between call sites.
  2. VERSIONING — each `Feature` carries a `version`; the whole registry hashes
     to a `SCHEMA_VERSION`. Improve Relative Strength 18 months from now → bump
     its version → the schema hash changes → every new snapshot records which
     schema produced it, and OLD experiments stay reproducible.
  3. VALIDATION — every feature self-validates: MISSING / IMPOSSIBLE (out of hard
     range) / OUTLIER (past a sane soft band) / STALE (too old). If Delivery
     suddenly reads 250%, the platform rejects it — not the scanner.

Pure and dependency-free (stdlib + the values handed in); unit-tested.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# Validation verdicts (one vocabulary, used everywhere downstream).
OK = "OK"
MISSING = "MISSING"
IMPOSSIBLE = "IMPOSSIBLE"     # violates a hard physical bound (delivery > 100%)
OUTLIER = "OUTLIER"          # inside hard bounds but past a sane soft band
STALE = "STALE"             # older than the feature tolerates


@dataclass(frozen=True)
class Feature:
    """One canonical feature. `version` bumps whenever its COMPUTATION changes,
    so historical snapshots remain attributable to the exact definition that
    produced them. Numeric features carry hard [lo, hi] bounds (violations are
    IMPOSSIBLE) and optional soft [soft_lo, soft_hi] bounds (violations are
    OUTLIERs). Categorical features carry the allowed `categories`."""
    name: str
    version: int
    dtype: str                       # "float" | "int" | "categorical"
    description: str
    lo: float | None = None
    hi: float | None = None
    soft_lo: float | None = None
    soft_hi: float | None = None
    categories: tuple[str, ...] | None = None
    # a feature may go stale — max age in DAYS relative to the observation's own
    # timestamp for point-in-time features; None = timeless (regime, sector).
    max_age_days: float | None = None

    def validate(self, value, age_days: float | None = None) -> tuple[str, str]:
        """Return (verdict, human reason). Pure; no I/O."""
        if value is None or (isinstance(value, float) and value != value):  # NaN
            return MISSING, f"{self.name} absent"
        if self.dtype == "categorical":
            if self.categories and str(value) not in self.categories:
                return IMPOSSIBLE, f"{self.name}={value!r} not a known category"
            return OK, ""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return IMPOSSIBLE, f"{self.name}={value!r} not numeric"
        if v != v:  # NaN slipped through
            return MISSING, f"{self.name} is NaN"
        if self.lo is not None and v < self.lo:
            return IMPOSSIBLE, f"{self.name}={v:g} below hard floor {self.lo:g}"
        if self.hi is not None and v > self.hi:
            return IMPOSSIBLE, f"{self.name}={v:g} above hard cap {self.hi:g}"
        if age_days is not None and self.max_age_days is not None \
                and age_days > self.max_age_days:
            return STALE, f"{self.name} is {age_days:.1f}d old (max {self.max_age_days:g}d)"
        if self.soft_lo is not None and v < self.soft_lo:
            return OUTLIER, f"{self.name}={v:g} below sane band {self.soft_lo:g}"
        if self.soft_hi is not None and v > self.soft_hi:
            return OUTLIER, f"{self.name}={v:g} above sane band {self.soft_hi:g}"
        return OK, ""


# ══════════════════════════════════════════════════════════════════════════════
# THE canonical registry — the single source of truth for what a feature IS.
# Adding a feature or bumping a version here changes SCHEMA_VERSION below, which
# is stamped on every new snapshot. Order is irrelevant (the hash sorts).
# ══════════════════════════════════════════════════════════════════════════════

_FEATURES: tuple[Feature, ...] = (
    # ── price / trend structure ──
    Feature("rsi", 1, "float", "14-period RSI (momentum)", 0, 100, 5, 95),
    Feature("atr_pct", 1, "float", "ATR as % of price (volatility)", 0, 100, 0, 25),
    Feature("dist_from_high_pct", 1, "float",
            "% below the 52-week high (0 = at highs)", 0, 100, 0, 60),
    Feature("rel_strength", 1, "float",
            "relative strength vs the index (z-like, +ve = leader)", -5, 5, -3, 3),
    Feature("clv", 1, "float",
            "Close Location Value in the day's range (-1 low … +1 high)", -1, 1),
    Feature("volume_z", 1, "float",
            "volume z-score vs its own recent history", -5, 20, -3, 10),
    Feature("delivery_pct", 1, "float",
            "% of volume taken to delivery (conviction)", 0, 100, 5, 100),
    Feature("adr_pct", 1, "float", "average daily range %", 0, 100, 0, 20),
    Feature("liquidity_cr", 1, "float",
            "median daily turnover, ₹ crore (tradability)", 0, 1e6, 0.5, 1e5),
    # ── setup scores (the scanner's own read) ──
    Feature("quality_score", 1, "float", "scanner composite quality 0-100", 0, 100),
    Feature("accum_score", 1, "float", "accumulation footprint 0-100", 0, 100),
    # ── market context ──
    Feature("regime", 1, "categorical", "market regime label", categories=(
        "TRENDING_BULL", "EXPANSION", "MIXED", "RANGE", "CONTRACTION",
        "DISTRIBUTION", "TRENDING_BEAR", "BEAR", "UNKNOWN")),
    Feature("breadth_pct_above_50dma", 1, "float",
            "% of market above its 50-DMA (breadth)", 0, 100),
    Feature("sector_strength", 1, "float",
            "parent sector's relative strength (-ve = laggard sector)", -5, 5, -3, 3),
    Feature("index_trend", 1, "categorical", "index trend label",
            categories=("UP", "FLAT", "DOWN", "UNKNOWN")),
    # ── macro (news-blind stack's radar) ──
    Feature("vix", 1, "float", "India VIX (fear)", 0, 200, 5, 60, max_age_days=3),
    Feature("usdinr", 1, "float", "USD/INR", 20, 200, 60, 120, max_age_days=3),
    Feature("crude_usd", 1, "float", "Brent crude, USD/bbl", 0, 500, 20, 200,
            max_age_days=3),
)

FEATURE_REGISTRY: dict[str, Feature] = {f.name: f for f in _FEATURES}
FEATURE_NAMES: tuple[str, ...] = tuple(sorted(FEATURE_REGISTRY))


def _schema_hash(registry: dict[str, Feature]) -> str:
    """Deterministic fingerprint of the WHOLE schema. Any name/version/bound
    change flips it, so a snapshot can always be traced to the exact schema that
    produced it."""
    payload = json.dumps(
        [[f.name, f.version, f.dtype, f.lo, f.hi, f.soft_lo, f.soft_hi,
          list(f.categories) if f.categories else None, f.max_age_days]
         for f in sorted(registry.values(), key=lambda x: x.name)],
        sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


SCHEMA_VERSION: str = _schema_hash(FEATURE_REGISTRY)
SCHEMA_TAG: str = f"fs_{SCHEMA_VERSION}"


# ══════════════════════════════════════════════════════════════════════════════
# Canonicalisation + whole-vector validation
# ══════════════════════════════════════════════════════════════════════════════

def canonicalize(raw: dict) -> dict:
    """Project an arbitrary producer dict onto the canonical schema: keep only
    known features, coerce numerics to float / ints, pass categoricals as str,
    and set anything missing or uncoercible to None. Unknown keys are dropped
    (they have no definition, so they cannot be part of an observation)."""
    out: dict = {}
    for name, feat in FEATURE_REGISTRY.items():
        if name not in raw or raw[name] is None:
            out[name] = None
            continue
        v = raw[name]
        if feat.dtype == "categorical":
            out[name] = str(v)
        else:
            try:
                out[name] = float(v)
            except (TypeError, ValueError):
                out[name] = None
    return out


@dataclass
class ValidationReport:
    ok: bool                              # no MISSING-required / IMPOSSIBLE issues
    verdicts: dict                        # name → (verdict, reason)
    problems: list                        # [(name, verdict, reason)] non-OK only
    schema_version: str = SCHEMA_VERSION


def validate_vector(canonical: dict, ages: dict | None = None) -> ValidationReport:
    """Validate a canonicalised vector feature-by-feature. `ages` optionally maps
    feature → age-in-days for staleness. `ok` is False if ANY feature is
    IMPOSSIBLE or STALE (hard data-quality failures); MISSING and OUTLIER are
    recorded but don't by themselves fail the vector (a partial observation is
    still a real observation — the consumer decides)."""
    ages = ages or {}
    verdicts: dict = {}
    problems: list = []
    hard_fail = False
    for name, feat in FEATURE_REGISTRY.items():
        verdict, reason = feat.validate(canonical.get(name), ages.get(name))
        verdicts[name] = (verdict, reason)
        if verdict != OK:
            problems.append((name, verdict, reason))
            if verdict in (IMPOSSIBLE, STALE):
                hard_fail = True
    return ValidationReport(ok=not hard_fail, verdicts=verdicts, problems=problems)
