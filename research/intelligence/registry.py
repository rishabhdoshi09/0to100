"""
📇 Production strategy registry (Phase 11).

Builds the set of frozen strategies the loop may run, and validates each at startup. An invalid
strategy is DISABLED with an explicit reason — it never crashes the system and never silently
disappears. The registry is the single source of truth for "what is registered, what the runtime
supports, and why something is unsupported" (surfaced by the Strategy Coverage UI).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.intelligence import strategy_runtime as RT
from research.strategy_studio import grammar as G


@dataclass
class RegisteredStrategy:
    spec: object
    family: str
    version: int
    rules_hash: str
    runtime_supported: bool
    enabled: bool
    disabled_reasons: tuple = ()
    owner_enabled: bool = True

    @property
    def strategy_id(self):
        return self.spec.strategy_id

    def as_dict(self):
        return {"strategy_id": self.strategy_id, "family": self.family,
                "version": self.version, "rules_hash": self.rules_hash,
                "runtime_supported": self.runtime_supported, "enabled": self.enabled,
                "disabled_reasons": list(self.disabled_reasons),
                "owner_enabled": self.owner_enabled,
                "cross_sectional": RT.is_cross_sectional(self.family)}


class StrategyRegistry:
    def __init__(self):
        self.by_id: dict[str, RegisteredStrategy] = {}
        self.duplicates: list[RegisteredStrategy] = []   # rejected dup ids/hashes (kept for audit)

    def build(self, specs, *, owner_enabled: dict | None = None) -> "StrategyRegistry":
        """Register frozen specs, validating each. Duplicate ids / duplicate result-hashes /
        unknown families / missing runtime adapters are disabled with reasons — never fatal."""
        owner_enabled = owner_enabled or {}
        seen_hash: dict[str, str] = {}
        for spec in specs:
            reasons = []
            fam = getattr(spec, "family", "")
            if fam not in G.FAMILY_BLOCKS:
                reasons.append(f"unknown family {fam!r}")
            supported = RT.is_supported(fam)
            if not supported:
                reasons.append("runtime adapter missing")
            if spec.strategy_id in self.by_id:
                reasons.append("duplicate strategy id")
            try:
                h = spec.config_hash()
            except Exception:
                h = ""; reasons.append("invalid spec (no config hash)")
            if h and h in seen_hash and seen_hash[h] != spec.strategy_id:
                reasons.append(f"duplicate rules hash of {seen_hash[h]}")
            if getattr(spec, "max_holding_days", 0) <= 0:
                reasons.append("invalid max_holding_days")
            if not getattr(spec, "entry_rules", ()):
                reasons.append("no entry rules")
            oe = bool(owner_enabled.get(spec.strategy_id, True))
            is_dup = spec.strategy_id in self.by_id
            enabled = (not reasons) and oe and not is_dup
            if h and h not in seen_hash:
                seen_hash[h] = spec.strategy_id
            rec = RegisteredStrategy(
                spec=spec, family=fam, version=getattr(spec, "version", 0), rules_hash=h,
                runtime_supported=supported, enabled=enabled,
                disabled_reasons=tuple(reasons), owner_enabled=oe)
            if is_dup:
                self.duplicates.append(rec)             # keep the ORIGINAL; audit the duplicate
            else:
                self.by_id[spec.strategy_id] = rec
        return self


    def replace_version(self, spec) -> RegisteredStrategy:
        """Promote a validated successor as the current version for one strategy id.

        The previous frozen version is retained in ``duplicates`` as an audit archive; callers must
        have completed evidence/lifecycle checks before invoking this operation.
        """
        candidate = StrategyRegistry().build([spec])
        rec = candidate.by_id.get(spec.strategy_id)
        if rec is None or not rec.enabled:
            reasons = rec.disabled_reasons if rec is not None else ("registration failed",)
            raise ValueError("successor is not deployable: " + "; ".join(reasons))
        previous = self.by_id.get(spec.strategy_id)
        if previous is not None:
            self.duplicates.append(previous)
        self.by_id[spec.strategy_id] = rec
        return rec

    # ── views ─────────────────────────────────────────────────────────────────────
    def all(self):
        return list(self.by_id.values()) + list(self.duplicates)

    def enabled(self):
        return [r for r in self.by_id.values() if r.enabled]

    def supported(self):
        return [r for r in self.by_id.values() if r.runtime_supported]

    def unsupported(self):
        return [r for r in self.by_id.values() if not r.runtime_supported]

    def deployable_specs(self):
        """Frozen specs the loop may evaluate: enabled + runtime-supported + owner-enabled."""
        return [r.spec for r in self.by_id.values()
                if r.enabled and r.runtime_supported and r.owner_enabled]

    def coverage(self):
        return [r.as_dict() for r in sorted(self.by_id.values(),
                                            key=lambda r: (not r.runtime_supported, r.family))]
