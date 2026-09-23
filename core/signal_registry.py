"""Canonical, versioned identities for production trading signals.

The registry is deliberately pure and immutable: research/replay code may refer to
signals only through a versioned specification whose identity is content-derived.
Changing production-thesis semantics therefore creates a new identity instead of
silently contaminating historical evidence collected under an older thesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping


REGISTRY_SCHEMA_VERSION = 1


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"non-canonical signal value: {type(value).__name__}")


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SignalSpec:
    """Immutable production-thesis signal definition.

    ``implementation_version`` must be bumped when signal semantics change. The
    fingerprint additionally covers parameters and required feature identities,
    making accidental version reuse observable rather than silently accepted.
    """

    name: str
    implementation_version: str
    thesis_version: str
    parameters: Mapping[str, Any]
    required_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.implementation_version.strip() or not self.thesis_version.strip():
            raise ValueError("signal name, implementation_version and thesis_version are required")
        canonical_parameters = _canonical(dict(self.parameters))
        object.__setattr__(self, "parameters", MappingProxyType(canonical_parameters))
        object.__setattr__(self, "required_features", tuple(sorted(set(self.required_features))))

    @property
    def fingerprint(self) -> str:
        return _digest(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "name": self.name,
            "implementation_version": self.implementation_version,
            "thesis_version": self.thesis_version,
            "parameters": _canonical(self.parameters),
            "required_features": list(self.required_features),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result


class SignalRegistry:
    """Fail-closed registry for exact signal identities used by replay/research."""

    def __init__(self, specs: Iterable[SignalSpec] = ()) -> None:
        by_name: dict[str, SignalSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate signal name: {spec.name}")
            by_name[spec.name] = spec
        self._by_name = MappingProxyType(by_name)

    def require(self, name: str, *, fingerprint: str | None = None) -> SignalSpec:
        try:
            spec = self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unregistered signal: {name}") from exc
        if fingerprint is not None and spec.fingerprint != fingerprint:
            raise ValueError(f"signal identity mismatch for {name}")
        return spec

    @property
    def fingerprint(self) -> str:
        return _digest({"schema_version": REGISTRY_SCHEMA_VERSION, "signals": self.snapshot()["signals"]})

    def snapshot(self) -> dict[str, Any]:
        signals = [self._by_name[name].as_dict() for name in sorted(self._by_name)]
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "signals": signals,
            "registry_fingerprint": _digest({"schema_version": REGISTRY_SCHEMA_VERSION, "signals": signals}),
        }
