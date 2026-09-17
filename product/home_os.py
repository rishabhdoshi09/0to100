"""Public Home OS facade enforcing authoritative completed-session readiness.

The core projection remains unchanged in ``product.home_os_core``. This facade
normalizes only one contradictory state: when official NSE history proves that
its available completed session exactly matches the expected completed session,
has zero stale sessions, and is explicitly current, legacy readiness booleans
cannot keep Home stuck in DATA Waiting.
"""
from __future__ import annotations

from typing import Any, Mapping

from product import home_os_core as _core

# Preserve the complete public/private module surface for existing callers and
# tests while keeping the original implementation byte-for-byte in one module.
_CORE_EXPORTS: dict[str, Any] = {}
for _name in dir(_core):
    if _name.startswith("__"):
        continue
    _value = getattr(_core, _name)
    globals()[_name] = _value
    _CORE_EXPORTS[_name] = _value

_CORE_BUILD_HOME_OS = _core.build_home_os


def _verified_current_history(freshness: Mapping[str, Any] | None) -> bool:
    """True only when completed-session freshness is explicit and self-consistent."""
    row = dict(freshness or {})
    available = str(row.get("available_session") or "").strip()
    expected = str(row.get("expected_latest_completed_session") or "").strip()
    try:
        stale_sessions = int(row.get("stale_sessions"))
    except (TypeError, ValueError):
        return False
    return bool(
        row.get("current") is True
        and available
        and expected
        and available == expected
        and stale_sessions == 0
    )


def _official_freshness() -> dict[str, Any]:
    try:
        from data.bhavcopy_runtime import official_history_freshness

        return dict(official_history_freshness() or {})
    except Exception:
        return {}


def _ready_data_projection(
    data: Mapping[str, Any] | None,
    *,
    now: Any = None,
    fallback_to_official: bool = False,
) -> dict[str, Any] | None:
    """Return a copied data projection only when current history is proven."""
    data_d = dict(data or {})
    freshness: dict[str, Any]
    if data_d:
        freshness = dict(_core._history_freshness(data_d, None, now) or {})
    elif fallback_to_official:
        freshness = _official_freshness()
    else:
        return None

    if not _verified_current_history(freshness):
        return None

    bhav = dict(data_d.get("bhavcopy") or {})
    available = str(freshness.get("available_session") or "")
    bhav.update({
        "ready": True,
        "current": True,
        "latest_date": bhav.get("latest_date") or available,
        "available_session": bhav.get("available_session") or available,
        "expected_latest_completed_session": (
            bhav.get("expected_latest_completed_session")
            or freshness.get("expected_latest_completed_session")
        ),
        "stale_sessions": 0,
        "reason_code": bhav.get("reason_code") or freshness.get("reason_code") or "HISTORY_CURRENT",
        "source": bhav.get("source") or "official_bhavcopy",
    })
    data_d.update({
        "ready": True,
        "bhavcopy": bhav,
        "history_freshness": freshness,
    })
    return data_d


def _sync_core_test_overrides() -> None:
    """Keep monkeypatches against product.home_os effective after the facade split."""
    for name in _CORE_EXPORTS:
        if name == "build_home_os" or name.startswith("__"):
            continue
        if name in globals():
            setattr(_core, name, globals()[name])


def build_home_os(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build Home from core logic after reconciling proven official readiness."""
    _sync_core_test_overrides()

    explicit_data = kwargs.get("data")
    dashboard = kwargs.get("dashboard")
    now = kwargs.get("now")

    if explicit_data is not None:
        normalized = _ready_data_projection(explicit_data, now=now)
        if normalized is not None:
            kwargs["data"] = normalized
    elif isinstance(dashboard, Mapping):
        dash = dict(dashboard)
        dashboard_data = dash.get("data")
        normalized = _ready_data_projection(
            dashboard_data if isinstance(dashboard_data, Mapping) else None,
            now=now,
            fallback_to_official=not bool(dashboard_data),
        )
        if normalized is not None:
            dash["data"] = normalized
            kwargs["dashboard"] = dash
    else:
        # Radar/Home callers commonly pass scan + autonomy but no data object.
        # Use the same official completed-session authority that core already
        # loads for freshness, and inject readiness only when it proves itself.
        normalized = _ready_data_projection(None, now=now, fallback_to_official=True)
        if normalized is not None:
            kwargs["data"] = normalized

    return _CORE_BUILD_HOME_OS(*args, **kwargs)
