"""Paper-learning façade with canonical live-execution safety truth.

The learning engine remains in ``_paper_learning_loop_core``. This module only
replaces the operator dashboard projection so learning evidence can never stamp
or preserve a positive broker-safety claim on its own.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_CORE_PATH = Path(__file__).with_name("_paper_learning_loop_core.py")
_SPEC = importlib.util.spec_from_file_location("_quantterm_paper_learning_loop_core", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load paper-learning core from {_CORE_PATH}")
_core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_core)

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def learning_dashboard(*, policy_path=None) -> dict:
    """Existing evidence dashboard with canonical, fail-closed live safety."""
    from product.live_safety import live_safety_projection

    payload = dict(_core.learning_dashboard(policy_path=policy_path) or {})
    safety = live_safety_projection()
    payload.update(safety)

    # The legacy dashboard included Forward Soak as a nested projection and its
    # exception fallback stamped live_locked=True. Current broker truth overrides
    # that field too; errors stay errors and never become positive safety proof.
    forward = payload.get("forward_soak")
    if isinstance(forward, dict):
        forward = dict(forward)
        forward.update(safety)
        payload["forward_soak"] = forward
    return payload
