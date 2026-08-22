"""Historical research must not silently fetch the live web."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


class NetworkForbidden(RuntimeError):
    """Raised when a historical experiment attempts a live HTTP call."""


def _block(*a, **k):
    raise NetworkForbidden("historical research cannot perform live network I/O")


@contextmanager
def forbid_network() -> Iterator[None]:
    """Patch common HTTP clients for the duration of a research read."""
    patched: list[tuple[object, str, object]] = []

    def _patch(mod_name: str, attr: str) -> None:
        try:
            mod = __import__(mod_name, fromlist=[attr])
        except Exception:
            return
        if not hasattr(mod, attr):
            return
        patched.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, _block)

    _patch("requests", "get")
    _patch("requests", "post")
    _patch("requests", "request")
    try:
        import requests
        if hasattr(requests, "Session"):
            orig = requests.Session.request
            patched.append((requests.Session, "request", orig))
            requests.Session.request = _block  # type: ignore[method-assign]
    except Exception:
        pass
    try:
        yield
    finally:
        for obj, attr, val in patched:
            try:
                setattr(obj, attr, val)
            except Exception:
                pass
