from __future__ import annotations

from pathlib import Path
import re

from terminal_product_api_parallel import app


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "src"
TEMPLATE_EXPR = re.compile(r"\$\{[^}]+\}")
FASTAPI_PARAM = re.compile(r"\{[^}]+\}")


def _quoted_literals(text: str):
    """Yield JS/TS quoted literals without being confused by quotes inside backticks.

    The old one-regex scanner could terminate a template literal on a quote that lived
    inside ``${...}``, producing fake API paths such as
    ``/api/controls/${action === 'history' ``. This tiny lexer only needs to find the
    matching outer quote; it is intentionally not a JavaScript parser.
    """
    i = 0
    size = len(text)
    while i < size:
        quote = text[i]
        if quote not in {"'", '"', '`'}:
            i += 1
            continue
        start = i + 1
        i = start
        escaped = False
        while i < size:
            ch = text[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if ch == quote:
                yield text[start:i]
                i += 1
                break
            i += 1
        else:
            break


def _normalise(path: str) -> str:
    # Replace interpolation before stripping a literal query string. Dynamic query
    # suffixes (``/path${query}``) become a trailing {}, which the cross-wire check
    # may safely collapse only when the static backend path actually exists.
    path = TEMPLATE_EXPR.sub("{}", path)
    path = path.split("?", 1)[0]
    path = FASTAPI_PARAM.sub("{}", path)
    return path.rstrip("/") or "/"


def _frontend_api_paths() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in FRONTEND.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx"} or ".test." in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        for raw in _quoted_literals(text):
            if not raw.startswith("/api/") or any(ch in raw for ch in ("\n", "\r")):
                continue
            normal = _normalise(raw)
            found.setdefault(normal, set()).add(str(path.relative_to(ROOT)))
    return found


def _backend_api_paths() -> set[str]:
    return {
        _normalise(str(getattr(route, "path", "")))
        for route in app.routes
        if str(getattr(route, "path", "")).startswith("/api/")
    }


def _resolves(frontend_path: str, backend: set[str]) -> bool:
    if frontend_path in backend:
        return True
    # A template variable can be a query suffix, e.g. ``/framework-audit${query}``.
    # Only treat it as such when removing the trailing placeholder lands on an exact
    # static backend route. Dynamic path parameters still require the {} route.
    if frontend_path.endswith("{}") and frontend_path[:-2] in backend:
        return True
    return False


def test_every_frontend_api_literal_resolves_to_a_backend_route():
    """No visible frontend call may point at a route the canonical backend does not expose."""
    frontend = _frontend_api_paths()
    backend = _backend_api_paths()
    missing = {
        path: sorted(files)
        for path, files in frontend.items()
        if not _resolves(path, backend)
    }
    assert not missing, f"Frontend API path(s) are not cross-wired to FastAPI: {missing}"


def test_crosswire_audit_covers_the_primary_product_clients():
    frontend = _frontend_api_paths()
    expected = {
        "/api/dashboard",
        "/api/health",
        "/api/radar-home",
        "/api/recommendations-workspace",
        "/api/market-reports-workspace",
        "/api/stock-intelligence/{}",
        "/api/due-diligence/{}",
        "/api/decision-simulator",
    }
    missing = sorted(expected - set(frontend))
    assert not missing, f"Cross-wire scanner stopped seeing primary frontend clients: {missing}"


def test_template_literal_scanner_keeps_inner_quotes_inside_expression():
    text = "const x = `/api/controls/${action === 'history' ? 'A' : 'B'}`"
    literals = list(_quoted_literals(text))
    assert "/api/controls/${action === 'history' ? 'A' : 'B'}" in literals
    assert _normalise(next(item for item in literals if item.startswith('/api/'))) == "/api/controls/{}"
