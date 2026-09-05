from __future__ import annotations

from pathlib import Path
import re

from terminal_product_api_parallel import app


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "src"
API_LITERAL = re.compile(r"(?P<quote>['\"`])(?P<path>/api/.*?)(?P=quote)")
TEMPLATE_EXPR = re.compile(r"\$\{[^}]+\}")
TRAILING_QUERY_EXPR = re.compile(
    r"\$\{(?:query|params|searchParams)(?:\.toString\(\))?\}$"
)
FASTAPI_PARAM = re.compile(r"\{[^}]+\}")


def _normalise(path: str) -> str:
    path = path.split("?", 1)[0]
    # A template variable named query/params appended directly to a static path
    # represents an optional query string, not another backend path segment.
    # Strip it before normalising real dynamic path parameters.
    path = TRAILING_QUERY_EXPR.sub("", path)
    path = TEMPLATE_EXPR.sub("{}", path)
    path = FASTAPI_PARAM.sub("{}", path)
    return path.rstrip("/") or "/"


def _frontend_api_paths() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in FRONTEND.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx"} or ".test." in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        for match in API_LITERAL.finditer(text):
            raw = match.group("path")
            # Ignore malformed/incomplete fragments rather than pretending they are
            # routes. This matters for regex matches that begin inside a JavaScript
            # template expression containing nested quotes.
            if not raw.startswith("/api/") or any(ch in raw for ch in ("\n", "\r")):
                continue
            if raw.count("${") != raw.count("}"):
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


def test_every_frontend_api_literal_resolves_to_a_backend_route():
    """No visible frontend call may point at a route the canonical backend does not expose."""
    frontend = _frontend_api_paths()
    backend = _backend_api_paths()
    missing = {
        path: sorted(files)
        for path, files in frontend.items()
        if path not in backend
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
