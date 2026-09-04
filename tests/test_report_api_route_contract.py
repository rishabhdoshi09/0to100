from __future__ import annotations

import re

import report_api


def _routes() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for route in report_api.app.routes:
        path = re.sub(r"\{[^}]+\}", "{}", str(getattr(route, "path", "") or ""))
        for method in set(getattr(route, "methods", set()) or set()):
            out.add((method.upper(), path))
    return out


def test_report_service_exposes_every_desk_report_and_evidence_surface():
    routes = _routes()
    expected = {
        ("GET", "/health"),
        ("GET", "/reports/equity/{}"),
        ("GET", "/reports/basket/long-term"),
        ("GET", "/evidence/{}"),
        ("POST", "/evidence/{}/actions/auto-acquire"),
        ("POST", "/evidence/{}/actions/refresh-fundamentals"),
        ("GET", "/evidence/templates/{}.csv"),
        ("POST", "/evidence/{}/{}"),
        ("GET", "/evidence/{}/files/{}"),
    }
    missing = sorted(expected - routes)
    assert not missing, f"Research report/evidence route(s) are not wired: {missing}"


def test_report_service_does_not_expose_broker_order_mutations():
    paths = {path.lower() for _method, path in _routes()}
    assert not any("order" in path or "gtt" in path or "broker" in path for path in paths)
