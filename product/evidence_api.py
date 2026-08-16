"""Evidence intake routes — same handlers on the terminal API and report API.

System → Data must work from one origin (:8765). The report process on :8766
stays available for PDFs; it reuses these handlers so the two apps cannot drift.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Request, Response
from fastapi.responses import FileResponse


def runtime_as_of(symbol: str) -> dict[str, str]:
    dates = {
        "price_as_of": "",
        "scan_as_of": "",
        "long_term_as_of": "",
        "news_as_of": "",
        "fno_as_of": "",
    }
    try:
        from data.bhavcopy_runtime import get_ohlcv

        frame = get_ohlcv(symbol)
        if frame is not None and len(frame):
            value = frame.index[-1]
            dates["price_as_of"] = str(getattr(value, "date", lambda: value)())
    except Exception:
        pass
    try:
        from product.scan_store import load_scan

        dates["scan_as_of"] = str((load_scan() or {}).get("scanned_at", ""))
    except Exception:
        pass
    try:
        from product.long_term_store import load_long_term_scan

        dates["long_term_as_of"] = str((load_long_term_scan() or {}).get("scanned_at", ""))
    except Exception:
        pass
    try:
        from news.curator_store import NewsCuratorStore
        from reporting.evidence_intake import ROOT

        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            rows = store.recent(hours=24 * 30, limit=1, symbol=symbol.upper())
        finally:
            store.close()
        if rows:
            item = rows[0].as_dict()
            dates["news_as_of"] = str(item.get("published_at") or item.get("fetched_at") or "")
    except Exception:
        pass
    try:
        import json
        from reporting.evidence_intake import ROOT

        path = ROOT / "logs" / "product" / "fno_universe.json"
        if path.exists():
            dates["fno_as_of"] = str(json.loads(path.read_text(encoding="utf-8")).get("generated_at", ""))
    except Exception:
        pass
    return dates


def evidence_status(symbol: str) -> dict[str, Any]:
    try:
        from reporting.evidence_intake import clean_symbol, evidence_requirements

        clean = clean_symbol(symbol)
        return evidence_requirements(clean, **runtime_as_of(clean))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence status failed: {exc}") from exc


def refresh_fundamentals(symbol: str) -> dict[str, Any]:
    try:
        from reporting.evidence_intake import clean_symbol
        from fundamentals.resolver import next_actions, resolve

        clean = clean_symbol(symbol)
        data, steps = resolve(clean, force_refresh=True, write_cache=True)
        if data is None:
            return {
                "accepted": False,
                "symbol": clean,
                "outcome": "MISSING",
                "steps": steps,
                "next_actions": next_actions(clean),
                "message": steps[-1]["message"] if steps else "All fundamentals sources exhausted",
                "status": evidence_status(clean),
            }
        return {
            "accepted": True,
            "symbol": clean,
            "outcome": "READY",
            "source": str(data.get("_source") or ""),
            "steps": steps,
            "next_actions": next_actions(clean),
            "sections": {
                "about": bool(data.get("about")),
                "quarterly_results": len(data.get("quarterly_results", []) or []),
                "profit_loss": len(data.get("profit_loss", []) or []),
                "balance_sheet": len(data.get("balance_sheet", []) or []),
                "cash_flow": len(data.get("cash_flow", []) or []),
                "shareholding": len(data.get("shareholding", []) or []),
                "peer_comparison": len(data.get("peer_comparison", []) or []),
            },
            "status": evidence_status(clean),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Deep fundamentals refresh failed: {exc}") from exc


def evidence_template(kind: str) -> Response:
    try:
        from reporting.evidence_intake import template_csv

        content = template_csv(kind)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="quantterm_{kind}_template.csv"'},
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def evidence_worked_example(kind: str) -> Response:
    try:
        from reporting.evidence_intake import worked_example_csv

        content = worked_example_csv(kind)
        return Response(
            content=content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="quantterm_{kind}_worked_example.csv"',
                "X-QuantTerm-Example": "SAMPLE_NOT_LIVE_EXCHANGE_DATA",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def install_worked_example_action(symbol: str) -> dict[str, Any]:
    try:
        from reporting.evidence_intake import install_worked_example

        return install_worked_example(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Worked-example install failed: {exc}") from exc


async def upload_evidence(
    symbol: str,
    kind: str,
    request: Request,
    as_of: str = Query(..., description="Source data date, for example 2026-06-30"),
    source_url: str = Query(default=""),
) -> dict[str, Any]:
    try:
        from reporting.evidence_intake import save_upload

        content = await request.body()
        filename = request.headers.get("x-filename", "evidence.bin")
        item = save_upload(
            symbol,
            kind,
            content,
            filename=filename,
            as_of=as_of,
            source_url=source_url,
        )
        return {"accepted": True, "evidence": item, "status": evidence_status(symbol)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence upload failed: {exc}") from exc


def download_evidence(symbol: str, evidence_id: str) -> FileResponse:
    try:
        from reporting.evidence_intake import upload_path

        status = evidence_status(symbol)
        path = upload_path(symbol, evidence_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Evidence file not found")
        item = next(
            (entry for entry in status.get("uploads", []) if entry.get("evidence_id") == evidence_id),
            {},
        )
        return FileResponse(
            path=str(path),
            filename=str(item.get("filename") or path.name),
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence download failed: {exc}") from exc


def install_evidence_routes(app) -> None:
    """Mount the same /evidence tree the report API exposes."""
    app.add_api_route("/evidence/{symbol}", evidence_status, methods=["GET"], name="evidence_status")
    app.add_api_route(
        "/evidence/{symbol}/actions/refresh-fundamentals",
        refresh_fundamentals,
        methods=["POST"],
        name="evidence_refresh_fundamentals",
    )
    app.add_api_route(
        "/evidence/templates/{kind}.csv",
        evidence_template,
        methods=["GET"],
        name="evidence_template",
    )
    app.add_api_route(
        "/evidence/examples/{kind}.csv",
        evidence_worked_example,
        methods=["GET"],
        name="evidence_worked_example",
    )
    app.add_api_route(
        "/evidence/{symbol}/actions/install-worked-example",
        install_worked_example_action,
        methods=["POST"],
        name="evidence_install_worked_example",
    )
    app.add_api_route(
        "/evidence/{symbol}/{kind}",
        upload_evidence,
        methods=["POST"],
        name="evidence_upload",
    )
    app.add_api_route(
        "/evidence/{symbol}/files/{evidence_id}",
        download_evidence,
        methods=["GET"],
        name="evidence_download",
    )
