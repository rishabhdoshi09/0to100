"""Local API for QuantTerm research reports and evidence intake."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="QuantTerm Research Report API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class _QuietHealthAccess(logging.Filter):
    """Stack watch loops hit /health often — keep the console readable."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietHealthAccess())


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "quantterm-research-report-api", "version": app.version}


def _pdf_response(path: Path, *, download: bool = False) -> FileResponse:
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=500, detail="Report generator did not produce a PDF")
    safe_name = path.name.replace('"', "")
    # Prefer Starlette's disposition helper so browsers render inline by default
    # (Safari often downloads when Content-Disposition is missing or attachment).
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=safe_name,
        content_disposition_type="attachment" if download else "inline",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _runtime_as_of(symbol: str) -> dict[str, str]:
    dates = {"price_as_of": "", "scan_as_of": "", "long_term_as_of": "", "news_as_of": "", "fno_as_of": ""}
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


@app.get("/reports/equity/{symbol}")
def equity_report(symbol: str, download: bool = Query(False)) -> FileResponse:
    try:
        from reporting.research_dossier import generate_equity_report

        return _pdf_response(generate_equity_report(symbol), download=download)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Equity report generation failed: {exc}") from exc


@app.get("/reports/basket/long-term")
def long_term_basket_report(
    limit: int = Query(default=3, ge=1, le=10),
    download: bool = Query(False),
) -> FileResponse:
    try:
        from reporting.research_dossier import generate_basket_report

        return _pdf_response(generate_basket_report(limit=limit), download=download)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Basket report generation failed: {exc}") from exc


@app.get("/reports/market/institutional")
def institutional_market_report(
    days: int = Query(default=30, ge=5, le=365),
    symbol_limit: int = Query(default=4, ge=1, le=8),
    download: bool = Query(False),
) -> FileResponse:
    try:
        from reporting.market_brief import generate_institutional_market_report

        return _pdf_response(
            generate_institutional_market_report(days=days, symbol_limit=symbol_limit),
            download=download,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Institutional market report failed: {exc}") from exc


@app.get("/reports/market/street-pulse")
def street_pulse_report(
    force: bool = Query(default=True, description="Rebuild pulse before rendering PDF"),
    download: bool = Query(False),
) -> FileResponse:
    """Daily Street Pulse PDF — research digest from stores, not a buy desk."""
    try:
        from reporting.street_pulse_report import generate_street_pulse_report

        return _pdf_response(generate_street_pulse_report(force=force), download=download)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Street pulse report failed: {exc}") from exc


@app.get("/evidence/{symbol}")
def evidence_status(symbol: str) -> dict:
    try:
        from reporting.evidence_intake import clean_symbol, evidence_requirements

        clean = clean_symbol(symbol)
        return evidence_requirements(clean, **_runtime_as_of(clean))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence status failed: {exc}") from exc


@app.post("/evidence/{symbol}/actions/refresh-fundamentals")
def refresh_fundamentals(symbol: str) -> dict:
    try:
        from fundamentals.fetcher import get_deep_fundamentals
        from reporting.evidence_intake import clean_symbol

        clean = clean_symbol(symbol)
        payload = get_deep_fundamentals(clean, force_refresh=True)
        return {
            "accepted": True,
            "symbol": clean,
            "sections": {
                "about": bool(payload.get("about")),
                "quarterly_results": len(payload.get("quarterly_results", []) or []),
                "profit_loss": len(payload.get("profit_loss", []) or []),
                "balance_sheet": len(payload.get("balance_sheet", []) or []),
                "cash_flow": len(payload.get("cash_flow", []) or []),
                "shareholding": len(payload.get("shareholding", []) or []),
                "peer_comparison": len(payload.get("peer_comparison", []) or []),
            },
            "status": evidence_status(clean),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Deep fundamentals refresh failed: {exc}") from exc


@app.post("/evidence/{symbol}/actions/autofetch")
def evidence_autofetch(symbol: str, body: dict = Body(default_factory=dict)) -> dict:
    """Download/export evidence from known source links and attach automatically."""
    try:
        from reporting.evidence_autofetch import autofetch_evidence
        from reporting.evidence_intake import clean_symbol

        payload = body or {}
        kinds = payload.get("kinds")
        refresh = payload.get("refresh_screener", True)
        return autofetch_evidence(
            clean_symbol(symbol),
            kinds=kinds,
            refresh_screener=bool(refresh),
            only_missing=not bool(payload.get("force", False)),
            max_link_downloads=int(payload.get("max_link_downloads") or 3),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Evidence autofetch failed: {exc}") from exc


@app.get("/evidence/templates/{kind}.csv")
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


@app.post("/evidence/{symbol}/{kind}")
async def upload_evidence(
    symbol: str,
    kind: str,
    request: Request,
    as_of: str = Query(..., description="Source data date, for example 2026-06-30"),
    source_url: str = Query(default=""),
) -> dict:
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


@app.get("/evidence/{symbol}/files/{evidence_id}")
def download_evidence(symbol: str, evidence_id: str) -> FileResponse:
    try:
        from reporting.evidence_intake import upload_path

        status = evidence_status(symbol)
        path = upload_path(symbol, evidence_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Evidence file not found")
        item = next((entry for entry in status.get("uploads", []) if entry.get("evidence_id") == evidence_id), {})
        return FileResponse(
            path=str(path),
            filename=str(item.get("filename") or path.name),
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence download failed: {exc}") from exc
