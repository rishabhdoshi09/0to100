"""Local API for QuantTerm research reports and evidence intake."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from product.evidence_api import evidence_template, install_evidence_routes

app = FastAPI(title="QuantTerm Research Report API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


install_evidence_routes(app)


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
