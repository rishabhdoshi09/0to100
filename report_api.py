"""Local download API for QuantTerm professional research reports."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="QuantTerm Research Report API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "quantterm-research-report-api", "version": app.version}


def _pdf_response(path: Path) -> FileResponse:
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=500, detail="Report generator did not produce a PDF")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=path.name,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/reports/equity/{symbol}")
def equity_report(symbol: str) -> FileResponse:
    try:
        from reporting.research_dossier import generate_equity_report

        return _pdf_response(generate_equity_report(symbol))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Equity report generation failed: {exc}") from exc


@app.get("/reports/basket/long-term")
def long_term_basket_report(limit: int = Query(default=3, ge=1, le=10)) -> FileResponse:
    try:
        from reporting.research_dossier import generate_basket_report

        return _pdf_response(generate_basket_report(limit=limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Basket report generation failed: {exc}") from exc
