"""Document-type-aware parse + sample validation.

Parsing success is not evidence quality. Confidence and errors are stored.
Publication dates are never guessed from filenames or mtime.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from product.pit_warehouse import classify_document, persist

ROOT = Path(__file__).resolve().parents[1]
PARSER_VERSION = "pit_parse.v1"


def _sha_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def parse_local_document(path: Path, *, symbol: str = "") -> dict[str, Any]:
    from product.due_diligence.acquire import bytes_to_text

    target = Path(path)
    report = {
        "path": str(target),
        "symbol": str(symbol or "").upper(),
        "document_type": classify_document(target.name),
        "parser_version": PARSER_VERSION,
        "ok": False,
        "confidence": 0.0,
        "errors": [],
        "numbers_parsed": False,
        "text_chars": 0,
        "content_sha": "",
    }
    try:
        blob = target.read_bytes()
    except OSError as exc:
        report["errors"].append(f"CORRUPT_DOCUMENT:{exc}")
        report["reason_code"] = "CORRUPT_DOCUMENT"
        return report
    report["content_sha"] = _sha_bytes(blob)
    report["bytes"] = len(blob)
    text = bytes_to_text(blob, target.suffix, max_pages=12)
    report["text_chars"] = len(text)
    if not text.strip():
        report["errors"].append("PARSER_FAILED:empty_text")
        report["reason_code"] = "PARSER_FAILED"
        return report
    report["ok"] = True
    report["document_type"] = classify_document(f"{target.name} {text[:400]}")
    needles = ("revenue", "profit", "pat", "ebitda", "shareholding", "promoter")
    hits = sum(1 for n in needles if n in text.lower())
    report["confidence"] = min(0.85, 0.25 + 0.1 * hits)
    report["keyword_hits"] = hits
    # Do not invent financial facts from keyword presence.
    report["numbers_parsed"] = False
    report["note"] = "Text extracted. Numeric facts stay unknown until a typed parser confirms them."
    return report


def validate_sample(paths: list[Path], *, symbol: str = "") -> dict[str, Any]:
    rows = [parse_local_document(path, symbol=symbol) for path in paths]
    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r.get("ok")),
        "failed": sum(1 for r in rows if not r.get("ok")),
        "numbers_parsed": sum(1 for r in rows if r.get("numbers_parsed")),
        "details": rows,
        "note": "A successful text extract is not a verified financial snapshot.",
    }


def index_unverified_pdf(path: Path, *, symbol: str, warehouse_path=None) -> dict[str, Any]:
    """Store the artifact. Do not guess a publication date from the filename."""
    parsed = parse_local_document(path, symbol=symbol)
    persist({
        "symbol": symbol,
        "evidence_type": parsed.get("document_type"),
        "document_type": parsed.get("document_type"),
        "publication_date": "",
        "available_from": "",
        "source": "local filing artifact",
        "source_url": str(path),
        "source_identity": f"local:{parsed.get('content_sha')}",
        "raw_artifact_id": parsed.get("content_sha"),
        "parser_version": PARSER_VERSION,
        "extracted": {
            "text_chars": parsed.get("text_chars"),
            "confidence": parsed.get("confidence"),
            "numbers_parsed": False,
        },
        "pit_status": "PIT_UNVERIFIED",
        "reason_code": "PUBLICATION_DATE_UNKNOWN",
    }, path=warehouse_path)
    return parsed
