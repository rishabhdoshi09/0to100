"""Earnings-call analyser.

* Receives a transcript PDF URL or local path, extracts text via
  ``pdfplumber`` (or naive PDF reader fallback), then asks Claude to
  pull structured highlights + forward guidance JSON.
* Persists results into ``earnings_calls`` (permanent — TTL = 0 in
  ``cache_set``).
* When no transcript or LLM is available, returns a deterministic
  synthesised summary so the UI is never empty.
"""
from __future__ import annotations

import json
import re
from typing import Any

from sq_ai.backend.cache import cached
from sq_ai.backend.llm_clients import ClaudeClient
from sq_ai.portfolio.tracker import PortfolioTracker


HIGHLIGHTS_SYSTEM = (
    "You are an Indian-equities equity research analyst. From the earnings-call "
    "transcript snippet, extract a strict JSON of:\n"
    '{"highlights": ["bullet 1", ...max 6],'
    ' "guidance": {"revenue":"...","margin":"...","capex":"...","other":"..."},'
    ' "tone": "positive|neutral|cautious"}\n'
    "No prose, JSON only."
)


def _extract_text(url_or_path: str, max_chars: int = 18000) -> str:
    """Download (if URL) and extract text. Returns empty string on failure."""
    text = ""
    try:
        if url_or_path.startswith(("http://", "https://")):
            import requests  # noqa: WPS433
            r = requests.get(url_or_path, timeout=15)
            r.raise_for_status()
            data = r.content
            try:
                import pdfplumber  # noqa: WPS433
                from io import BytesIO
                with pdfplumber.open(BytesIO(data)) as pdf:
                    for page in pdf.pages[:25]:
                        text += page.extract_text() or ""
            except Exception:
                text = data.decode("latin-1", errors="ignore")
        else:
            import pdfplumber  # noqa: WPS433
            with pdfplumber.open(url_or_path) as pdf:
                for page in pdf.pages[:25]:
                    text += page.extract_text() or ""
    except Exception:
        return ""
    return text[:max_chars]


def _fallback_highlights(symbol: str, quarter: str) -> dict[str, Any]:
    return {
        "highlights": [
            f"{symbol} {quarter}: management commentary not available in cache.",
            "Use the transcript URL to populate this section.",
        ],
        "guidance": {"revenue": "", "margin": "", "capex": "", "other": ""},
        "tone": "neutral",
        "source": "fallback",
    }


@cached("earnings_call", ttl_seconds=0)        # transcripts are permanent
def analyse_call(symbol: str, quarter: str,
                 transcript_url: str | None = None,
                 call_date: str | None = None,
                 tracker: PortfolioTracker | None = None,
                 claude: ClaudeClient | None = None) -> dict[str, Any]:
    tracker = tracker or PortfolioTracker()
    claude = claude or ClaudeClient()
    text = _extract_text(transcript_url) if transcript_url else ""
    parsed: dict[str, Any] | None = None

    if text and claude.available:
        prompt = (
            f"Transcript snippet for {symbol} {quarter}:\n\n"
            f"{text[:14000]}\n\n"
            "Return JSON only."
        )
        raw = claude.generate(prompt, max_tokens=600, temperature=0.1,
                              system=HIGHLIGHTS_SYSTEM)
        if raw:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    parsed["source"] = "claude"
                except json.JSONDecodeError:
                    parsed = None

    if parsed is None:
        parsed = _fallback_highlights(symbol, quarter)

    tracker.earnings_save(
        symbol, quarter,
        call_date=call_date, transcript_url=transcript_url,
        highlights=parsed, guidance=parsed.get("guidance", {}),
    )
    return {"symbol": symbol, "quarter": quarter,
            "transcript_url": transcript_url, "call_date": call_date,
            **parsed}


def list_calls(symbol: str,
               tracker: PortfolioTracker | None = None) -> list[dict]:
    """Return cached earnings calls + a couple of synthesised placeholder
    rows so the UI tab is never empty for a fresh symbol."""
    tracker = tracker or PortfolioTracker()
    rows = tracker.earnings_list(symbol)
    if rows:
        return rows
    # placeholders for the 4 most recent quarters
    out = []
    for i, q in enumerate(["Q3-2024", "Q2-2024", "Q1-2024", "Q4-2023"]):
        out.append({
            "symbol": symbol, "quarter": q, "call_date": None,
            "transcript_url": None,
            "highlights": _fallback_highlights(symbol, q),
            "guidance": {},
        })
    return out


__all__ = ["analyse_call", "list_calls"]
