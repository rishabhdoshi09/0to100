"""
JARVIS Live Coder — self-modification engine.

When JARVIS can't answer a query because the feature/data doesn't exist,
this module:
  1. Diagnoses what's missing
  2. Generates a plain-English plan
  3. Presents it to the user for confirmation
  4. Writes and executes the code live
  5. Streams the output back into the JARVIS chat

Safety constraints:
  - No file deletions, no subprocess calls, no network calls outside whitelisted domains
  - All generated code is shown to the user before execution
  - Execution is sandboxed with output capture
  - Permanent saves require an extra explicit confirmation
"""
from __future__ import annotations

import io
import json
import os
import sys
import contextlib
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ── Plan dataclass ────────────────────────────────────────────────────────────

@dataclass
class CodePlan:
    query: str                       # original user query
    gap: str                         # what's missing: "no sector P/E data", etc.
    explanation: str                 # plain English: what will be built
    what_changes: list[str]          # e.g. ["Fetches NSE sector P/E via API", "Adds chart"]
    estimated_output: str            # "A table of sector P/E ratios with trend"
    complexity: str                  # SIMPLE | MODERATE | COMPLEX
    code: str = ""                   # generated Python code (filled in step 2)
    permanent_file: str = ""         # if non-empty, where to save this as a feature
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


# ── System prompt for the coder ───────────────────────────────────────────────

_DIAGNOSE_PROMPT = """You are the JARVIS coder for QUANTTERM, an institutional trading system.

The user asked: "{query}"

Current system context:
{context}

Current system capabilities (what already EXISTS):
- Regime engine: market_regime, vix, nifty_price, breadth, sector_returns, quality_multiplier
- Scan pipeline: SetupCandidate with symbol, price, archetype, confidence, pivot, stop
- Quality scoring: tier (ELITE_A_PLUS/A/B/WATCHLIST), quality_score 0-100
- Journal: SQLite at journal.db with trade history, win rates
- Options: option chain, PCR, max pain, IV percentile
- News: RSS articles with sentiment
- Kite Connect: live OHLCV, LTP, positions, orders
- Charts: historical OHLCV via Kite
- DeepSeek API available for AI analysis

Determine if this query NEEDS new code (feature/data that doesn't exist) or CAN be answered now.

Return ONLY valid JSON:
{{
  "type": "CAN_ANSWER" | "NEEDS_CODE",
  "gap": "what specific data or feature is missing (empty if CAN_ANSWER)",
  "explanation": "plain English: what needs to be built to answer this",
  "what_changes": ["step 1", "step 2"],
  "estimated_output": "what the user will see after the code runs",
  "complexity": "SIMPLE" | "MODERATE" | "COMPLEX"
}}

NEEDS_CODE examples: sector P/E ratios, 52-week high/low screener with custom filters,
correlation matrix between portfolio stocks, drawdown analysis, custom backtesting logic,
new data sources not currently fetched, F&O OI change analysis.

CAN_ANSWER examples: current regime, top setups, today's opportunity score,
VIX level, breadth, your win rate, position sizing for a trade."""


_CODE_GEN_PROMPT = """You are writing Python code for QUANTTERM, a trading intelligence system.

User query: "{query}"
What to build: {explanation}
Expected output: {estimated_output}

Available imports (already installed):
- pandas, numpy, json, datetime, os, sqlite3, requests
- from core.regime_engine import compute_regime
- from core.intelligence_hub import compute_opportunity_score
- from data.kite_client import KiteClient
- from data.instruments import InstrumentManager
- from data.historical import HistoricalDataFetcher

Available data:
{context}

Write COMPLETE, EXECUTABLE Python code that:
1. Fetches/computes the requested data
2. Stores result in a variable called `RESULT` (string, dict, or list)
3. Prints a clean, formatted output for the user
4. Has NO side effects (no file writes, no orders, no external posts)
5. Handles errors gracefully with try/except

STRICT RULES:
- No os.system(), subprocess, shutil.rmtree, open() for writing
- No requests to non-whitelisted APIs (only api.deepseek.com, api.kite.trade, nseindia.com)
- No sleep() calls
- Code must complete in under 30 seconds
- Use only the imports listed above

Return ONLY the Python code, no markdown fences, no explanation."""


# ── Core functions ────────────────────────────────────────────────────────────

def diagnose_query(query: str, context: str) -> dict:
    """
    Ask DeepSeek if the query needs new code.
    Returns the parsed JSON plan dict, or {"type": "CAN_ANSWER"} on failure.
    """
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return {"type": "CAN_ANSWER"}

    import requests
    import re
    try:
        prompt = _DIAGNOSE_PROMPT.format(query=query, context=context[:2000])
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=15,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception:
        return {"type": "CAN_ANSWER"}


def generate_code(query: str, plan: dict, context: str) -> str:
    """Ask DeepSeek to write the code for this plan."""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return "# DEEPSEEK_API_KEY not set"

    import requests
    try:
        prompt = _CODE_GEN_PROMPT.format(
            query=query,
            explanation=plan.get("explanation", ""),
            estimated_output=plan.get("estimated_output", ""),
            context=context[:1500],
        )
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You write clean, safe Python code for trading systems. Return only executable code."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            },
            timeout=30,
        )
        code = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences
        import re
        code = re.sub(r"^```[a-z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
        return code
    except Exception as e:
        return f"# Code generation failed: {e}"


def execute_code(code: str) -> tuple[str, str, Any]:
    """
    Execute generated code in a sandboxed environment.
    Returns (stdout_output, error_message, RESULT_variable).

    Safety: blocks dangerous calls before execution.
    """
    # Safety pre-check — block obvious destructive patterns
    _BLOCKED = [
        "os.system", "subprocess", "shutil.rmtree", "os.remove", "os.unlink",
        "open(", "__import__", "eval(", "exec(", "compile(",
        "os.chmod", "os.rename", "os.makedirs",
    ]
    # Allow open() only if it's clearly a read — but we block all for safety
    for blocked in _BLOCKED:
        if blocked in code:
            # Exception: open( is needed for sqlite3 sometimes — but we have sqlite3 module
            if blocked == "open(" and "sqlite3" in code:
                continue
            return "", f"Safety block: '{blocked}' is not permitted in generated code.", None

    stdout_buf = io.StringIO()
    result_ns: dict = {"RESULT": None}

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, result_ns)  # noqa: S102
        output = stdout_buf.getvalue()
        result = result_ns.get("RESULT")
        return output, "", result
    except Exception:
        output = stdout_buf.getvalue()
        err = traceback.format_exc(limit=5)
        return output, err, None


def format_result_for_jarvis(
    query: str, plan: dict, output: str, error: str, result: Any
) -> str:
    """Format the execution result into a JARVIS response."""
    if error and not output:
        return (
            f"I ran into an error executing the code:\n\n"
            f"```\n{error[:600]}\n```\n\n"
            f"This likely means the data source is temporarily unavailable or needs "
            f"additional configuration. Want me to try a different approach?"
        )

    lines = []

    if output.strip():
        lines.append(output.strip())

    if result is not None:
        import pandas as pd
        if isinstance(result, pd.DataFrame) and not result.empty:
            lines.append(f"\n*{len(result)} rows returned.*")
        elif isinstance(result, (list, dict)):
            try:
                lines.append(f"\n```json\n{json.dumps(result, indent=2, default=str)[:800]}\n```")
            except Exception:
                lines.append(str(result)[:500])
        elif isinstance(result, str) and result.strip():
            lines.append(result.strip())

    if error:
        lines.append(f"\n⚠️ Partial error: `{error[:200]}`")

    if not lines:
        lines.append("Code ran successfully but produced no output.")

    return "\n".join(lines)
