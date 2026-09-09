"""Moneycontrol fallback for sector KPIs when official tables are thin.

Personal-research adapter. Does not log in, solve CAPTCHAs, or paywall-bypass.
A 401/403/captcha-like page is recorded as source_unavailable.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from product.due_diligence.extract import extract_rates_from_text, html_to_text
from product.due_diligence.providers.base import (
    FetchResult,
    ProviderPolicy,
    archive_bytes,
    content_hash,
    host_allowed,
    respect_rate_limit,
)

POLICY = ProviderPolicy(
    timeout_s=18,
    retries=1,
    rate_gap_s=1.2,
    allowed_hosts=("www.moneycontrol.com", "moneycontrol.com"),
)

_SUGGEST = "https://www.moneycontrol.com/mccode/common/autosuggion.php?query={q}&type=1&format=json"
_CAPTCHA = ("captcha", "access denied", "cf-challenge", "verify you are human")


def _blocked(status: int | None, text: str) -> bool:
    if status in {401, 403, 429}:
        return True
    blob = (text or "").lower()
    return any(tok in blob for tok in _CAPTCHA)


def fetch_moneycontrol_kpis(symbol: str, *, session=None) -> dict[str, Any]:
    """Return {ok, kpis, error, status} — never raises into the research page."""
    import requests

    retrieved = datetime.now(timezone.utc).isoformat()
    out: dict[str, Any] = {
        "ok": False,
        "provider": "moneycontrol.com",
        "source_type": "established_financial_data",
        "kpis": {},
        "url": "",
        "error": "",
        "status": "source_unavailable",
        "retrieved_at": retrieved,
    }
    own_session = session is None
    sess = session or requests.Session()
    response = None
    page = None
    try:
        sess.headers.setdefault("User-Agent", POLICY.user_agent)
        sess.headers.setdefault("Accept", "application/json, text/html;q=0.9")
        sess.headers.setdefault("Accept-Language", "en-IN,en;q=0.9")
        respect_rate_limit("moneycontrol", POLICY.rate_gap_s)
        suggest_url = _SUGGEST.format(q=quote(symbol))
        try:
            response = sess.get(suggest_url, timeout=POLICY.timeout_s)
        except Exception as exc:
            out["error"] = str(exc)[:240]
            return out
        if _blocked(response.status_code, response.text):
            out["error"] = f"HTTP {response.status_code} — source unavailable"
            out["status"] = "source_unavailable"
            return out
        if response.status_code != 200:
            out["error"] = f"HTTP {response.status_code}"
            out["status"] = "acquisition_failed"
            return out
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            out["error"] = "Moneycontrol suggest did not return JSON"
            out["status"] = "acquisition_failed"
            return out
        rows = payload if isinstance(payload, list) else payload.get("data") or []
        link = ""
        for row in rows:
            if not isinstance(row, dict):
                continue
            sc_id = str(row.get("sc_id") or row.get("id") or "")
            stock = str(row.get("stock_name") or row.get("pdt_dis_nm") or row.get("name") or "")
            href = str(row.get("link_src") or row.get("link") or row.get("url") or "")
            if href and (symbol.upper() in href.upper() or symbol.upper() in stock.upper() or sc_id):
                link = href
                break
        if not link:
            out["error"] = "No public Moneycontrol company page matched this ticker"
            out["status"] = "source_unavailable"
            return out
        if link.startswith("/"):
            link = "https://www.moneycontrol.com" + link
        if not host_allowed(link, POLICY.allowed_hosts):
            out["error"] = "host not on the Moneycontrol allow-list"
            return out
        respect_rate_limit("moneycontrol", POLICY.rate_gap_s)
        try:
            page = sess.get(link, timeout=POLICY.timeout_s)
        except Exception as exc:
            out["error"] = str(exc)[:240]
            return out
        out["url"] = link
        if _blocked(page.status_code, page.text):
            out["error"] = f"HTTP {page.status_code} — source unavailable"
            return out
        if page.status_code != 200:
            out["error"] = f"HTTP {page.status_code}"
            out["status"] = "acquisition_failed"
            return out
        body = page.content or b""
        archive_bytes(symbol, "moneycontrol", body, suffix=".html")
        text = html_to_text(page.text or "")
        kpis = extract_rates_from_text(text, source="Moneycontrol company page", source_url=link)
        if not kpis:
            out["error"] = "Metric not reported on the public Moneycontrol page"
            out["status"] = "metric_not_reported"
            out["ok"] = True
            return out
        for snap in kpis.values():
            snap["source"] = "Moneycontrol company page"
            snap["source_url"] = link
            snap["retrieved_at"] = retrieved
        out.update({
            "ok": True,
            "kpis": kpis,
            "status": "current",
            "content_hash": content_hash(body),
        })
        return out
    finally:
        for obj in (response, page):
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        if own_session:
            closer = getattr(sess, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
