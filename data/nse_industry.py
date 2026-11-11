"""Official NSE industry per symbol, captured from data QuantTerm already fetches.

The static sector map in ``scan/sector_heat.py`` is parsed from comments in
``data/nse_universe.py`` and covers ~337 symbols: the large caps in the core
list (TCS, HDFCBANK, PNB, BANKBARODA ...) and most PSU banks have no sector,
so sector packs and the portfolio sector cap could not see them.

NSE's ``equity-stockIndices`` rows (already pulled by ``data/nse_live.py``
for the intraday overlay) carry ``meta.industry`` for every constituent. This
module persists that official label (no extra request) and maps it onto the
static group names so one sector is never split across two spellings.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from logger import get_logger

log = get_logger(__name__)

_cache: dict[str, Any] = {"mtime": None, "industries": {}}

# Ordered: first match wins. Right side is the static group name used by
# scan.sector_heat so NSE-labelled and comment-labelled symbols share buckets.
_GROUP_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("edible oil", "vanaspati"), "FMCG / Consumer Staples"),
    (("housing finance",), "Housing Finance"),
    (("microfinance", "micro finance"), "Microfinance"),
    (("bank",), "Banking & Finance"),
    (("insurance",), "Insurance"),
    (("capital market", "stock broking", "broking", "exchange", "asset management", "depositor"),
     "Capital Markets"),
    (("finance", "financial", "nbfc", "investment compan"), "NBFC"),
    (("software", "computers", "information technology", "it enabled", "it - "), "IT / Software"),
    (("pharma", "healthcare", "hospital", "diagnostic", "medical", "drug"), "Pharma & Healthcare"),
    (("tyre",), "Tyres"),
    (("auto",), "Auto & Auto Ancillary"),
    (("cement",), "Cement"),
    (("paint",), "Paints"),
    (("steel", "metal", "mining", "aluminium", "copper", "zinc", "iron", "coal"), "Metals & Mining"),
    (("gas",), "Gas & Energy Distribution"),
    (("power", "electricity", "refiner", "petroleum", "oil", "energy"), "Energy & Power"),
    (("telecom", "media", "entertainment", "broadcast", "publishing"), "Telecom & Media"),
    (("realty", "real estate", "residential", "commercial projects"), "Real Estate"),
    (("construction", "infrastructure", "civil", "engineering - ind", "roads"),
     "Infrastructure & Construction"),
    (("chemical",), "Specialty Chemicals"),
    (("fertili", "agro", "agri", "pesticide", "sugar", "seed"), "Agri & Fertilizers"),
    (("textile", "apparel", "garment", "yarn"), "Textiles"),
    (("hotel", "hospitality", "tourism", "restaurant"), "Hospitality"),
    (("shipping", "logistic", "aviation", "airline", "transport", "ports", "courier"),
     "Logistics & Transport"),
    (("defence", "defense", "aerospace"), "Defence"),
    (("cable", "wire"), "Wires & Cables"),
    (("consumer durable", "household appliance", "electronic"), "White Goods / Consumer Durables"),
    (("fmcg", "consumer goods", "food", "beverage", "tobacco", "personal care", "dairy",
      "consumer staple"), "FMCG / Consumer Staples"),
    (("retail",), "Consumer & Retail / Apparel"),
    (("jewel", "gems"), "Gems & Jewellery"),
    (("paper", "packaging"), "Paper & Packaging"),
    (("footwear", "leather"), "Footwear"),
    (("railway",), "Railways"),
    (("ceramic", "building material", "glass", "sanitary", "plywood", "tiles"),
     "Ceramics & Building Materials"),
    (("electrical equipment", "capital goods", "industrial machinery", "engineering",
      "machinery", "industrial products"), "Manufacturing & Capital Goods"),
)


def map_path() -> Path:
    from core.runtime_paths import logs_path

    return Path(logs_path("nse_industry_map.json"))


def normalise_industry(industry: str) -> str:
    """NSE industry label -> static sector group (raw label when unknown)."""
    label = " ".join(str(industry or "").split())
    if not label:
        return ""
    low = label.lower()
    for needles, group in _GROUP_RULES:
        if any(n in low for n in needles):
            return group
    return label.title() if label.isupper() else label


def industries_from_index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
        symbol = str(row.get("symbol") or meta.get("symbol") or "").strip().upper()
        industry = " ".join(str(meta.get("industry") or "").split())
        if symbol and industry and industry.upper() not in {"-", "NA", "N/A", "NULL"}:
            out[symbol] = industry
    return out


def record_industries(found: Mapping[str, str], *, path: Path | None = None) -> int:
    """Merge newly observed official labels into the persisted map.

    Returns the number of symbols added or changed. Never raises.
    """
    if not found:
        return 0
    target = Path(path) if path else map_path()
    try:
        current: dict[str, Any] = {}
        if target.exists():
            current = json.loads(target.read_text(encoding="utf-8") or "{}")
        industries = dict(current.get("industries") or {})
        changed = {s: i for s, i in found.items() if industries.get(s) != i}
        if not changed:
            return 0
        industries.update(changed)
        from core.atomic_io import atomic_write_text

        atomic_write_text(target, json.dumps({
            "source": "NSE equity-stockIndices meta.industry",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "industries": dict(sorted(industries.items())),
        }, indent=1))
        return len(changed)
    except Exception as exc:
        log.debug("nse_industry_record_failed", error=str(exc)[:200])
        return 0


def industry_of(symbol: str, *, path: Path | None = None) -> str:
    """Official NSE industry label on file for ``symbol`` ('' when unknown)."""
    target = Path(path) if path else map_path()
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return ""
    key = (str(target), mtime)
    if _cache.get("mtime") != key:
        try:
            data = json.loads(target.read_text(encoding="utf-8") or "{}")
            _cache["industries"] = dict(data.get("industries") or {})
        except Exception:
            _cache["industries"] = {}
        _cache["mtime"] = key
    return str(_cache["industries"].get(str(symbol or "").strip().upper()) or "")
