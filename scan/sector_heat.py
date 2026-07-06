"""
Sector Heat — breakouts hunt in packs.

When 3+ stocks from the same sector fire signals in the same scan,
something sector-wide is happening (rotation, policy, results wave).
Each of those stocks gets a conviction boost and a plain-English
reason naming its fellow movers.

Sector map is parsed once from the commented groups in
data/nse_universe.py's NIFTY500 list — no external data needed.
"""
from __future__ import annotations

import re
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_SECTOR_MAP: dict[str, str] = {}     # symbol -> sector
_loaded = False

# Comment-group header → clean sector name
_SECTION_RE = re.compile(r"^\s*#\s*([A-Za-z][A-Za-z &/'-]+)$")
_SYM_RE = re.compile(r'"([A-Z0-9&\-]+)"')


def _load_map() -> dict[str, str]:
    global _SECTOR_MAP, _loaded
    if _loaded:
        return _SECTOR_MAP
    try:
        src = (Path(__file__).resolve().parent.parent / "data" / "nse_universe.py"
               ).read_text()
        # Only parse inside the NIFTY500 block
        start = src.find("NIFTY500")
        end = src.find("]))", start)
        block = src[start:end]
        sector = ""
        mapping: dict[str, str] = {}
        for line in block.splitlines():
            m = _SECTION_RE.match(line)
            if m:
                sector = m.group(1).strip()
                continue
            if sector:
                for sym in _SYM_RE.findall(line):
                    mapping.setdefault(sym, sector)
        _SECTOR_MAP = mapping
        log.info("sector_map_loaded", symbols=len(mapping),
                 sectors=len(set(mapping.values())))
    except Exception as exc:
        log.warning("sector_map_failed", error=str(exc))
    _loaded = True
    return _SECTOR_MAP


def sector_of(symbol: str) -> str:
    return _load_map().get(symbol.upper(), "")


def apply_sector_heat(results: list[dict], min_pack: int = 3,
                      boost: float = 6.0) -> dict[str, int]:
    """
    Mutates serialized scan results: stocks in a hot sector get a score
    boost + a reason naming their pack. Returns {sector: count} for
    display. Only BUY/WATCH signals count toward the pack.
    """
    smap = _load_map()
    if not smap or not results:
        return {}

    by_sector: dict[str, list[dict]] = {}
    for r in results:
        sec = smap.get(r.get("symbol", ""))
        if sec:
            by_sector.setdefault(sec, []).append(r)

    hot: dict[str, int] = {s: len(rs) for s, rs in by_sector.items()
                           if len(rs) >= min_pack}

    for sec, count in hot.items():
        pack = by_sector[sec]
        names = [p["symbol"] for p in pack]
        for r in pack:
            others = [x for x in names if x != r["symbol"]][:3]
            r["score"] = min(100.0, float(r.get("score", 0)) + boost)
            r["sector"] = sec
            r.setdefault("reasons", []).append(
                f"🔥 Sector heat ({sec}): {', '.join(others)} bhi setup bana rahe hain")
            if r.get("checks") is not None:
                r["checks"].append(
                    f"✓ Sector heat: {sec} ke {count} stocks ek saath active")
    if hot:
        log.info("sector_heat", hot_sectors={k: v for k, v in hot.items()})
    return hot
