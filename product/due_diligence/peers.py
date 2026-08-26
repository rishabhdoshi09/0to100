"""Peer ranking from tables on file. No estimated relatives."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.series import _f


def _quartile(rank: int, n: int) -> str:
    if n <= 1:
        return "Unranked — only this company is on the peer table."
    pct = (rank - 1) / max(n - 1, 1)
    if pct <= 0.25:
        return "Top quartile"
    if pct <= 0.45:
        return "Above median"
    if abs(pct - 0.5) <= 0.1:
        return "Median"
    if pct <= 0.75:
        return "Below median"
    return "Bottom quartile"


def rank_peers(rows: Sequence[Mapping[str, Any]], *, company: str, symbol: str) -> dict[str, Any]:
    """Quartile vs names already on the Screener peer table. Empty stays empty."""
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or row.get("Name") or "").strip()
        cells = dict(row.get("cells") or {})
        if not cells and name:
            cells = {
                str(k): v for k, v in row.items()
                if str(k) not in {"name", "Name", "cells", "fact", "row_label"} and v not in (None, "")
            }
        if not name or not cells:
            continue
        parsed.append({"name": name, "cells": cells})
    if not parsed:
        return {
            "available": False,
            "detail": "Data unavailable — no peer comparison table on file.",
            "rows": [],
            "ranks": [],
            "note": "Peers are not inferred from sector membership alone.",
        }

    want = {company.lower(), symbol.lower()}
    self_row = next(
        (r for r in parsed if r["name"].lower() in want or symbol.lower() in r["name"].lower()
         or company.lower() in r["name"].lower()),
        None,
    )
    numeric_keys: list[str] = []
    for row in parsed:
        for key, value in row["cells"].items():
            if _f(value) is not None and key not in numeric_keys:
                numeric_keys.append(key)
    ranks: list[dict[str, Any]] = []
    higher_is_better = {
        "cmp": False, "pe": False, "p/e": False, "pb": False, "p/b": False,
        "mar cap": False, "div yld": True, "np qtr": True, "qtr profit": True,
        "sales qtr": True, "roce": True, "roe": True,
    }
    for key in numeric_keys[:8]:
        scored: list[tuple[float, str]] = []
        for row in parsed:
            number = _f(row["cells"].get(key))
            if number is None:
                continue
            scored.append((number, row["name"]))
        if len(scored) < 2:
            continue
        hib = True
        key_l = key.lower()
        for token, flag in higher_is_better.items():
            if token in key_l:
                hib = flag
                break
        ordered = sorted(scored, key=lambda item: item[0], reverse=hib)
        names = [name for _value, name in ordered]
        self_name = (self_row or {}).get("name")
        if self_name not in names:
            continue
        rank = names.index(self_name) + 1
        self_value = next(v for v, n in ordered if n == self_name)
        ranks.append({
            "metric": key,
            "value": self_value,
            "rank": rank,
            "n": len(ordered),
            "quartile": _quartile(rank, len(ordered)),
            "higher_is_better": hib,
            "formula": f"Rank {rank} of {len(ordered)} on '{key}' (file order, not a model).",
        })
    return {
        "available": True,
        "detail": (
            f"{len(parsed)} names from the peer table on file. "
            "QuantTerm does not invent a peer set."
        ),
        "rows": parsed[:12],
        "self": (self_row or {}).get("name") or "Data unavailable",
        "ranks": ranks,
        "note": "Avoid reading a shared-sector name as a true business comparable without the table.",
    }
