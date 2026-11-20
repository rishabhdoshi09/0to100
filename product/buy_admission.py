"""One BUY-admission status for Home, Opportunities and Telegram (pure).

"No BUY" has four different causes. Each needs a different action, so the
surfaces must not collapse them into one "nothing to buy":

* SCAN_NOT_CURRENT: the official data or the saved scan is not the current
  session. Earlier setups are not signals; refresh the data or rescan. If
  Kite is logged out and the scan needs broker data, log in.
* EVIDENCE_UNAVAILABLE: the walk-forward evidence cannot certify anything
  (it is missing, stale, unprovable, or under another rulebook). No setup can
  qualify until the weekly walk-forward regenerates valid evidence.
* NO_PROVEN_EDGE: setups were found, and the evidence gate withheld them as
  WATCH because none has a proven edge.
* NO_SETUP: nothing in the current scan qualified.

ELIGIBLE_BUY means at least one BUY cleared every gate. The order is the
order of precedence: a stale scan is reported as stale even when its
evidence is also unusable.
"""

from __future__ import annotations

from typing import Any

ELIGIBLE_BUY = "ELIGIBLE_BUY"
SCAN_NOT_CURRENT = "SCAN_NOT_CURRENT"
EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
NO_PROVEN_EDGE = "NO_PROVEN_EDGE"
NO_SETUP = "NO_SETUP"
CODES = (ELIGIBLE_BUY, SCAN_NOT_CURRENT, EVIDENCE_UNAVAILABLE, NO_PROVEN_EDGE, NO_SETUP)

LOGIN_ACTION = "Log in to Kite (python main.py login) so the next scan has current broker data."


def classify(*, buys: int, scan_current: bool, evidence_note: str = "", withheld: int = 0,
             login_required: bool = False, scan_session: str = "", expected_session: str = "",
             closest: str = "") -> dict[str, Any]:
    """{code, headline, detail, action} for the current BUY admission."""
    note = str(evidence_note or "").strip().rstrip(".")
    if buys > 0 and scan_current:
        return {"code": ELIGIBLE_BUY, "headline": f"{int(buys)} BUY cleared every gate", "detail": "", "action": ""}
    if not scan_current:
        when = ""
        if scan_session and expected_session and scan_session < expected_session:
            when = f" The saved scan is for {scan_session}; the current session is {expected_session}."
        return {"code": SCAN_NOT_CURRENT, "headline": "Scan not current",
                "detail": "Earlier setups are not current signals." + when,
                "action": LOGIN_ACTION if login_required else "Refresh official data, then run a market scan."}
    if note:
        return {"code": EVIDENCE_UNAVAILABLE, "headline": "No BUY possible: walk-forward evidence unavailable",
                "detail": f"{note[:1].upper()}{note[1:]}. Setups are WATCH research only.",
                "action": "The weekly walk-forward must regenerate valid evidence before any setup can qualify."}
    if withheld > 0:
        return {"code": NO_PROVEN_EDGE, "headline": "No BUY today: no proven edge",
                "detail": (f"{int(withheld)} setup(s) looked technically ready, but none has a proven edge"
                           f"{f' (closest: {closest})' if closest else ''}, so they are WATCH."),
                "action": "Nothing to do: WATCH is the correct output until evidence proves an edge."}
    return {"code": NO_SETUP, "headline": "No BUY today: no setup qualified",
            "detail": "No setup in the current scan qualified.", "action": "Nothing to do."}


def from_scan_payload(payload: dict | None, *, scan_current: bool = True, login_required: bool = False,
                      expected_session: str = "") -> dict[str, Any]:
    """Classify straight from a saved scan payload (Opportunities, Telegram)."""
    payload = dict(payload or {})
    records = [r for r in payload.get("records") or [] if isinstance(r, dict)]
    buys = sum(1 for r in records if str(r.get("verdict") or "").upper() in {"BUY", "STRONG BUY"})
    gated = [r for r in records if r.get("edge_gated")]
    gated.sort(key=lambda r: (r.get("edge_r") is None, -float(r.get("edge_r") or 0.0)))
    closest = ""
    if gated and gated[0].get("edge_r") is not None:
        closest = f"{str(gated[0].get('symbol') or '').upper()} at {float(gated[0]['edge_r']):+.2f}R/trade"
    prov = dict(payload.get("provenance") or {})
    session = str(prov.get("market_session_date") or payload.get("as_of_session") or "")[:10]
    current = bool(scan_current) and not (session and expected_session and session < expected_session)
    note = str(dict(payload.get("evidence_identity") or {}).get("note") or "")
    return classify(buys=buys, scan_current=current, evidence_note=note, withheld=len(gated),
                    login_required=login_required, scan_session=session, expected_session=expected_session,
                    closest=closest)
