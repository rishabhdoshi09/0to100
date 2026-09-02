"""Desk note honesty: sourced wrap, empty slots, no invented blog numbers."""
from __future__ import annotations

import pytest

from product.desk_note import MIX_SHIFT_DESKS, build_desk_note, wrap_from_news
from product.education_feed import build_education_feed
from product.recommendations_workspace import build_market_reports_workspace


def _article(**kwargs):
    base = {
        "article_id": "a1",
        "headline": "Headline",
        "summary": "Summary",
        "why_it_matters": "",
        "category": "market",
        "event_type": "news",
        "source": "Example",
        "url": "https://example.com/a",
        "official": False,
        "published_at": "2026-08-24T06:00:00+00:00",
        "mentioned_symbols": [],
        "fno_symbols": [],
        "impact_score": 70,
    }
    base.update(kwargs)
    return base


@pytest.fixture(autouse=True)
def _isolate_case_db(tmp_path, monkeypatch):
    import product.case_memory as cm
    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")


INVENTED = (
    "related-party transaction rules",
    "₹15,000",
    "15000 crore",
    "1.6–1.63",
    "1.63 lakh",
    "₹600 crore",
    "+43%",
    "20,000 MTPA",
    "65% pre-booked",
    "7.3%",
    "Q1 FY27",
    "EBITDA per wheel: roughly",
)


def test_empty_news_does_not_invent_the_blog_wrap(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "product.market_view.peek_cached_market_view",
        lambda: SimpleNamespace(nifty_change_1d=None, nifty_price=0, summary="", leaders=(), laggards=()),
    )
    monkeypatch.setattr("data.index_store.latest_index_print", lambda *_a, **_k: None)
    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *_a, **_k: [])
    note = build_desk_note(articles=[], scan_payload={})
    assert note["places_orders"] is False
    assert note["wrap_sourced"] == 0
    assert note["wrap_empty"] == 5
    for bullet in note["wrap"]:
        assert bullet["available"] is False
        assert bullet["headline"] == ""
        assert "invent" in bullet["empty_detail"].lower() or "no sourced" in bullet["empty_detail"].lower()
    blob = " ".join(str(v) for v in note.values()) + " ".join(
        str(b) for b in note["wrap"]
    ) + " ".join(str(d) for d in note["desks"])
    for phrase in INVENTED:
        assert phrase not in blob
        assert phrase not in note["theme"]["body"]
    for line in note.get("daily_wrap") or []:
        assert "HDFC Bank" not in line["text"]
        assert "lawsuit" not in line["text"].lower()
        assert "Happiest Minds" not in line["text"]
        assert "nvidia" not in line["text"].lower()
        assert "490 million" not in line["text"]
        assert "Sensex dropped" not in line["text"]
        assert "Last market scan" not in line["text"]


def test_daily_wrap_uses_official_session_and_sourced_news(monkeypatch):
    from types import SimpleNamespace

    from product.desk_note import daily_wrap

    monkeypatch.setattr(
        "product.market_view.peek_cached_market_view",
        lambda: SimpleNamespace(
            nifty_change_1d=-0.5,
            nifty_price=0,
            summary="Market condition is weak.",
            leaders=("pharma",),
            laggards=("banks",),
        ),
    )
    monkeypatch.setattr("data.index_store.latest_index_print", lambda *_a, **_k: None)
    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *_a, **_k: [])
    lines = daily_wrap(
        articles=[
            _article(
                article_id="hdfc-news",
                headline="HDFC Bank fell after a US lawsuit alleged securities violations",
                summary="The bank denied the claims.",
                impact_score=90,
                mentioned_symbols=["HDFCBANK"],
            )
        ],
        scan_payload={"records": [{"symbol": "TATAPOWER", "status": "Ready to trade", "signals": ["BREAKOUT_52W"]}]},
    )
    texts = " ".join(item["text"] for item in lines)
    assert "falling 0.5%" in texts
    assert "Nifty" in texts
    assert "HDFC Bank fell after a US lawsuit" in texts
    assert "Last market scan" not in texts
    assert "TATAPOWER" not in texts
    assert "490 million" not in texts
    assert lines[0]["id"] == "session_indices"
    assert lines[0]["official"] is True


def test_daily_wrap_magazine_tape_from_official_prints(monkeypatch):
    from types import SimpleNamespace

    from product.desk_note import daily_wrap

    prints = {
        "^NSEI": {"price": 24087.0, "chg_pct": -0.5},
        "^NSEBANK": {"price": 51200.0, "chg_pct": -0.8},
        "^CNXPHARMA": {"price": 22100.0, "chg_pct": 0.6},
        "^CNXFMCG": {"price": 55000.0, "chg_pct": 0.4},
    }
    monkeypatch.setattr(
        "product.market_view.peek_cached_market_view",
        lambda: SimpleNamespace(
            nifty_change_1d=-0.5,
            nifty_price=24087.0,
            summary="",
            leaders=(),
            laggards=(),
        ),
    )
    monkeypatch.setattr("data.index_store.latest_index_print", lambda ticker: prints.get(ticker))
    monkeypatch.setattr(
        "data.index_store.recent_index_closes",
        lambda ticker, n=4: [24350.0, 24208.0, 24087.0] if ticker == "^NSEI" else [],
    )
    lines = daily_wrap(
        articles=[
            _article(
                article_id="happiest",
                headline="Happiest Minds fell 6% on reports that ITC Infotech may acquire the promoter’s stake",
                summary="The company could be delisted if the deal goes through.",
                impact_score=80,
                mentioned_symbols=["HAPPSTMNDS"],
            ),
            _article(
                article_id="us-open",
                headline="US markets are set for a stronger open after Nvidia earnings",
                summary="S&P 500 futures were higher.",
                impact_score=70,
            ),
        ],
        scan_payload={},
    )
    texts = [item["text"] for item in lines]
    session = texts[0]
    assert "extended losses for the second straight session" in session
    assert "falling 0.5%" in session
    assert "24,087" in session
    assert "slipping below 24,100" in session
    assert "Bank Nifty fell" in session
    assert "Pharma" in session and "ended positive" in session
    assert "Sensex dropped" not in session
    assert any("Happiest Minds fell 6%" in line for line in texts)
    assert texts[-1].startswith("US markets are set for a stronger open")
    assert "Last market scan" not in " ".join(texts)


def test_lodr_sebi_filing_does_not_fill_policy_slot():
    articles = [
        _article(
            article_id="lodr",
            headline="Larsen & Toubro Limited — disclosure pursuant to the provisions of Regulation 30",
            summary="Listing Obligations and Disclosure Requirements. Bagging/Receiving of orders/contracts.",
            source="NSE",
            url="https://nsearchives.nseindia.com/corporate/LT.pdf",
            mentioned_symbols=["LT"],
        )
    ]
    wrap = wrap_from_news(articles)
    policy = next(b for b in wrap if b["id"] == "policy")
    assert policy["available"] is False
    assert "related-party" not in (policy["headline"] + policy["summary"]).lower()


def test_real_sebi_rss_and_distinctive_needles_win():
    articles = [
        _article(
            article_id="lodr",
            headline="Disclosure pursuant to the provisions of Regulation 30 of Listing Obligations and Disclosure Requirements",
            source="NSE",
            mentioned_symbols=["LT"],
        ),
        _article(
            article_id="sebi-rss",
            headline="SEBI studies retail participation in cash and derivatives markets",
            summary="A SEBI working paper on household market participation.",
            source="SEBI",
            url="https://www.sebi.gov.in/reports/123.html",
            official=True,
            category="regulation",
        ),
        _article(
            article_id="lt-order",
            headline="L&T bags ultra-mega Middle East gas compression order",
            summary="Order book visibility from a Middle East gas compression contract.",
            source="Business Standard",
            url="https://www.business-standard.com/lt-gas",
            mentioned_symbols=["LT", "GKENERGY", "KPEL"],
            impact_score="88",
        ),
        _article(
            article_id="gold",
            headline="Muthoot, Manappuram, IIFL Finance gain as gold crosses ₹1.63 lakh",
            summary="Gold-loan NBFCs rally with bullion.",
            source="Economic Times",
            url="https://economictimes.indiatimes.com/gold-loan",
            mentioned_symbols=["MUTHOOTFIN", "MANAPPURAM", "IIFL", "DOLLAR", "LTF"],
        ),
        _article(
            article_id="fed",
            headline="US futures dip as Treasury yields pressure tech ahead of inflation data and Fed chair comments",
            source="Mint",
            url="https://www.livemint.com/us-futures",
        ),
        _article(
            article_id="mf",
            headline="Mutual fund SIPs keep drawing household savings into markets",
            source="AMFI",
            url="https://www.amfiindia.com/sip",
        ),
    ]
    note = build_desk_note(articles=articles, scan_payload={})
    by_id = {b["id"]: b for b in note["wrap"]}
    assert by_id["policy"]["available"] is True
    assert "retail participation" in by_id["policy"]["headline"].lower()
    assert "related-party" not in by_id["policy"]["headline"].lower()
    assert by_id["orders"]["available"] is True
    assert "gas compression" in by_id["orders"]["headline"].lower()
    assert by_id["gold_loan"]["available"] is True
    assert "muthoot" in by_id["gold_loan"]["headline"].lower()
    assert "1.63 lakh" in by_id["gold_loan"]["headline"]
    assert set(by_id["gold_loan"]["symbols"]) <= {"MUTHOOTFIN", "MANAPPURAM", "IIFL"}
    assert "DOLLAR" not in by_id["gold_loan"]["symbols"]
    assert by_id["orders"]["symbols"] == ["LT"]
    assert by_id["global"]["available"] is True
    assert by_id["flows"]["available"] is True
    gold_explainer = next(e for e in note["explainers"] if e["id"] == "concept-gold-loan-collateral")
    assert "collateral" in gold_explainer["teach_point"].lower()
    assert any(d["symbol"] == "LT" and d["available"] for d in note["desks"])
    muthoot = next(d for d in note["desks"] if d["symbol"] == "MUTHOOTFIN")
    assert muthoot["available"] is True
    assert muthoot["is_recommendation"] is False
    extras = {d["symbol"] for d in note["desks"]}
    assert "LT" in extras
    assert "DOLLAR" not in extras
    assert "GKENERGY" not in extras


def test_sswl_stays_empty_without_news_or_scan_and_does_not_invent_numbers():
    note = build_desk_note(articles=[], scan_payload={"records": []})
    sswl = next(d for d in note["desks"] if d["symbol"] == "SSWL")
    assert sswl["available"] is False
    assert sswl["source_headline"] == ""
    assert "not get invented" in sswl["empty_detail"]
    blob = " ".join([sswl["source_headline"], sswl["source_summary"], sswl["empty_detail"], *sswl["watch"]])
    assert "₹600" not in blob
    assert "+43%" not in blob
    assert "+27%" not in blob
    for frame in MIX_SHIFT_DESKS:
        desk = next(d for d in note["desks"] if d["symbol"] == frame["symbol"])
        assert desk["available"] is False
        assert desk["is_recommendation"] is False


def test_scan_row_makes_desk_available_without_inventing_filings():
    note = build_desk_note(
        articles=[],
        scan_payload={
            "records": [
                {"symbol": "SSWL", "status": "WATCH", "reason": "Breakout watch", "reasons": ["volume"]},
            ]
        },
    )
    sswl = next(d for d in note["desks"] if d["symbol"] == "SSWL")
    assert sswl["available"] is True
    assert sswl["scan_status"] == "WATCH"
    assert "₹600 crore" not in sswl["scan_reason"]
    assert sswl["source_headline"] == ""


def test_education_includes_gold_loan_and_mix_shift_concepts():
    payload = build_education_feed(articles=[], include_concepts=True)
    ids = {c["id"] for c in payload["cards"]}
    assert "concept-gold-loan-collateral" in ids
    assert "concept-mix-shift" in ids
    mix = next(c for c in payload["cards"] if c["id"] == "concept-mix-shift")
    assert "product mix" in mix["teach_point"].lower()
    assert mix["kind"] == "CONCEPT"
    assert mix["url"] == ""


def test_desk_note_module_does_not_hardcode_blog_prints():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "product" / "desk_note.py").read_text(encoding="utf-8")
    for phrase in (
        "₹600 crore", "20,000 MTPA", "65% pre-booked", "Q1 FY27", "+43%", "₹15,000",
        "HDFC Bank fell", "Happiest Minds", "490 million", "Sensex dropped 539",
    ):
        assert phrase not in src


def test_market_reports_workspace_embeds_desk_note(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        "reports.street_pulse.build_pulse",
        lambda **_k: {"date": "24 August 2026", "takeaways": ["Nifty steady"], "gainers": [], "losers": [], "breakouts_today": []},
    )
    payload = build_market_reports_workspace(
        persist_today=True,
        news_payload={
            "articles": [
                _article(
                    article_id="lt-order",
                    headline="L&T bags Middle East gas compression order",
                    mentioned_symbols=["LT"],
                )
            ]
        },
        scan_payload={"records": []},
    )
    note = payload["desk_note"]
    orders = next(b for b in note["wrap"] if b["id"] == "orders")
    assert orders["available"] is True
    assert "gas compression" in orders["headline"].lower()
    policy = next(b for b in note["wrap"] if b["id"] == "policy")
    assert policy["available"] is False
    assert note["places_orders"] is False
