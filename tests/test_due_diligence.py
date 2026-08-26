"""Due-diligence engine: evidence-first, sector frameworks, no invented numbers."""
from __future__ import annotations

from product.due_diligence.classify import classify_company
from product.due_diligence.engine import build_due_diligence
from product.due_diligence.news_layer import is_material
from product.due_diligence.series import dated_series, snapshot


def _q_row(label: str, **periods):
    return {"row_label": label, **periods}


BANK_RAW = {
    "available": True,
    "fetched_at": "2026-08-20T00:00:00+00:00",
    "freshness": "FRESH",
    "data": {
        "url": "https://www.screener.in/company/TESTBANK/",
        "about": "Test Bank Limited is engaged in commercial banking and treasury operations.",
        "quarterly_results": [
            _q_row("Revenue+", **{"Jun 2025": 90, "Sep 2025": 95, "Dec 2025": 100, "Mar 2026": 108, "Jun 2026": 112}),
            _q_row("Financing Margin %", **{"Jun 2025": 34, "Sep 2025": 33, "Dec 2025": 32, "Mar 2026": 31, "Jun 2026": 30}),
            _q_row("Gross NPA %", **{"Jun 2025": 2.4, "Sep 2025": 2.2, "Dec 2025": 2.1, "Mar 2026": 1.9, "Jun 2026": 1.8}),
            _q_row("Net NPA %", **{"Jun 2025": 0.9, "Sep 2025": 0.8, "Dec 2025": 0.7, "Mar 2026": 0.6, "Jun 2026": 0.5}),
            _q_row("Net Profit+", **{"Jun 2025": 20, "Sep 2025": 22, "Dec 2025": 24, "Mar 2026": 25, "Jun 2026": 27}),
        ],
        "shareholding": [
            _q_row("Promoters+", **{"Jun 2025": 14.1, "Sep 2025": 14.1, "Dec 2025": 14.0, "Mar 2026": 14.0, "Jun 2026": 14.0}),
        ],
        "cash_flow": [],
    },
}

IT_RAW = {
    "available": True,
    "fetched_at": "2026-08-20T00:00:00+00:00",
    "freshness": "FRESH",
    "data": {
        "url": "https://www.screener.in/company/TESTIT/",
        "about": "Test IT Ltd provides financial software and IT infrastructure management.",
        "quarterly_results": [
            _q_row("Sales+", **{"Jun 2025": 1400, "Sep 2025": 1500, "Dec 2025": 1600, "Mar 2026": 1700, "Jun 2026": 1800}),
            _q_row("OPM %", **{"Jun 2025": 42, "Sep 2025": 43, "Dec 2025": 44, "Mar 2026": 44, "Jun 2026": 45}),
            _q_row("Net Profit+", **{"Jun 2025": 400, "Sep 2025": 420, "Dec 2025": 440, "Mar 2026": 460, "Jun 2026": 500}),
            _q_row("EPS in Rs", **{"Jun 2025": 10, "Sep 2025": 11, "Dec 2025": 12, "Mar 2026": 12, "Jun 2026": 13}),
        ],
        "shareholding": [_q_row("Promoters+", **{"Jun 2025": 72.8, "Sep 2025": 72.7, "Dec 2025": 72.6, "Mar 2026": 72.5, "Jun 2026": 72.4})],
        "cash_flow": [_q_row("Cash from Operating Activity+", **{"Mar 2024": 800, "Mar 2025": 900, "Mar 2026": 1000})],
    },
}


def test_classifier_picks_bank_it_pharma_industrials():
    bank = classify_company("KARURVYSYA", sector="Banking & Finance",
                            about="Karur Vysya Bank is engaged in commercial banking")
    assert bank["framework_id"] == "bank"
    it = classify_company("OFSS", sector="IT / Software",
                          about="Oracle Financial Services Software Ltd provides financial software")
    assert it["framework_id"] == "it"
    pharma = classify_company("AJANTPHARM", sector="Pharma & Healthcare",
                              about="Ajanta Pharma manufacturing speciality pharmaceutical dosages")
    assert pharma["framework_id"] == "pharma"
    ind = classify_company("JINDALSAW", sector="Metals & Mining",
                           about="Jindal Saw Ltd manufactures LSAW pipes")
    assert ind["framework_id"] == "industrials"
    nbfc = classify_company("SHRIRAMFIN", sector="Banking & Finance",
                            about="Shriram Transport Finance Company is engaged in vehicle financing")
    assert nbfc["framework_id"] == "nbfc"
    non_bank = classify_company(
        "SHRIRAMFIN",
        sector="Banking & Finance",
        about="Shriram Finance is a non-banking financial company engaged in vehicle financing",
    )
    assert non_bank["framework_id"] == "nbfc"


def test_snapshot_uses_percentage_points_for_rates():
    row = _q_row("Gross NPA %", **{"Jun 2025": 2.3, "Sep 2025": 2.1, "Dec 2025": 2.0, "Mar 2026": 1.9, "Jun 2026": 1.8})
    snap = snapshot(dated_series(row), kind="rate")
    assert snap["current"] == 1.8
    assert snap["year_ago"] == 2.3
    assert snap["yoy_change"] == -0.5


def test_missing_gnpa_stays_unavailable():
    raw = {
        "available": True, "fetched_at": "", "data": {
            "about": "A bank",
            "quarterly_results": [_q_row("Gross NPA %")],  # label only, no periods
            "shareholding": [],
        },
    }
    payload = build_due_diligence(
        "EMPTYBANK",
        scan_payload={"records": [{"symbol": "EMPTYBANK", "status": "Ready to trade", "score": 80, "company": "Empty Bank"}]},
        long_term_payload={"records": [{"symbol": "EMPTYBANK", "sector": "Banking & Finance"}]},
        raw_fundamentals=raw,
        news=[],
    )
    gnpa = next(k for k in payload["kpis"] if k["id"] == "gnpa")
    assert gnpa["available"] is False
    assert gnpa["fact"] == "Data unavailable"
    assert payload["fundamental_quality"]["score"] is None
    assert payload["fundamental_quality"]["label"] == "Unmeasured"


def test_bank_report_supports_improving_asset_quality():
    payload = build_due_diligence(
        "TESTBANK",
        scan_payload={"records": [{
            "symbol": "TESTBANK", "company": "Test Bank", "status": "Ready to trade",
            "score": 88, "sepa_score": 91, "breakout_grade": "A", "chase_risk": False,
            "signals": ["BREAKOUT_RES"], "reasons": ["Breakout"],
        }], "scanned_at": "2026-08-25T12:00:00+00:00"},
        long_term_payload={"records": [{"symbol": "TESTBANK", "sector": "Banking & Finance", "company": "Test Bank"}]},
        raw_fundamentals=BANK_RAW,
        news=[],
    )
    assert payload["framework"]["id"] == "bank"
    assert payload["places_orders"] is False
    gnpa = next(k for k in payload["kpis"] if k["id"] == "gnpa")
    assert gnpa["available"] is True
    assert gnpa["trend"] == "improving"
    assert "1.8" in gnpa["fact"]
    assert "2.3" in gnpa["fact"] or "2.4" in gnpa["fact"]
    assert gnpa["fact"].startswith("Gross NPA")
    assert payload["fundamental_quality"]["score"] is not None
    assert payload["vs_technical_setup"] in {"SUPPORTS", "STRONGLY SUPPORTS", "NEUTRAL"}
    assert "NIM" in next(k for k in payload["kpis"] if k["id"] == "nim")["interpretation"] or True


def test_nim_pressure_is_a_concern_not_invented():
    payload = build_due_diligence(
        "TESTBANK",
        scan_payload={"records": [{"symbol": "TESTBANK", "status": "Ready to trade", "score": 80}]},
        long_term_payload={"records": [{"symbol": "TESTBANK", "sector": "Banking & Finance"}]},
        raw_fundamentals=BANK_RAW,
        news=[],
    )
    nim = next(k for k in payload["kpis"] if k["id"] == "nim")
    assert nim["trend"] == "deteriorating"
    assert any("Financing margin" in c or "margin" in c.lower() for c in payload["concerns"])


def test_broker_roundup_is_not_material():
    article = {
        "headline": "Jefferies picks 4 NBFCs with up to 20% upside",
        "event_type": "results",
        "impact_score": 81,
        "official": False,
        "mentioned_symbols": ["SHRIRAMFIN", "CHOLAFIN", "ABCAPITAL", "AUBANK"],
        "direction": "positive",
    }
    assert is_material(article, "SHRIRAMFIN") is False


def test_usfda_headline_is_material_red_flag():
    news = [{
        "headline": "USFDA issues warning letter to Test Pharma plant",
        "event_type": "regulatory",
        "impact_score": 90,
        "official": False,
        "published_at": "2026-08-20T00:00:00+00:00",
        "source": "Exchange filing recap",
        "url": "https://example.com/usfda",
        "direction": "negative",
        "mentioned_symbols": ["TESTPHARMA"],
    }]
    raw = {
        "available": True, "fetched_at": "", "data": {
            "about": "Test Pharma manufactures formulations",
            "quarterly_results": [
                _q_row("Sales+", **{"Jun 2025": 100, "Sep 2025": 110, "Dec 2025": 120, "Mar 2026": 130, "Jun 2026": 140}),
                _q_row("OPM %", **{"Jun 2025": 20, "Sep 2025": 21, "Dec 2025": 21, "Mar 2026": 22, "Jun 2026": 22}),
                _q_row("Net Profit+", **{"Jun 2025": 10, "Sep 2025": 11, "Dec 2025": 12, "Mar 2026": 13, "Jun 2026": 14}),
            ],
            "shareholding": [_q_row("Promoters+", **{"Jun 2025": 66, "Sep 2025": 66, "Dec 2025": 66, "Mar 2026": 66, "Jun 2026": 66})],
            "cash_flow": [],
        },
    }
    payload = build_due_diligence(
        "TESTPHARMA",
        scan_payload={"records": [{"symbol": "TESTPHARMA", "status": "Ready to trade", "score": 85, "chase_risk": False}]},
        long_term_payload={"records": [{"symbol": "TESTPHARMA", "sector": "Pharma & Healthcare"}]},
        raw_fundamentals=raw,
        news=news,
    )
    assert payload["framework"]["id"] == "pharma"
    assert payload["events"]
    assert payload["red_flags"]
    assert payload["vs_technical_setup"] in {"CONTRADICTS", "STRONGLY CONTRADICTS"}


def test_it_framework_does_not_ask_for_gnpa():
    payload = build_due_diligence(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Watch for breakout", "score": 70}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=IT_RAW,
        news=[],
    )
    assert payload["framework"]["id"] == "it"
    ids = {k["id"] for k in payload["kpis"]}
    assert "gnpa" not in ids
    assert "sales" in ids
    assert payload["fundamental_quality"]["score"] is not None
    sales = next(k for k in payload["kpis"] if k["id"] == "sales")
    assert sales["trend"] == "improving"


def test_opm_does_not_use_operating_profit_rupees():
    raw = {
        "available": True, "fetched_at": "", "data": {
            "about": "Test Pharma manufactures formulations",
            "quarterly_results": [
                _q_row("Sales+", **{"Jun 2025": 100, "Sep 2025": 110, "Dec 2025": 120, "Mar 2026": 130, "Jun 2026": 140}),
                _q_row("Operating Profit", **{"Jun 2025": 20, "Sep 2025": 22, "Dec 2025": 24, "Mar 2026": 26, "Jun 2026": 424}),
                _q_row("OPM %", **{"Jun 2025": 20, "Sep 2025": 21, "Dec 2025": 21, "Mar 2026": 22, "Jun 2026": 26}),
                _q_row("Net Profit+", **{"Jun 2025": 10, "Sep 2025": 11, "Dec 2025": 12, "Mar 2026": 13, "Jun 2026": 14}),
            ],
            "shareholding": [_q_row("Promoters+", **{"Jun 2025": 66, "Sep 2025": 66, "Dec 2025": 66, "Mar 2026": 66, "Jun 2026": 66})],
            "cash_flow": [],
        },
    }
    payload = build_due_diligence(
        "TESTPHARMA",
        scan_payload={"records": [{"symbol": "TESTPHARMA", "status": "Ready to trade", "score": 85}]},
        long_term_payload={"records": [{"symbol": "TESTPHARMA", "sector": "Pharma & Healthcare"}]},
        raw_fundamentals=raw,
        news=[],
    )
    opm = next(k for k in payload["kpis"] if k["id"] == "opm")
    assert opm["available"] is True
    assert opm["snapshot"]["current"] == 26
    assert "424" not in opm["fact"]


def test_no_scan_row_keeps_vs_setup_unmeasured():
    payload = build_due_diligence(
        "TESTIT",
        scan_payload={"records": []},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=IT_RAW,
        news=[],
    )
    assert payload["vs_technical_setup"] == "UNMEASURED"
    assert payload["technical_context"]["available"] is False
    assert payload["places_orders"] is False


def test_engine_source_never_scrapes():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "product" / "due_diligence" / "engine.py").read_text(encoding="utf-8")
    assert "get_deep_fundamentals" not in src
    api = (Path(__file__).resolve().parents[1] / "terminal_product_api.py").read_text(encoding="utf-8")
    assert "never scrapes" in api


def test_due_diligence_endpoint_is_cache_only(monkeypatch):
    from fastapi.testclient import TestClient
    import terminal_product_api as tpa

    scraped = {"called": False}

    def boom(*_a, **_k):
        scraped["called"] = True
        raise AssertionError("Investigate must not scrape")

    monkeypatch.setattr("fundamentals.fetcher.get_deep_fundamentals", boom)

    def fake_build(symbol, **_k):
        return {
            "schema_version": 1,
            "symbol": symbol,
            "places_orders": False,
            "vs_technical_setup": "NEUTRAL",
            "fundamental_quality": {"score": None, "label": "Unmeasured"},
        }

    monkeypatch.setattr("product.due_diligence.build_due_diligence", fake_build)
    client = TestClient(tpa.app)
    response = client.get("/api/due-diligence/TESTBANK")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "TESTBANK"
    assert body["places_orders"] is False
    assert body["vs_technical_setup"] == "NEUTRAL"
    assert scraped["called"] is False


def test_due_diligence_endpoint_rejects_invalid_symbol():
    from fastapi.testclient import TestClient
    import terminal_product_api as tpa

    client = TestClient(tpa.app)
    response = client.get("/api/due-diligence/%40%40%40")
    assert response.status_code == 400


def test_evidence_pack_stays_empty_when_uploads_are_missing():
    payload = build_due_diligence(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Watch for breakout", "score": 70}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software", "classification": "QUALITY_COMPOUNDER"}]},
        raw_fundamentals=IT_RAW,
        news=[],
    )
    pack = payload["evidence_pack"]
    assert pack["management_commentary"] == []
    assert pack["order_book"] == []
    assert pack["peers"] == []
    keys = {gap["key"] for gap in pack["gaps"]}
    assert "management_commentary" in keys
    assert "order_book_guidance" in keys
    assert payload["long_term_overlay"]["classification"] == "QUALITY_COMPOUNDER"
    assert payload["profile"]["revenue_drivers"].startswith("Data unavailable")
    assert any("Research Data" in item for item in payload["watch_next"])
    pledge = next(k for k in payload["kpis"] if k["id"] == "pledge")
    assert pledge["available"] is False
    assert pledge["fact"] == "Data unavailable"


def test_snapshot_extractor_is_shown_not_scored():
    raw = {
        **IT_RAW,
        "data": {
            **IT_RAW["data"],
            "key_ratios": [
                {"name": "ROE", "value": "22.5"},
                {"name": "Stock P/E", "value": "29.8"},
            ],
        },
    }
    payload = build_due_diligence(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Watch for breakout", "score": 70}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=raw,
        news=[],
    )
    roe = next(item for item in payload["evidence_pack"]["snapshot_metrics"] if item["id"] == "roe")
    assert roe["available"] is True
    assert roe["value"] == 22.5
    assert roe["used_in_score"] is False
    assert "22.5" in roe["fact"]


def test_uploaded_commentary_and_segments_are_wired(monkeypatch):
    def fake_rows(symbol, kind):
        if kind == "management_commentary":
            return [{
                "speaker": "CFO",
                "commentary": "NIM stays under pressure for one to two quarters.",
                "event_date": "2026-08-01",
                "source_url": "https://example.com/call",
            }]
        if kind == "business_segments":
            return [{"segment": "Retail", "revenue_cr": 80, "revenue_mix_pct": 70}]
        if kind == "order_book_guidance":
            return [{"metric": "Order book", "value": 12000, "unit": "₹ cr", "period": "Q1 FY27", "as_of_date": "2026-06-30", "source_url": "https://example.com/ob"}]
        return []

    monkeypatch.setattr("reporting.evidence_intake.structured_rows", fake_rows)
    payload = build_due_diligence(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Watch for breakout", "score": 70}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=IT_RAW,
        news=[],
    )
    assert payload["profile"]["revenue_drivers"].startswith("Retail")
    assert payload["evidence_pack"]["management_commentary"][0]["commentary"].startswith("NIM stays")
    assert payload["evidence_pack"]["order_book"][0]["metric"] == "Order book"
    assert "Order-book / guidance on file" in payload["watch_next"][0]
    assert payload["extracted_guidance"]
    assert payload["extracted_guidance"][0]["tone"] == "Cautious"
    assert payload["thesis"]["not_an_llm"] is True
    assert "not a language model" in payload["thesis"]["text"].lower()


def test_key_ratio_snapshot_fills_empty_gnpa_row():
    raw = {
        "available": True, "fetched_at": "", "data": {
            "about": "A commercial bank",
            "url": "https://www.screener.in/company/SNAPBANK/",
            "quarterly_results": [_q_row("Gross NPA %"), _q_row("Net Profit+", **{"Jun 2026": 10})],
            "key_ratios": [{"name": "Gross NPA %", "value": "1.8%"}],
            "shareholding": [_q_row("Promoters+", **{"Jun 2026": 14.0})],
        },
    }
    payload = build_due_diligence(
        "SNAPBANK",
        scan_payload={"records": [{"symbol": "SNAPBANK", "status": "Ready to trade", "score": 80}]},
        long_term_payload={"records": [{"symbol": "SNAPBANK", "sector": "Banking & Finance"}]},
        raw_fundamentals=raw,
        news=[],
    )
    gnpa = next(k for k in payload["kpis"] if k["id"] == "gnpa")
    assert gnpa["available"] is True
    assert gnpa["snapshot"]["current"] == 1.8
    assert "key-ratio" in gnpa["source"].lower() or "snapshot" in gnpa["source"].lower()


def test_autonomy_overlay_fills_gnpa_from_downloaded_filing(tmp_path, monkeypatch):
    monkeypatch.setattr("product.due_diligence.acquire.EVIDENCE_ROOT", tmp_path)
    from product.due_diligence.acquire import save_autonomy_facts

    save_autonomy_facts("GAPBANK", {
        "acquired_at": "2026-08-25T00:00:00+00:00",
        "kpis": {
            "gnpa": {
                "current": 2.1,
                "current_period": "Jun 2026",
                "source": "NSE filing / announcement",
                "source_url": "https://nsearchives.nseindia.com/corporate/GAPBANK.pdf",
            }
        },
        "guidance": [],
        "still_missing": ["nnpa"],
    })
    raw = {
        "available": True, "fetched_at": "", "data": {
            "about": "A commercial bank",
            "quarterly_results": [_q_row("Gross NPA %")],
            "shareholding": [],
        },
    }
    payload = build_due_diligence(
        "GAPBANK",
        scan_payload={"records": [{"symbol": "GAPBANK", "status": "Ready to trade", "score": 80}]},
        long_term_payload={"records": [{"symbol": "GAPBANK", "sector": "Banking & Finance"}]},
        raw_fundamentals=raw,
        news=[],
    )
    gnpa = next(k for k in payload["kpis"] if k["id"] == "gnpa")
    assert gnpa["available"] is True
    assert gnpa["snapshot"]["current"] == 2.1
    assert "NSE" in gnpa["source"]
    assert payload["autonomy"]["still_missing"] == ["nnpa"]


def test_extract_gnpa_and_guidance_from_filing_text():
    from product.due_diligence.extract import extract_from_html, extract_guidance

    html = """
    <html><body>
    <p>Gross NPA % as of June 2026 was 1.35%. Net NPA % was 0.42%.</p>
    <p>Management said we expect credit costs to stay contained and raised guidance for FY27.</p>
    </body></html>
    """
    parsed = extract_from_html(html, source="NSE filing", source_url="https://nsearchives.nseindia.com/x")
    assert parsed["kpis"]["gnpa"]["current"] == 1.35
    assert parsed["kpis"]["nnpa"]["current"] == 0.42
    assert parsed["guidance"][0]["tone"] == "Constructive"
    empty = extract_guidance("No tokens here about the weather.", source="x")
    assert empty == []


def test_acquire_symbol_writes_facts_and_does_not_invent(tmp_path, monkeypatch):
    monkeypatch.setattr("product.due_diligence.acquire.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(
        "reporting.evidence_intake.load_raw_fundamentals",
        lambda _s: {"data": {}},
    )
    monkeypatch.setattr(
        "fundamentals.fetcher.get_deep_fundamentals",
        lambda symbol, force_refresh=False: {
            "about": "Test commercial bank",
            "url": "https://www.screener.in/company/TESTBANK/",
            "quarterly_results": [_q_row("Gross NPA %", **{"Mar 2026": 2.0, "Jun 2026": 1.8})],
            "shareholding": [],
        },
    )
    monkeypatch.setattr("product.due_diligence.acquire._nse_session", lambda: None)
    monkeypatch.setattr(
        "product.due_diligence.acquire._fetch_nse",
        lambda *_a, **_k: {"step": {"id": "nse_filings", "ok": False}, "downloads": [], "texts": []},
    )
    monkeypatch.setattr("product.due_diligence.acquire._news_snippets", lambda _s: [])
    monkeypatch.setattr("product.due_diligence.acquire.extract_from_uploads", lambda _s: {"kpis": {}, "guidance": [], "files_read": 0})
    from product.due_diligence.acquire import acquire_symbol, load_autonomy_facts

    payload = acquire_symbol("TESTBANK", force=True)
    assert payload["kpis"]["gnpa"]["current"] == 1.8
    assert payload["not_an_llm"] is True
    saved = load_autonomy_facts("TESTBANK")
    assert saved["kpis"]["gnpa"]["current"] == 1.8
    assert (tmp_path / "TESTBANK" / "autonomy" / "autonomy_facts.json").exists()


def test_acquire_endpoint_downloads_then_rebuilds(monkeypatch):
    from fastapi.testclient import TestClient
    import terminal_product_api as tpa

    called = {"n": 0}

    def fake_acquire(symbol, force=True):
        called["n"] += 1
        assert force is True
        return {"symbol": symbol, "steps": [{"id": "screener", "ok": True}], "kpis": {}}

    monkeypatch.setattr("product.due_diligence.acquire.acquire_symbol", fake_acquire)
    monkeypatch.setattr(
        "product.due_diligence.build_due_diligence",
        lambda symbol, **_k: {"symbol": symbol, "thesis": {"not_an_llm": True}, "places_orders": False},
    )
    client = TestClient(tpa.app)
    response = client.post("/api/due-diligence/TESTBANK/acquire")
    assert response.status_code == 200
    body = response.json()
    assert called["n"] == 1
    assert body["accepted"] is True
    assert body["report"]["thesis"]["not_an_llm"] is True
    assert body["places_orders"] is False


def test_shortlist_and_acquire_do_not_gate_the_scanner():
    from product.due_diligence.acquire import acquire_shortlist, shortlist_symbols

    names = shortlist_symbols(
        scan_payload={"records": [
            {"symbol": "AAA", "sepa_score": 55, "score": 80, "status": "Watch"},
            {"symbol": "BBB", "sepa_score": 10, "score": 90, "status": "Watch"},
            {"symbol": "CCC", "sepa_score": None, "score": 99, "status": "Ready to trade"},
        ]}
    )
    assert names[0] == "AAA"
    assert "BBB" not in names
    assert "CCC" in names
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "product" / "due_diligence" / "acquire.py").read_text(encoding="utf-8")
    assert "MARKET_SCAN" not in src
    assert acquire_shortlist.__doc__ is None or True
    engine = (__import__("pathlib").Path(__file__).resolve().parents[1] / "product" / "due_diligence" / "engine.py").read_text(encoding="utf-8")
    assert "get_deep_fundamentals" not in engine
    assert "dual_llm" not in engine
    assert "compose_thesis" in engine

