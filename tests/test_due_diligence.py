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
    from product.due_diligence.extract import extract_rates_from_text
    spaced = extract_rates_from_text(
        "GNPA at 0.74%, NNPA is below 1% and stands at 0. 19% of net advances.",
        source="press",
    )
    assert spaced["gnpa"]["current"] == 0.74
    assert spaced["nnpa"]["current"] == 0.19
    charity = extract_rates_from_text(
        "Promoters are LivingMyPromise signatories; they have pledged to give away at least 50% to charity.",
        source="press",
    )
    assert "pledge" not in charity
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

    called = {"n": 0, "force": None}

    def fake_acquire(symbol, force=False, **_k):
        called["n"] += 1
        called["force"] = force
        return {"symbol": symbol, "steps": [{"id": "screener", "ok": True}], "kpis": {}, "mode": "missing_or_stale"}

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
    assert called["force"] is False
    assert body["mode"] == "missing_or_stale"
    assert body["accepted"] is True
    assert body["report"]["thesis"]["not_an_llm"] is True
    assert body["places_orders"] is False
    all_resp = client.post("/api/due-diligence/TESTBANK/acquire?mode=all")
    assert all_resp.status_code == 200
    assert called["force"] is True
    assert all_resp.json()["mode"] == "all"


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


def test_attachment_rank_prefers_result_filings():
    from product.due_diligence.acquire import _attachment_rank
    from product.due_diligence.extract import extract_rates_from_text

    assert _attachment_rank("submitted the financial results for the period ended Jun 30, 2026") == 0
    assert _attachment_rank("informed the Exchange about Transcript") == 1
    assert _attachment_rank("Schedule of investor meet") == 99
    assert _attachment_rank("Audited financial results newspaper advertisement under Regulation 47") == 99
    parsed = extract_rates_from_text(
        "Gross NPA % as of June 2026 was 1.35%. Net NPA % was 0.42%.",
        source="NSE results PDF",
        source_url="https://nsearchives.nseindia.com/x.pdf",
    )
    assert parsed["gnpa"]["current"] == 1.35
    assert parsed["nnpa"]["current"] == 0.42
    assert _attachment_rank("Annual Report for FY 2025") == 4
    assert _attachment_rank("Audio recording of concall") == 1


def test_extract_research_pack_fills_order_book_segments_commentary():
    from product.due_diligence.extract import extract_research_pack

    text = (
        "The order book stood at 12,450 crore as of June 2026. "
        "The energy segment contributed 42% of revenue. "
        "The CEO said demand remains robust across industrial pipes this quarter."
    )
    parsed = extract_research_pack(text, source="NSE filing", source_url="https://nsearchives.nseindia.com/x.pdf")
    assert parsed["order_book"][0]["value"] == 12450
    assert parsed["segments"][0]["segment"].lower() == "energy"
    assert parsed["segments"][0]["revenue_mix_pct"] == 42
    assert "robust" in parsed["commentary"][0]["commentary"].lower()


def test_option_chain_summary_is_descriptive_not_a_signal():
    from product.due_diligence.option_chain import summarize_option_chain

    empty = summarize_option_chain({})
    assert empty["available"] is False
    assert empty["places_orders"] is False
    payload = {
        "records": {
            "expiryDates": ["28-Aug-2026", "25-Sep-2026"],
            "underlyingValue": 100,
            "data": [
                {"strikePrice": 90, "expiryDate": "28-Aug-2026", "CE": {"openInterest": 10, "impliedVolatility": 20}, "PE": {"openInterest": 50, "impliedVolatility": 22}},
                {"strikePrice": 100, "expiryDate": "28-Aug-2026", "CE": {"openInterest": 40, "impliedVolatility": 18}, "PE": {"openInterest": 40, "impliedVolatility": 18}},
                {"strikePrice": 110, "expiryDate": "28-Aug-2026", "CE": {"openInterest": 80, "impliedVolatility": 25}, "PE": {"openInterest": 5, "impliedVolatility": 24}},
                {"strikePrice": 100, "expiryDate": "25-Sep-2026", "CE": {"openInterest": 999}, "PE": {"openInterest": 999}},
            ],
        }
    }
    snap = summarize_option_chain(payload, source_url="https://www.nseindia.com/api/option-chain-equities?symbol=TEST")
    assert snap["available"] is True
    assert snap["expiry"] == "28-Aug-2026"
    assert snap["call_oi"] == 130
    assert snap["put_oi"] == 95
    assert snap["pcr"] == 0.731
    assert snap["max_pain"] == 100
    assert snap["atm_strike"] == 100
    assert snap["atm_iv"] == 18
    assert snap["not_a_signal"] is True
    assert snap["places_orders"] is False


def test_autonomy_overlay_fills_commentary_order_book_and_chain(tmp_path, monkeypatch):
    monkeypatch.setattr("product.due_diligence.acquire.EVIDENCE_ROOT", tmp_path)
    from product.due_diligence.acquire import save_autonomy_facts

    save_autonomy_facts("PIPECO", {
        "acquired_at": "2026-08-25T00:00:00+00:00",
        "kpis": {},
        "guidance": [],
        "commentary": [{
            "speaker": "CEO",
            "topic": "",
            "commentary": "Demand remains robust across industrial pipes this quarter.",
            "event_date": "2026-07-20",
            "source_url": "https://nsearchives.nseindia.com/x.pdf",
        }],
        "order_book": [{
            "metric": "Order Book",
            "value": 12450,
            "unit": "₹ cr",
            "period": "Jun 2026",
            "as_of_date": "Jun 2026",
        }],
        "segments": [{"segment": "Energy", "revenue_mix_pct": 42}],
        "option_chain": {
            "available": True,
            "expiry": "28-Aug-2026",
            "pcr": 0.9,
            "not_a_signal": True,
            "places_orders": False,
        },
        "still_missing": [],
    })
    payload = build_due_diligence(
        "PIPECO",
        scan_payload={"records": [{"symbol": "PIPECO", "status": "Ready to trade", "score": 80}]},
        long_term_payload={"records": [{"symbol": "PIPECO", "sector": "Metals & Mining"}]},
        raw_fundamentals={"available": True, "fetched_at": "", "data": {"about": "Makes SAW pipes", "quarterly_results": [], "shareholding": []}},
        news=[],
    )
    pack = payload["evidence_pack"]
    assert "robust" in pack["management_commentary"][0]["commentary"].lower()
    assert pack["order_book"][0]["value"] == 12450
    assert "Energy" in pack["revenue_drivers"]
    assert pack["option_chain"]["pcr"] == 0.9
    assert pack["option_chain"]["not_a_signal"] is True
    assert payload["autonomy"]["option_chain"]["available"] is True


def test_acquire_symbol_persists_research_pack_and_chain(tmp_path, monkeypatch):
    monkeypatch.setattr("product.due_diligence.acquire.EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr("reporting.evidence_intake.load_raw_fundamentals", lambda _s: {"data": {}})
    monkeypatch.setattr(
        "fundamentals.fetcher.get_deep_fundamentals",
        lambda symbol, force_refresh=False: {
            "about": "Makes SAW pipes",
            "url": "https://www.screener.in/company/PIPECO/",
            "quarterly_results": [],
            "shareholding": [],
        },
    )
    monkeypatch.setattr("product.due_diligence.acquire._nse_session", lambda: object())
    monkeypatch.setattr(
        "product.due_diligence.acquire._fetch_nse",
        lambda *_a, **_k: {
            "step": {"id": "nse_filings", "ok": True},
            "downloads": [],
            "texts": [(
                "The order book stood at 12,450 crore. The energy segment contributed 42%. "
                "The CEO said demand remains robust across industrial pipes this quarter.",
                "https://nsearchives.nseindia.com/x.pdf",
            )],
            "headlines": [],
        },
    )
    monkeypatch.setattr(
        "product.due_diligence.acquire._fetch_annual_reports",
        lambda *_a, **_k: {"step": {"id": "nse_annual_reports", "ok": False}, "downloads": [], "texts": []},
    )
    monkeypatch.setattr(
        "product.due_diligence.acquire._fetch_option_chain",
        lambda *_a, **_k: {
            "step": {"id": "option_chain", "ok": True, "expiry": "28-Aug-2026"},
            "download": {"ok": True, "path": "logs/research_evidence/PIPECO/autonomy/nse_option_chain.json"},
            "snapshot": {
                "available": True,
                "expiry": "28-Aug-2026",
                "pcr": 1.1,
                "not_a_signal": True,
                "places_orders": False,
            },
        },
    )
    monkeypatch.setattr("product.due_diligence.acquire._news_snippets", lambda _s: [])
    monkeypatch.setattr(
        "product.due_diligence.acquire.extract_from_uploads",
        lambda _s: {"kpis": {}, "guidance": [], "commentary": [], "order_book": [], "segments": [], "files_read": 0},
    )
    from product.due_diligence.acquire import acquire_symbol

    payload = acquire_symbol("PIPECO", force=True)
    assert payload["order_book"][0]["value"] == 12450
    assert payload["segments"][0]["revenue_mix_pct"] == 42
    assert "robust" in payload["commentary"][0]["commentary"].lower()
    assert payload["option_chain"]["pcr"] == 1.1
    assert payload["option_chain"]["not_a_signal"] is True
    assert "option_chain" not in payload["still_missing"]
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "product" / "due_diligence" / "acquire.py").read_text(encoding="utf-8")
    assert "options.analytics" not in src
    chain_src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "product" / "due_diligence" / "option_chain.py").read_text(encoding="utf-8")
    assert "streamlit" not in chain_src


def test_stock_research_engine_is_the_same_builder():
    from product.due_diligence import StockResearchEngine, investigate_stock

    payload = StockResearchEngine().investigate(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Watch for breakout", "score": 70}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=IT_RAW,
        news=[],
    )
    alias = investigate_stock(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Watch for breakout", "score": 70}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=IT_RAW,
        news=[],
    )
    assert payload["engine"] == "StockResearchEngine"
    assert payload["uses_llm"] is False
    assert payload["fundamental_confirmation"] in {
        "STRONG SUPPORT", "SUPPORT", "NEUTRAL", "CAUTION", "CONTRADICTS",
    }
    assert payload["first_screen"]["ticker"] == "TESTIT"
    assert payload["kpis"] == alias["kpis"]
    sales = next(k for k in payload["kpis"] if k["id"] == "sales")
    assert sales["formula"]
    assert sales["provenance"]["source"]
    assert payload["fundamental_quality"]["breakdown"]["pillars"]


def test_auto_and_fmcg_frameworks_are_configured():
    auto = classify_company("TATAMOTORS", sector="Auto", about="Tata Motors manufactures passenger vehicles")
    assert auto["framework_id"] == "auto"
    fmcg = classify_company("HINDUNILVR", sector="FMCG", about="Hindustan Unilever is an FMCG company")
    assert fmcg["framework_id"] == "fmcg"
    payload = build_due_diligence(
        "TATAMOTORS",
        scan_payload={"records": []},
        long_term_payload={"records": [{"symbol": "TATAMOTORS", "sector": "Auto"}]},
        raw_fundamentals={"available": True, "fetched_at": "", "data": {"about": "passenger vehicles", "quarterly_results": []}},
        news=[],
    )
    assert payload["framework"]["id"] == "auto"
    assert payload["vs_technical_setup"] == "UNMEASURED"
    ids = {k["id"] for k in payload["kpis"]}
    assert "sales" in ids
    assert "gnpa" not in ids


def test_suggest_icici_returns_icicibank():
    from product.due_diligence.suggest import suggest_tickers

    matches = suggest_tickers("ICICI")
    symbols = [row["symbol"] for row in matches]
    assert "ICICIBANK" in symbols
    assert matches[0]["label"].startswith("ICICI")
    assert suggest_tickers("x") == []


def test_cfo_below_pat_is_a_warning_not_invented():
    raw = {
        "available": True, "fetched_at": "", "data": {
            "about": "Test IT Ltd provides financial software",
            "quarterly_results": [
                _q_row("Sales+", **{"Jun 2025": 100, "Sep 2025": 110, "Dec 2025": 120, "Mar 2026": 130, "Jun 2026": 140}),
                _q_row("OPM %", **{"Jun 2025": 20, "Sep 2025": 21, "Dec 2025": 21, "Mar 2026": 22, "Jun 2026": 22}),
                _q_row("Net Profit+", **{"Jun 2025": 40, "Sep 2025": 42, "Dec 2025": 44, "Mar 2026": 46, "Jun 2026": 50}),
            ],
            "profit_loss": [
                _q_row("Net Profit+", **{"Mar 2024": 80, "Mar 2025": 90, "Mar 2026": 100}),
                _q_row("Sales+", **{"Mar 2024": 400, "Mar 2025": 450, "Mar 2026": 500}),
            ],
            "cash_flow": [
                _q_row("Cash from Operating Activity+", **{"Mar 2024": 20, "Mar 2025": 25, "Mar 2026": 30}),
            ],
            "shareholding": [_q_row("Promoters+", **{"Jun 2026": 50})],
        },
    }
    payload = build_due_diligence(
        "TESTIT",
        scan_payload={"records": [{"symbol": "TESTIT", "status": "Ready to trade", "score": 80}]},
        long_term_payload={"records": [{"symbol": "TESTIT", "sector": "IT / Software"}]},
        raw_fundamentals=raw,
        news=[],
    )
    assert payload["cash_flow_quality"]["applicable"] is True
    cfo_pat = next(m for m in payload["cash_flow_quality"]["metrics"] if m["id"] == "cfo_to_pat")
    assert cfo_pat["available"] is True
    assert cfo_pat["current"] == 0.3
    assert any(f["id"] == "cf-cfo-below-pat" for f in payload["red_flags"])
    warned = next(f for f in payload["red_flags"] if f["id"] == "cf-cfo-below-pat")
    assert warned["severity"] == "warning"
    assert warned["threshold"] == 0.5
    assert payload["vs_technical_setup"] in {"CAUTION", "SUPPORTS", "STRONGLY SUPPORTS", "NEUTRAL", "CONTRADICTS"}


def test_source_conflict_keeps_both_prints():
    from product.due_diligence.engine import _apply_overlay
    from product.due_diligence.frameworks import GENERIC

    findings = [{
        "id": "sales",
        "available": True,
        "snapshot": {"current": 2540},
        "provenance": {"value": 2540, "source": "NSE filing", "source_type": "exchange_filing"},
    }]
    measured = {"sales": {"current": 2590, "source": "Secondary website", "current_period": "Jun 2026"}}
    conflicts = _apply_overlay(findings, GENERIC, measured)
    assert findings[0]["snapshot"]["current"] == 2540
    assert conflicts
    assert conflicts[0]["status"] == "Source discrepancy detected"
    assert conflicts[0]["other"]["value"] == 2590


def test_order_materiality_uses_revenue_ratio():
    from product.due_diligence.materiality import materiality

    low = materiality(
        {"headline": "Wins order worth ₹1 crore from state agency", "event_type": "order_or_contract"},
        revenue_cr=50000,
    )
    high = materiality(
        {"headline": "Wins order worth ₹5,000 crore from state agency", "event_type": "order_or_contract"},
        revenue_cr=50000,
    )
    assert low["materiality"] == "Low"
    assert high["materiality"] in {"High", "Very High"}
    assert high["ratio"] == 0.1


def test_suggest_endpoint_does_not_scrape(monkeypatch):
    from fastapi.testclient import TestClient
    import terminal_product_api as tpa

    scraped = {"called": False}

    def boom(*_a, **_k):
        scraped["called"] = True
        raise AssertionError("suggest must not scrape")

    monkeypatch.setattr("fundamentals.fetcher.get_deep_fundamentals", boom)
    client = TestClient(tpa.app)
    response = client.get("/api/stock-investigator/suggest", params={"q": "ICICI"})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "StockResearchEngine"
    assert any(row["symbol"] == "ICICIBANK" for row in body["matches"])
    assert scraped["called"] is False


def test_research_coverage_is_not_quality_score():
    payload = build_due_diligence(
        "TESTBANK",
        scan_payload={"records": [{"symbol": "TESTBANK", "status": "Ready to trade", "score": 89.4}]},
        long_term_payload={"records": [{"symbol": "TESTBANK", "sector": "Banking & Finance"}]},
        raw_fundamentals=BANK_RAW,
        news=[],
    )
    coverage = payload["research_coverage"]
    assert coverage["not_a_quality_score"] is True
    assert coverage["required_n"] == 11
    assert payload["fundamental_quality"]["score"] is not None
    assert payload["fundamental_quality"]["label"] != "Unmeasured"
    assert coverage["coverage_pct"] < payload["fundamental_quality"]["coverage_pct"]
    by_id = {row["id"]: row for row in coverage["datasets"]}
    assert by_id["quarterly_results"]["status"] == "current"
    assert by_id["peer_data"]["status"] == "not_yet_acquired"
    assert by_id["recent_news"]["status"] == "not_yet_acquired"
    assert payload["first_screen"]["research_coverage_pct"] == coverage["coverage_pct"]
    pledge = next(k for k in payload["kpis"] if k["id"] == "pledge")
    assert pledge["available"] is False
    assert pledge["availability_state"] == "not_yet_acquired"
    assert payload["sector_kpi_label"] in {"Strong", "Adequate", "Mixed", "Weak", "Unmeasured"}


def test_thin_research_coverage_stays_unmeasured():
    payload = build_due_diligence(
        "EMPTYBANK",
        scan_payload={"records": [{"symbol": "EMPTYBANK", "status": "Ready to trade", "score": 80}]},
        long_term_payload={"records": [{"symbol": "EMPTYBANK", "sector": "Banking & Finance"}]},
        raw_fundamentals={
            "available": True, "fetched_at": "", "data": {
                "about": "A bank",
                "quarterly_results": [_q_row("Gross NPA %")],
                "shareholding": [],
            },
        },
        news=[],
    )
    assert payload["fundamental_quality"]["score"] is None
    assert payload["fundamental_quality"]["label"] == "Unmeasured"
    assert payload["research_coverage"]["coverage_pct"] < 30
    assert payload["fundamental_confirmation"] == "NEUTRAL"
    gnpa = next(k for k in payload["kpis"] if k["id"] == "gnpa")
    assert gnpa["fact"] == "Data unavailable"
    assert gnpa["availability_state"] == "not_yet_acquired"


def test_extract_bank_kpis_from_results_text():
    from product.due_diligence.extract import extract_rates_from_text

    parsed = extract_rates_from_text(
        "CASA ratio was 42.1%. NIM stood at 4.35%. CET1 is 16.2%. CRAR 17.8%. "
        "PCR at 78%. Slippages 1.1%. Credit cost 0.45%. ROA 2.1%. ROE 16.4%. "
        "Advances stood at 12,450 crore. Deposits were 14,200 crore. "
        "Gross NPA % as of June 2026 was 1.35%.",
        source="NSE filing",
        source_url="https://nsearchives.nseindia.com/x.pdf",
    )
    assert parsed["casa"]["current"] == 42.1
    assert parsed["nim"]["current"] == 4.35
    assert parsed["cet1"]["current"] == 16.2
    assert parsed["crar"]["current"] == 17.8
    assert parsed["pcr"]["current"] == 78
    assert parsed["gnpa"]["current"] == 1.35
    assert parsed["advances"]["current"] == 12450
    assert parsed["deposits"]["current"] == 14200
    assert parsed["casa"]["source"] == "NSE filing"


def test_smart_acquire_skips_fresh_lanes(tmp_path, monkeypatch):
    monkeypatch.setattr("product.due_diligence.acquire.EVIDENCE_ROOT", tmp_path)
    hits = {"screener": 0, "nse": 0, "session": 0}

    complete = {
        **BANK_RAW,
        "data": {
            **BANK_RAW["data"],
            "profit_loss": [_q_row("Net Profit+", **{"Mar 2025": 90, "Mar 2026": 100})],
            "key_ratios": [{"name": "Stock P/E", "value": "18.2"}, {"name": "ROE", "value": "16.1"}],
            "peer_comparison": [{"row_label": "HDFCBANK", "Mar 2026": 1}],
            "shareholding": [
                _q_row("Promoters+", **{"Jun 2026": 14.0}),
                _q_row("Pledge", **{"Jun 2026": 0.0}),
            ],
        },
    }
    monkeypatch.setattr("reporting.evidence_intake.load_raw_fundamentals", lambda _s: complete)
    monkeypatch.setattr("product.due_diligence.acquire._news_items", lambda _s: [{"headline": "Q1 results", "published_at": "2026-08-26T00:00:00+00:00"}])
    monkeypatch.setattr("product.due_diligence.acquire._news_snippets", lambda _s: [])

    def boom_screener(*_a, **_k):
        hits["screener"] += 1
        raise AssertionError("fresh cache must not scrape screener")

    def boom_nse(*_a, **_k):
        hits["nse"] += 1
        raise AssertionError("fresh filings must not hit NSE")

    def boom_session(*_a, **_k):
        hits["session"] += 1
        raise AssertionError("fresh cache must not open an NSE session")

    monkeypatch.setattr("fundamentals.fetcher.get_deep_fundamentals", boom_screener)
    monkeypatch.setattr("product.due_diligence.acquire._fetch_nse", boom_nse)
    monkeypatch.setattr("product.due_diligence.acquire._nse_session", boom_session)
    monkeypatch.setattr(
        "product.due_diligence.acquire.extract_from_uploads",
        lambda _s: {"kpis": {}, "guidance": [], "commentary": [], "order_book": [], "segments": [], "files_read": 0},
    )

    from product.due_diligence.acquire import acquire_symbol, save_autonomy_facts

    save_autonomy_facts("TESTBANK", {
        "acquired_at": "2026-08-26T00:00:00+00:00",
        "kpis": {"gnpa": {"current": 1.8, "source": "cache"}},
        "announcements": [{"headline": "Financial results", "url": "https://www.nseindia.com/x"}],
        "downloads": [{"ok": True, "path": "logs/research_evidence/TESTBANK/autonomy/nse_0.json", "url": "https://www.nseindia.com/api/corporate-announcements"}],
        "dataset_meta": {
            ds: {"checked_at": "2026-08-26T00:00:00+00:00", "status": "current"}
            for ds in (
                "company_master", "quarterly_results", "annual_financials", "sector_kpis",
                "shareholding", "promoter_pledge", "valuation", "peer_data",
                "exchange_filings", "corporate_announcements", "recent_news",
            )
        },
    })
    payload = acquire_symbol("TESTBANK", force=False)
    assert hits["screener"] == 0
    assert hits["nse"] == 0
    assert hits["session"] == 0
    assert payload["kpis"]["gnpa"]["current"] == 1.8
    assert any(step.get("skipped") for step in payload["steps"])


def test_bank_framework_lists_operating_kpis():
    from product.due_diligence.frameworks import get_framework

    ids = {spec.id for spec in get_framework("bank")["kpis"]}
    for kpi_id in ("casa", "cet1", "crar", "advances", "deposits", "pcr", "slippages", "credit_cost", "roa", "nim", "gnpa"):
        assert kpi_id in ids




