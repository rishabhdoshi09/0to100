"""Evidence-only Growth & Financial Outlook."""
from __future__ import annotations


def test_double_engine_from_sales_and_margin(monkeypatch):
    from product import growth_outlook as GO

    monkeypatch.setattr(GO, "_guidance_rows", lambda symbol: ([], [], []))
    raw = {
        "data": {
            "profit_loss": [
                {"row_label": "Sales", "FY22": 100, "FY23": 130, "FY24": 170, "FY25": 220},
                {"row_label": "Net Profit", "FY22": 10, "FY23": 15, "FY24": 24, "FY25": 40},
                {"row_label": "Operating Profit", "FY22": 15, "FY23": 22, "FY24": 34, "FY25": 48},
            ],
            "key_ratios": [
                {"name": "OPM %", "value": "21.8%"},
                {"name": "ROE", "value": "18%"},
            ],
        }
    }
    fundamentals = {
        "metrics": [
            {"key": "sales_growth_3y", "value": 30.0},
            {"key": "profit_growth_3y", "value": 58.0},
            {"key": "roe", "value": 18.0},
            {"key": "pe", "value": 22.0},
        ],
        "key_ratios": [{"name": "OPM %", "value": "21.8%"}],
        "raw_values": {},
        "fetched_at": "2026-08-01",
        "section_as_of": {"financial_history": "2026-03-31"},
    }
    out = GO.build_growth_outlook(
        "RKFORGE",
        fundamentals=fundamentals,
        technical={"available": True, "close": 704.0, "trend": "UPTREND", "trend_explanation": "Above 20/50.", "latest_date": "2026-08-04"},
        company="Ramkrishna Forgings",
        sector="Auto Ancillaries",
        raw_fundamentals=raw,
    )
    assert out["available"] is True
    assert out["places_orders"] is False
    assert "DOUBLE ENGINE" in out["thesis"]["label"]
    by_key = {c["key"]: c for c in out["claims"]}
    assert by_key["sales_cagr_3y"]["value"] == 30.0
    assert by_key["opm"]["status"] == "AVAILABLE"
    assert any("guidance" in g.lower() or "concall" in g.lower() for g in out["gaps"])
    # Must not invent FY28 targets
    blob = str(out)
    assert "FY28" not in blob
    assert "20% CAGR by" not in blob


def test_missing_guidance_stays_missing(monkeypatch):
    from product import growth_outlook as GO

    monkeypatch.setattr(GO, "_guidance_rows", lambda symbol: ([], [], []))
    out = GO.build_growth_outlook(
        "TESTCO",
        fundamentals={"metrics": [], "key_ratios": [], "raw_values": {}},
        technical={"available": False},
        raw_fundamentals={"data": {}},
    )
    assert out["guidance"] == []
    assert any("guidance" in g.lower() or "concall" in g.lower() for g in out["gaps"])
    assert "INCOMPLETE" in out["thesis"]["label"]


def test_uploaded_guidance_is_cited(monkeypatch):
    from product import growth_outlook as GO

    mgmt = [
        {
            "event_date": "2026-05-10",
            "speaker": "CFO",
            "topic": "margin",
            "commentary": "Expect EBITDA margin near 19-20% if mix holds.",
            "guidance_metric": "EBITDA margin",
            "guidance_value": "19-20%",
            "guidance_period": "FY26",
            "source_url": "https://example.com/transcript",
            "as_of_date": "2026-05-10",
        }
    ]
    monkeypatch.setattr(GO, "_guidance_rows", lambda symbol: (mgmt, [], []))
    out = GO.build_growth_outlook(
        "RKFORGE",
        fundamentals={
            "metrics": [{"key": "sales_growth_3y", "value": 22.0}],
            "key_ratios": [],
            "raw_values": {},
        },
        technical={"available": True, "close": 700, "trend": "MIXED", "trend_explanation": "Mixed"},
        raw_fundamentals={"data": {"profit_loss": []}},
    )
    assert len(out["guidance"]) == 1
    assert "19-20%" in out["guidance"][0]["guidance_value"]
    assert not any("concall" in g.lower() and "missing" in g.lower() for g in out["gaps"])


def test_workspace_includes_growth_outlook(monkeypatch):
    from product import stock_workspace as SW

    monkeypatch.setattr(
        SW,
        "_default_inputs",
        lambda symbol: {
            "scan": {"records": [], "scanned_at": ""},
            "long_term": {"records": [], "scanned_at": ""},
            "raw": {
                "available": True,
                "fetched_at": "2026-08-01",
                "data": {
                    "profit_loss": [
                        {"row_label": "Sales", "a": 100, "b": 120, "c": 150},
                        {"row_label": "Net Profit", "a": 10, "b": 14, "c": 20},
                    ],
                    "about": "Test company",
                },
            },
            "frame": None,
            "news": [],
            "fno": {"underlyings": [], "generated_at": ""},
        },
    )
    monkeypatch.setattr(
        "product.growth_outlook._guidance_rows",
        lambda symbol: ([], [], []),
    )
    ws = SW.build_stock_workspace("TESTCO")
    assert "growth_outlook" in ws
    assert ws["growth_outlook"]["places_orders"] is False
    assert isinstance(ws["growth_outlook"].get("claims"), list)
