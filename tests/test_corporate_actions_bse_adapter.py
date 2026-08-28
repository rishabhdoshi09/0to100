from __future__ import annotations

from data import corporate_actions_bse_adapter as BSE


def test_bse_sample_date_formats_are_supported():
    assert BSE.parse_bse_date("25 Oct 2023") == "2023-10-25"
    assert BSE.parse_bse_date("20231025") == "2023-10-25"


def test_bse_exact_short_name_mapping_and_long_name_fallback():
    maps = (
        {"INFY", "JAYBARMARU"},
        {"INE009A01021": "INFY"},
        {"INFOSYSLTD": "INFY", "JAYBHARATMARUTILTD": "JAYBARMARU"},
    )
    assert BSE.map_bse_symbol({"short_name": "INFY"}, maps) == "INFY"
    assert BSE.map_bse_symbol({"long_name": "JAY BHARAT MARUTI LTD."}, maps) == "JAYBARMARU"


def test_bse_official_sample_split_and_bonus_are_normalized(monkeypatch):
    payload = [
        {
            "scrip_code": 520066,
            "short_name": "JAYBARMARU",
            "Ex_date": "26 Oct 2023",
            "Purpose": "Stock Split From Rs.5/- to Rs.2/-",
            "RD_Date": "26 Oct 2023",
            "long_name": "JAY BHARAT MARUTI LTD.",
        },
        {
            "scrip_code": 543464,
            "short_name": "SPITZE",
            "Ex_date": "27 Oct 2023",
            "Purpose": "Bonus issue 1:1",
            "RD_Date": "27 Oct 2023",
            "long_name": "Maruti Interior Products Ltd",
        },
        {
            "scrip_code": 500209,
            "short_name": "INFY",
            "Ex_date": "25 Oct 2023",
            "Purpose": "Interim Dividend - Rs. - 18.0000",
            "RD_Date": "25 Oct 2023",
            "long_name": "INFOSYS LTD.",
        },
    ]

    class Response:
        status_code = 200
        def json(self):
            return payload

    class Session:
        headers = {}
        def get(self, *_args, **_kwargs):
            return Response()

    maps = ({"JAYBARMARU", "SPITZE", "INFY"}, {}, {})
    rows = BSE.fetch_bse_window(
        __import__("datetime").date(2023, 10, 1),
        __import__("datetime").date(2023, 10, 31),
        session=Session(),
        maps=maps,
    )

    assert [(r["symbol"], r["type"], r["factor"], r["ex_date"]) for r in rows] == [
        ("JAYBARMARU", "split", 2.5, "2023-10-26"),
        ("SPITZE", "bonus", 2.0, "2023-10-27"),
    ]
    assert all(r["source"] == "bse_api" for r in rows)
