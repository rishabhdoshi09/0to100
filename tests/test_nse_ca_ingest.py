"""Tests for NSE corporate-action subject parsing (no invented factors)."""
from data.nse_ca_ingest import parse_share_factor, rows_from_nse_payload, ADJUSTMENT_POLICY


def test_bonus_factor():
    assert parse_share_factor("Bonus 1:1") == ("bonus", 2.0)
    assert parse_share_factor("Bonus 3:1") == ("bonus", 4.0)
    assert parse_share_factor("Bonus 2:5") == ("bonus", 1.4)
    assert parse_share_factor("Bonus 4:1") == ("bonus", 5.0)


def test_split_factor_spaced():
    typ, fac = parse_share_factor("Face Value Split (Sub-Division) - From Rs 10 to Rs 2")
    assert typ == "split"
    assert abs(fac - 5.0) < 1e-9


def test_split_factor_nse_slash_variants():
    # Official NSE subject strings often use Rs10/- and Re 1/-
    cases = [
        ("Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share", 10.0),
        ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share", 5.0),
        ("Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share", 2.0),
    ]
    for subject, expected in cases:
        parsed = parse_share_factor(subject)
        assert parsed is not None, subject
        typ, fac = parsed
        assert typ == "split"
        assert abs(fac - expected) < 1e-9, (subject, fac, expected)


def test_demerger_unparseable_without_ratio():
    assert parse_share_factor("Demerger") is None


def test_unparseable_returns_none():
    assert parse_share_factor("Annual General Meeting") is None
    assert parse_share_factor("Buy Back") is None
    assert parse_share_factor("Dividend - Rs 10 Per Share") is None


def test_rows_separate_dividends_from_adjusting():
    payload = [
        {"symbol": "AAA", "series": "EQ", "subject": "Bonus 1:1", "exDate": "02-Jan-2024", "isin": "INE1"},
        {"symbol": "BBB", "series": "EQ", "subject": "Dividend - Rs 2 Per Share", "exDate": "03-Jan-2024", "isin": "INE2"},
        {"symbol": "CCC", "series": "EQ", "subject": "Mystery Event", "exDate": "04-Jan-2024", "isin": "INE3"},
        {"symbol": "DDD", "series": "EQ",
         "subject": "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share",
         "exDate": "05-Jan-2024", "isin": "INE4"},
    ]
    packed = rows_from_nse_payload(payload, source_sha256="abc")
    assert len(packed["adjusting"]) == 2
    factors = {r["symbol"]: r["factor"] for r in packed["adjusting"]}
    assert factors["AAA"] == 2.0
    assert factors["DDD"] == 5.0
    assert len(packed["dividends"]) == 1
    assert packed["dividends"][0]["type"] == "dividend"
    assert ADJUSTMENT_POLICY["dividend_adjustment"] == "NONE_BY_DEFAULT"
