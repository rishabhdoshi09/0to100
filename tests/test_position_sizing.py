from research.intelligence.runtime.position_sizing import size_long_cash


def test_position_sizing_uses_percentage_points_and_house_cap():
    quarter = size_long_cash(
        capital=100_000,
        entry=100,
        stop=90,
        requested_risk_pct=0.25,
        max_risk_fraction=0.01,
        max_position_fraction=0.10,
    )
    oversized = size_long_cash(
        capital=100_000,
        entry=100,
        stop=90,
        requested_risk_pct=5.0,
        max_risk_fraction=0.01,
        max_position_fraction=0.10,
    )

    assert quarter.ok is True
    assert quarter.quantity == 25
    assert quarter.risk_amount == 250
    assert oversized.ok is True
    assert oversized.capped_risk_pct == 1.0
    assert oversized.quantity == 100


def test_exact_target_quantity_is_revalidated():
    accepted = size_long_cash(
        capital=100_000,
        entry=100,
        stop=90,
        requested_risk_pct=0.5,
        max_risk_fraction=0.01,
        max_position_fraction=0.10,
        requested_quantity=30,
    )
    refused = size_long_cash(
        capital=100_000,
        entry=100,
        stop=90,
        requested_risk_pct=0.5,
        max_risk_fraction=0.01,
        max_position_fraction=0.10,
        requested_quantity=60,
    )

    assert accepted.ok is True
    assert accepted.quantity == 30
    assert accepted.risk_amount == 300
    assert refused.ok is False
    assert refused.reason_code == "QUANTITY_EXCEEDS_APPROVED_LIMIT"


def test_position_sizing_distinguishes_risk_and_concentration_failures():
    risk_too_small = size_long_cash(
        capital=1_000,
        entry=1_000,
        stop=1,
        requested_risk_pct=0.25,
        max_risk_fraction=0.01,
        max_position_fraction=1.0,
    )
    position_cap_too_small = size_long_cash(
        capital=1_000,
        entry=2_000,
        stop=1_990,
        requested_risk_pct=1.0,
        max_risk_fraction=0.01,
        max_position_fraction=0.10,
    )

    assert risk_too_small.reason_code == "RISK_BUDGET_TOO_SMALL"
    assert position_cap_too_small.reason_code == "POSITION_CAP_TOO_SMALL"


def test_invalid_inputs_fail_closed():
    result = size_long_cash(
        capital=100_000,
        entry=100,
        stop=90,
        requested_risk_pct=0,
        max_risk_fraction=0.01,
        max_position_fraction=0.10,
    )

    assert result.ok is False
    assert result.quantity == 0
    assert result.reason_code == "NON_POSITIVE_RISK"
