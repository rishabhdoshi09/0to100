from scan.unified_scanner import _breakout_note_requires_wait, grade_breakout


def test_gap_exhaustion_breakout_is_machine_readable_do_not_chase():
    ok, grade, note = grade_breakout(
        price=110.0,
        level=100.0,
        atr=2.0,
        vratio=3.0,
        day_change=9.0,
        clv=0.9,
        rsi=60.0,
    )

    assert ok is False
    assert grade == ""
    assert "chase nahi" in note
    assert _breakout_note_requires_wait(note) is True


def test_hard_rsi_breakout_is_machine_readable_do_not_chase():
    ok, grade, note = grade_breakout(
        price=103.0,
        level=100.0,
        atr=2.0,
        vratio=3.0,
        day_change=2.0,
        clv=0.9,
        rsi=85.0,
    )

    assert ok is False
    assert grade == ""
    assert "chase nahi" in note
    assert _breakout_note_requires_wait(note) is True


def test_plain_confirmation_wait_is_not_promoted_to_chase_risk():
    ok, grade, note = grade_breakout(
        price=101.0,
        level=100.0,
        atr=2.0,
        vratio=0.8,
        day_change=1.0,
        clv=0.9,
        rsi=60.0,
    )

    assert ok is False
    assert grade == ""
    assert "marginal break" in note
    assert _breakout_note_requires_wait(note) is False
