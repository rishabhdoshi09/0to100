"""Paper trading must not be frozen by a daily-expiring broker token.

Zerodha access tokens expire every day. AUTH_MISSING/AUTH_EXPIRED sat in the
paper-entry block set, so from the moment a token lapsed until someone logged
in by hand the desk took no paper entries -- and therefore settled no outcomes,
accumulated no forward evidence, and learned nothing. The paper autopilot never
calls the broker (it prices fills from the bhavcopy-derived scan) and the
live-execution interlock makes a real order structurally impossible, so the
gate protected nothing while silently stopping the whole evidence chain.
"""
from __future__ import annotations

from research.autonomy import health as H


def _entries(failures, **kw):
    return H.capabilities(set(failures), **kw)["new_paper_entries"]


def test_missing_broker_token_does_not_stop_paper_entries():
    assert _entries({H.AUTH_MISSING}, live_authorized=False) == H.ALLOWED


def test_expired_broker_session_does_not_stop_paper_entries():
    assert _entries({H.AUTH_EXPIRED}, live_authorized=False) == H.ALLOWED


def test_broker_state_still_blocks_when_live_execution_is_authorized():
    """The gate is about placing real orders, so it must return under live."""
    assert _entries({H.AUTH_MISSING}, live_authorized=True) == H.BLOCKED
    assert _entries({H.AUTH_EXPIRED}, live_authorized=True) == H.BLOCKED


def test_data_trustworthiness_still_blocks_paper_entries():
    """These are the real prerequisites for a defensible paper fill."""
    for failure in (H.PROVIDER_UNAVAILABLE, H.SNAPSHOT_STALE, H.EVENT_STORE_FAILURE,
                    H.RISK_GOVERNOR_UNHEALTHY, H.UNRECONCILED):
        assert _entries({failure}, live_authorized=False) == H.BLOCKED, failure


def test_owner_pause_is_still_respected():
    assert _entries({H.OWNER_PAUSED}, live_authorized=False) == H.BLOCKED


def test_clean_paper_desk_allows_entries():
    assert _entries(set(), live_authorized=False) == H.ALLOWED


def test_default_reads_the_interlock_and_the_locked_desk_allows_paper():
    """With no explicit flag the matrix consults the interlock, which is locked."""
    assert H.live_execution_authorized() is False
    assert _entries({H.AUTH_MISSING}) == H.ALLOWED


def test_auth_note_no_longer_claims_paper_is_paused():
    notes = " ".join(H.capabilities({H.AUTH_MISSING}, live_authorized=False)["notes"])
    assert "Paper entries continue" in notes
