"""Broker-auth provider outages must not masquerade as paper market-data outages."""
from __future__ import annotations

from research.autonomy import auth as AUTH
from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS


class _BrokerUnavailable:
    def auth_health(self):
        return AUTH.AuthHealth(
            AUTH.PROVIDER_UNAVAILABLE,
            "2026-09-13T10:00:00+00:00",
            error_code="TIMEOUTERROR",
            reason="Zerodha profile probe timed out",
        )


def test_broker_provider_outage_does_not_block_paper_mode():
    assert H.capabilities(
        {H.BROKER_PROVIDER_UNAVAILABLE}, live_authorized=False
    )["new_paper_entries"] == H.ALLOWED


def test_broker_provider_outage_blocks_entries_when_live_is_authorized():
    assert H.capabilities(
        {H.BROKER_PROVIDER_UNAVAILABLE}, live_authorized=True
    )["new_paper_entries"] == H.BLOCKED


def test_auth_probe_provider_failure_is_not_promoted_to_market_data_failure():
    result = JOBS.run_auth_health(JOBS._Ctx(_BrokerUnavailable()))

    assert result.status == JS.RETRYABLE_FAILED
    assert H.BROKER_PROVIDER_UNAVAILABLE in result.failures
    assert H.PROVIDER_UNAVAILABLE not in result.failures


def test_real_market_data_provider_failure_still_blocks_paper():
    assert H.capabilities(
        {H.PROVIDER_UNAVAILABLE}, live_authorized=False
    )["new_paper_entries"] == H.BLOCKED
