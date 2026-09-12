"""One machine-level fact should not be reported as four provider outages.

A container behind a deny-all proxy produced, repeatedly:

    nsearchives.nseindia.com  Tunnel connection failed: 403 Forbidden
    www.nseindia.com          Tunnel connection failed: 403 Forbidden
    api.bseindia.com          connect_rejected
    query1.finance.yahoo.com  403 Host not in allowlist

Line by line that reads as NSE, BSE and Yahoo all breaking on the same
afternoon. It is one firewall. Reporting it four times is four wrong
statements about the world, and it sends the operator looking at exchanges
instead of at their own network.
"""
from __future__ import annotations

import pytest

from data.egress import (
    AUTH_FAILURE,
    DNS_FAILURE,
    ENVIRONMENT_EGRESS_BLOCKED,
    MARKET_EGRESS_BLOCKED,
    PARSER_CHANGED,
    PROVIDER_DOWN,
    RATE_LIMITED,
    TIMEOUT,
    ProbeResult,
    classify_environment,
    classify_failure,
    retry_is_futile,
)


@pytest.mark.parametrize("error,expected", [
    ("ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden'))",
     ENVIRONMENT_EGRESS_BLOCKED),
    ("HTTP Error 403: Host not in allowlist: query2.finance.yahoo.com",
     ENVIRONMENT_EGRESS_BLOCKED),
    ("connect_rejected", ENVIRONMENT_EGRESS_BLOCKED),
    ("gaierror: Name or service not known", DNS_FAILURE),
    ("HTTPSConnectionPool: Read timed out", TIMEOUT),
    ("401 Unauthorized", AUTH_FAILURE),
    ("429 Too Many Requests", RATE_LIMITED),
    ("JSONDecodeError: Expecting value", PARSER_CHANGED),
    ("500 Internal Server Error", PROVIDER_DOWN),
])
def test_failures_are_classified_by_shape(error, expected):
    assert classify_failure(error) == expected


def test_a_proxy_refusal_outranks_the_403_that_carries_it():
    """403 usually means auth. Behind a tunnel refusal it means the tunnel."""
    assert classify_failure("Tunnel connection failed: 403 Forbidden") == (
        ENVIRONMENT_EGRESS_BLOCKED
    )


def test_status_codes_are_honoured_when_given():
    assert classify_failure("blocked", status_code=429) == RATE_LIMITED
    assert classify_failure("nope", status_code=503) == PROVIDER_DOWN


# ── the aggregate verdict ──────────────────────────────────────────────────
def _blocked(host):
    return ProbeResult(host, host, False, ENVIRONMENT_EGRESS_BLOCKED, "tunnel refused")


def test_everything_unreachable_is_one_fact_about_the_machine():
    verdict = classify_environment([
        _blocked("www.nseindia.com"), _blocked("api.bseindia.com"),
        _blocked("query1.finance.yahoo.com"),
    ])
    assert verdict.failure_class == ENVIRONMENT_EGRESS_BLOCKED
    assert verdict.egress_blocked is True
    assert retry_is_futile(verdict)


def test_two_hostnames_of_one_exchange_are_one_provider():
    """nsearchives and www are both NSE. Two of them is not corroboration."""
    verdict = classify_environment([
        _blocked("www.nseindia.com"), _blocked("nsearchives.nseindia.com"),
    ])
    assert verdict.failure_class != ENVIRONMENT_EGRESS_BLOCKED
    assert "not enough to blame the machine" in verdict.reason


def test_a_working_control_turns_it_into_a_selective_market_block():
    """The environment this was written in: internet fine, exchanges denied."""
    verdict = classify_environment([
        _blocked("www.nseindia.com"), _blocked("nsearchives.nseindia.com"),
        _blocked("api.kite.trade"),
        ProbeResult("control", "pypi.org", True),
    ], control_hosts=["pypi.org"])
    assert verdict.failure_class == MARKET_EGRESS_BLOCKED
    assert verdict.market_blocked is True
    assert verdict.egress_blocked is False, "the internet is not down"
    assert "selective egress policy" in verdict.reason
    assert retry_is_futile(verdict), "the exchanges are no more reachable for that"


def test_a_provider_side_failure_is_never_blamed_on_the_machine():
    verdict = classify_environment([
        _blocked("www.nseindia.com"), _blocked("api.bseindia.com"),
        ProbeResult("y", "query1.finance.yahoo.com", False, PROVIDER_DOWN, "500"),
    ])
    assert verdict.failure_class != ENVIRONMENT_EGRESS_BLOCKED
    assert "from the providers themselves" in verdict.reason


def test_one_reachable_provider_means_egress_works():
    verdict = classify_environment([
        ProbeResult("a", "www.nseindia.com", True),
        _blocked("api.bseindia.com"),
    ])
    assert verdict.egress_blocked is False
    assert not retry_is_futile(verdict)
    assert verdict.reachable_hosts == ("www.nseindia.com",)


def test_probing_nothing_claims_nothing():
    verdict = classify_environment([])
    assert verdict.egress_blocked is False
    assert verdict.failure_class == ""
