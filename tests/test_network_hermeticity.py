"""The canonical suite must behave identically with and without a network.

Before this policy it did not. On a machine with no route to NSE the downloads
failed and the code fell back; on CI, with real egress, the same tests pulled
410 index CSVs off the exchange into the checkout. Both runs were green, and a
gate that is green either way is measuring nothing.

The rule these tests pin: the ATTEMPT is the violation. Code that reaches for
an external host and swallows the failure is still an uncontrolled external
dependency — it just hides when the network happens to be down, which is
exactly how this defect survived for so long.
"""
from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request

import pytest

from tests import network_policy


def test_the_gate_is_closed_for_an_ordinary_test():
    assert not network_policy.is_open()


@pytest.mark.network
def test_the_marker_opens_the_gate():
    """Marked tests are also deselected from the canonical run entirely."""
    assert network_policy.is_open()


def test_an_external_connection_is_refused():
    with pytest.raises(network_policy.NetworkAccessDenied):
        socket.create_connection(("api.nseindia.com", 443), timeout=1)
    attempts = network_policy.take_attempts()
    assert any("api.nseindia.com" in str(a) for a in attempts)


def test_dns_for_an_external_host_is_refused():
    with pytest.raises(network_policy.NetworkAccessDenied):
        socket.getaddrinfo("www1.nseindia.com", 443)
    assert network_policy.take_attempts()


def test_urllib_is_refused_the_same_way_on_every_machine():
    """Deterministic offline, not 'whatever error this network happens to give'."""
    with pytest.raises(urllib.error.URLError, match="offline"):
        urllib.request.urlopen("https://api.nseindia.com/", timeout=1)
    assert any("nseindia" in row for row in network_policy.take_stubbed())


def test_requests_is_refused_with_the_exception_production_expects():
    """Production catches ConnectionError; the stub must not change that shape."""
    requests = pytest.importorskip("requests")
    with pytest.raises(requests.exceptions.ConnectionError, match="offline"):
        requests.get("https://www.nseindia.com/api/marketStatus", timeout=1)
    assert any("nseindia" in row for row in network_policy.take_stubbed())


def test_a_client_that_bypasses_the_stub_still_trips_the_guard():
    """The stub is for determinism; the socket guard is the actual boundary."""
    with pytest.raises(network_policy.NetworkAccessDenied):
        socket.create_connection(("www.nseindia.com", 443), timeout=1)
    assert network_policy.take_attempts()


def test_a_loopback_proxy_is_not_treated_as_local():
    """The hole this policy nearly shipped with.

    This environment runs an agent proxy on 127.0.0.1. Allowing loopback
    unconditionally would have let every blocked request reach the internet
    through it, with the guard reporting a clean run.
    """
    network_policy._PROXY_ENDPOINTS.add(("127.0.0.1", 33871))
    try:
        with pytest.raises(network_policy.NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", 33871), timeout=1)
        assert network_policy.take_attempts()
    finally:
        network_policy._PROXY_ENDPOINTS.discard(("127.0.0.1", 33871))


def test_proxy_configuration_is_removed_for_the_canonical_run():
    """A client that honours HTTPS_PROXY must find nothing to honour."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        assert name not in os.environ, f"{name} survived into the canonical run"
    assert os.environ.get("NO_PROXY") == "*"


def test_a_swallowed_attempt_is_still_recorded():
    """The defect class: code that catches its own network failure."""
    try:
        socket.create_connection(("api.nseindia.com", 443), timeout=1)
    except Exception:
        pass  # exactly what the production fallback paths do

    attempts = network_policy.take_attempts()
    assert attempts, "a swallowed attempt must not disappear"
    message = network_policy.failure_message(attempts)
    assert "api.nseindia.com" in message
    assert "the violation" in message


def test_loopback_stays_open():
    """A test that talks to a server it started itself is still hermetic."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    finally:
        server.close()
    assert not network_policy.take_attempts()


def test_localhost_by_name_stays_open():
    assert socket.getaddrinfo("localhost", 0)
    assert not network_policy.take_attempts()


def test_unix_sockets_stay_open(tmp_path):
    path = str(tmp_path / "s.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(path)
        client.close()
    finally:
        server.close()
    assert not network_policy.take_attempts()


def test_attempts_do_not_leak_between_tests():
    """Each test is judged on what it did, not what the one before it did."""
    assert network_policy.take_attempts() == []


# ---------------------------------------------------------------------------
# A production defect the hermetic guard exposed.
#
# news.fetcher tries requests.get(timeout=10), and on ANY exception falls back
# to feedparser.parse(url), which does its own HTTP through urllib with no
# timeout at all. The "fallback" was a second, unbounded network path through a
# different stack: a hung feed would have held the news lane open indefinitely,
# and the guard only found it because the fallback slipped past a urlopen patch
# by building its own opener.
# ---------------------------------------------------------------------------
def test_the_rss_fallback_is_bounded_and_restores_the_default_timeout():
    from news.fetcher import NewsFetcher

    assert NewsFetcher._RSS_FALLBACK_TIMEOUT_S > 0

    before = socket.getdefaulttimeout()
    seen: list[float | None] = []

    import feedparser
    import news.fetcher as fetcher_module

    original_parse = feedparser.parse

    def record_timeout(*args, **kwargs):
        seen.append(socket.getdefaulttimeout())
        raise RuntimeError("feed is hung")

    feedparser.parse = record_timeout
    try:
        try:
            NewsFetcher()._fetch_rss("https://www.livemint.com/rss/markets", 24)
        except Exception:
            pass
    finally:
        feedparser.parse = original_parse

    assert seen, "the fallback should have been reached"
    assert seen[0] == NewsFetcher._RSS_FALLBACK_TIMEOUT_S
    assert socket.getdefaulttimeout() == before, (
        "the fallback must not leave a global socket timeout behind it"
    )
    assert fetcher_module is not None
