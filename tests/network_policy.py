"""
Hard network policy for the canonical suite.

The canonical suite used to pass in two different worlds. On a machine with no
route to NSE the downloads failed and the code fell back; on CI, with real
egress, the same tests pulled 410 index CSVs off the exchange into the checkout.
Both runs were green. A gate that behaves differently depending on whether the
exchange answers is not a deterministic gate, and its green tells you nothing
about the code.

So the canonical suite forbids the network outright:

    pytest                        NETWORK = FORBIDDEN
    pytest -m integration         controlled external dependencies
    pytest -m live_source         real-source validation

The ATTEMPT is the violation, not the failure. A module that reaches for
api.nseindia.com and swallows the error is still an uncontrolled external
dependency — it just happens to be one that hides when the network is down.
Every attempt is therefore recorded and fails the test that made it, whether or
not the calling code noticed.

Loopback and AF_UNIX stay open: a test that talks to a server it started itself
is hermetic. Everything else raises before a single byte leaves the machine.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

_LOCAL_HOSTS = {
    "127.0.0.1", "::1", "localhost", "localhost.localdomain",
    "0.0.0.0", "::", "",
}

# Markers that buy a test the right to leave the machine.
ALLOWED_MARKERS = ("network", "integration", "live_source")


class NetworkAccessDenied(RuntimeError):
    """Raised instead of opening a socket to anything outside the machine."""


@dataclass(frozen=True)
class Attempt:
    host: str
    port: object
    via: str

    def __str__(self) -> str:
        where = f"{self.host}:{self.port}" if self.port not in (None, "") else str(self.host)
        return f"{where} (via {self.via})"


_attempts: list[Attempt] = []
_allowed = False
_installed = False

_REAL = {}

# Proxy variables the HTTP clients in this stack honour.
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
    "NO_PROXY", "no_proxy",
)

#: Loopback endpoints that are NOT hermetic because they forward off the box.
#: This is not hypothetical: the environment this policy was written in runs an
#: agent proxy on 127.0.0.1, so allowing loopback unconditionally would have let
#: every blocked request reach the real internet through it while the guard
#: reported a clean run.
_PROXY_ENDPOINTS: set[tuple[str, int]] = set()
_SAVED_PROXY_ENV: dict[str, str] = {}


def _capture_proxies() -> None:
    """Record every configured proxy endpoint, then take the configuration away.

    Removing the variables stops urllib/requests/httpx from routing through a
    proxy at all; denying the endpoints catches anything that hardcodes one.
    """
    _PROXY_ENDPOINTS.clear()
    _SAVED_PROXY_ENV.clear()
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        _SAVED_PROXY_ENV[name] = value
        if name.lower().startswith("no_"):
            continue
        parsed = urlsplit(value if "://" in value else f"http://{value}")
        if parsed.hostname:
            _PROXY_ENDPOINTS.add((parsed.hostname, int(parsed.port or 80)))
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)
    # Belt and braces for clients that read a system proxy configuration.
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _restore_proxies() -> None:
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)
    os.environ.update(_SAVED_PROXY_ENV)
    _PROXY_ENDPOINTS.clear()
    _SAVED_PROXY_ENV.clear()


def _is_proxy_endpoint(address) -> bool:
    if not isinstance(address, (tuple, list)) or len(address) < 2:
        return False
    host = str(address[0])
    try:
        port = int(address[1])
    except (TypeError, ValueError):
        return False
    return (host, port) in _PROXY_ENDPOINTS


def _host_of(address) -> str:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return str(address)


def _is_local(address, family=None) -> bool:
    if _is_proxy_endpoint(address):
        # Loopback, but it forwards off the machine. Not hermetic.
        return False
    if family is not None and family == getattr(socket, "AF_UNIX", object()):
        return True
    if isinstance(address, (str, bytes)):
        # A bare path is an AF_UNIX endpoint.
        return True
    host = _host_of(address)
    if host in _LOCAL_HOSTS:
        return True
    return host.startswith("127.")


def _deny(address, via: str, family=None):
    """Record the attempt, then refuse it."""
    if _allowed or _is_local(address, family):
        return False
    attempt = Attempt(host=_host_of(address),
                      port=address[1] if isinstance(address, (tuple, list)) and len(address) > 1 else None,
                      via=via)
    _attempts.append(attempt)
    raise NetworkAccessDenied(
        f"canonical tests may not reach the network: {attempt}. "
        "Stub the source, or mark the test with @pytest.mark.live_source and "
        "move it out of the canonical gate."
    )


class OfflineError(Exception):
    """Base for the offline failures the canonical suite injects."""


def install_http_stub() -> None:
    """Make every HTTP fetch fail the same way, everywhere, every run.

    The socket guard alone would leave the suite's behaviour depending on
    HOW a fetch fails: a refused connection here, a DNS error there, a
    timeout on a slow runner. The product's fallback paths are written to
    catch those, so the assertions would still pass — but for a slightly
    different reason on every machine, which is how the original defect
    stayed invisible.

    Failing at the adapter layer instead makes "the upstream is not
    reachable" a fixed, identical fact for the canonical run, while keeping
    the exception types the production code already expects. Anything that
    bypasses this and reaches a real socket still trips the guard and fails
    its test.
    """
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from requests.exceptions import ConnectionError as RequestsConnectionError
    except Exception:  # pragma: no cover - requests is a hard dependency
        requests = None
    else:
        if "http_adapter_send" not in _REAL:
            _REAL["http_adapter_send"] = HTTPAdapter.send

            def send(self, request, *args, **kwargs):
                if _allowed:
                    return _REAL["http_adapter_send"](self, request, *args, **kwargs)
                url = getattr(request, "url", "")
                _note_stubbed(url, "requests")
                raise RequestsConnectionError(
                    f"canonical tests are offline: refused {url}"
                )

            HTTPAdapter.send = send

    import urllib.request
    from urllib.error import URLError

    # OpenerDirector.open, not urlopen: feedparser and friends build their own
    # opener, which bypasses the module-level function entirely. Patching the
    # director covers urlopen (which uses one) and every custom opener.
    if "opener_open" not in _REAL:
        _REAL["opener_open"] = urllib.request.OpenerDirector.open

        def opener_open(self, fullurl, *args, **kwargs):
            if _allowed:
                return _REAL["opener_open"](self, fullurl, *args, **kwargs)
            target = str(getattr(fullurl, "full_url", fullurl))
            if target.startswith(("file:", "data:")):
                return _REAL["opener_open"](self, fullurl, *args, **kwargs)
            _note_stubbed(target, "urllib")
            raise URLError(f"canonical tests are offline: refused {target}")

        urllib.request.OpenerDirector.open = opener_open


def uninstall_http_stub() -> None:
    if "http_adapter_send" in _REAL:
        from requests.adapters import HTTPAdapter

        HTTPAdapter.send = _REAL.pop("http_adapter_send")
    if "opener_open" in _REAL:
        import urllib.request

        urllib.request.OpenerDirector.open = _REAL.pop("opener_open")


_stubbed: list[str] = []


def _note_stubbed(url: str, via: str) -> None:
    _stubbed.append(f"{url} (via {via})")


def take_stubbed() -> list[str]:
    """Fetches the canonical run refused. Diagnostic, not a failure."""
    global _stubbed
    drained, _stubbed = _stubbed, []
    return drained


def install() -> None:
    """Patch the socket layer. Every HTTP client in the stack bottoms out here."""
    global _installed
    if _installed:
        return
    _capture_proxies()
    _REAL["connect"] = socket.socket.connect
    _REAL["connect_ex"] = socket.socket.connect_ex
    _REAL["create_connection"] = socket.create_connection
    _REAL["getaddrinfo"] = socket.getaddrinfo

    def connect(self, address, *args, **kwargs):
        _deny(address, "socket.connect", getattr(self, "family", None))
        return _REAL["connect"](self, address, *args, **kwargs)

    def connect_ex(self, address, *args, **kwargs):
        _deny(address, "socket.connect_ex", getattr(self, "family", None))
        return _REAL["connect_ex"](self, address, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        _deny(address, "socket.create_connection")
        return _REAL["create_connection"](address, *args, **kwargs)

    def getaddrinfo(host, port, *args, **kwargs):
        _deny((host, port), "socket.getaddrinfo")
        return _REAL["getaddrinfo"](host, port, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    install_http_stub()
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    socket.socket.connect = _REAL["connect"]
    socket.socket.connect_ex = _REAL["connect_ex"]
    socket.create_connection = _REAL["create_connection"]
    socket.getaddrinfo = _REAL["getaddrinfo"]
    uninstall_http_stub()
    _restore_proxies()
    _installed = False


def allow(enabled: bool) -> None:
    """Open the gate for a test that legitimately validates a real source."""
    global _allowed
    _allowed = enabled


def is_open() -> bool:
    """Whether the gate is currently open for the running test."""
    return _allowed


def failure_message(attempts) -> str:
    """The message a violating test fails with."""
    unique = sorted({str(a) for a in attempts})
    return (
        "test reached for the network in the canonical suite:\n  "
        + "\n  ".join(unique)
        + "\n\nThe attempt is the violation: code that swallows the failure is "
        "still an uncontrolled external dependency, it just hides when the "
        "network is down. Stub the source, or mark the test "
        "@pytest.mark.live_source and run it in the live-source gate."
    )


def take_attempts() -> list[Attempt]:
    """Drain and return everything attempted since the last drain."""
    global _attempts
    drained, _attempts = _attempts, []
    return drained
