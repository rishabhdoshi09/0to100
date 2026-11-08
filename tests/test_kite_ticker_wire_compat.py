"""KiteTicker ↔ installed autobahn/Twisted wire compatibility (loopback only).

kiteconnect pins ``autobahn[twisted]`` exactly, and that pin carries known
advisories, so any autobahn change is a release decision that needs evidence
the real KiteTicker still works. This drives the UNMODIFIED KiteTicker against
a local autobahn WebSocket server speaking Kite's binary tick protocol:
handshake, subscribe/mode JSON, binary tick parsing and clean shutdown. It
also pins the property the advisory triage relies on: the client never offers
permessage-deflate.

Market data only — no broker endpoint, credential or order API is touched.
The Twisted reactor is global and cannot be restarted, so the exchange runs
in a child interpreter.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("kiteconnect")
pytest.importorskip("autobahn.twisted.websocket")

_CHILD = textwrap.dedent(
    r"""
    import json, struct, sys
    from twisted.internet import reactor
    from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol
    from kiteconnect import KiteTicker
    import autobahn

    TOKEN = 738561          # NSE segment (token & 0xff == 1)
    LTP_PAISE = 295055
    seen = {"client_messages": [], "offered_extensions": None, "headers": {}}
    result = {"ticks": [], "connected": False, "autobahn": autobahn.__version__}

    def ltp_frame():
        packet = struct.pack(">II", TOKEN, LTP_PAISE)
        return struct.pack(">H", 1) + struct.pack(">H", len(packet)) + packet

    class Server(WebSocketServerProtocol):
        def onConnect(self, request):
            seen["offered_extensions"] = [str(e) for e in (request.extensions or [])]
            seen["headers"] = {k.lower(): v for k, v in (request.headers or {}).items()}

        def onMessage(self, payload, isBinary):
            seen["client_messages"].append(json.loads(payload.decode("utf-8")))
            if len(seen["client_messages"]) == 2:
                self.sendMessage(ltp_frame(), isBinary=True)

    factory = WebSocketServerFactory()
    factory.protocol = Server
    port = reactor.listenTCP(0, factory, interface="127.0.0.1")

    kt = KiteTicker("compat_test_key", "compat_test_token",
                    root="ws://127.0.0.1:%d" % port.getHost().port,
                    reconnect=False, connect_timeout=5)

    def on_connect(ws, response):
        result["connected"] = True
        ws.subscribe([TOKEN])
        ws.set_mode(ws.MODE_LTP, [TOKEN])

    def on_ticks(ws, ticks):
        result["ticks"].extend(ticks)
        ws.close()
        reactor.callLater(0.2, reactor.stop)

    kt.on_connect = on_connect
    kt.on_ticks = on_ticks
    reactor.callLater(10, reactor.stop)
    kt.connect(threaded=False)
    result.update(seen)
    print("RESULT " + json.dumps(result))
    """
)


def _exchange() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=60,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"child produced no result (rc={proc.returncode}): {proc.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT "):])


@pytest.fixture(scope="module")
def exchange() -> dict:
    return _exchange()


def test_ticker_handshake_carries_kite_protocol_headers(exchange):
    assert exchange["connected"] is True
    assert exchange["headers"].get("x-kite-version") == "3"


def test_ticker_subscribe_and_mode_messages_reach_server(exchange):
    assert exchange["client_messages"][:2] == [
        {"a": "subscribe", "v": [738561]},
        {"a": "mode", "v": ["ltp", [738561]]},
    ]


def test_binary_ltp_tick_is_parsed_exactly(exchange):
    assert exchange["ticks"] == [{
        "tradable": True,
        "mode": "ltp",
        "instrument_token": 738561,
        "last_price": 2950.55,
    }]


def test_client_never_offers_permessage_deflate(exchange):
    # CVE-2026-77528 (GHSA-hxp9-w8x3-p566) is a maxMessagePayloadSize bypass
    # that needs permessage-deflate to be negotiated. The Kite client must not
    # offer it; if a kiteconnect/autobahn change starts offering compression,
    # the advisory triage in docs/audits must be redone.
    assert exchange["offered_extensions"] == []
