"""QT-PRA-078: host alerts must reach Telegram from the owner's macOS Python.

On the owner's Mac the python.org interpreter has no default CA store
(ssl.get_default_verify_paths().cafile is None): a bare urllib HTTPS call to
api.telegram.org fails with CERTIFICATE_VERIFY_FAILED, which is why the
2026-11-09 crash-loop alert was recorded as "URLError: alert delivery failed".
requests (used by scan alerts) worked because it ships certifi. Host alerts now
use a verifying context built on the same certifi bundle.
"""
from __future__ import annotations

import ssl

import product.host_alerts as HA


def test_host_alert_tls_context_verifies_with_a_loaded_ca_bundle():
    ctx = HA._https_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    # A real trust store is loaded even when the interpreter has none by default.
    assert ctx.cert_store_stats()["x509_ca"] > 50


def test_host_alert_uses_the_verifying_context_for_webhook_too(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(HA, "_configured_destinations", lambda: ("", "", "https://hooks.example.invalid/x"))
    seen = []

    class _Resp:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=0, context=None):
        seen.append(context)
        return _Resp()

    monkeypatch.setattr(HA.urllib.request, "urlopen", fake_urlopen)
    result = HA.send_operational_alert("host check")
    assert result.delivered is True and result.channel == "webhook"
    assert isinstance(seen[0], ssl.SSLContext) and seen[0].verify_mode == ssl.CERT_REQUIRED
