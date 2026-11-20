"""The evidence that admits a BUY is bound to its statistical RESULT, not only its metadata (cross-review C-2).

The alpha ledger used to bind a look to report metadata (code, data span,
family size and membership), so a report whose rows were edited afterwards,
for example a failed combination flipped to family_significant, still
verified. Now a look is FINALISED to the digest of the result it produced,
the gate recomputes that digest from the rows it is about to use, and both
must match the ledger.
"""
from __future__ import annotations

import json
import uuid

import pytest

from research import alpha_ledger as AL
from scan import signal_backtest as SB
from scan import unified_scanner as US
from tests.evidence_fixtures import proven_provenance


def test_a_finalised_matching_result_certifies():
    assert US.evidence_provenance_note(proven_provenance()) == ""


def test_rows_changed_after_finalisation_certify_nothing():
    prov = proven_provenance()
    prov["result_sha256_recomputed"] = "0" * 64              # the rows now digest to something else
    assert "differ from the result" in US.evidence_provenance_note(prov)


def test_a_report_without_a_result_digest_certifies_nothing():
    prov = proven_provenance(result_sha256="", result_sha256_recomputed="")
    assert "no finalised result digest" in US.evidence_provenance_note(prov)


def test_a_reserved_but_never_finalised_look_certifies_nothing():
    """A crash between the reservation and the saved report leaves an unfinalised look."""
    prov = proven_provenance()
    binding = {k: prov.get(k) for k in AL.BINDING_FIELDS}
    prov["alpha_accounting"] = AL.reserve(SB.gate_program(), f"crash-{uuid.uuid4().hex}", family_size=44,
                                          members_sha256="m" * 64, binding=binding)
    assert "never finalised" in US.evidence_provenance_note(prov)


def test_one_digest_cannot_vouch_for_another_result():
    prov = proven_provenance()
    look = prov["alpha_accounting"]["look_id"]
    assert AL.finalize(look, prov["result_sha256"]) is True            # idempotent for the same digest
    with pytest.raises(AL.LedgerConflict):
        AL.finalize(look, "f" * 64)
    forged = dict(prov, result_sha256="f" * 64, result_sha256_recomputed="f" * 64)
    assert "differs from the one finalised" in US.evidence_provenance_note(forged)


@pytest.mark.parametrize("field,value", [("family_significant", True), ("lower_r", 0.9),
                                         ("p_one_sided", 1e-9), ("trades", 9999), ("independent_sessions", 99)])
def test_every_admission_field_is_inside_the_digest(field, value):
    report = {"signals": {"VCP": {"trades": 40, "lower_r": -0.1, "family_significant": False,
                                  "p_one_sided": 0.4, "independent_sessions": 21}},
              "combos": {}, "evidence_provenance": {"code_sha": "x"}}
    before = SB.result_digest(report)
    edited = json.loads(json.dumps(report))
    edited["signals"]["VCP"][field] = value
    assert SB.result_digest(edited) != before
    assert SB.result_digest(json.loads(json.dumps(report))) == before     # a JSON round trip is stable


def test_a_real_walk_forward_finalises_its_look_to_the_saved_result(tmp_path, monkeypatch):
    from tests.test_alpha_ledger import _stub_walk_forward

    sb = _stub_walk_forward(monkeypatch, tmp_path)
    monkeypatch.setattr(sb, "MIN_SYMBOLS_TO_SAVE", 1)
    sb.run_backtest()
    saved = json.loads((tmp_path / "signal_backtest.json").read_text())
    prov = saved["evidence_provenance"]
    assert prov["result_sha256"] == SB.result_digest(saved)
    import sqlite3
    conn = sqlite3.connect(AL.ledger_path())
    finalised = conn.execute("SELECT result_sha256 FROM looks WHERE look_id=?",
                             (prov["alpha_accounting"]["look_id"],)).fetchone()[0]
    assert finalised == prov["result_sha256"]
    from scan.calibration_snapshot import _provenance_with_result_check
    assert _provenance_with_result_check(saved)["result_sha256_recomputed"] == prov["result_sha256"]
