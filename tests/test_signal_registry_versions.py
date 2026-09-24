import json

import pytest

from scan import signal_registry as SR
from scan import signal_registry_versions as SV


def test_canonical_manifest_matches_registry_truth():
    row = SV.canonical_manifest()
    assert row["registry_version"] == SR.registry_version()
    assert row["signal_ids"] == list(SR.signal_ids())
    assert row["definitions"] == SR.SIGNAL_DEFINITIONS


def test_version_persistence_is_immutable_and_idempotent(tmp_path):
    first = SV.persist_canonical_version(directory=tmp_path)
    second = SV.persist_canonical_version(directory=tmp_path)
    assert first["registry_version"] == second["registry_version"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_collision_fails_closed(tmp_path):
    manifest = SV.canonical_manifest()
    target = tmp_path / f"{manifest['registry_version']}.json"
    target.write_text(json.dumps({**manifest, "definitions": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="collision"):
        SV.persist_canonical_version(directory=tmp_path)


def test_load_version_rejects_bad_or_mismatched_identity(tmp_path):
    with pytest.raises(ValueError, match="invalid"):
        SV.load_version("../current", directory=tmp_path)
    manifest = SV.canonical_manifest()
    target = tmp_path / f"{manifest['registry_version']}.json"
    target.write_text(json.dumps({**manifest, "registry_version": "deadbeef"}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        SV.load_version(manifest["registry_version"], directory=tmp_path)
