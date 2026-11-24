"""The storage preflight's OPT-IN stale-attachment recovery (off by default; enabling it is
the owner's decision). After a sleep the USB disk re-enumerates and the configured image is
left attached with revoked descriptors: it can never mount and re-attaching fails forever
(19, 21, 22 and 23 Nov 2026). With QT_STORAGE_PREFLIGHT_STALE_DETACH=1 the preflight clears
exactly that attachment and nothing else: a plain mount of its own volume first, a plain
detach (never -force) only if that fails, then the normal attach and identity checks.

Stateful fakes stand in for hdiutil and diskutil; every invocation is recorded.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "quantterm_storage_preflight.sh"
APFS_VOLUME = "41504653-0000-11AA-AA11-00306543ECAC"

_HDIUTIL = r'''#!/bin/sh
S="$FAKE_STATE"
echo "hdiutil $*" >> "$S/calls.log"
if [ "$1" = "info" ] && [ "$2" = "-plist" ]; then
  exec "$FAKE_PY" -c 'import json, plistlib, sys; sys.stdout.buffer.write(plistlib.dumps({"images": json.load(open(sys.argv[1]))}))' "$S/images.json"
fi
if [ "$1" = "detach" ]; then
  if [ -e "$S/detach_fails" ]; then echo "hdiutil: couldn't detach $2 - Resource busy" >&2; exit 16; fi
  "$FAKE_PY" -c 'import json, sys; p=sys.argv[1]; ims=json.load(open(p)); json.dump([i for i in ims if not any(e.get("dev-entry","").startswith(sys.argv[2]) for e in i.get("system-entities",[]))], open(p,"w"))' "$S/images.json" "$2"
  exit 0
fi
if [ "$1" = "attach" ]; then
  if "$FAKE_PY" -c 'import json, os, sys; sys.exit(0 if any(os.path.realpath(i["image-path"]) == os.path.realpath(sys.argv[2]) for i in json.load(open(sys.argv[1]))) else 1)' "$S/images.json" "$3"; then
    echo "hdiutil: attach failed - Resource temporarily unavailable" >&2; exit 16      # already (stale) attached
  fi
  "$FAKE_PY" -c 'import json, sys; p=sys.argv[1]; ims=json.load(open(p)); ims.append({"image-path": sys.argv[2], "system-entities": [{"dev-entry": "/dev/disk9"}, {"dev-entry": "/dev/disk9s1", "mount-point": sys.argv[3], "content-hint": "41504653-0000-11AA-AA11-00306543ECAC"}]}); json.dump(ims, open(p,"w"))' "$S/images.json" "$3" "$QT_STORAGE_MOUNT"
  touch "$S/mounted"
  exit 0
fi
exit 99
'''

_DISKUTIL = r'''#!/bin/sh
S="$FAKE_STATE"
echo "diskutil $*" >> "$S/calls.log"
if [ "$1" = "info" ]; then
  [ -e "$S/mounted" ] || exit 1
  printf '   Mount Point:              %s\n' "$QT_STORAGE_MOUNT"
  printf '   File System Personality:  APFS\n'
  exit 0
fi
if [ "$1" = "mount" ]; then
  [ -e "$S/mount_ok" ] || { echo "Volume on $2 failed to mount" >&2; exit 1; }
  "$FAKE_PY" -c 'import json, sys; p=sys.argv[1]; ims=json.load(open(p))
for i in ims:
    for e in i.get("system-entities", []):
        if e.get("dev-entry") == sys.argv[2]: e["mount-point"] = sys.argv[3]
json.dump(ims, open(p,"w"))' "$S/images.json" "$2" "$QT_STORAGE_MOUNT"
  touch "$S/mounted"
  exit 0
fi
exit 2
'''


def _stale(path: str, top: str = "/dev/disk3", mounted: str | None = None) -> dict:
    return {"image-path": path, "system-entities": [
        {"dev-entry": top}, {"dev-entry": top + "s1"}, {"dev-entry": top + "s2"},
        {"dev-entry": "/dev/disk4"},
        {"dev-entry": "/dev/disk4s1", "content-hint": APFS_VOLUME, **({"mount-point": mounted} if mounted else {})},
    ]}


@pytest.fixture
def layout(tmp_path):
    external = tmp_path / "Expansion"
    bundle = external / "QuantTermStorage_WORKING_20261107.sparsebundle"
    old = external / "QuantTermStorage.sparsebundle"
    mount = tmp_path / "QuantTermStorage"
    runtime = mount / "QuantTerm" / "runtime"
    canonical = tmp_path / "home" / "Library" / "Application Support" / "QuantTerm" / "runtime"
    for d in (bundle, old, runtime, canonical.parent):
        d.mkdir(parents=True)
    canonical.symlink_to(runtime)
    state, fake_bin = tmp_path / "state", tmp_path / "bin"
    state.mkdir()
    fake_bin.mkdir()
    for name, body in (("hdiutil", _HDIUTIL), ("diskutil", _DISKUTIL)):
        (fake_bin / name).write_text(body)
        (fake_bin / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}", "FAKE_STATE": str(state),
           "FAKE_PY": sys.executable, "QT_STORAGE_PREFLIGHT_PYTHON": sys.executable,
           "QT_PREFLIGHT_UNAME": "Darwin", "QT_STORAGE_PREFLIGHT_REQUIRED": "1",
           "QT_STORAGE_EXTERNAL_VOLUME": str(external), "QT_STORAGE_BUNDLE": str(bundle),
           "QT_STORAGE_MOUNT": str(mount), "QT_STORAGE_RUNTIME": str(runtime), "QT_RUNTIME_LINK": str(canonical)}
    env.pop("QT_STORAGE_PREFLIGHT_STALE_DETACH", None)
    return {"env": env, "state": state, "bundle": bundle, "old": old}


def _run(layout, images: list, *, enabled: bool, mount_ok: bool = False, detach_fails: bool = False):
    s = layout["state"]
    (s / "images.json").write_text(json.dumps(images))
    (s / "calls.log").write_text("")
    for flag, on in (("mount_ok", mount_ok), ("detach_fails", detach_fails)):
        if on:
            (s / flag).touch()
    env = dict(layout["env"])
    if enabled:
        env["QT_STORAGE_PREFLIGHT_STALE_DETACH"] = "1"
    proc = subprocess.run(["bash", str(PREFLIGHT)], env=env, capture_output=True, text=True, timeout=120)
    calls = (s / "calls.log").read_text().splitlines()
    assert not any("-force" in c for c in calls), calls                    # never, in any scenario
    return proc, calls


def _detaches(calls):
    return [c for c in calls if c.startswith("hdiutil detach")]


def test_off_by_default_a_stale_attachment_still_fails_closed(layout):
    proc, calls = _run(layout, [_stale(str(layout["bundle"]))], enabled=False)
    assert proc.returncode == 78 and "could not attach sparsebundle" in proc.stderr
    assert _detaches(calls) == []


def test_enabled_a_stale_configured_image_is_detached_then_attached(layout):
    proc, calls = _run(layout, [_stale(str(layout["bundle"]))], enabled=True)
    assert proc.returncode == 0, proc.stderr
    assert "PASS: APFS runtime verified" in proc.stdout
    assert "diskutil mount /dev/disk4s1" in calls                          # a plain mount first
    assert _detaches(calls) == ["hdiutil detach /dev/disk3"]               # then exactly that device
    assert calls.index("hdiutil detach /dev/disk3") < max(i for i, c in enumerate(calls) if c.startswith("hdiutil attach"))


def test_enabled_a_mountable_volume_is_mounted_never_detached(layout):
    proc, calls = _run(layout, [_stale(str(layout["bundle"]))], enabled=True, mount_ok=True)
    assert proc.returncode == 0, proc.stderr
    assert _detaches(calls) == [] and not any(c.startswith("hdiutil attach") for c in calls)


def test_enabled_another_image_is_never_touched(layout):
    other = _stale(str(layout["old"]), top="/dev/disk5")
    proc, calls = _run(layout, [other], enabled=True)
    assert _detaches(calls) == []                                          # only the configured image
    assert proc.returncode == 0, proc.stderr                               # the configured one attaches normally


def test_enabled_a_partly_mounted_image_is_never_detached(layout):
    elsewhere = _stale(str(layout["bundle"]), mounted="/Volumes/QuantTermStorage 1")
    proc, calls = _run(layout, [elsewhere], enabled=True)
    assert _detaches(calls) == []
    assert proc.returncode == 78                                           # fail closed, as before


def test_enabled_an_ambiguous_inventory_is_never_acted_on(layout):
    twice = [_stale(str(layout["bundle"])), _stale(str(layout["bundle"]), top="/dev/disk6")]
    proc, calls = _run(layout, twice, enabled=True)
    assert _detaches(calls) == [] and proc.returncode == 78


def test_enabled_a_failed_detach_fails_closed_with_its_reason(layout):
    proc, calls = _run(layout, [_stale(str(layout["bundle"]))], enabled=True, detach_fails=True)
    assert proc.returncode == 78 and "could not be detached" in proc.stderr and "Resource busy" in proc.stderr
    assert _detaches(calls) == ["hdiutil detach /dev/disk3"]
    assert not any(c.startswith("hdiutil attach") for c in calls)


def test_the_installed_service_does_not_enable_it():
    """Enabling is a separate owner decision: nothing in the repo turns it on."""
    import re

    offenders = []
    for path in list((ROOT / "product").rglob("*.py")) + list((ROOT / "scripts").rglob("*")) + list((ROOT / "deploy").rglob("*")):
        if path.is_file() and path.name != PREFLIGHT.name:
            text = path.read_text(errors="replace")
            if re.search(r"QT_STORAGE_PREFLIGHT_STALE_DETACH\s*[=:]\s*['\"]?1", text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
