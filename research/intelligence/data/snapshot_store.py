"""
🧊 Immutable, content-addressed NSE snapshot store.

A committed snapshot is never mutated in place — corrections create a SUCCESSOR. The snapshot id
is a deterministic hash of the normalized content plus schema/parser versions, so identical data
under identical versions commits to the same id (idempotent), and any change yields a new id.

Activation is atomic: the active pointer is swapped with `os.replace` (crash leaves the OLD or the
NEW snapshot active, never a partial state) and writes an audit line. Nothing here trades.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
PARSER_VERSION = 1


def _canonical_csv(rows, header) -> str:
    """Deterministic CSV: fixed header + rows sorted, so content addressing is stable."""
    out = [",".join(header)]
    for r in sorted(rows):
        out.append(",".join(str(x) for x in r))
    return "\n".join(out) + "\n"


class SnapshotStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else (
            Path(__file__).resolve().parents[3] / "logs" / "snapshots")
        self.root.mkdir(parents=True, exist_ok=True)

    # ── commit (immutable + content-addressed) ───────────────────────────────────
    def commit_snapshot(self, equity_rows, *, index_rows=None, parent_id="",
                        extra_manifest=None) -> str:
        """`equity_rows`: iterable of (symbol, date, open, high, low, close, volume, series).
        Returns the snapshot id. Idempotent: committing identical content returns the same id
        and does not rewrite an existing snapshot."""
        eq_header = ["symbol", "date", "open", "high", "low", "close", "volume", "series"]
        ix_header = ["name", "date", "open", "high", "low", "close"]
        eq_csv = _canonical_csv([tuple(r) for r in equity_rows], eq_header)
        ix_csv = _canonical_csv([tuple(r) for r in (index_rows or [])], ix_header)
        blob = (eq_csv + "\x1e" + ix_csv + f"\x1e{SCHEMA_VERSION}.{PARSER_VERSION}").encode()
        sid = hashlib.sha256(blob).hexdigest()[:16]
        sdir = self.root / sid
        if sdir.exists():
            return sid                                    # already committed — immutable, idempotent

        eq_rows = list(eq_csv_to_rows(eq_csv))
        ix_rows = list(eq_csv_to_rows(ix_csv))
        dates = sorted({r[1] for r in eq_rows})
        symbols = sorted({r[0] for r in eq_rows})
        manifest = {
            "snapshot_id": sid, "parent_snapshot_id": parent_id,
            "schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION,
            "equity_sha256": hashlib.sha256(eq_csv.encode()).hexdigest(),
            "index_sha256": hashlib.sha256(ix_csv.encode()).hexdigest(),
            "date_range": [dates[0], dates[-1]] if dates else [None, None],
            "last_trading_date": dates[-1] if dates else None,
            "instrument_count": len(symbols),
            "equity_bar_count": len(eq_rows),
            "index_bar_count": len(ix_rows),
            "has_benchmark": len(ix_rows) > 0,
        }
        manifest.update(extra_manifest or {})
        manifest["manifest_checksum"] = _manifest_checksum(manifest)

        # write atomically into a temp dir, then rename — no partial snapshot is ever visible
        tmp = Path(tempfile.mkdtemp(prefix=f".stage_{sid}_", dir=self.root))
        try:
            (tmp / "bars_equity.csv").write_text(eq_csv, encoding="utf-8")
            if ix_csv.strip():
                (tmp / "index_daily.csv").write_text(ix_csv, encoding="utf-8")
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(tmp, sdir)                         # atomic publish
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return sid

    # ── verify ───────────────────────────────────────────────────────────────────
    def verify_snapshot(self, snapshot_id: str) -> tuple:
        sdir = self.root / snapshot_id
        fails = []
        if not sdir.exists():
            return False, ["snapshot directory missing"]
        mpath = sdir / "manifest.json"
        if not mpath.exists():
            return False, ["manifest missing"]
        try:
            m = json.loads(mpath.read_text())
        except Exception as e:
            return False, [f"manifest unreadable: {e}"]
        chk = m.get("manifest_checksum")
        if chk != _manifest_checksum({k: v for k, v in m.items() if k != "manifest_checksum"}):
            fails.append("manifest checksum mismatch")
        eqp = sdir / "bars_equity.csv"
        if not eqp.exists():
            fails.append("equity data file missing")
        elif hashlib.sha256(eqp.read_text().encode()).hexdigest() != m.get("equity_sha256"):
            fails.append("equity data hash mismatch")
        if m.get("schema_version") != SCHEMA_VERSION:
            fails.append("unsupported schema version")
        return (not fails), fails

    # ── activation (atomic pointer swap + audit) ─────────────────────────────────
    def activate_snapshot(self, snapshot_id: str, *, actor="system", reason="") -> dict:
        ok, fails = self.verify_snapshot(snapshot_id)
        if not ok:
            raise ValueError(f"cannot activate {snapshot_id}: {fails}")
        prev = self.get_active_snapshot()
        ptr = {"snapshot_id": snapshot_id, "activated_by": actor, "reason": reason,
               "previous_snapshot_id": prev or ""}
        tmp = self.root / ".ACTIVE.tmp"
        tmp.write_text(json.dumps(ptr), encoding="utf-8")
        os.replace(tmp, self.root / "ACTIVE")            # atomic
        with open(self.root / "activation_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ptr) + "\n")
        return ptr

    def get_active_snapshot(self) -> str | None:
        p = self.root / "ACTIVE"
        if not p.exists():
            return None
        try:
            sid = json.loads(p.read_text())["snapshot_id"]
        except Exception:
            return None
        return sid if (self.root / sid).exists() else None   # pointer to missing snap ⇒ None

    def open_snapshot(self, snapshot_id: str):
        from research.intelligence.data.snapshot import Snapshot
        ok, fails = self.verify_snapshot(snapshot_id)
        if not ok:
            raise ValueError(f"snapshot {snapshot_id} failed verification: {fails}")
        return Snapshot(self.root / snapshot_id)

    def open_active(self):
        sid = self.get_active_snapshot()
        return self.open_snapshot(sid) if sid else None

    def list_snapshots(self) -> list:
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and (d / "manifest.json").exists())


def eq_csv_to_rows(csv_text: str):
    lines = csv_text.strip().split("\n")
    for line in lines[1:]:
        if line:
            yield tuple(line.split(","))


def _manifest_checksum(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()
