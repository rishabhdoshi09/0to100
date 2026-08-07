"""
🕸️ Evidence Graph — provenance for every belief the system holds.

The Research OS has many stores (features, experiments, beliefs, drift,
counterfactuals) but until now they were islands: you could see a gate was
active, not WHY. This is the connective tissue — every object is a node, every
causal link an edge, so the system understands provenance and can answer, as an
audit trail rather than a guess:

    Why is this gate active?
      GATE  extension_guard
        ← GATES        BELIEF   "breakouts work in healthy breadth"
        ← PROMOTED_TO  EXPERIMENT  hyp19  (184 observations)
        ← TESTED_BY    HYPOTHESIS  "breadth conditions breakout edge"
        ← DEPENDS_ON   SCHEMA   fs_20d1805d…   (last validated 21d ago)

That chain is what turns internal research into TRUST — and, downstream, into an
explainability feature ("why didn't you recommend this stock?" answered with the
evidence behind the rejecting gate).

Directed edges read src→dst as "src leads to dst" (HYPOTHESIS →tested_by→
EXPERIMENT →promoted_to→ BELIEF →gates→ GATE). `ancestry()` walks INCOMING edges
to reconstruct why a node exists; `descendants()` walks forward to see what a
thing led to. Pure graph walks (`_walk`) are unit-tested; the SQLite layer uses a
monkeypatchable path and fails safe (a broken graph degrades to "no provenance",
never to a wrong one).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "evidence_graph.db"

# node kinds — the object types the research lifecycle produces
KINDS = ("FEATURE", "SCHEMA", "HYPOTHESIS", "EXPERIMENT", "BELIEF", "GATE",
         "OBSERVATION", "DRIFT_EVENT", "RETIREMENT")

# edge relations — the causal verbs between them
RELATIONS = ("DERIVED_FROM", "DEPENDS_ON", "TESTED_BY", "PROMOTED_TO", "GATES",
             "EVIDENCED_BY", "DRIFTED", "RETIRED_AS")

_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,        -- '{kind}:{ref}', deterministic
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    label TEXT,
    attrs TEXT,                      -- json
    created_at TEXT NOT NULL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    relation TEXT NOT NULL,
    attrs TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.commit()
    return c


def node_id(kind: str, ref: str) -> str:
    """Deterministic id so the same object always maps to the same node (linking
    is idempotent, provenance never forks a duplicate)."""
    return f"{kind}:{ref}"


# ══════════════════════════════════════════════════════════════════════════════
# Pure graph walk (unit-tested without a DB)
# ══════════════════════════════════════════════════════════════════════════════

def _walk(adjacency: dict[str, list[tuple[str, str]]], start: str,
          max_depth: int) -> list[dict]:
    """Breadth-first traversal from `start` over `adjacency` (node → list of
    (neighbour, relation)). Returns [{node, relation, depth}] in visit order,
    each node once, cycle-safe. Pure."""
    seen = {start}
    out: list[dict] = []
    frontier = [(start, None, 0)]
    while frontier:
        node, rel, depth = frontier.pop(0)
        if rel is not None:
            out.append({"node": node, "relation": rel, "depth": depth})
        if depth >= max_depth:
            continue
        for nbr, r in adjacency.get(node, []):
            if nbr not in seen:
                seen.add(nbr)
                frontier.append((nbr, r, depth + 1))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Write — idempotent nodes + edges
# ══════════════════════════════════════════════════════════════════════════════

def add_node(kind: str, ref: str, label: str | None = None,
             attrs: dict | None = None) -> str:
    """Create or refresh a node; returns its id. Idempotent on (kind, ref):
    re-adding updates label/attrs but keeps identity + created_at. Fail-open."""
    nid = node_id(kind, ref)
    if kind not in KINDS:
        return nid
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        c = _conn()
        try:
            row = c.execute("SELECT 1 FROM nodes WHERE node_id=?", (nid,)).fetchone()
            if row:
                c.execute("UPDATE nodes SET label=COALESCE(?,label), "
                          "attrs=COALESCE(?,attrs), updated_at=? WHERE node_id=?",
                          (label, json.dumps(attrs) if attrs is not None else None,
                           now, nid))
            else:
                c.execute("INSERT INTO nodes (node_id, kind, ref, label, attrs, "
                          "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                          (nid, kind, ref, label,
                           json.dumps(attrs) if attrs is not None else None,
                           now, now))
            c.commit()
        finally:
            c.close()
    except Exception:
        pass
    return nid


def link(src_id: str, dst_id: str, relation: str,
         attrs: dict | None = None) -> bool:
    """Add a directed edge src→dst (idempotent on (src,dst,relation)). Fail-open
    → False."""
    if relation not in RELATIONS:
        return False
    try:
        c = _conn()
        try:
            c.execute("INSERT OR IGNORE INTO edges (src, dst, relation, attrs, "
                      "created_at) VALUES (?,?,?,?,?)",
                      (src_id, dst_id, relation,
                       json.dumps(attrs) if attrs is not None else None,
                       time.strftime("%Y-%m-%dT%H:%M:%S")))
            c.commit()
        finally:
            c.close()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Read — nodes, provenance (ancestry), descendants, plain-English explain
# ══════════════════════════════════════════════════════════════════════════════

def get_node(nid: str) -> dict | None:
    try:
        c = _conn()
        try:
            row = c.execute("SELECT * FROM nodes WHERE node_id=?", (nid,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["attrs"] = json.loads(d["attrs"]) if d.get("attrs") else {}
            return d
        finally:
            c.close()
    except Exception:
        return None


def _adjacency(reverse: bool) -> dict[str, list[tuple[str, str]]]:
    """Whole-graph adjacency. reverse=True → incoming edges (for ancestry:
    dst → [(src, relation)]); reverse=False → outgoing (for descendants)."""
    adj: dict[str, list[tuple[str, str]]] = {}
    try:
        c = _conn()
        try:
            rows = c.execute("SELECT src, dst, relation FROM edges").fetchall()
        finally:
            c.close()
    except Exception:
        return {}
    for r in rows:
        if reverse:
            adj.setdefault(r["dst"], []).append((r["src"], r["relation"]))
        else:
            adj.setdefault(r["src"], []).append((r["dst"], r["relation"]))
    return adj


def ancestry(nid: str, max_depth: int = 8) -> list[dict]:
    """The provenance chain BEHIND a node — walk incoming edges to reconstruct
    why it exists. Each entry {node, relation, depth} plus a resolved node record.
    Fail-open → []."""
    steps = _walk(_adjacency(reverse=True), nid, max_depth)
    for s in steps:
        s["detail"] = get_node(s["node"])
    return steps


def descendants(nid: str, max_depth: int = 8) -> list[dict]:
    """What a node LED TO — walk outgoing edges (e.g. belief → gate → …).
    Fail-open → []."""
    steps = _walk(_adjacency(reverse=False), nid, max_depth)
    for s in steps:
        s["detail"] = get_node(s["node"])
    return steps


def explain(nid: str, max_depth: int = 8) -> str:
    """Human-readable audit trail for a node — its provenance chain rendered as
    'X ← relation Y ← relation Z'. The literal answer to 'why is this active?'.
    Fail-open → a short stub."""
    head = get_node(nid)
    if head is None:
        return f"{nid}: no provenance recorded."
    lines = [f"{head['kind']} {head.get('label') or head['ref']}"]
    for s in ancestry(nid, max_depth):
        d = s.get("detail") or {}
        kind = d.get("kind", "?")
        name = d.get("label") or d.get("ref") or s["node"]
        indent = "  " * s["depth"]
        lines.append(f"{indent}← {s['relation']} {kind} {name}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Convenience recorders — the common lifecycle links, one call each
# ══════════════════════════════════════════════════════════════════════════════

def record_promotion(hypothesis_ref: str, experiment_ref: str, belief_ref: str,
                     *, hypothesis_label: str = "", belief_label: str = "",
                     schema_version: str | None = None,
                     evidence_n: int | None = None) -> str:
    """Record the spine HYPOTHESIS →tested_by→ EXPERIMENT →promoted_to→ BELIEF
    (optionally BELIEF →depends_on→ SCHEMA) in one call. Returns the belief node
    id. Fail-open."""
    h = add_node("HYPOTHESIS", hypothesis_ref, hypothesis_label or None)
    e = add_node("EXPERIMENT", experiment_ref,
                 attrs={"evidence_n": evidence_n} if evidence_n is not None else None)
    b = add_node("BELIEF", belief_ref, belief_label or None,
                 attrs={"evidence_n": evidence_n} if evidence_n is not None else None)
    link(h, e, "TESTED_BY")
    link(e, b, "PROMOTED_TO")
    if schema_version:
        s = add_node("SCHEMA", schema_version)
        # schema → belief ("belief ← DEPENDS_ON schema") so the schema is part of
        # the belief's provenance and surfaces in any ancestry/explain trace.
        link(s, b, "DEPENDS_ON")
    return b


def record_gate(gate_ref: str, belief_ref: str, *, gate_label: str = "") -> str:
    """Link a production GATE to the BELIEF that justifies it (BELIEF →gates→
    GATE). Returns the gate node id. Fail-open."""
    b = add_node("BELIEF", belief_ref)
    g = add_node("GATE", gate_ref, gate_label or None)
    link(b, g, "GATES")
    return g


def record_drift(belief_ref: str, drift_ref: str, *, status: str = "") -> str:
    """Attach a DRIFT_EVENT to a belief (BELIEF →drifted→ DRIFT_EVENT). Fail-open."""
    b = add_node("BELIEF", belief_ref)
    d = add_node("DRIFT_EVENT", drift_ref, status or None, attrs={"status": status})
    link(b, d, "DRIFTED")
    return d


def record_retirement(belief_ref: str, reason: str = "") -> str:
    """Mark a belief RETIRED in the graph (BELIEF →retired_as→ RETIREMENT).
    Fail-open."""
    b = add_node("BELIEF", belief_ref)
    r = add_node("RETIREMENT", belief_ref, reason or None, attrs={"reason": reason})
    link(b, r, "RETIRED_AS")
    return r
