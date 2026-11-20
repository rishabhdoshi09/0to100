"""Reconcile legacy 5-session counterfactual labels with the one rulebook (dry run on COPIES).

Until 2026-11-20 the forward ledger judged rejected candidates by a 5-session
close-to-close move that ignored the stop, while taken trades used the
10-session rulebook. Two kinds of row on disk are affected:

* LEGACY_REJECTION: a settled rejection without ``eval_spec``. Its label is not
  comparable with taken trades.
* BLOCKED_TAKEN: an entered (taken) row that the outcome job settled as a
  counterfactual before the paper trade closed. A row settles once, so its
  real paper result could never attach.

This module re-judges every LEGACY_REJECTION under the rulebook from official
bars in an SSD copy of the store, and reports the label transitions and the
per-gate missed-winner rate before and after. It lists every BLOCKED_TAKEN row.

It never writes a production path:

* The input ledgers are read only and must not be on /Volumes (the live
  runtime).
* ``--stage DIR`` writes migrated COPIES for review. A legacy outcome moves
  to ``legacy_later_outcome``; a re-judged rejection gets a rulebook
  ``later_outcome``; a blocked taken row has its counterfactual cleared, so
  the real close can attach. Swapping them in is a separate step that needs
  the owner's approval (docs/audits/2026-11-20-counterfactual-horizon-migration.md).

    python -m research.counterfactual_reconcile --ledger <copy>/forward_evidence.jsonl \
        --counterfactuals <copy>/counterfactuals.jsonl --store-root <SSD copy root> \
        --out docs/research/counterfactual_reconcile_2026-11-20.json [--stage <dir>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

LEGACY = "LEGACY_REJECTION"
BLOCKED_TAKEN = "BLOCKED_TAKEN"


def _refuse_live(*paths: str | Path | None) -> None:
    for p in paths:
        if p and str(Path(p).resolve()).startswith("/Volumes/"):
            raise SystemExit(f"refusing {p}: work on an SSD copy, never the live runtime")


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _unit(row: Mapping[str, Any], day: str) -> tuple[str, str, str]:
    return (day, str(row.get("symbol") or "").upper(), str(row.get("reason_code") or ""))


def classify_rows(ledger: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split a forward ledger into LEGACY_REJECTION, BLOCKED_TAKEN and the rest."""
    from product.counterfactual_learning import FORWARD_OUTCOME_CLASSES, is_rulebook
    out: dict[str, list[dict[str, Any]]] = {LEGACY: [], BLOCKED_TAKEN: [], "OTHER": []}
    for raw in ledger:
        row = dict(raw)
        later = dict(row.get("later_outcome") or {})
        cls = str(row.get("counterfactual_classification") or "")
        if row.get("entered"):
            # a taken row settled with a rejection-style label and no real result
            if later and row.get("gross_R") is None and row.get("execution_adjusted_R") is None:
                out[BLOCKED_TAKEN].append(row)
            else:
                out["OTHER"].append(row)
            continue
        if later and cls in FORWARD_OUTCOME_CLASSES and not is_rulebook(later.get("eval_spec")):
            out[LEGACY].append(row)
        else:
            out["OTHER"].append(row)
    return out


def rejudge(rows: Iterable[Mapping[str, Any]], outcome_fn: Callable[..., Mapping[str, Any] | None],
            *, levels=("entry", "stop", "target"), day_of=None) -> list[dict[str, Any]]:
    """Re-judge settled rejections under the rulebook. Rows whose bars are not complete stay PENDING."""
    from product.counterfactual_learning import classify_rulebook
    day_of = day_of or (lambda r: str(r.get("market_timestamp") or (r.get("pit_proof") or {}).get("as_of")
                                      or r.get("as_of") or "")[:10])
    out = []
    for raw in rows:
        row = dict(raw)
        day = day_of(row)
        sym = str(row.get("symbol") or "").upper()
        res = outcome_fn(sym, day, row.get(levels[0]), row.get(levels[1]), row.get(levels[2])) if day else None
        old = str(row.get("counterfactual_classification") or row.get("classification") or "")
        new = "PENDING" if res is None else classify_rulebook(res)
        out.append({"row": row, "day": day, "symbol": sym, "reason": str(row.get("reason_code") or ""),
                    "old": old, "new": new, "outcome": None if res is None else dict(res)})
    return out


def _rates(judged: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Per-gate missed-winner rate over independent decision units (first settlement wins)."""
    per: dict[str, dict[tuple, str]] = defaultdict(dict)
    for j in judged:
        label = j[key]
        if label in ("PENDING", "UNMEASURED"):
            continue
        per[j["reason"]].setdefault(_unit(j["row"], j["day"]), label)
    out = {}
    for reason, units in sorted(per.items()):
        labels = Counter(units.values())
        n = sum(labels.values())
        out[reason] = {"n": n, "missed_winner_rate": round(labels.get("MISSED_WINNER", 0) / n, 4) if n else None,
                       "labels": dict(labels)}
    return out


def reconcile(ledger_rows: list[dict[str, Any]], cf_rows: list[dict[str, Any]],
              outcome_fn: Callable[..., Mapping[str, Any] | None]) -> dict[str, Any]:
    from product.counterfactual_learning import FORWARD_OUTCOME_CLASSES, is_rulebook
    split = classify_rows(ledger_rows)
    judged = rejudge(split[LEGACY], outcome_fn)
    cf_legacy = [r for r in cf_rows if str(r.get("classification") or "") in FORWARD_OUTCOME_CLASSES
                 and not is_rulebook(r.get("eval_spec") or (r.get("outcome") or {}).get("eval_spec"))]
    cf_judged = rejudge(cf_legacy, outcome_fn, levels=("hypothetical_entry", "hypothetical_stop",
                                                      "hypothetical_target"),
                        day_of=lambda r: str(r.get("as_of") or "")[:10])
    transitions = Counter((j["old"], j["new"]) for j in judged)
    units_flipped = {(_unit(j["row"], j["day"])) for j in judged if j["new"] not in ("PENDING", j["old"])}
    return {
        "forward_ledger_rows": len(ledger_rows),
        "legacy_rejection_rows": len(split[LEGACY]),
        "legacy_rejection_units": len({_unit(j["row"], j["day"]) for j in judged}),
        "blocked_taken_rows": [{"decision_id": r.get("decision_id"), "symbol": r.get("symbol"),
                                "as_of": str(r.get("market_timestamp") or "")[:10],
                                "stamped": r.get("counterfactual_classification")} for r in split[BLOCKED_TAKEN]],
        "rejudged_pending": sum(1 for j in judged if j["new"] == "PENDING"),
        "label_transitions": {f"{a} -> {b}": n for (a, b), n in transitions.most_common()},
        "units_whose_label_changes": len(units_flipped),
        "per_gate_before": _rates(judged, "old"),
        "per_gate_after": _rates(judged, "new"),
        "counterfactual_jsonl_legacy_rows": len(cf_legacy),
        "counterfactual_jsonl_transitions": {f"{a} -> {b}": n for (a, b), n in
                                             Counter((j["old"], j["new"]) for j in cf_judged).most_common()},
        "_judged": judged, "_cf_judged": cf_judged, "_split": split,
    }


def stage(report: Mapping[str, Any], ledger_rows: list[dict[str, Any]], cf_rows: list[dict[str, Any]],
          stage_dir: str | Path) -> dict[str, str]:
    """Write migrated COPIES (never in place). Legacy outcomes are preserved under legacy_* keys."""
    stage_dir = Path(stage_dir)
    _refuse_live(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    by_decision = {str(j["row"].get("decision_id")): j for j in report["_judged"]}
    blocked = {str(r.get("decision_id")) for r in report["_split"][BLOCKED_TAKEN]}
    out_rows = []
    for raw in ledger_rows:
        row = dict(raw)
        did = str(row.get("decision_id"))
        if did in blocked:
            row["legacy_later_outcome"] = row.pop("later_outcome", None)
            row["legacy_counterfactual_classification"] = row.pop("counterfactual_classification", None)
            row["later_outcome"] = None
            row["counterfactual_classification"] = None
        elif did in by_decision and by_decision[did]["new"] != "PENDING":
            j = by_decision[did]
            o = j["outcome"] or {}
            row["legacy_later_outcome"] = row.get("later_outcome")
            row["legacy_counterfactual_classification"] = row.get("counterfactual_classification")
            row["later_outcome"] = {**dict(row.get("later_outcome") or {}),
                                    "forward_return_pct": o.get("return_pct"), "eval_spec": o.get("eval_spec"),
                                    "counterfactual_R": o.get("net_r"), "reconciled": True}
            row["counterfactual_classification"] = j["new"]
        out_rows.append(row)
    cf_out = []
    cf_keyed = {(str(j["row"].get("symbol")), str(j["row"].get("as_of")), str(j["row"].get("reason_code"))): j
                for j in report["_cf_judged"]}
    for raw in cf_rows:
        row = dict(raw)
        j = cf_keyed.get((str(row.get("symbol")), str(row.get("as_of")), str(row.get("reason_code"))))
        if j is not None and j["new"] != "PENDING" and str(row.get("classification") or "") == j["old"]:
            o = j["outcome"] or {}
            row["legacy_classification"] = row.get("classification")
            row["legacy_outcome"] = row.get("outcome")
            row["classification"] = j["new"]
            row["eval_spec"] = o.get("eval_spec")
            row["counterfactual_R"] = o.get("net_r")
            row["outcome"] = {**dict(row.get("outcome") or {}), "forward_return_pct": o.get("return_pct"),
                              "eval_spec": o.get("eval_spec"), "net_r": o.get("net_r"), "reconciled": True}
        cf_out.append(row)
    paths = {"forward_evidence": str(stage_dir / "forward_evidence.migrated.jsonl"),
             "counterfactuals": str(stage_dir / "counterfactuals.migrated.jsonl")}
    Path(paths["forward_evidence"]).write_text("".join(json.dumps(r, default=str) + "\n" for r in out_rows))
    Path(paths["counterfactuals"]).write_text("".join(json.dumps(r, default=str) + "\n" for r in cf_out))
    return paths


def store_outcome_fn(store_root: str | Path) -> Callable[..., Mapping[str, Any] | None]:
    """counterfactual_outcome over a CA-adjusted SSD copy of the bhav store."""
    import pickle

    from core.outcome_resolver import counterfactual_outcome
    from data.corporate_actions import adjust_frame, load_events
    import os

    from core.runtime_paths import logs_path
    root = Path(store_root).resolve()
    _refuse_live(root)
    os.environ["QT_RUNTIME_ROOT"] = str(root)              # every path resolves inside the copy
    with open(logs_path("bhav", "store_cache.pkl"), "rb") as fh:
        store = pickle.load(fh)["store"]
    events = load_events(logs_path("ca_events.json"))
    cache: dict[str, Any] = {}

    def frame(sym: str):
        if sym not in cache:
            df = store.get(sym)
            ev = events.get(sym)
            cache[sym] = adjust_frame(df.copy(), ev, copy=False) if (df is not None and ev) else df
        return cache[sym]

    def fn(sym, day, entry, stop, target):
        df = frame(str(sym).upper())
        if df is None:
            return None
        return counterfactual_outcome(sym, day, entry, stop, target, frame=df)
    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--counterfactuals")
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--out")
    ap.add_argument("--stage")
    args = ap.parse_args(argv)
    _refuse_live(args.ledger, args.counterfactuals, args.store_root, args.out)
    import os
    os.environ["QT_RUNTIME_ROOT"] = str(Path(args.store_root).resolve())     # any runtime path -> the copy
    ledger, cf = _read_jsonl(args.ledger), _read_jsonl(args.counterfactuals)
    report = reconcile(ledger, cf, store_outcome_fn(args.store_root))
    staged = stage(report, ledger, cf, args.stage) if args.stage else None
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    public["staged"] = staged
    text = json.dumps(public, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
