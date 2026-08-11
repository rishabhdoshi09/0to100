"""
🏛️ EXP-006 Historical Evidence Runner.

Runs the FROZEN, pre-registered EXP-006 framework (commit 6e7968e) against a data
provider and produces an auditable evidence verdict: PASS / FAIL / INCONCLUSIVE.

The hypothesis is NOT assumed valid. The runner:
  1. runs the data-quality gate and FAILS CLOSED (→ INCONCLUSIVE) on corruption or
     absent data — no fabricated numbers;
  2. freezes a dataset snapshot manifest;
  3. generates candidates chronologically (one event per breakout, structural dedup);
  4. simulates the pre-registered primary + secondary exits (gap-aware, next-bar entry);
  5. runs the frozen ablations and benchmark comparisons;
  6. hands the R-stream to the EXISTING harness (DSR / alpha / block-CI / BH-FDR);
  7. maps the harness verdict to PASS/FAIL/INCONCLUSIVE with a RESEARCH-GRADE gate:
     a would-be PASS on survivorship-biased / unadjusted data is DOWNGRADED to
     INCONCLUSIVE (a biased PASS is not defensible; a FAIL stays meaningful);
  8. writes machine-readable artifacts.

Research-only. Imports nothing from execution/, alerts/, the broker, or GTT.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.momentum_breakout.config import MomentumBreakoutConfig, primary_config
from research.momentum_breakout.detector import BarSeries, scan_symbol
from research.momentum_breakout import pit_safety as PS
from research.momentum_breakout import observation as OBS
from research.momentum_breakout import experiment as EXP
from research.momentum_breakout import dataset as DS

_ARTIFACT_ROOT = Path(__file__).resolve().parent.parent.parent / "logs" / "experiments" / "EXP-006"


def _regime_at(bench: np.ndarray, i: int) -> str:
    if i < 200 or not np.isfinite(bench[i]):
        return "UNKNOWN"
    sma = float(np.mean(bench[i - 199:i + 1]))
    return "RISK_ON" if bench[i] >= sma else "RISK_OFF"


def _series_for(provider, sym, cal, bench) -> BarSeries | None:
    d = provider.ohlcv(sym)
    if d is None:
        return None
    return BarSeries(symbol=sym, exchange="NSE", dates=cal,
                     open=np.asarray(d["open"], float), high=np.asarray(d["high"], float),
                     low=np.asarray(d["low"], float), close=np.asarray(d["close"], float),
                     volume=np.asarray(d["volume"], float), bench_close=np.asarray(bench, float))


def _write(out_dir: Path, name: str, obj) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    if name.endswith(".jsonl"):
        with open(p, "w") as f:
            for row in obj:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    else:
        p.write_text(json.dumps(obj, sort_keys=True, indent=2, default=str))
    return str(p)


def _generate(provider, cfg, cal, bench, provenance, eligible_only=False,
              sector_of_fn=None):
    """Chronological candidate generation across the universe with a shared event
    registry (global dedup)."""
    reg = PS.EventRegistry()
    eligible, rejected = [], []
    for sym in provider.symbols():
        s = _series_for(provider, sym, cal, bench)
        if s is None:
            continue
        def _sec(ss, i, _sym=sym):
            return provider.sector_ctx(_sym, i)
        def _val(ss, i, _sym=sym):
            return provider.valuation(_sym, i)
        obs = scan_symbol(s, cfg, sector_ctx_fn=_sec, valuation_fn=_val,
                          provenance=provenance, registry=reg, eligible_only=False)
        for o in obs:
            (eligible if o.eligibility == OBS.ELIGIBLE else rejected).append((s, o))
    if eligible_only:
        return eligible
    return eligible, rejected


def _simulate(eligible, cfg, variant):
    trades, no_fills = [], []
    for s, o in eligible:
        try:
            sig_i = s.dates.index(o.candidate_date)
        except ValueError:
            continue
        t = EXP.simulate_trade(s, sig_i, o.structural_stop, cfg, variant=variant)
        if t is None:
            no_fills.append({"symbol": o.symbol, "candidate_date": o.candidate_date,
                             "event_id": o.event_id(), "reason": "no_realistic_fill"})
        else:
            t._regime = _regime_at(s.bench_close, sig_i + 1)
            t._high_expectation = OBS.FLAG_HIGH_EXPECTATION_RISK in o.data_quality_flags
            t._val_available = bool(o.valuation.get("available"))
            trades.append(t)
    return trades, no_fills


def run_evidence(provider, cfg: MomentumBreakoutConfig | None = None,
                 out_dir: str | Path | None = None, sector_of_fn=None,
                 write: bool = True) -> dict:
    """Execute the full EXP-006 evidence run. Returns a result dict; writes
    artifacts under logs/experiments/EXP-006/<snapshot_id>/ unless write=False."""
    cfg = cfg or primary_config()

    # ── 1. data-quality gate (fail closed) ──
    quality = DS.data_quality_report(provider, cfg)
    manifest = DS.snapshot_manifest(provider, cfg, quality)
    out = Path(out_dir) if out_dir else (_ARTIFACT_ROOT / manifest["snapshot_id"])
    spec = EXP.spec(cfg)
    artifacts = {}

    def _emit(name, obj):
        if write:
            artifacts[name] = _write(out, name, obj)

    _emit("data_quality.json", quality.as_dict())
    _emit("snapshot_manifest.json", manifest)
    _emit("experiment_spec.json", spec)
    _emit("config_snapshot.json", cfg.to_dict())

    if not quality.ok:
        verdict = {"experiment_id": EXP.EXPERIMENT_ID, "verdict": "INCONCLUSIVE",
                   "reason": "; ".join(quality.fatal_reasons),
                   "primary_exit": EXP.PRIMARY_EXIT,
                   "limitations": quality.limitations,
                   "snapshot_id": manifest["snapshot_id"],
                   "note": ("Data-quality gate failed closed — no defensible PASS/FAIL "
                            "is possible on this dataset. This is NOT strategy evidence.")}
        _emit("limitations.json", {"limitations": quality.limitations,
                                   "fatal": quality.fatal_reasons})
        _emit("verdict.json", verdict)
        if write:
            _emit("artifact_index.json", {"snapshot_id": manifest["snapshot_id"],
                                          "artifacts": sorted(artifacts)})
        return {"verdict": verdict, "quality": quality.as_dict(),
                "manifest": manifest, "artifacts": artifacts}

    cal = provider.calendar(); bench = provider.benchmark_close()
    provenance = {"experiment_id": EXP.EXPERIMENT_ID, "config_hash": cfg.config_hash(),
                  "dataset_snapshot_id": manifest["snapshot_id"],
                  "code_commit": manifest["code_commit"],
                  "survivorship_complete": provider.universe_policy().get("survivorship_complete"),
                  "research_grade": provider.universe_policy().get("research_grade"),
                  "universe_source": provider.universe_policy().get("source")}

    # ── 2. chronological candidate generation ──
    eligible, rejected = _generate(provider, cfg, cal, bench, provenance,
                                   sector_of_fn=sector_of_fn)
    _emit("observations.jsonl", [o.as_dict() for _, o in eligible])
    _emit("rejected_candidates.jsonl", [o.as_dict() for _, o in rejected])

    # ── 3. simulate the PRIMARY + secondary exits ──
    primary_trades, no_fills = _simulate(eligible, cfg, EXP.PRIMARY_EXIT)
    _emit("trade_ledger.jsonl", [_trade_row(t) for t in primary_trades])
    _emit("no_fills.jsonl", no_fills)

    family_hyps = 1 + (len(EXP.EXIT_VARIANTS) - 1) + len(EXP.ablation_configs(cfg))
    primary = EXP.evaluate_trades(primary_trades, n_trials=family_hyps,
                                  require_alpha=True, require_block_ci=True)
    primary["n_candidates"] = len(eligible) + len(rejected)
    primary["n_eligible"] = len(eligible)
    primary["n_no_fill"] = len(no_fills)
    primary["max_drawdown_R"] = _max_dd_R(primary_trades)
    primary["turnover"] = len(primary_trades)
    primary["cost_drag_pct"] = cfg.cost_pct_roundtrip
    primary["regime_breakdown"] = _by_regime(primary_trades)
    primary["sector_concentration"] = _sector_conc(eligible, sector_of_fn)
    _emit("primary_metrics.json", primary)

    exit_variants = {}
    for v in EXP.EXIT_VARIANTS:
        tv, nfv = _simulate(eligible, cfg, v)
        res = EXP.evaluate_trades(tv, n_trials=family_hyps, require_alpha=False)
        res["is_primary"] = (v == EXP.PRIMARY_EXIT)
        res["n_no_fill"] = len(nfv)
        exit_variants[v] = res
    _emit("exit_variants.json", exit_variants)

    # ── 4. pre-registered ablations (each RE-scans with a relaxed config) ──
    ablations = {}
    p_values = {"primary": primary.get("p_value", 1.0)}
    for name, acfg in EXP.ablation_configs(cfg).items():
        el = _generate(provider, acfg, cal, bench, provenance, eligible_only=True,
                       sector_of_fn=sector_of_fn)
        tr, _ = _simulate(el, acfg, EXP.PRIMARY_EXIT)
        res = EXP.evaluate_trades(tr, n_trials=family_hyps, require_alpha=False)
        res["n_eligible"] = len(el)
        ablations[name] = res
        p_values[name] = res.get("p_value", 1.0)
    _emit("ablations.json", ablations)

    # ── 5. benchmark comparisons (registered alternatives) ──
    bench_cmp = {
        "nifty_buyhold_avg_R": _avg_bench_R(primary_trades),
        "simple_cross_sectional_momentum": ablations.get("prior_only", {}).get("expectancy_R"),
        "breakout_without_strong_sector": ablations.get("prior_base_risk", {}).get("expectancy_R"),
        "primary_expectancy_R": primary.get("expectancy_R"),
        "alpha_beta": primary.get("stats", {}).get("alpha_beta"),
        "note": "benchmark R aligned to each trade's actual entry→exit window",
    }
    _emit("benchmark_comparisons.json", bench_cmp)

    # ── 6. valuation & sector breakdowns ──
    valuation_bd = _valuation_breakdown(primary_trades, eligible)
    _emit("valuation_breakdown.json", valuation_bd)
    sector_bd = {"limitation": "SECTOR_MEMBERSHIP_NOT_PIT",
                 "concentration": _sector_conc(eligible, sector_of_fn)}
    _emit("sector_breakdown.json", sector_bd)
    _emit("regime.json", primary["regime_breakdown"])

    # ── 7. multiple-testing control across the family ──
    from research import harness as H
    names = list(p_values); pv = [p_values[n] for n in names]
    bh = H.benjamini_hochberg(pv)
    multiple_testing = {"hypotheses": names, "p_values": pv, "bh_result": bh,
                        "family_size": family_hyps}
    _emit("multiple_testing.json", multiple_testing)

    # ── 8. verdict mapping with the RESEARCH-GRADE gate ──
    verdict = _decide(primary, quality, manifest, provider, spec)
    _emit("limitations.json", {"limitations": quality.limitations})
    _emit("verdict.json", verdict)
    if write:
        _emit("artifact_index.json", {"snapshot_id": manifest["snapshot_id"],
                                      "artifacts": sorted(artifacts)})

    return {"verdict": verdict, "primary": primary, "exit_variants": exit_variants,
            "ablations": ablations, "benchmark": bench_cmp, "valuation": valuation_bd,
            "sector": sector_bd, "multiple_testing": multiple_testing,
            "quality": quality.as_dict(), "manifest": manifest, "artifacts": artifacts}


# ── helpers ────────────────────────────────────────────────────────────────────

def _trade_row(t) -> dict:
    d = {k: getattr(t, k) for k in ("symbol", "entry_date", "exit_date", "entry_price",
                                    "exit_price", "stop_price", "holding_period",
                                    "gross_R", "net_R", "exit_reason", "mae_R", "mfe_R",
                                    "benchmark_return")}
    d["regime"] = getattr(t, "_regime", "UNKNOWN")
    return d


def _max_dd_R(trades) -> float:
    if not trades:
        return 0.0
    eq = np.cumsum([t.net_R for t in trades])
    peak = np.maximum.accumulate(eq)
    return round(float(np.max(peak - eq)), 3)


def _by_regime(trades) -> dict:
    out = {}
    for t in trades:
        r = getattr(t, "_regime", "UNKNOWN")
        out.setdefault(r, []).append(t.net_R)
    return {k: {"n": len(v), "mean_R": round(float(np.mean(v)), 3)} for k, v in out.items()}


def _avg_bench_R(trades) -> float:
    bs = [t.benchmark_return for t in trades if t.benchmark_return is not None]
    return round(float(np.mean(bs)), 4) if bs else None


def _sector_conc(eligible, sector_of_fn) -> dict:
    if sector_of_fn is None:
        return {"note": "sector attribution not supplied"}
    out = {}
    for _, o in eligible:
        sec = sector_of_fn(o.symbol) or "UNKNOWN"
        out[sec] = out.get(sec, 0) + 1
    return out


def _valuation_breakdown(trades, eligible) -> dict:
    # valuation is CONTEXT; split trades by their observation's HIGH_EXPECTATION flag
    if not any(getattr(t, "_val_available", False) for t in trades):
        return {"status": OBS.FLAG_VALUATION_UNAVAILABLE,
                "note": "no point-in-time valuation data — extreme/non-extreme split "
                        "cannot be computed; current fundamentals NOT substituted"}
    extreme = [t for t in trades if getattr(t, "_high_expectation", False)]
    normal = [t for t in trades if not getattr(t, "_high_expectation", False)]
    return {"status": "AVAILABLE",
            "extreme": {"n": len(extreme),
                        "mean_R": round(float(np.mean([t.net_R for t in extreme])), 3) if extreme else None},
            "normal": {"n": len(normal),
                       "mean_R": round(float(np.mean([t.net_R for t in normal])), 3) if normal else None}}


def _decide(primary: dict, quality, manifest: dict, provider, spec: dict) -> dict:
    """Map harness verdict → PASS/FAIL/INCONCLUSIVE. RESEARCH-GRADE gate: a would-be
    PASS on non-research-grade data (incomplete/inferred survivorship or unadjusted CA)
    is DOWNGRADED to INCONCLUSIVE — a biased PASS is not defensible. A FAIL is retained
    when limitations are one-directional favourable (meaningful even on optimistic bias)."""
    hv = primary.get("verdict")
    reasons = []
    up = provider.universe_policy() or {}
    survivorship_ok = bool(up.get("survivorship_complete"))
    if "research_grade" in up:
        universe_rg = bool(up.get("research_grade"))
    else:
        # Legacy providers: survivorship_complete alone meant research-grade membership,
        # unless the source is explicitly the bhav bootstrap.
        universe_rg = survivorship_ok and str(up.get("source") or "") != "bhav_inferred"
    ca_raw = "RAW" in json.dumps(provider.adjustment_policy())
    research_grade = bool(universe_rg) and not ca_raw
    directions = _limitation_directions(universe_rg, ca_raw)
    if hv == "PROMOTE":
        if not research_grade:
            reasons.append(
                "would-be PASS DOWNGRADED: dataset not research-grade "
                f"(survivorship_complete={survivorship_ok}, "
                f"universe_research_grade={universe_rg}, ca_raw={ca_raw}, "
                f"source={up.get('source') or ''}) "
                "— a PASS on optimistically-biased data is not defensible"
            )
            verdict = "INCONCLUSIVE"
        else:
            verdict = "PASS"; reasons.append(primary.get("insight", ""))
    elif hv == "REJECT":
        # A FAIL is only trustworthy when the data limitations are demonstrably
        # ONE-DIRECTIONAL and FAVOURABLE to the hypothesis (biased toward MORE apparent
        # edge). If any active limitation can bias either way / against the hypothesis /
        # is unknown, a FAIL might be an artefact of bad data → INCONCLUSIVE.
        bad = [f"{k}={v}" for k, v in directions.items()
               if v not in ("FAVOURABLE", "NEUTRAL", "NONE")]
        if bad:
            reasons.append("FAIL DOWNGRADED: data limitations are not one-directional "
                           f"favourable ({'; '.join(bad)}) — a FAIL could be a data "
                           "artefact, not a real absence of edge")
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAIL"; reasons.append(primary.get("insight", ""))
    else:                                       # UNDERPOWERED / INCONCLUSIVE
        verdict = "INCONCLUSIVE"; reasons.append(primary.get("insight", ""))
    return {"experiment_id": EXP.EXPERIMENT_ID, "verdict": verdict,
            "harness_verdict": hv, "primary_exit": EXP.PRIMARY_EXIT,
            "expectancy_R": primary.get("expectancy_R"), "n_trades": primary.get("n_trades"),
            "research_grade_data": research_grade,
            "limitation_directions": directions, "reasons": reasons,
            "limitations": quality.limitations, "snapshot_id": manifest["snapshot_id"],
            "config_hash": manifest["experiment_config_hash"],
            "note": "Secondary exits / ablations / slices NEVER override the primary verdict."}


def _limitation_directions(universe_research_grade, ca_raw) -> dict:
    """Classify each active data limitation by the direction it biases the primary
    hypothesis (FAVOURABLE = makes the edge look better; UNFAVOURABLE = worse; EITHER =
    can go both ways; NEUTRAL = does not affect the primary momentum result; NONE = not
    active). Used to decide whether a FAIL is trustworthy under incomplete data.

    ``universe_research_grade`` is True only for an official listing/delisting archive —
    bhav-inferred membership still carries optimistic survivorship bias.
    """
    d = {}
    # survivorship: survivors / inferred membership inflate returns → edge looks BETTER
    d["survivorship"] = "NONE" if universe_research_grade else "FAVOURABLE"
    # unadjusted corporate actions: phantom split/bonus gaps fabricate BOTH fake
    # breakouts (favourable) and fake stop-hits/breakdowns (unfavourable) → EITHER
    d["corporate_actions"] = "EITHER" if ca_raw else "NONE"
    # valuation is context only and never a primary input → NEUTRAL
    d["valuation"] = "NEUTRAL"
    return d


# ── operator CLI ────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    """`python -m research.momentum_breakout.runner` — run on the real NSE bhav
    provider (requires the bhavcopy store / network where the data lives)."""
    import argparse
    ap = argparse.ArgumentParser(description="EXP-006 historical evidence run")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        provider = DS.BhavDataProvider(max_symbols=args.max_symbols)
    except DS.DatasetUnavailable as exc:
        # honest fail-closed path (this is what happens with no data / no network)
        res = run_evidence(_EmptyProvider(str(exc)), out_dir=args.out)
        print(json.dumps(res["verdict"], indent=2, default=str))
        return 0
    res = run_evidence(provider, out_dir=args.out)
    print(json.dumps(res["verdict"], indent=2, default=str))
    return 0


class _EmptyProvider:
    """Represents an unavailable dataset so the runner can still emit an honest
    INCONCLUSIVE(DATA_UNAVAILABLE) verdict + artifacts."""
    def __init__(self, reason): self._reason = reason
    def calendar(self): raise DS.DatasetUnavailable(self._reason)
    def benchmark_close(self): raise DS.DatasetUnavailable(self._reason)
    def benchmark_id(self): return "unavailable"
    def symbols(self): raise DS.DatasetUnavailable(self._reason)
    def ohlcv(self, sym): return None
    def sector_ctx(self, sym, i): return None
    def valuation(self, sym, i): return None
    def source_identities(self): return {"status": f"unavailable: {self._reason}"}
    def universe_policy(self):
        return {
            "survivorship_complete": False,
            "research_grade": False,
            "source": "",
            "note": self._reason,
        }
    def adjustment_policy(self): return {"corporate_actions": "unavailable (RAW)"}


if __name__ == "__main__":
    raise SystemExit(main())
