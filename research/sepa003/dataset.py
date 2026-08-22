"""Reconstruct frozen F setups with PIT regime/sector features. No threshold change."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from research.sepa.ablation_r import sepa_fill_sim
from research.sepa.ca_audit import CATimeline, build_timeline, unresolved_events
from research.sepa.config import R2_CONFIG
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.frames import iso_date
from research.sepa.signal_study import forward_path_study
from research.sepa.universe_pit import FastInvestable, load_store_frames
from research.sepa003.constants import (
    FILL_SEARCH_SESSIONS,
    HORIZON,
    OUT_DIR,
    R2_DIR,
)
from research.sepa003.features import pack_features
from research.sepa003.regime import (
    breadth_as_of,
    build_index_level,
    classify_regime_level,
    regime_at,
)
from research.sepa003.sector import load_sector_map_v1, sector_context, sector_ranks_as_of


def load_r2_f_setups() -> list[dict[str, Any]]:
    path = R2_DIR / "setups.jsonl"
    out = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("variant") == "F":
                out.append(row)
    return out


def load_g_rows(limit: int | None = None) -> list[dict[str, Any]]:
    path = R2_DIR / "g_signal_rows.jsonl"
    out = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            out.append(json.loads(line))
            if limit is not None and i + 1 >= limit:
                break
    return out


def _fwd_pct(fwd: pd.DataFrame | None, n: int = 20) -> float | None:
    if fwd is None or len(fwd) < n:
        return None
    o = float(fwd["open"].iloc[0])
    c = float(fwd["close"].iloc[n - 1])
    if o <= 0:
        return None
    return c / o - 1.0


def reconstruct(
    *,
    frames: dict[str, pd.DataFrame] | None = None,
    max_setups: int | None = None,
    collect_controls: bool = True,
    controls_per_date: int = 6,
) -> dict[str, Any]:
    """Replay frozen F setups. Does not modify R2.1 files."""
    frames = frames or load_store_frames(min_bars=80)
    setups = load_r2_f_setups()
    if max_setups is not None:
        setups = setups[: int(max_setups)]
    cfg = R2_CONFIG
    unresolved = unresolved_events(frames, sample=None)
    timeline = build_timeline(unresolved)
    calendar = []
    acc = set()
    for df in frames.values():
        if df is None or len(df) == 0:
            continue
        acc.update(pd.DatetimeIndex(df.index).tz_localize(None).normalize())
    calendar = sorted(acc)
    timeline.annotate_calendar(calendar)
    cal_pos = {pd.Timestamp(t): i for i, t in enumerate(calendar)}

    index_level, index_source = build_index_level(frames)
    regime_table = classify_regime_level(index_level) if len(index_level) else None
    sector_pack = load_sector_map_v1()
    smap = sector_pack["map"]
    fast = FastInvestable(frames)
    from research.sepa003.fastrs import FastRS
    fastrs = FastRS(fast, cfg)

    by_date: dict[str, list[dict]] = defaultdict(list)
    for s in setups:
        by_date[str(s.get("detection_date"))].append(s)

    f_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    skipped = defaultdict(int)
    rs_cache: dict[str, Any] = {}
    sector_rank_cache: dict[str, dict[str, float]] = {}
    breadth_cache: dict[str, dict[str, Any]] = {}

    dates = sorted(by_date)
    for di, det in enumerate(dates):
        group = by_date[det]
        snap = fast.snapshot(det, min_sessions=260, min_price=20.0, min_turnover=5_000_000.0)
        if det not in rs_cache:
            rs_cache[det] = fastrs.table(det, list(snap.investable))
        rs_table = rs_cache[det]
        if det not in sector_rank_cache:
            sector_rank_cache[det] = sector_ranks_as_of(det, frames, smap)
        if det not in breadth_cache:
            # Use investable names only — PIT as-of membership.
            breadth_cache[det] = breadth_as_of(frames, det, list(snap.investable), min_n=200)
        meta = {
            "universe_complete": False,
            "universe_source": "bhav_inferred",
            "ca_complete": False,
            "pit_class": "PIT_DEGRADED",
        }
        seen_ctrl = set()
        for setup in group:
            sym = str(setup.get("symbol") or "").upper()
            hist, _ = fast.hist_fwd(sym, det, HORIZON + FILL_SEARCH_SESSIONS, timeline=timeline)
            if hist is None or len(hist) < 80:
                skipped["no_hist"] += 1
                continue
            sepa0 = evaluate_sepa_eligibility(
                sym, det, frame=hist, frames=frames, universe=list(snap.investable),
                rs_table=rs_table, config=cfg, pit_meta=meta, compute_vcp=True,
            )
            det_i = cal_pos.get(pd.Timestamp(det))
            filled = None
            fill_sepa = sepa0
            fill_asof = det
            fill_fwd = None
            fill_class = "NO_FILL"
            if det_i is None:
                skipped["det_not_on_calendar"] += 1
                continue
            for step in range(0, FILL_SEARCH_SESSIONS):
                as_i = det_i + step
                if as_i >= len(calendar):
                    break
                as_of = iso_date(calendar[as_i])
                hist_i, fwd_i = fast.hist_fwd(sym, as_of, HORIZON, timeline=timeline)
                if hist_i is None or fwd_i is None or len(fwd_i) < 1:
                    continue
                if step == 0:
                    sepa = sepa0
                    use_rs = rs_table
                else:
                    if as_of not in rs_cache:
                        snap_i = fast.snapshot(
                            as_of, min_sessions=260, min_price=20.0, min_turnover=5_000_000.0,
                        )
                        rs_cache[as_of] = fastrs.table(as_of, list(snap_i.investable))
                    use_rs = rs_cache[as_of]
                    sepa = evaluate_sepa_eligibility(
                        sym, as_of, frame=hist_i, frames=frames,
                        universe=list(snap.investable), rs_table=use_rs,
                        config=cfg, pit_meta=meta, compute_vcp=True,
                    )
                if not sepa.trend_template_pass or not sepa.vcp_detected or sepa.structural_stop is None:
                    continue
                if timeline.horizon_crosses(sym, as_of, iso_date(fwd_i.index[min(HORIZON, len(fwd_i)) - 1])):
                    fill_class = "CA_CENSORED_OUTCOME"
                    fill_sepa = sepa
                    fill_asof = as_of
                    fill_fwd = fwd_i
                    break
                packed = sepa_fill_sim(
                    fwd_i, stop=float(sepa.structural_stop), pivot=sepa.pivot,
                    buy_zone_low=sepa.buy_zone_low, buy_zone_high=sepa.buy_zone_high,
                    horizon=HORIZON,
                )
                fill_class = packed.get("class") or "NO_BAR"
                if fill_class == "VALID_FILL" and packed.get("sim"):
                    filled = packed["sim"]
                    fill_sepa = sepa
                    fill_asof = as_of
                    fill_fwd = fwd_i
                    break
                if fill_class in {"GAP_THROUGH", "EXTENDED"}:
                    fill_sepa = sepa
                    fill_asof = as_of
                    fill_fwd = fwd_i
                    # keep searching; a later session may still fill
                    continue
            rd = regime_at(regime_table, det)
            re = regime_at(regime_table, fill_asof)
            sctx = sector_context(sym, fill_asof, frames, smap, stock_rs=fill_sepa.rs_percentile)
            ranks = sector_rank_cache.get(det) or sector_ranks_as_of(fill_asof, frames, smap)
            if sctx["sector"] in ranks:
                sctx["sector_rs"] = ranks[sctx["sector"]]
            row = pack_features(
                sepa=fill_sepa, sim=filled, fill_class=fill_class if filled else fill_class,
                detection_date=det, entry_date=fill_asof if filled else None,
                regime_detect=rd, regime_entry=re, sector_ctx=sctx,
                breadth=breadth_cache.get(det),
                fwd_pct_20=_fwd_pct(fill_fwd),
                ca_censored=fill_class == "CA_CENSORED_OUTCOME",
                setup_id=str(setup.get("setup_id") or fill_sepa.setup_id),
                source_variant="F",
            )
            if filled:
                f_rows.append(row)
            else:
                skipped[fill_class] += 1
                if fill_class == "CA_CENSORED_OUTCOME":
                    f_rows.append(row)

        if collect_controls and group:
            rng = np.random.default_rng(abs(hash(det)) % (2**32))
            candidates = [s for s in snap.investable if s not in {x.get("symbol") for x in group}]
            rng.shuffle(candidates)
            got = 0
            for sym in candidates:
                if got >= controls_per_date:
                    break
                if sym in seen_ctrl:
                    continue
                hist, fwd = fast.hist_fwd(sym, det, HORIZON, timeline=timeline)
                if hist is None or fwd is None or len(fwd) < HORIZON:
                    continue
                sepa = evaluate_sepa_eligibility(
                    sym, det, frame=hist, frames=frames, universe=list(snap.investable),
                    rs_table=rs_table, config=cfg, pit_meta=meta, compute_vcp=False,
                )
                if not (sepa.structure_pass and sepa.rs_pass):
                    continue
                sepa = evaluate_sepa_eligibility(
                    sym, det, frame=hist, frames=frames, universe=list(snap.investable),
                    rs_table=rs_table, config=cfg, pit_meta=meta, compute_vcp=True,
                )
                if sepa.vcp_detected:
                    continue
                if timeline.horizon_crosses(sym, det, iso_date(fwd.index[min(HORIZON, len(fwd)) - 1])):
                    continue
                seen_ctrl.add(sym)
                sctx = sector_context(sym, det, frames, smap, stock_rs=sepa.rs_percentile)
                if sctx["sector"] in sector_rank_cache[det]:
                    sctx["sector_rs"] = sector_rank_cache[det][sctx["sector"]]
                path = forward_path_study(fwd)
                control_rows.append(pack_features(
                    sepa=sepa, sim=None, fill_class="CONTROL_NO_VCP",
                    detection_date=det, entry_date=iso_date(fwd.index[0]),
                    regime_detect=regime_at(regime_table, det),
                    regime_entry=regime_at(regime_table, det),
                    sector_ctx=sctx, breadth=breadth_cache[det],
                    fwd_pct_20=_fwd_pct(fwd),
                    setup_id=f"CTRL:{sym}:{det}",
                    source_variant="CONTROL",
                    is_control=True, control_kind="stage2_rs_no_vcp",
                ) | {
                    "fwd_pct_20": None if path is None else path.get("fwd_20d_pct"),
                    "mae_pct": None if path is None else path.get("mae_pct"),
                    "mfe_pct": None if path is None else path.get("mfe_pct"),
                })
                got += 1

        if di % 25 == 0:
            print(
                f"SEPA-003 {det} {di+1}/{len(dates)} F_fills={len(f_rows)} ctrl={len(control_rows)}",
                flush=True,
            )

    # Attach repaired regime/sector to G panel (no R, diagnosis only).
    g_annot = []
    for g in load_g_rows():
        as_of = str(g.get("as_of") or "")
        re = regime_at(regime_table, as_of)
        sec = sector_context(str(g.get("symbol") or ""), as_of, frames, smap, stock_rs=g.get("rs_percentile"))
        g_annot.append({
            **{k: g[k] for k in g if k not in {"regime", "sector"}},
            "regime": re.get("regime"),
            "sector": sec.get("sector"),
            "sector_rs": sec.get("sector_rs"),
            "era": "winning_era" if as_of <= "2023-12-31" else "weak_era",
            "rs_bucket": None,
        })
    from research.sepa003.constants import rs_bucket
    for g in g_annot:
        g["rs_bucket"] = rs_bucket(g.get("rs_percentile"))

    payload = {
        "experiment": "SEPA-003",
        "index_source": index_source,
        "regime_version": "regime_pit_v1",
        "sector": {k: sector_pack[k] for k in sector_pack if k != "map"},
        "n_f_setups_ledger": len(setups),
        "n_f_rows": len(f_rows),
        "n_f_fills": sum(1 for r in f_rows if r.get("net_r") is not None),
        "n_controls": len(control_rows),
        "n_g_annotated": len(g_annot),
        "skipped": dict(skipped),
        "features": f_rows,
        "controls": control_rows,
        "g_panel": g_annot,
        "not_validated_edge": True,
        "confirmation_already_observed": True,
    }
    return payload


def persist_dataset(payload: dict[str, Any]) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat = payload.get("features") or []
    ctrl = payload.get("controls") or []
    g = payload.get("g_panel") or []
    paths = {}
    df = pd.DataFrame(feat)
    pq = OUT_DIR / "sepa_003_features.parquet"
    try:
        df.to_parquet(pq, index=False)
        paths["features_parquet"] = str(pq)
    except Exception:
        jl = OUT_DIR / "sepa_003_features.jsonl"
        with jl.open("w") as fh:
            for row in feat:
                fh.write(json.dumps(row, default=str) + "\n")
        paths["features_jsonl"] = str(jl)
    cpath = OUT_DIR / "sepa_003_controls.jsonl"
    with cpath.open("w") as fh:
        for row in ctrl:
            fh.write(json.dumps(row, default=str) + "\n")
    paths["controls_jsonl"] = str(cpath)
    gpath = OUT_DIR / "sepa_003_g_panel.jsonl"
    with gpath.open("w") as fh:
        for row in g:
            fh.write(json.dumps(row, default=str) + "\n")
    paths["g_panel_jsonl"] = str(gpath)
    slim = {k: payload[k] for k in payload if k not in {"features", "controls", "g_panel"}}
    (OUT_DIR / "sepa_003_dataset_meta.json").write_text(json.dumps(slim, indent=2, default=str))
    paths["meta"] = str(OUT_DIR / "sepa_003_dataset_meta.json")
    return paths
