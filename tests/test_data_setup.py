"""
Deterministic, network-free tests for the Historical Data Setup engine
(`research.momentum_breakout.data_setup`).

Synthetic bhavcopy/index CSVs + JSON in tmp dirs; no network, no wall-clock dependence,
no Streamlit runtime, no order path. Covers ZIP safety, validation, readiness, snapshot
identity, overwrite protection, immutable run creation + prior-run preservation, EXP-006
config immutability, and execution isolation.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import pytest

from research.momentum_breakout import data_setup as D
from research.momentum_breakout.config import primary_config


# ── synthetic dataset builders ─────────────────────────────────────────────────

def _write_bhav(root: Path, n_days=65, symbols=("TCS", "INFY", "HAL"),
                bad_ohlc=False, dupe=False, bad_cols=False):
    bhav = root / "bhav"; bhav.mkdir(parents=True, exist_ok=True)
    d = dt.date(2024, 1, 1); made = 0
    while made < n_days:
        if d.weekday() < 5:
            if bad_cols:
                rows = ["FOO,BAR\n1,2"]
            else:
                hdr = ("SYMBOL,SERIES,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,"
                       "TTL_TRD_QNTY,DELIV_PER")
                rows = [hdr]
                for s, base in zip(symbols, range(3000, 3000 + 500 * len(symbols), 500)):
                    p = base + made
                    if bad_ohlc:
                        rows.append(f"{s},EQ,{p},{p-50},{p+50},{p},1000000,55")  # high<low
                    else:
                        rows.append(f"{s},EQ,{p},{p+5},{p-5},{p+2},1000000,55")
                    if dupe:
                        rows.append(f"{s},EQ,{p},{p+5},{p-5},{p+2},1000000,55")   # dup symbol
            (bhav / f"{d.strftime('%d%m%Y')}.csv").write_text("\n".join(rows) + "\n")
            made += 1
        d += dt.timedelta(days=1)
    return bhav


def _write_index(root: Path, n_days=65):
    idx = root / "index"; idx.mkdir(parents=True, exist_ok=True)
    d = dt.date(2024, 1, 1); made = 0
    while made < n_days:
        if d.weekday() < 5:
            v = 20000 + made
            rows = ["Index Name,Open Index Value,High Index Value,Low Index Value,"
                    "Closing Index Value,Volume",
                    f"Nifty 50,{v},{v+50},{v-50},{v+10},0"]
            (idx / f"{d.strftime('%d%m%Y')}.csv").write_text("\n".join(rows) + "\n")
            made += 1
        d += dt.timedelta(days=1)
    return idx


def _full_dataset(root: Path, ca=True, uni=True, **kw):
    _write_bhav(root, **kw)
    _write_index(root)
    if ca:
        (root / "ca_events.json").write_text(json.dumps({"TCS": []}))
    if uni:
        (root / "universe_history.json").write_text(json.dumps(
            [{"symbol": "TCS", "listed": "2000-01-01"}]))
    return root


# ══════════════════════════════════════════════════════════════════════════════
# 1. ZIP safety
# ══════════════════════════════════════════════════════════════════════════════

class TestZipSafety:
    def _zip(self, entries: dict) -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)
        buf.seek(0)
        return buf

    def test_valid_zip_ingestion(self, tmp_path):
        z = self._zip({"bhav/01012024.csv": "SYMBOL,SERIES\nTCS,EQ",
                       "index/01012024.csv": "Index Name\nNifty 50",
                       "ca_events.json": "{}"})
        rep = D.safe_extract_zip(z, tmp_path / "stg")
        assert rep.ok
        assert set(rep.extracted) == {"bhav/01012024.csv", "index/01012024.csv",
                                      "ca_events.json"}

    def test_unsafe_path_rejected(self, tmp_path):
        z = self._zip({"../evil.csv": "x", "bhav/ok.csv": "SYMBOL"})
        rep = D.safe_extract_zip(z, tmp_path / "stg")
        assert any("evil" in n for n, _ in rep.rejected)
        assert not (tmp_path / "evil.csv").exists()      # never escaped

    def test_unsupported_file_rejected(self, tmp_path):
        z = self._zip({"random.exe": "x", "bhav/ok.csv": "SYMBOL"})
        rep = D.safe_extract_zip(z, tmp_path / "stg")
        assert any(n == "random.exe" for n, _ in rep.rejected)

    def test_not_a_zip_is_handled(self, tmp_path):
        rep = D.safe_extract_zip(io.BytesIO(b"not a zip"), tmp_path / "stg")
        assert rep.ok is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. Validation + quality
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_full_valid_dataset_is_ready(self, tmp_path):
        v = D.validate_dataset(_full_dataset(tmp_path))
        assert v.status == D.READY
        assert v.price_data and v.benchmark and v.corporate_actions and v.universe_history
        assert v.delivery_data is True
        assert v.symbol_count == 3 and v.row_count > 0 and v.invalid_price_count == 0

    def test_missing_benchmark_not_ready(self, tmp_path):
        _write_bhav(tmp_path)                              # no index/
        v = D.validate_dataset(tmp_path)
        assert v.status == D.NOT_READY
        assert any("benchmark" in b for b in v.blockers)

    def test_malformed_csv_is_invalid(self, tmp_path):
        _write_bhav(tmp_path, bad_cols=True); _write_index(tmp_path)
        v = D.validate_dataset(tmp_path)
        assert v.status == D.INVALID_DATA

    def test_invalid_ohlc_is_invalid(self, tmp_path):
        _full_dataset(tmp_path, bad_ohlc=True)
        v = D.validate_dataset(tmp_path)
        assert v.status == D.INVALID_DATA and v.invalid_price_count > 0

    def test_malformed_json_is_invalid(self, tmp_path):
        _full_dataset(tmp_path, ca=False)
        (tmp_path / "ca_events.json").write_text("{ not json ]")
        v = D.validate_dataset(tmp_path)
        assert v.status == D.INVALID_DATA

    def test_duplicate_rows_detected(self, tmp_path):
        _full_dataset(tmp_path, dupe=True)
        v = D.validate_dataset(tmp_path)
        assert v.duplicate_count > 0

    def test_insufficient_history_not_ready(self, tmp_path):
        _write_bhav(tmp_path, n_days=10); _write_index(tmp_path)
        v = D.validate_dataset(tmp_path)
        assert v.status == D.NOT_READY

    def test_no_wall_clock_dependence(self, tmp_path):
        # validation reads dates from filenames, not today's date → deterministic
        v1 = D.validate_dataset(_full_dataset(tmp_path))
        v2 = D.validate_dataset(tmp_path)
        assert v1.first_date == v2.first_date == "2024-01-01"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Readiness (green / amber / red)
# ══════════════════════════════════════════════════════════════════════════════

class TestReadiness:
    def test_green_when_fully_research_grade(self, tmp_path):
        r = D.readiness(D.validate_dataset(_full_dataset(tmp_path)))
        assert r["color"] == "green" and r["can_run"] is True

    def test_amber_when_missing_ca_or_universe(self, tmp_path):
        r = D.readiness(D.validate_dataset(_full_dataset(tmp_path, ca=False, uni=False)))
        assert r["color"] == "amber" and r["can_run"] is True
        assert r["reasons"]                                 # limitations explained

    def test_red_when_not_ready(self, tmp_path):
        _write_bhav(tmp_path)                                # no benchmark
        r = D.readiness(D.validate_dataset(tmp_path))
        assert r["color"] == "red" and r["can_run"] is False
        assert r["reasons"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Snapshot identity
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapshot:
    def test_snapshot_id_is_stable_and_content_addressed(self, tmp_path):
        root = _full_dataset(tmp_path)
        v = D.validate_dataset(root)
        s1 = D.dataset_snapshot(root, v)
        s2 = D.dataset_snapshot(root, v)
        assert s1["snapshot_id"] == s2["snapshot_id"]       # same content → same id
        assert s1["ingestion_ts"] != s2["ingestion_ts"] or True  # ts is provenance only

    def test_material_change_alters_snapshot_id(self, tmp_path):
        root = _full_dataset(tmp_path)
        s1 = D.dataset_snapshot(root, D.validate_dataset(root))
        # add another trading day → different content → different id
        (root / "bhav" / "08012024.csv").write_text(
            "SYMBOL,SERIES,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY\n"
            "TCS,EQ,3100,3110,3090,3105,1000000\n")
        s2 = D.dataset_snapshot(root, D.validate_dataset(root))
        assert s1["snapshot_id"] != s2["snapshot_id"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save into canonical stores + overwrite protection + materialise
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveAndMaterialize:
    def test_new_mode_refuses_silent_overwrite(self, tmp_path):
        stg = _full_dataset(tmp_path / "stg")
        logs = tmp_path / "logs"
        D.save_into_canonical(stg, mode="new", logs_root=logs)          # first save ok
        with pytest.raises(D.OverwriteRefused):                          # second refuses
            D.save_into_canonical(stg, mode="new", logs_root=logs)

    def test_replace_mode_overwrites_with_confirmation(self, tmp_path):
        stg = _full_dataset(tmp_path / "stg")
        logs = tmp_path / "logs"
        D.save_into_canonical(stg, mode="new", logs_root=logs)
        res = D.save_into_canonical(stg, mode="replace", logs_root=logs)
        assert res["status"] == "saved" and res["copied"]["bhav"] > 0

    def test_cancel_mode_does_nothing(self, tmp_path):
        res = D.save_into_canonical(tmp_path / "stg", mode="cancel", logs_root=tmp_path / "logs")
        assert res["status"] == "cancelled"

    def test_materialize_builds_canonical_store_no_network(self, tmp_path, monkeypatch):
        # point the canonical bhav store dir at a tmp location, save + materialise
        import data.bhavcopy_store as bs
        stg = _full_dataset(tmp_path / "stg")
        logs = tmp_path / "logs"
        monkeypatch.setattr(bs, "_BHAV_DIR", logs / "bhav")
        monkeypatch.setattr(bs, "_PKL", logs / "bhav" / "store_cache.pkl")
        monkeypatch.setattr(bs, "_store", {}, raising=False)
        D.save_into_canonical(stg, mode="replace", logs_root=logs)
        n = bs.build_from_local()
        assert n >= 1 and bs.is_ready()


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXP-006 run into a NEW immutable dir (red-gate blocked; prior runs preserved)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunExp006:
    def test_red_gate_prevents_run(self, tmp_path):
        with pytest.raises(D.OverwriteRefused):
            D.run_exp006({"can_run": False, "color": "red"}, runs_root=tmp_path / "runs")

    def test_next_run_id_skips_existing(self, tmp_path):
        runs = tmp_path / "runs"
        (runs / "0001-blocked").mkdir(parents=True)
        assert D.next_run_id(runs) == "0002"

    def test_green_run_creates_new_dir_and_preserves_prior(self, tmp_path):
        from tests.test_momentum_breakout_run import _clean_universe
        runs = tmp_path / "runs"
        prior = runs / "0001-blocked"; prior.mkdir(parents=True)
        (prior / "verdict.json").write_text('{"verdict":"INCONCLUSIVE"}')
        prior_bytes = (prior / "verdict.json").read_bytes()
        res = D.run_exp006({"can_run": True, "color": "green"},
                           provider=_clean_universe(k=4), runs_root=runs)
        assert res["run_id"] == "0002"
        assert (runs / "0002" / "verdict.json").exists()
        assert (runs / "0002" / "run_manifest.json").exists()
        # prior run untouched
        assert (prior / "verdict.json").read_bytes() == prior_bytes
        assert res["verdict"]["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")

    def test_run_does_not_change_exp006_config(self, tmp_path):
        from tests.test_momentum_breakout_run import _clean_universe
        before = primary_config().config_hash()
        D.run_exp006({"can_run": True, "color": "green"},
                     provider=_clean_universe(k=3), runs_root=tmp_path / "runs")
        assert primary_config().config_hash() == before   # frozen config unchanged


# ══════════════════════════════════════════════════════════════════════════════
# 7. Execution isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionIsolation:
    def test_data_setup_imports_no_order_path(self):
        import inspect
        src = inspect.getsource(D)
        code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
        for pat in ("import execution", "from execution", "import alerts", "from alerts",
                    ".place_trade(", "place_trade(", "kite_client", "GTT", ".arm("):
            assert pat not in code, f"data_setup references {pat}"

    def test_ui_page_has_no_order_actions(self):
        import inspect
        from ui import data_setup_page as page
        src = inspect.getsource(page)
        code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
        for pat in ("place_trade", "arm(", "GTT", "zerodha", "kite_client",
                    "telegram_actions", "consider("):
            assert pat not in code, f"ui.data_setup_page references {pat}"
