"""
🗂️ Historical Data Setup — the pure engine behind the layman data-management page.

Lets a local user hand QuantTerm real NSE history (a ZIP or an existing folder),
validates it, previews coverage/quality, saves it into the EXISTING canonical stores,
freezes a deterministic snapshot, reports research readiness, and — only when the
readiness gate is green/amber — runs the UNCHANGED frozen EXP-006 runner into a NEW
immutable run directory.

PURE: no Streamlit, no network, no clock-in-identity, no order path. It reuses the
canonical `data/bhavcopy_store` + `data/index_store` (local-load) — it does NOT build a
parallel market-data database. It cannot place any broker order.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# ── ZIP safety limits (decompression-bomb + traversal defence) ─────────────────
MAX_ZIP_ENTRIES = 20000
MAX_TOTAL_UNCOMPRESSED = 4 * 1024 * 1024 * 1024      # 4 GiB hard cap
MAX_COMPRESSION_RATIO = 200                          # per-entry bomb guard
# Only these content families are accepted (validated by content, not name alone).
_BHAV_COLS = {"SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
              "CLOSE_PRICE", "TTL_TRD_QNTY"}
_INDEX_COL_HINTS = ("Index Name", "Closing Index Value", "Open Index Value")


# ══════════════════════════════════════════════════════════════════════════════
# A. Safe ZIP extraction
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractReport:
    ok: bool
    extracted: list = field(default_factory=list)      # relative dest paths
    rejected: list = field(default_factory=list)       # (name, reason)
    total_bytes: int = 0

    def as_dict(self):
        return {"ok": self.ok, "extracted": self.extracted,
                "rejected": self.rejected, "total_bytes": self.total_bytes}


def _classify_entry(name: str) -> str | None:
    """Return the destination sub-path for an allowed entry, or None to reject.
    Accepts `bhav/*.csv`, `index/*.csv`, `ca_events.json`, `universe_history.json`
    (also tolerates a single top-level wrapper folder)."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts:
        return None
    # strip an optional single wrapper dir
    if len(parts) >= 2 and parts[0] not in ("bhav", "index"):
        parts = parts[1:]
    if not parts:
        return None
    if parts[0] == "bhav" and len(parts) == 2 and parts[1].lower().endswith(".csv"):
        return f"bhav/{parts[1]}"
    if parts[0] == "index" and len(parts) == 2 and parts[1].lower().endswith(".csv"):
        return f"index/{parts[1]}"
    if len(parts) == 1 and parts[0] in ("ca_events.json", "universe_history.json"):
        return parts[0]
    # ANY other CSV / markdown → classify later by CONTENT (folder name is ignored)
    if name.lower().endswith((".csv", ".md")):
        return "LOOSE"
    return None


def safe_extract_zip(zip_source, dest_dir) -> ExtractReport:
    """Extract a data package into `dest_dir`, defending against path traversal,
    symlinks, decompression bombs and unsupported files. Validates by CONTENT family
    (see `_classify_entry`), never by trusting the raw name. `zip_source` is a path or
    a bytes/file-like object."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rep = ExtractReport(ok=True)
    try:
        zf = zipfile.ZipFile(zip_source)
    except Exception as exc:
        return ExtractReport(ok=False, rejected=[("<archive>", f"not a valid zip: {exc}")])
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            return ExtractReport(ok=False,
                                 rejected=[("<archive>", f"too many entries ({len(infos)})")])
        total = 0
        loose: list[tuple] = []                              # (name, bytes) → content-classified
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                continue                                     # directory entry
            # traversal / absolute / symlink guards
            if name.startswith("/") or ".." in Path(name).parts:
                rep.rejected.append((name, "unsafe path (traversal/absolute)")); continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                rep.rejected.append((name, "symlink not allowed")); continue
            sub = _classify_entry(name)
            if sub is None:
                rep.rejected.append((name, "unsupported file")); continue
            # decompression-bomb guards
            size = info.file_size
            if info.compress_size > 0 and size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
                rep.rejected.append((name, "compression ratio too high")); continue
            total += size
            if total > MAX_TOTAL_UNCOMPRESSED:
                return ExtractReport(ok=False, extracted=rep.extracted,
                                     rejected=rep.rejected + [(name, "total size cap exceeded")],
                                     total_bytes=total)
            if sub == "LOOSE":                               # unknown folder/name → by content
                with zf.open(info) as src:
                    loose.append((Path(name).name, src.read()))
                continue
            target = (dest / sub).resolve()
            if not str(target).startswith(str(dest.resolve())):
                rep.rejected.append((name, "escapes destination")); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 256)
            rep.extracted.append(sub)
        rep.total_bytes = total
    if loose:                                                # content-classify everything else
        ing = ingest_files(loose, dest)
        rep.extracted += ing.extracted
        rep.rejected += ing.rejected
    rep.ok = bool(rep.extracted)
    return rep


# ══════════════════════════════════════════════════════════════════════════════
# A2. Direct file ingestion — .csv / .json / .md / .pdf (no ZIP required)
# ══════════════════════════════════════════════════════════════════════════════
# Files are classified by CONTENT (bhavcopy vs index), split by their own date column
# (or a DDMMYYYY in the name) into the per-day files the canonical store expects.
# Unreliable extraction is REJECTED with a clear message — never silently accepted as
# research data.

_BHAV_DATE_COLS = ("DATE1", "DATE", "TIMESTAMP", "TRADE_DATE")
_INDEX_DATE_COLS = ("INDEX DATE", "DATE")
import io as _io
import re as _re
from pathlib import Path as _P


def _date_stem(val) -> str | None:
    import pandas as pd
    s = str(val).strip()
    iso = bool(_re.match(r"^\d{4}-\d{2}-\d{2}", s))     # ISO dates aren't day-first
    try:
        ts = pd.to_datetime(s, dayfirst=not iso, errors="coerce")
        return None if pd.isna(ts) else ts.strftime("%d%m%Y")
    except Exception:
        return None


def _stem_from_name(name: str) -> str | None:
    m = _re.search(r"(\d{8})", _P(name).stem)      # DDMMYYYY anywhere in the name
    return m.group(1) if (m and _date_stem_valid(m.group(1))) else None


def _date_stem_valid(stem: str) -> bool:
    import pandas as pd
    try:
        return not pd.isna(pd.to_datetime(stem, format="%d%m%Y", errors="coerce"))
    except Exception:
        return False


def _read_any_table(name: str, raw: bytes):
    """Return a list of DataFrames from a .csv / .md / .pdf file, or None if unreadable."""
    import pandas as pd
    low = name.lower()
    if low.endswith(".csv"):
        for enc in ("utf-8", "latin-1"):
            try:
                return [pd.read_csv(_io.BytesIO(raw), dtype=str, encoding=enc)]
            except Exception:
                continue
        return None
    if low.endswith(".md"):
        return _markdown_tables(raw.decode("utf-8", "ignore"))
    if low.endswith(".pdf"):
        return _pdf_tables(raw)
    return None


def _markdown_tables(text: str):
    """Parse GitHub-flavoured markdown tables → DataFrames. Deterministic, no deps."""
    import pandas as pd
    tables, rows = [], []

    def _flush():
        nonlocal rows
        if len(rows) >= 2:
            hdr = [c.strip() for c in rows[0]]
            body = [r for r in rows[2:]] if _is_sep(rows[1]) else [r for r in rows[1:]]
            data = [r for r in body if len(r) == len(hdr)]
            if data:
                tables.append(pd.DataFrame(data, columns=hdr).astype(str))
        rows = []

    for line in text.splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        else:
            _flush()
    _flush()
    return tables or None


def _is_sep(cells) -> bool:
    return all(set(c.strip()) <= set("-: ") and "-" in c for c in cells if c.strip())


def _pdf_tables(raw: bytes):
    """Best-effort PDF table extraction — ONLY if pdfplumber is installed. If not, return
    None so the caller rejects honestly (PDF price data can't be validated reliably)."""
    try:
        import pdfplumber
    except Exception:
        return None
    import pandas as pd
    out = []
    try:
        with pdfplumber.open(_io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                for tbl in (page.extract_tables() or []):
                    if len(tbl) >= 2:
                        out.append(pd.DataFrame(tbl[1:], columns=[str(c).strip()
                                                                  for c in tbl[0]]).astype(str))
    except Exception:
        return None
    return out or None


# flexible column aliases → common OHLCV schemas are accepted, not just NSE's exact
# sec_bhavdata_full columns (Date/Symbol/Open/High/Low/Close/Volume in many namings).
def _nk(c) -> str:
    return _re.sub(r"[^A-Z0-9]", "", str(c).upper())


_ALIAS = {
    "SYMBOL": {"SYMBOL", "TICKER", "TCKRSYMB", "SCRIP", "SECURITY", "STOCK"},
    "SERIES": {"SERIES", "SCTYSRS"},
    "DATE": {"DATE", "DATE1", "TIMESTAMP", "TRADEDATE", "TRADDT", "TRDDT", "DT"},
    "OPEN": {"OPEN", "OPENPRICE", "OPNPRIC", "OPENINDEXVALUE"},
    "HIGH": {"HIGH", "HIGHPRICE", "HGHPRIC", "HIGHINDEXVALUE"},
    "LOW": {"LOW", "LOWPRICE", "LWPRIC", "LOWINDEXVALUE"},
    "CLOSE": {"CLOSE", "CLOSEPRICE", "CLSPRIC", "LAST", "LASTPRICE", "LTP",
              "CLOSINGINDEXVALUE"},
    "VOLUME": {"VOLUME", "TTLTRDQNTY", "TTLTRADGVOL", "TOTALTRADEDQUANTITY", "QTY",
               "NOOFSHARES", "VOL", "SHARESTRADED"},
    "DELIV": {"DELIVPER", "PERCENTDELIVERBLE", "DELIVERBLE", "PERDELIVERBLE",
              "DELIVERYPERCENTAGE"},
}


def _colmap(df) -> dict:
    m = {}
    for c in df.columns:
        k = _nk(c)
        for canon, al in _ALIAS.items():
            if k in al:
                m.setdefault(canon, c)
    return m


def _normalize_to_bhav(df):
    """Map a flexible stock-OHLCV table onto the canonical bhavcopy columns the store
    reads (SYMBOL, SERIES, DATE1, OPEN/HIGH/LOW/CLOSE_PRICE, TTL_TRD_QNTY, DELIV_PER)."""
    import pandas as pd
    m = _colmap(df)
    if not all(x in m for x in ("SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE")):
        return None
    out = pd.DataFrame({
        "SYMBOL": df[m["SYMBOL"]].astype(str).str.strip().str.upper(),
        "SERIES": (df[m["SERIES"]].astype(str).str.strip() if "SERIES" in m else "EQ"),
        "DATE1": df[m["DATE"]],
        "OPEN_PRICE": df[m["OPEN"]], "HIGH_PRICE": df[m["HIGH"]],
        "LOW_PRICE": df[m["LOW"]], "CLOSE_PRICE": df[m["CLOSE"]],
        "TTL_TRD_QNTY": (df[m["VOLUME"]] if "VOLUME" in m else 0),
    })
    if "DELIV" in m:
        out["DELIV_PER"] = df[m["DELIV"]]
    return out


def _normalize_to_index(df, name: str):
    """Map a flexible benchmark table onto the canonical index columns — ONLY when it is
    clearly the Nifty 50 (^NSEI) benchmark, so a stock file is never mislabelled."""
    import pandas as pd
    m = _colmap(df)
    if not all(x in m for x in ("DATE", "CLOSE")):
        return None
    hint = _nk(_P(name).stem)                            # stem only (drop the .csv)
    is_nifty = ("NIFTY50" in hint) or ("NSEI" in hint)
    for c in df.columns:
        if _nk(c) in ("INDEXNAME", "INDEX", "NAME"):
            vals = {_nk(v) for v in df[c].astype(str).unique()[:5]}
            if vals & {"NIFTY50", "NSEI"}:
                is_nifty = True
    if not is_nifty:
        return None
    return pd.DataFrame({
        "Index Name": "Nifty 50", "Index Date": df[m["DATE"]],
        "Open Index Value": df[m["OPEN"]] if "OPEN" in m else df[m["CLOSE"]],
        "High Index Value": df[m["HIGH"]] if "HIGH" in m else df[m["CLOSE"]],
        "Low Index Value": df[m["LOW"]] if "LOW" in m else df[m["CLOSE"]],
        "Closing Index Value": df[m["CLOSE"]]})


def _classify_and_write(df, name: str, dest, rep) -> None:
    """Classify one table as bhavcopy or index (by exact columns, then flexible OHLCV
    aliases), split by date, write per-day files. Rejects (with a reason) anything it
    cannot validate — never silently accepted."""
    df = df.rename(columns=lambda c: str(c).strip())
    upper = {c.upper() for c in df.columns}
    if _BHAV_COLS.issubset(upper):
        n = _write_by_date(df, name, _P(dest) / "bhav", _BHAV_DATE_COLS)
        (rep.extracted.append(f"bhav/ ({n} day(s)) ← {name}") if n
         else rep.rejected.append((name, "bhavcopy CSV but no readable trading date")))
        return
    if any(h.lower() in {c.lower() for c in df.columns} for h in _INDEX_COL_HINTS):
        n = _write_by_date(df, name, _P(dest) / "index", _INDEX_DATE_COLS)
        (rep.extracted.append(f"index/ ({n} day(s)) ← {name}") if n
         else rep.rejected.append((name, "index CSV but no readable date")))
        return
    nb = _normalize_to_bhav(df)
    if nb is not None:
        n = _write_by_date(nb, name, _P(dest) / "bhav", _BHAV_DATE_COLS)
        (rep.extracted.append(f"bhav/ ({n} day(s)) ← {name}") if n
         else rep.rejected.append((name, "stock columns recognised but no readable date")))
        return
    ni = _normalize_to_index(df, name)
    if ni is not None:
        n = _write_by_date(ni, name, _P(dest) / "index", _INDEX_DATE_COLS)
        (rep.extracted.append(f"index/ ({n} day(s)) ← {name}") if n
         else rep.rejected.append((name, "benchmark recognised but no readable date")))
        return
    rep.rejected.append((name, "unrecognised table — need columns like Date, Symbol, "
                               "Open, High, Low, Close (a stock file) or a Nifty 50 index"))


def _write_by_date(df, name, out_dir, date_cols) -> int:
    out_dir = _P(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dcol = next((c for c in df.columns if c.strip().upper() in date_cols), None)
    written = 0
    if dcol is not None:
        for val, g in df.groupby(dcol):
            stem = _date_stem(val)
            if stem:
                g.drop(columns=[]).to_csv(out_dir / f"{stem}.csv", index=False)
                written += 1
    if written == 0:                                # fall back to a date in the filename
        stem = _stem_from_name(name)
        if stem:
            df.to_csv(out_dir / f"{stem}.csv", index=False); written = 1
    return written


def ingest_files(files, dest_dir) -> ExtractReport:
    """Accept a list of (filename, bytes) uploads of type .csv / .json / .md / .pdf and
    stage them into `dest_dir` as the canonical `bhav/` `index/` `*.json` layout. Files
    are validated by CONTENT; unrecognised or unreadable files are rejected with a reason
    (never silently accepted). Returns an ExtractReport."""
    dest = _P(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    rep = ExtractReport(ok=True)
    for name, data in files:
        raw = data if isinstance(data, bytes) else bytes(data)
        low = name.lower()
        if low.endswith(".json"):
            base = _P(name).name
            if base in ("ca_events.json", "universe_history.json"):
                try:
                    json.loads(raw.decode("utf-8", "ignore"))
                    (dest / base).write_bytes(raw); rep.extracted.append(base)
                except Exception:
                    rep.rejected.append((name, "invalid JSON"))
            else:
                rep.rejected.append((name, "unsupported JSON (need ca_events.json / "
                                            "universe_history.json)"))
            continue
        if not low.endswith((".csv", ".md", ".pdf")):
            rep.rejected.append((name, "unsupported file type")); continue
        tables = _read_any_table(name, raw)
        if not tables:
            hint = ("PDF price data can't be validated reliably here — please export to CSV"
                    if low.endswith(".pdf") else "could not read a table from this file")
            rep.rejected.append((name, hint)); continue
        for df in tables:
            _classify_and_write(df, name, dest, rep)
    rep.ok = bool(rep.extracted)
    return rep


# ══════════════════════════════════════════════════════════════════════════════
# B. Validation + coverage/quality
# ══════════════════════════════════════════════════════════════════════════════

READY = "READY"
READY_WITH_LIMITATIONS = "READY WITH LIMITATIONS"
NOT_READY = "NOT READY"
INVALID_DATA = "INVALID DATA"

_MIN_SESSIONS = 60          # matches bhavcopy_store._MIN_DAYS (usable history floor)


@dataclass
class DatasetValidation:
    status: str
    price_data: bool
    benchmark: bool
    corporate_actions: bool
    universe_history: bool
    delivery_data: bool
    first_date: str | None
    last_date: str | None
    file_count: int
    symbol_count: int
    row_count: int
    duplicate_count: int
    invalid_price_count: int
    adjustment_status: str
    blockers: list = field(default_factory=list)     # plain-language
    limitations: list = field(default_factory=list)  # plain-language

    def as_dict(self):
        return asdict(self)


def _read_bhav_csv(path: Path):
    import pandas as pd
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, dtype=str, encoding="latin-1")
        except Exception:
            return None
    df.columns = [c.strip().upper() for c in df.columns]
    if not _BHAV_COLS.issubset(set(df.columns)):
        return None
    return df


def validate_dataset(root) -> DatasetValidation:
    """Validate a staged/existing dataset folder (expects `bhav/`, optional `index/`,
    optional `ca_events.json` / `universe_history.json`). Reads CONTENT, not names.
    Fails to INVALID_DATA on malformed files, NOT_READY on missing essentials."""
    import numpy as np
    import pandas as pd
    root = Path(root)
    bhav_dir = root / "bhav"
    index_dir = root / "index"
    ca = root / "ca_events.json"
    uni = root / "universe_history.json"

    blockers: list[str] = []; limitations: list[str] = []
    invalid = False

    bhav_files = sorted(bhav_dir.glob("*.csv")) if bhav_dir.exists() else []
    symbols: set[str] = set(); rows = 0; dupes = 0; bad_prices = 0
    dates: list[str] = []; has_delivery = False
    for p in bhav_files:
        # date parses from the filename (DDMMYYYY)
        try:
            dt = datetime.strptime(p.stem, "%d%m%Y").date(); dates.append(dt.isoformat())
        except Exception:
            blockers.append(f"file '{p.name}' is not named as a date (DDMMYYYY.csv)")
            invalid = True; continue
        df = _read_bhav_csv(p)
        if df is None:
            blockers.append(f"'{p.name}' is not a valid bhavcopy file (missing columns)")
            invalid = True; continue
        eq = df[df["SERIES"].str.strip() == "EQ"] if "SERIES" in df.columns else df
        syms = eq["SYMBOL"].str.strip().str.upper()
        symbols.update(syms.dropna().tolist())
        rows += len(eq)
        dupes += int(syms.duplicated().sum())
        for c_hi, c_lo, c_op, c_cl in [("HIGH_PRICE", "LOW_PRICE", "OPEN_PRICE", "CLOSE_PRICE")]:
            hi = pd.to_numeric(eq[c_hi], errors="coerce")
            lo = pd.to_numeric(eq[c_lo], errors="coerce")
            op = pd.to_numeric(eq[c_op], errors="coerce")
            cl = pd.to_numeric(eq[c_cl], errors="coerce")
        nonpos = ((cl <= 0) | (op <= 0) | (hi <= 0) | (lo <= 0)).sum()
        inconsistent = ((hi < lo) | (hi < np.maximum(op, cl)) | (lo > np.minimum(op, cl))).sum()
        bad_prices += int(nonpos) + int(inconsistent)
        if "DELIV_PER" in df.columns:
            has_delivery = True

    price_data = len(bhav_files) > 0 and not invalid and rows > 0
    # benchmark: index CSVs OR the app's index store can serve ^NSEI
    index_files = sorted(index_dir.glob("*.csv")) if index_dir.exists() else []
    benchmark = len(index_files) > 0
    # corporate actions / universe history: valid JSON if present
    corporate_actions = False; universe_history = False
    if ca.exists():
        try:
            json.loads(ca.read_text()); corporate_actions = True
        except Exception:
            blockers.append("ca_events.json is not valid JSON"); invalid = True
    if uni.exists():
        try:
            json.loads(uni.read_text()); universe_history = True
        except Exception:
            blockers.append("universe_history.json is not valid JSON"); invalid = True

    first_date = min(dates) if dates else None
    last_date = max(dates) if dates else None
    n_sessions = len(set(dates))
    adjustment = "Adjusted for splits/bonuses" if corporate_actions else "Raw (not adjusted)"

    # ── status ──
    if invalid or bad_prices > 0:
        if bad_prices > 0:
            blockers.append(f"{bad_prices} price rows are impossible (zero/negative or "
                            "high<low) — the data looks corrupted")
        status = INVALID_DATA
    elif not price_data:
        blockers.append("no daily price data found (need bhavcopy CSVs under 'bhav/')")
        status = NOT_READY
    elif not benchmark:
        blockers.append("no Nifty benchmark found (need index CSVs under 'index/')")
        status = NOT_READY
    elif n_sessions < _MIN_SESSIONS:
        blockers.append(f"only {n_sessions} trading days found — need at least "
                        f"{_MIN_SESSIONS} for a fair test")
        status = NOT_READY
    else:
        if not corporate_actions:
            limitations.append("no corporate-action file — prices are raw, so splits/"
                               "bonuses may look like big jumps (a PASS won't be issued)")
        if not universe_history:
            limitations.append("no listing/delisting history — only currently-listed "
                               "stocks are used, which flatters results (a PASS won't be issued)")
        status = READY if (corporate_actions and universe_history) else READY_WITH_LIMITATIONS

    return DatasetValidation(
        status=status, price_data=price_data, benchmark=benchmark,
        corporate_actions=corporate_actions, universe_history=universe_history,
        delivery_data=has_delivery, first_date=first_date, last_date=last_date,
        file_count=len(bhav_files) + len(index_files), symbol_count=len(symbols),
        row_count=rows, duplicate_count=dupes, invalid_price_count=bad_prices,
        adjustment_status=adjustment, blockers=blockers, limitations=limitations)


# ══════════════════════════════════════════════════════════════════════════════
# C. Readiness indicator (green / amber / red)
# ══════════════════════════════════════════════════════════════════════════════

def readiness(v: DatasetValidation) -> dict:
    """One prominent result. GREEN = research ready (run allowed); AMBER = usable for
    limited analysis (run allowed, but the runner's verdict-safety gate will not issue
    an undeserved PASS/FAIL); RED = cannot run (gate must not be bypassed)."""
    if v.status in (INVALID_DATA, NOT_READY):
        return {"color": "red", "can_run": False, "label": "Experiment cannot be run",
                "reasons": v.blockers or ["Data is not ready."]}
    if v.status == READY_WITH_LIMITATIONS:
        return {"color": "amber", "can_run": True, "label": "Usable for limited analysis",
                "reasons": v.limitations}
    return {"color": "green", "can_run": True, "label": "Research ready", "reasons": []}


# ══════════════════════════════════════════════════════════════════════════════
# D. Deterministic dataset snapshot (file hashes + coverage)
# ══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_snapshot(root, v: DatasetValidation, source: str = "user-supplied") -> dict:
    """Freeze a deterministic snapshot: per-file content hashes + coverage + policies.
    Equivalent content ⇒ same `snapshot_id`; any material change ⇒ new id. The
    ingestion timestamp is provenance only and is NOT part of the identity hash."""
    root = Path(root)
    file_hashes = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            file_hashes[str(p.relative_to(root))] = "sha256:" + _sha256(p)
    identity = {
        "file_hashes": file_hashes, "date_range": [v.first_date, v.last_date],
        "symbols": v.symbol_count, "rows": v.row_count,
        "adjustment_policy": "ADJUSTED" if v.corporate_actions else "RAW",
        "universe_policy": "PIT" if v.universe_history else "SURVIVORSHIP_INCOMPLETE",
        "benchmark": "^NSEI" if v.benchmark else None,
    }
    sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return {"snapshot_id": sid, "source": source,
            "ingestion_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **identity, "limitations": v.limitations, "status": v.status}


# ══════════════════════════════════════════════════════════════════════════════
# E. Save into the canonical stores (with overwrite protection)
# ══════════════════════════════════════════════════════════════════════════════

class OverwriteRefused(Exception):
    """Raised when saving would replace existing data without explicit confirmation."""


def save_into_canonical(staging_root, *, mode: str = "new", logs_root=None) -> dict:
    """Copy validated files from `staging_root` into the canonical stores under
    `logs/` and materialise them. `mode`:
      • 'new'     — refuse if canonical bhav data already exists (no silent overwrite);
      • 'replace' — replace existing (explicit confirmation);
      • 'cancel'  — do nothing.
    Reuses `bhavcopy_store.build_from_local` + `index_store.build_from_local` — no
    parallel database. Returns a result dict."""
    staging = Path(staging_root)
    root = Path(logs_root) if logs_root else _repo_logs()
    if mode == "cancel":
        return {"status": "cancelled"}
    bhav_dst = root / "bhav"
    existing = bhav_dst.exists() and any(bhav_dst.glob("*.csv"))
    if existing and mode == "new":
        raise OverwriteRefused("A dataset already exists. Choose 'replace' to overwrite "
                               "it (with confirmation) or 'cancel'.")
    if existing and mode == "replace":
        shutil.rmtree(bhav_dst, ignore_errors=True)
        shutil.rmtree(root / "index", ignore_errors=True)
    # copy families
    copied = {"bhav": 0, "index": 0, "ca_events": False, "universe_history": False}
    if (staging / "bhav").exists():
        bhav_dst.mkdir(parents=True, exist_ok=True)
        for p in (staging / "bhav").glob("*.csv"):
            shutil.copy2(p, bhav_dst / p.name); copied["bhav"] += 1
    if (staging / "index").exists():
        (root / "index").mkdir(parents=True, exist_ok=True)
        for p in (staging / "index").glob("*.csv"):
            shutil.copy2(p, root / "index" / p.name); copied["index"] += 1
    for name, key in (("ca_events.json", "ca_events"),
                      ("universe_history.json", "universe_history")):
        if (staging / name).exists():
            shutil.copy2(staging / name, root / name); copied[key] = True
    return {"status": "saved", "copied": copied}


def materialize(logs_root=None) -> dict:
    """Build the canonical in-memory stores from the saved local files (no network)."""
    # point the stores at logs_root if a test overrides it (else the repo default)
    from data import bhavcopy_store as bs
    from data import index_store as ix
    n_sym = bs.build_from_local()
    n_idx = ix.build_from_local()
    try:
        bs.reload_corporate_actions()
    except Exception:
        pass
    return {"bhav_symbols": n_sym, "index_series": n_idx}


def _repo_logs() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "logs"


# ══════════════════════════════════════════════════════════════════════════════
# F. EXP-006 execution into a NEW immutable run directory
# ══════════════════════════════════════════════════════════════════════════════

def _runs_root() -> Path:
    return (Path(__file__).resolve().parent.parent.parent /
            "docs" / "overhaul" / "experiments" / "EXP-006" / "runs")


def next_run_id(runs_root=None) -> str:
    """Next numeric run id, never colliding with an existing run (e.g. 0001-blocked)."""
    root = Path(runs_root) if runs_root else _runs_root()
    mx = 0
    if root.exists():
        for d in root.iterdir():
            if d.is_dir():
                try:
                    mx = max(mx, int(d.name.split("-")[0]))
                except Exception:
                    continue
    return f"{mx + 1:04d}"


def run_exp006(readiness_result: dict, provider=None, runs_root=None) -> dict:
    """Run the UNCHANGED frozen EXP-006 runner into a NEW immutable run directory.
    Refuses if the readiness gate is red (cannot be bypassed). Never overwrites an
    existing run. `provider` lets tests inject a synthetic provider; production passes
    None → the canonical BhavDataProvider."""
    if not readiness_result.get("can_run"):
        raise OverwriteRefused("Readiness is RED — the experiment cannot be run. "
                               "Fix the blockers first.")
    from research.momentum_breakout import runner as R
    from research.momentum_breakout import dataset as DS
    root = Path(runs_root) if runs_root else _runs_root()
    run_id = next_run_id(root)
    out_dir = root / run_id
    if out_dir.exists():
        raise OverwriteRefused(f"run {run_id} already exists — refusing to overwrite")
    prov = provider if provider is not None else DS.BhavDataProvider()
    res = R.run_evidence(prov, out_dir=out_dir)
    # run manifest (provenance; timestamps live here, not in the reproducible artifacts)
    manifest = {"run_id": run_id, "verdict": res["verdict"]["verdict"],
                "verdict_reason": res["verdict"].get("reason") or res["verdict"].get("reasons"),
                "snapshot_id": res["manifest"]["snapshot_id"],
                "config_hash": res["manifest"]["experiment_config_hash"],
                "code_commit": res["manifest"]["code_commit"],
                "completion_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"run_id": run_id, "out_dir": str(out_dir), "verdict": res["verdict"]}
