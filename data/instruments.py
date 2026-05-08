"""
Instrument token management.

Kite requires numeric instrument_token for historical/tick subscriptions.
This module downloads the full instrument dump once per day, caches it,
and exposes fast symbol → token lookups.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from config import settings
from logger import get_logger

log = get_logger(__name__)

_INSTRUMENT_URL = "https://api.kite.trade/instruments"
_CACHE_FILE = Path("logs/instruments_cache.csv")
_CACHE_TTL_SECONDS = 86_400  # 24 h


class InstrumentManager:
    def __init__(self) -> None:
        self._token_map: Dict[str, int] = {}   # symbol → instrument_token
        self._meta_map: Dict[str, Dict] = {}    # symbol → full row
        self._load()

    def _load(self) -> None:
        if _CACHE_FILE.exists():
            age = time.time() - _CACHE_FILE.stat().st_mtime
            if age < _CACHE_TTL_SECONDS:
                self._parse_csv(_CACHE_FILE.read_text(encoding="utf-8"))
                log.info("instruments_loaded_from_cache", count=len(self._token_map))
                return

        # Skip network download when Kite credentials are not configured
        if not settings.kite_api_key:
            log.info("instruments_skipped_no_kite_key")
            return

        self._download()

    def _download(self) -> None:
        log.info("downloading_instrument_list")
        try:
            resp = requests.get(_INSTRUMENT_URL, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("instruments_download_failed", error=str(exc))
            return
        text = resp.text
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(text, encoding="utf-8")
        self._parse_csv(text)
        log.info("instruments_downloaded", count=len(self._token_map))

    def _parse_csv(self, text: str) -> None:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            exch = row.get("exchange", "")
            sym = row.get("tradingsymbol", "")
            try:
                token = int(row["instrument_token"])
            except (KeyError, ValueError):
                continue
            if exch == settings.exchange:
                self._token_map[sym] = token
                self._meta_map[sym] = row

    def token(self, symbol: str) -> Optional[int]:
        return self._token_map.get(symbol.upper())

    def tokens_for(self, symbols: List[str]) -> Dict[str, int]:
        return {s: t for s in symbols if (t := self._token_map.get(s.upper()))}

    def meta(self, symbol: str) -> Optional[Dict]:
        return self._meta_map.get(symbol.upper())

    def refresh(self) -> None:
        self._download()
