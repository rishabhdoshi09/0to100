"""
QuestDB client — writes OHLCV time series via ILP (InfluxDB Line Protocol)
and reads via the REST HTTP API.
"""

from __future__ import annotations

import socket
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from config.settings import settings


class QuestDBClient:
    """
    Writes using ILP (port 9009) for throughput.
    Reads using the HTTP REST API (port 9000) for flexibility.
    Silently degrades if QuestDB is unreachable.
    """

    _TABLE_OHLCV = "ohlcv"

    def __init__(self) -> None:
        self._host = settings.questdb_host
        self._ilp_port = settings.questdb_port
        self._http_base = f"http://{self._host}:{settings.questdb_http_port}"
        self._available: Optional[bool] = None  # lazy probe

    # ── Connectivity probe ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self._http_base}/", timeout=2)
            self._available = r.status_code < 500
        except Exception:
            self._available = False
            logger.warning("QuestDB not reachable — writes will be skipped")
        return self._available

    # ── ILP write ─────────────────────────────────────────────────────────

    def write_ohlcv(
        self,
        symbol: str,
        df: pd.DataFrame,
        interval: str = "day",
    ) -> None:
        """Write a DataFrame of OHLCV rows via ILP."""
        if not self.is_available():
            return

        lines = []
        for ts, row in df.iterrows():
            # ILP timestamp is nanoseconds since epoch
            ts_ns = int(pd.Timestamp(ts).timestamp() * 1e9)
            line = (
                f"{self._TABLE_OHLCV},"
                f"symbol={symbol},interval={interval} "
                f"open={row['open']:.4f},"
                f"high={row['high']:.4f},"
                f"low={row['low']:.4f},"
                f"close={row['close']:.4f},"
                f"volume={int(row['volume'])}i "
                f"{ts_ns}"
            )
            lines.append(line)

        self._ilp_send("\n".join(lines))
        logger.debug(f"QuestDB: wrote {len(lines)} rows for {symbol}/{interval}")

    def _ilp_send(self, payload: str) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self._host, self._ilp_port))
                sock.sendall((payload + "\n").encode())
        except Exception as exc:
            logger.error(f"QuestDB ILP send failed: {exc}")
            self._available = False

    # ── REST queries ──────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return result as DataFrame."""
        if not self.is_available():
            return pd.DataFrame()
        try:
            r = requests.get(
                f"{self._http_base}/exec",
                params={"query": sql},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            columns = [col["name"] for col in data.get("columns", [])]
            rows = data.get("dataset", [])
            return pd.DataFrame(rows, columns=columns)
        except Exception as exc:
            logger.error(f"QuestDB query failed: {exc}")
            return pd.DataFrame()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        limit: int = 500,
    ) -> pd.DataFrame:
        sql = (
            f"SELECT timestamp, open, high, low, close, volume "
            f"FROM {self._TABLE_OHLCV} "
            f"WHERE symbol='{symbol}' AND interval='{interval}' "
            f"ORDER BY timestamp DESC LIMIT {limit}"
        )
        df = self.query(sql)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df
