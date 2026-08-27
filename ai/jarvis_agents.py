"""
JARVIS Specialized Agents — each has domain tools and a ReAct loop.

Agents:
  DataAgent      — live market data, Kite quotes, options, positions
  ResearchAgent  — internet search, URL fetch, news scraping
  CodeAgent      — file writes, git ops, Python execution (full system access)
  AnalysisAgent  — regime, technical indicators, patterns, opportunity score
  SystemAgent    — process control, logs, env, service restart

All agents use DeepSeek for reasoning (ReAct: think → act → observe → repeat).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ai.agent_bus import AgentBus, AgentMessage


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    agent_name: str
    task: str
    steps: list[dict] = field(default_factory=list)   # [{thought, action, args, observation}]
    result: str = ""
    success: bool = True
    error: str = ""


# ── LLM call helper ───────────────────────────────────────────────────────────

def _llm(messages: list[dict], temperature: float = 0.1, max_tokens: int = 800) -> str:
    from config import settings
    key = settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return '{"thought":"No API key","action":"DONE","result":"DEEPSEEK_API_KEY not set in .env"}'
    import requests as _req
    try:
        resp = _req.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": settings.deepseek_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=25,
        )
        data = resp.json()
        if "choices" not in data:
            err = data.get("message", data.get("error", {}).get("message", str(data)))
            return json.dumps({"thought": "API error", "action": "DONE", "result": f"DeepSeek API error: {err}"})
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"thought": str(e), "action": "DONE", "result": f"LLM error: {e}"})


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"thought": text, "action": "DONE", "result": text}


# ── Base agent ────────────────────────────────────────────────────────────────

_REACT_SYSTEM = """You are a specialist agent in the JARVIS multi-agent trading system.
You have a set of tools to complete your task. Use them step by step.

At each step respond with ONLY a JSON object:
{
  "thought": "your reasoning",
  "action": "tool_name_or_DONE",
  "args": {"key": "value"},
  "result": "only when action is DONE — your final answer"
}

Be efficient — complete tasks in as few steps as possible. Max 6 steps."""


class BaseJarvisAgent:
    name: str = "BaseAgent"
    role: str = "Generic agent"

    def __init__(self, bus: AgentBus) -> None:
        self.bus = bus
        self.tools: dict[str, Callable] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        """Override in subclass to register tools."""

    def _tool_descriptions(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            lines.append(f"  {name}: {doc}")
        return "\n".join(lines)

    def _log(self, msg_type: str, content: str, data: Any = None) -> None:
        self.bus.publish(AgentMessage(
            from_agent=self.name,
            to_agent="ORCHESTRATOR",
            msg_type=msg_type,
            content=content,
            data=data,
        ))

    def run(self, task: str, context: str = "", max_steps: int = 6) -> AgentResult:
        self._log("STATUS", f"Starting: {task[:80]}")
        result = AgentResult(agent_name=self.name, task=task)

        system_prompt = (
            f"{_REACT_SYSTEM}\n\nYour role: {self.role}\n\n"
            f"Available tools:\n{self._tool_descriptions()}\n\n"
            f"Context:\n{context[:1000] if context else 'none'}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]

        for step in range(max_steps):
            raw = _llm(messages, temperature=0.1, max_tokens=600)
            parsed = _parse_json(raw)

            thought = parsed.get("thought", "")
            action = parsed.get("action", "DONE")
            args = parsed.get("args", {})

            step_record: dict = {"thought": thought, "action": action, "args": args, "observation": ""}

            if action == "DONE":
                result.result = parsed.get("result", thought)
                result.success = True
                step_record["observation"] = result.result
                result.steps.append(step_record)
                self._log("RESULT", result.result[:200])
                break

            # Execute tool
            if action not in self.tools:
                observation = f"Unknown tool '{action}'. Available: {list(self.tools.keys())}"
            else:
                self._log("LOG", f"→ {action}({json.dumps(args)[:80]})")
                try:
                    observation = str(self.tools[action](**args))
                except Exception as exc:
                    observation = f"Tool error: {traceback.format_exc(limit=3)}"

            step_record["observation"] = observation[:500]
            result.steps.append(step_record)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation[:600]}\n\nContinue."})
        else:
            result.result = result.steps[-1].get("observation", "Max steps reached")
            result.success = False

        return result


# ── DataAgent ─────────────────────────────────────────────────────────────────

class DataAgent(BaseJarvisAgent):
    name = "DataAgent"
    role = "Fetches live and historical market data from Kite Connect and NSE"

    def _register_tools(self) -> None:
        self.tools = {
            "get_quote": self._get_quote,
            "get_ohlcv": self._get_ohlcv,
            "get_positions": self._get_positions,
            "get_indices": self._get_indices,
            "get_option_chain": self._get_option_chain,
        }

    def _get_quote(self, symbol: str) -> dict:
        """Get live LTP and quote data for a symbol."""
        try:
            from data.kite_client import KiteClient
            kite = KiteClient()
            token = kite._get_token(f"NSE:{symbol}")
            quotes = kite.kite.quote([f"NSE:{symbol}"])
            q = quotes.get(f"NSE:{symbol}", {})
            return {"symbol": symbol, "ltp": q.get("last_price"), "volume": q.get("volume"),
                    "ohlc": q.get("ohlc", {}), "change_pct": q.get("net_change")}
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    def _get_ohlcv(self, symbol: str, days: int = 60) -> str:
        """Get historical OHLCV bars for a symbol (returns summary stats)."""
        try:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            kite = KiteClient()
            im = InstrumentManager(kite)
            fetcher = HistoricalDataFetcher(kite, im)
            df = fetcher.fetch(symbol, interval="day", days=days)
            if df is None or df.empty:
                return f"No data for {symbol}"
            close = df["Close"] if "Close" in df.columns else df["close"]
            return (f"{symbol}: last={close.iloc[-1]:.2f}, 20d_high={close[-20:].max():.2f}, "
                    f"20d_low={close[-20:].min():.2f}, rows={len(df)}")
        except Exception as e:
            return f"Error: {e}"

    def _get_positions(self) -> dict:
        """Get current open positions from Kite."""
        try:
            from data.kite_client import KiteClient
            kite = KiteClient()
            pos = kite.kite.positions()
            return {"net": pos.get("net", []), "day": pos.get("day", [])}
        except Exception as e:
            return {"error": str(e)}

    def _get_indices(self) -> dict:
        """Get current values for Nifty 50, VIX, Bank Nifty."""
        try:
            from core.regime_engine import compute_regime
            r = compute_regime()
            return {
                "nifty50": r.nifty_price,
                "vix": r.vix,
                "regime": r.market_regime,
                "breadth": r.breadth_label,
                "sector_returns": r.sector_returns,
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_option_chain(self, symbol: str = "NIFTY") -> dict:
        """Get option chain PCR, max pain, IV for a symbol."""
        try:
            from options.chain import OptionChainAnalyser
            oc = OptionChainAnalyser()
            result = oc.analyse(symbol)
            return result if result else {"error": "No option data"}
        except Exception as e:
            return {"error": str(e)}


# ── ResearchAgent ─────────────────────────────────────────────────────────────

class ResearchAgent(BaseJarvisAgent):
    name = "ResearchAgent"
    role = "Searches the internet, fetches URLs, and scrapes financial data"

    def _register_tools(self) -> None:
        self.tools = {
            "web_search": self._web_search,
            "fetch_url": self._fetch_url,
            "fetch_news": self._fetch_news,
            "nse_data": self._nse_data,
        }

    def _web_search(self, query: str, n: int = 5) -> list[dict]:
        """Search DuckDuckGo and return top results."""
        import requests as _req
        try:
            # DuckDuckGo HTML scrape
            resp = _req.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
                timeout=10,
            )
            results = []
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for r in soup.find_all("div", class_="result", limit=n):
                    title_el = r.find("a", class_="result__a")
                    snip_el = r.find("a", class_="result__snippet")
                    if title_el:
                        results.append({
                            "title": title_el.get_text(strip=True),
                            "url": title_el.get("href", ""),
                            "snippet": snip_el.get_text(strip=True) if snip_el else "",
                        })
            except ImportError:
                # Regex fallback without BeautifulSoup
                for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)', resp.text):
                    results.append({"url": m.group(1), "title": m.group(2), "snippet": ""})
                    if len(results) >= n:
                        break
            return results[:n] if results else [{"error": "No results found"}]
        except Exception as e:
            return [{"error": str(e)}]

    def _fetch_url(self, url: str) -> str:
        """Fetch a URL and return extracted text content (max 3000 chars)."""
        import requests as _req
        try:
            resp = _req.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12,
            )
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
            except ImportError:
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text)
            return text[:3000]
        except Exception as e:
            return f"Fetch error: {e}"

    def _fetch_news(self, symbol: str = "") -> list[dict]:
        """Fetch recent news articles, optionally filtered by symbol."""
        try:
            from news.fetcher import NewsFetcher
            from news.normalizer import NewsNormalizer
            fetcher = NewsFetcher()
            articles = fetcher.fetch_all(max_age_hours=24)
            if symbol:
                sym_l = symbol.lower()
                articles = [a for a in articles if sym_l in (a.headline + a.summary).lower()]
            return [{"headline": a.headline, "source": a.source, "sentiment": getattr(a, "sentiment", "")}
                    for a in articles[:10]]
        except Exception as e:
            return [{"error": str(e)}]

    def _nse_data(self, endpoint: str = "equity-stockIndices") -> dict:
        """Fetch data from NSE India API (e.g. equity-stockIndices, option-chain)."""
        import requests as _req
        NSE_BASE = "https://www.nseindia.com/api/"
        try:
            session = _req.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            })
            session.get("https://www.nseindia.com/", timeout=5)
            resp = session.get(f"{NSE_BASE}{endpoint}", timeout=10)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


# ── CodeAgent ─────────────────────────────────────────────────────────────────

class CodeAgent(BaseJarvisAgent):
    name = "CodeAgent"
    role = "Writes, deploys, and manages code — full file/git access"

    def _register_tools(self) -> None:
        self.tools = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_files": self._list_files,
            "run_python": self._run_python,
            "git_status": self._git_status,
            "git_add": self._git_add,
            "git_commit": self._git_commit,
            "git_push": self._git_push,
            "git_pull": self._git_pull,
            "create_branch": self._create_branch,
            "merge_branch": self._merge_branch,
            "shell": self._shell,
        }

    def _read_file(self, path: str) -> str:
        """Read a file and return its contents."""
        try:
            return Path(path).read_text()[:4000]
        except Exception as e:
            return f"Error: {e}"

    def _write_file(self, path: str, content: str) -> dict:
        """Write content to a file (creates parent dirs if needed)."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {"success": True, "path": str(p), "bytes": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_files(self, directory: str = ".", pattern: str = "*.py") -> list[str]:
        """List files matching a pattern in a directory."""
        try:
            return [str(p) for p in Path(directory).rglob(pattern)][:50]
        except Exception as e:
            return [f"Error: {e}"]

    def _run_python(self, code: str) -> str:
        """Execute Python code and return stdout + result."""
        import io, contextlib
        buf = io.StringIO()
        ns: dict = {}
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)  # noqa: S102
            out = buf.getvalue()
            result = ns.get("RESULT", "")
            return f"{out}\nRESULT: {result}" if result else out or "OK (no output)"
        except Exception:
            return f"{buf.getvalue()}\nERROR: {traceback.format_exc(limit=4)}"

    def _git_status(self) -> str:
        """Run git status and return output."""
        r = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
        return r.stdout or r.stderr

    def _git_add(self, files: list | str = ".") -> str:
        """Stage files for commit."""
        if isinstance(files, str):
            files = [files]
        r = subprocess.run(["git", "add"] + files, capture_output=True, text=True)
        return r.stdout + r.stderr or "staged"

    def _git_commit(self, message: str) -> dict:
        """Commit staged changes."""
        r = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True,
        )
        return {"success": r.returncode == 0, "output": (r.stdout + r.stderr)[:400]}

    def _git_push(self, branch: str = "") -> dict:
        """Push to remote. Optionally specify branch."""
        cmd = ["git", "push"]
        if branch:
            cmd += ["-u", "origin", branch]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return {"success": r.returncode == 0, "output": (r.stdout + r.stderr)[:400]}

    def _git_pull(self) -> str:
        """Pull latest from remote."""
        r = subprocess.run(["git", "pull"], capture_output=True, text=True)
        return r.stdout + r.stderr

    def _create_branch(self, name: str) -> dict:
        """Create and switch to a new git branch."""
        r = subprocess.run(["git", "checkout", "-b", name], capture_output=True, text=True)
        return {"success": r.returncode == 0, "output": r.stdout + r.stderr}

    def _merge_branch(self, branch: str) -> dict:
        """Merge a branch into the current branch."""
        r = subprocess.run(["git", "merge", branch, "--no-edit"], capture_output=True, text=True)
        return {"success": r.returncode == 0, "output": (r.stdout + r.stderr)[:400]}

    def _shell(self, command: str) -> str:
        """Run a shell command safely (no shell=True). Split by spaces."""
        try:
            parts = command.split()
            r = subprocess.run(parts, capture_output=True, text=True, timeout=30)
            return (r.stdout + r.stderr)[:1000]
        except Exception as e:
            return f"Error: {e}"


# ── AnalysisAgent ─────────────────────────────────────────────────────────────

class AnalysisAgent(BaseJarvisAgent):
    name = "AnalysisAgent"
    role = "Deep market analysis — regime, technical indicators, patterns, opportunity score"

    def _register_tools(self) -> None:
        self.tools = {
            "compute_regime": self._compute_regime,
            "run_scan": self._run_scan,
            "get_opportunity_score": self._get_opp_score,
            "detect_patterns": self._detect_patterns,
            "compute_indicators": self._compute_indicators,
            "run_pre_trade_checklist": self._run_pre_trade_checklist,
            "rank_vs_sector_peers": self._rank_vs_peers,
            "get_options_flow": self._get_options_flow,
            "get_journal_stats": self._get_journal_stats,
            "get_fno_ban_list": self._get_fno_ban,
            "get_block_deals": self._get_block_deals,
            "get_circuit_stocks": self._get_circuits,
        }

    def _compute_regime(self) -> dict:
        """Compute current market regime (bull/bear/choppy, VIX, breadth)."""
        try:
            from core.regime_engine import compute_regime
            r = compute_regime()
            return {
                "regime": r.market_regime,
                "score": r.regime_score,
                "vix": r.vix,
                "nifty": r.nifty_price,
                "breadth": r.breadth_label,
                "volatility_regime": r.volatility_regime,
            }
        except Exception as e:
            return {"error": str(e)}

    def _run_scan(self, universe: list[str] = None, top_n: int = 10) -> list[dict]:
        """Run the full setup scan pipeline and return top candidates."""
        try:
            from scan.pipeline import ScanPipeline
            from core.regime_engine import compute_regime
            regime = compute_regime()
            if universe is None:
                from data.nse_universe import get_nse_universe
                universe = get_nse_universe()
            uni = universe
            results = ScanPipeline(max_workers=8, min_quality_score=40).run(
                uni, top_n=top_n, regime_state=regime, skip_liquidity_filter=True
            )
            return [{"symbol": r.symbol, "archetype": r.archetype, "tier": r.tier,
                     "quality_score": r.quality_score, "price": r.price} for r in results]
        except Exception as e:
            return [{"error": str(e)}]

    def _get_opp_score(self) -> dict:
        """Calculate the 0-100 opportunity score for current conditions."""
        try:
            from core.intelligence_hub import compute_opportunity_score
            from core.regime_engine import compute_regime
            from core.adaptive_engine import AdaptiveEngine
            regime = compute_regime()
            edge = AdaptiveEngine()
            opp = compute_opportunity_score(regime, [], edge)
            return {
                "total": opp.total,
                "grade": opp.grade,
                "label": opp.label,
                "regime_score": opp.regime_score,
                "breadth_score": opp.breadth_score,
                "volatility_score": opp.volatility_score,
                "aggression": opp.recommended_aggression,
            }
        except Exception as e:
            return {"error": str(e)}

    def _detect_patterns(self, symbol: str) -> dict:
        """Detect chart patterns for a symbol (golden cross, wedge, breakout etc)."""
        try:
            from ui.bulk_pattern import _analyse
            r = _analyse(symbol)
            return {
                "symbol": symbol,
                "patterns": r.patterns,
                "signal": r.signal,
                "score": r.score,
                "rsi": r.rsi,
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    def _compute_indicators(self, symbol: str) -> dict:
        """Compute RSI, ATR, SMA20/50/200, momentum for a symbol."""
        try:
            from agents.tools import get_technical_indicators
            return get_technical_indicators(symbol)
        except Exception as e:
            return {"error": str(e)}

    def _rank_vs_peers(self, symbol: str) -> dict:
        """Rank a stock vs all peers in its sector — who's the leader?"""
        try:
            from core.peer_ranker import rank_vs_peers
            result = rank_vs_peers(symbol)
            if result is None:
                return {"error": f"Sector not found for {symbol}"}
            return {
                "symbol": result.symbol,
                "sector": result.sector,
                "rank": result.rank,
                "total_peers": result.total_peers,
                "verdict": result.verdict,
                "score": result.score,
                "momentum_rank": result.momentum_rank,
                "rs_rank": result.rs_rank,
                "top_3": [p["symbol"] for p in result.peers_ranked[:3]],
                "bottom_3": [p["symbol"] for p in result.peers_ranked[-3:]],
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_options_flow(self, symbol: str) -> dict:
        """Check options flow for a symbol — PCR, IV rank, max pain, OI walls."""
        try:
            from options.analytics import (
                get_option_chain, compute_pcr, compute_max_pain,
                get_atm_iv, get_oi_buildup, get_iv_percentile,
            )
            df, spot_str = get_option_chain(symbol)
            if df is None or df.empty:
                return {"error": f"No options data for {symbol}"}
            spot = float(spot_str) if spot_str else 0.0
            return {
                "symbol": symbol, "spot": spot,
                "pcr": round(compute_pcr(df), 3),
                "max_pain": compute_max_pain(df),
                "atm_iv": round(get_atm_iv(df, spot), 1),
                "iv_percentile": round(get_iv_percentile(df), 1),
                "oi_buildup": get_oi_buildup(df, spot),
            }
        except Exception as e:
            return {"error": str(e)}

    def _run_pre_trade_checklist(
        self,
        symbol: str,
        action: str = "BUY",
        price: float = 0.0,
        stop: float = 0.0,
        target: float = 0.0,
        qty: int = 1,
        account_size: float = 1_000_000,
    ) -> dict:
        """Run the full pre-trade checklist for a symbol — volume, momentum, news, sector, pattern, regime, R:R."""
        try:
            from ui.pre_trade_checklist import run_checklist
            result = run_checklist(
                symbol=symbol,
                action=action,
                price=price,
                stop=stop,
                target=target,
                qty=qty,
                account_size=account_size,
            )
            return {
                "symbol": result.symbol,
                "action": result.action,
                "grade": result.grade,
                "score": result.score,
                "summary": result.summary,
                "checks": [
                    {
                        "question": item.question,
                        "passed": item.passed,
                        "detail": item.detail,
                        "weight": item.weight,
                    }
                    for item in result.items
                ],
            }
        except Exception as e:
            return {"error": str(e), "symbol": symbol}

    def _get_journal_stats(self) -> dict:
        """Get win rate, P&L, and edge from the trade journal."""
        try:
            from paper_trading import get_trading_summary, init_db
            init_db()
            return get_trading_summary() or {"no_data": True}
        except Exception as e:
            return {"error": str(e)}

    def _get_fno_ban(self) -> list[str]:
        """Get the current NSE F&O securities ban list."""
        try:
            from data.fno_ban import get_fno_ban_list
            return sorted(get_fno_ban_list())
        except Exception as e:
            return [f"error: {e}"]

    def _get_block_deals(self, min_value_cr: float = 25.0) -> list[dict]:
        """Get significant block/bulk deals from NSE (institutional transactions)."""
        try:
            from data.block_deals import get_significant_deals
            deals = get_significant_deals([], min_value_cr=min_value_cr)
            return [{"symbol": d.symbol, "client": d.client_name, "deal_type": d.deal_type,
                     "qty": d.quantity, "price": d.price, "value_cr": d.value_cr} for d in deals[:10]]
        except Exception as e:
            return [{"error": str(e)}]

    def _get_circuits(self) -> dict:
        """Get stocks hitting upper/lower price circuits today."""
        try:
            from data.circuit_breakers import get_circuit_stocks
            stocks = get_circuit_stocks("both")
            return {
                "upper": [{"symbol": c.symbol, "pct": c.pct_change, "days": c.consecutive_days}
                          for c in stocks if c.circuit_type == "UPPER"],
                "lower": [{"symbol": c.symbol, "pct": c.pct_change, "days": c.consecutive_days}
                          for c in stocks if c.circuit_type == "LOWER"],
            }
        except Exception as e:
            return {"error": str(e)}


# ── SystemAgent ───────────────────────────────────────────────────────────────

class SystemAgent(BaseJarvisAgent):
    name = "SystemAgent"
    role = "System control — process management, logs, environment, service restart"

    def _register_tools(self) -> None:
        self.tools = {
            "check_logs": self._check_logs,
            "list_processes": self._list_processes,
            "get_env": self._get_env,
            "disk_usage": self._disk_usage,
            "restart_desk": self._restart_desk,
            "pip_install": self._pip_install,
        }

    def _check_logs(self, lines: int = 50, path: str = "logs/simplequant.log") -> str:
        """Read the last N lines from the application log."""
        try:
            p = Path(path)
            if not p.exists():
                log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
                if not log_files:
                    return "No log files found"
                p = log_files[0]
            log_lines = p.read_text().splitlines()
            return "\n".join(log_lines[-lines:])
        except Exception as e:
            return f"Error: {e}"

    def _list_processes(self) -> list[dict]:
        """List running Python processes."""
        try:
            r = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True,
            )
            procs = []
            for line in r.stdout.splitlines():
                if "python" in line.lower() or "uvicorn" in line.lower() or "vite" in line.lower():
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        procs.append({"pid": parts[1], "cpu": parts[2], "mem": parts[3], "cmd": parts[10][:80]})
            return procs
        except Exception as e:
            return [{"error": str(e)}]

    def _get_env(self) -> dict:
        """Show configured environment (no secrets — keys only)."""
        keys = ["DEEPSEEK_API_KEY", "KITE_API_KEY", "ANTHROPIC_API_KEY",
                "UNIVERSE", "MAX_CAPITAL_EXPOSURE", "MAX_OPEN_POSITIONS"]
        result = {}
        for k in keys:
            v = os.getenv(k, "")
            if v and "KEY" in k:
                result[k] = f"{'*' * 6}{v[-4:]}" if len(v) > 4 else "set"
            else:
                result[k] = v or "not set"
        return result

    def _disk_usage(self) -> dict:
        """Check disk space usage."""
        try:
            r = subprocess.run(["df", "-h", "."], capture_output=True, text=True)
            lines = r.stdout.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[1].split()
                return {"filesystem": parts[0], "size": parts[1], "used": parts[2],
                        "available": parts[3], "use_pct": parts[4]}
            return {"raw": r.stdout}
        except Exception as e:
            return {"error": str(e)}

    def _restart_desk(self) -> dict:
        """Desk is Vite/React. Restart it with bash scripts/run_desk.sh."""
        return {
            "success": False,
            "error": "The product desk is Vite/React. Restart with: bash scripts/run_desk.sh then open http://127.0.0.1:5173",
        }

    def _pip_install(self, package: str) -> dict:
        """Install a Python package via pip."""
        if any(c in package for c in [";", "&", "|", "$", "`"]):
            return {"success": False, "error": "Invalid package name"}
        r = subprocess.run(
            ["pip", "install", package, "-q"],
            capture_output=True, text=True, timeout=60,
        )
        return {"success": r.returncode == 0, "output": (r.stdout + r.stderr)[:400]}
