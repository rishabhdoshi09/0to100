"""
News summarizer: condenses a list of NormalizedArticles into a compact
bullet-point text that can be safely inserted into the LLM prompt.

⚠️  The summary is CONTEXT — not analysis. The LLM does the analysis.
"""

from __future__ import annotations

from typing import List

from news.normalizer import NormalizedArticle
from logger import get_logger

log = get_logger(__name__)

_MAX_ARTICLES_PER_SYMBOL = 3
_MAX_GLOBAL_ARTICLES = 5
_MAX_SUMMARY_CHARS = 3000  # hard cap before LLM context


class NewsSummarizer:
    def summarize_for_symbol(
        self,
        symbol: str,
        articles: List[NormalizedArticle],
        max_items: int = _MAX_ARTICLES_PER_SYMBOL,
    ) -> str:
        """Return a bullet-point string of the most recent news for symbol."""
        relevant = [a for a in articles if symbol in a.mentioned_symbols]
        relevant = relevant[:max_items]

        if not relevant:
            return f"No recent specific news for {symbol}."

        lines = [f"Recent news for {symbol}:"]
        for a in relevant:
            ts = a.published_at[:16]  # YYYY-MM-DDTHH:MM
            lines.append(f"  [{ts}] {a.headline} (src: {a.source})")
            if a.summary and len(a.summary) > 10:
                lines.append(f"    → {a.summary[:200]}")

        return "\n".join(lines)

    def summarize_macro(
        self,
        articles: List[NormalizedArticle],
        max_items: int = _MAX_GLOBAL_ARTICLES,
    ) -> str:
        """Return a bullet-point string of top macro/market news."""
        macro = [a for a in articles if not a.mentioned_symbols]
        macro = macro[:max_items]

        if not macro:
            return "No macro news available."

        lines = ["Macro / Market news:"]
        for a in macro:
            ts = a.published_at[:16]
            lines.append(f"  [{ts}] {a.headline} (src: {a.source})")

        return "\n".join(lines)

    def build_context_block(
        self,
        symbol: str,
        articles: List[NormalizedArticle],
    ) -> str:
        """
        Combine symbol-specific and macro news into one block
        for injection into the LLM prompt. Hard-capped at 3000 chars.
        """
        symbol_block = self.summarize_for_symbol(symbol, articles)
        macro_block = self.summarize_macro(articles)
        combined = f"{symbol_block}\n\n{macro_block}"
        if len(combined) > _MAX_SUMMARY_CHARS:
            combined = combined[:_MAX_SUMMARY_CHARS] + "\n[…truncated]"
        return combined
