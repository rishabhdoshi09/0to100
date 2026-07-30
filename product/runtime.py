"""Idempotent startup wiring for the retail entrypoint."""
from __future__ import annotations


def ensure_runtime_started() -> dict:
    """Start persisted PAPER_AUTO and the read-only news-curator worker.

    Repeated calls are safe. The news worker only fetches and stores context; it
    cannot place or modify broker orders.
    """
    state = {
        "paper_auto_enabled": False,
        "worker_running": False,
        "news_curator_running": False,
        "error": "",
        "news_error": "",
    }
    try:
        from research.auto_research.scheduler import get_brain
        brain = get_brain()
        state["paper_auto_enabled"] = bool(brain.is_paper_auto_enabled())
        if state["paper_auto_enabled"]:
            brain.start()
        state["worker_running"] = bool(brain.state.running)
        state["error"] = str(brain.state.last_error or "")
    except Exception as exc:
        state["error"] = str(exc)

    try:
        from news.curator_service import get_news_curator_service
        service = get_news_curator_service()
        service.start()
        state["news_curator_running"] = service.running
        state["news_error"] = service.last_error
    except Exception as exc:
        state["news_error"] = str(exc)
    return state
