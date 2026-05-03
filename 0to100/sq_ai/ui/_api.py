"""Tiny shared HTTP helper for the Streamlit pages."""
from __future__ import annotations

import os

import httpx

HOST = os.environ.get("SQ_API_HOST", "127.0.0.1")
PORT = os.environ.get("SQ_API_PORT", "8000")
BASE = f"http://{HOST}:{PORT}"


def get(path: str, params: dict | None = None, timeout: float = 8.0):
    try:
        r = httpx.get(f"{BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        return {"error": str(exc)}
    return {"error": f"HTTP {r.status_code}"}


def post(path: str, json: dict | None = None, timeout: float = 30.0):
    try:
        r = httpx.post(f"{BASE}{path}", json=json or {}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}", "body": r.text}
    except Exception as exc:
        return {"error": str(exc)}


def delete(path: str, timeout: float = 8.0):
    try:
        r = httpx.delete(f"{BASE}{path}", timeout=timeout)
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}
