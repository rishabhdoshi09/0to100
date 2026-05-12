from __future__ import annotations
from fastapi import APIRouter, Request
from sqlalchemy import text
from app.db.engine import get_session_factory

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health(request: Request):
    checks = {"api": "ok", "db": "unknown", "redis": "unknown"}

    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    try:
        redis = request.app.state.redis
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}
