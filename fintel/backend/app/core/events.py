from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("fintel_starting")

    # Warm DB pool
    from app.db.engine import get_engine
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    log.info("db_pool_ready")

    # Warm Redis
    from app.db.redis import get_redis_pool
    redis = await get_redis_pool()
    await redis.ping()
    app.state.redis = redis
    log.info("redis_ready")

    yield

    # Shutdown
    from app.db.engine import dispose_engine
    from app.db.redis import close_redis
    await dispose_engine()
    await close_redis()
    log.info("fintel_shutdown")
