from __future__ import annotations
from celery import Celery
from celery.schedules import crontab
from app.core.config import settings

celery_app = Celery(
    "fintel",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.tasks.ingestion",
        "app.workers.tasks.scraping",
        "app.workers.tasks.ai_extraction",
        "app.workers.tasks.embeddings",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_routes={
        "app.workers.tasks.scraping.*": {"queue": "scraping"},
        "app.workers.tasks.ingestion.*": {"queue": "ingestion"},
        "app.workers.tasks.ai_extraction.*": {"queue": "ai"},
        "app.workers.tasks.embeddings.*": {"queue": "embeddings"},
    },
    beat_schedule={
        "sync-nse-filings": {
            "task": "app.workers.tasks.ingestion.sync_nse_filings",
            "schedule": crontab(minute="*/30"),
        },
        "update-price-snapshots": {
            "task": "app.workers.tasks.ingestion.update_price_snapshots",
            "schedule": crontab(minute="*/5", hour="9-16", day_of_week="1-5"),
        },
        "process-pending-filings": {
            "task": "app.workers.tasks.ai_extraction.process_pending_filings",
            "schedule": crontab(minute=0, hour="*/2"),
        },
    },
)
