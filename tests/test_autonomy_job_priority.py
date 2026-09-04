from research.autonomy import job_store as JS


def test_scan_and_news_are_leased_before_long_overnight_grind(tmp_path):
    now = 1_800_000_000.0
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: now)
    try:
        outcome = store.enqueue(
            "outcome_resolution",
            scheduled_for=now - 300,
            idempotency_key="outcome:session",
            critical=True,
        )
        news = store.enqueue(
            "news_refresh",
            scheduled_for=now - 120,
            idempotency_key="news:session",
        )
        scan = store.enqueue(
            "market_scan",
            scheduled_for=now - 60,
            idempotency_key="scan:session",
        )

        first = store.lease_due("worker")
        assert first is not None and first.job_id == scan.job_id
        store.complete(first.job_id, JS.SUCCEEDED)

        second = store.lease_due("worker")
        assert second is not None and second.job_id == news.job_id
        store.complete(second.job_id, JS.SUCCEEDED)

        third = store.lease_due("worker")
        assert third is not None and third.job_id == outcome.job_id
    finally:
        store.close()


def test_poll_wait_cannot_starve_real_due_work(tmp_path):
    now = 1_800_000_000.0
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: now)
    try:
        poll = store.enqueue(
            "data_refresh",
            scheduled_for=now - 600,
            idempotency_key="poll",
            critical=True,
        )
        store.complete(
            poll.job_id,
            JS.PENDING,
            error_code="DATA_REFRESH_IN_PROGRESS",
            error_message="waiting for another worker",
        )
        research = store.enqueue(
            "research_cycle",
            scheduled_for=now - 30,
            idempotency_key="research",
        )

        leased = store.lease_due("worker")
        assert leased is not None and leased.job_id == research.job_id
    finally:
        store.close()
