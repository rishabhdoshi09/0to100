from research.autonomy.research_loop import _kick_counterfactual_learning


def test_counterfactual_hook_does_not_duplicate_running_replay():
    called = {"runner": 0}

    def runner(**_kwargs):
        called["runner"] += 1
        raise AssertionError("runner must not be called while replay is already running")

    result = _kick_counterfactual_learning(
        loader=lambda: {"status": "RUNNING", "provenance": "BACKTEST", "decisions_tested": 83},
        runner=runner,
    )

    assert result["status"] == "RUNNING"
    assert result["already_running"] is True
    assert result["decisions_tested"] == 83
    assert called["runner"] == 0


def test_counterfactual_hook_starts_bounded_async_replay_when_idle():
    captured = {}

    def runner(**kwargs):
        captured.update(kwargs)
        return {
            "accepted": True,
            "status": "RUNNING",
            "provenance": "BACKTEST",
            "message": "started",
        }

    result = _kick_counterfactual_learning(
        loader=lambda: {"status": "SUCCEEDED", "provenance": "BACKTEST"},
        runner=runner,
    )

    assert captured == {"async_job": True, "sessions": 12, "universe_limit": 32}
    assert result["status"] == "RUNNING"
    assert result["accepted"] is True
    assert result["not_forward_evidence"] is True
    assert result["live_locked"] is True
