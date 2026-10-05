import copy
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from product.fo_desk import build_fo_desk
from research.autonomy.telegram_notifications import TelegramNotifier

NOW = datetime(2026, 10, 5, 10, 30, tzinfo=ZoneInfo('Asia/Kolkata'))


def scan():
    return {
        'available': True, 'status': 'READY', 'as_of': '2026-10-05',
        'generated_at': NOW.timestamp(), 'decision': 'PAPER_CANDIDATES',
        'candidates': [{
            'symbol': 'TEST', 'direction': 'LONG', 'decision': 'PAPER_OPTION_CANDIDATE',
            'setup': {'score': 82},
            'selected_contract': {
                'symbol': 'TEST26OCT100CE', 'option_type': 'CE', 'strike': 100,
                'expiry': '2026-10-29', 'premium': 50, 'eligible': True,
                'lot_size': 25, 'score': 86,
                'trade_plan': {'entry': 50, 'stop': 40, 'target': 70},
            },
        }],
    }


def paper():
    return {'available': True, 'open_positions': [{
        'trade_id': 'open-1', 'underlying': 'TEST', 'option_symbol': 'TEST26OCT100CE',
        'opened_at': NOW.isoformat(), 'entry_price': 50, 'stop_price': 40,
        'target_price': 70, 'quantity': 25,
    }], 'recent_closed_trades': [{
        'trade_id': 'closed-1', 'underlying': 'TEST', 'option_symbol': 'TEST26OCT100PE',
        'settled_at': NOW.isoformat(), 'exit_price': 70, 'net_pnl': 490,
        'exit_reason': 'TARGET', 'cost_model_status': 'PRICED', 'quantity': 25,
    }]}


class Engine:
    def __init__(self):
        self.messages = []
        self.fail = False
        self.last_error = ''

    def is_configured(self):
        return True

    def send(self, text):
        if self.fail:
            return False
        self.messages.append(text)
        return True


def notifier(path, engine):
    return TelegramNotifier(path, now_fn=lambda: NOW, engine_factory=lambda: engine)


def test_shared_projection_keeps_eligible_candidates_and_actual_paper_separate():
    source = scan()
    before = copy.deepcopy(source)
    desk = build_fo_desk(source, paper(), now=NOW)
    assert desk['candidate_count'] == 1
    assert desk['candidates'][0]['paper_only'] is True
    assert desk['candidates'][0]['live_execution_allowed'] is False
    assert desk['open_positions'][0]['trade_id'] == 'open-1'
    assert source == before


@pytest.mark.parametrize('change', [
    {'as_of': '2026-10-02'}, {'generated_at': None},
    {'generated_at': (NOW - timedelta(minutes=31)).timestamp()},
    {'status': 'BLOCKED', 'code': 'NIFTY_HISTORY_UNAVAILABLE'},
])
def test_stale_or_blocked_scan_cannot_be_a_current_candidate(change):
    source = scan()
    source.update(change)
    desk = build_fo_desk(source, paper(), now=NOW)
    assert desk['candidates'] == []
    assert desk['open_positions'][0]['trade_id'] == 'open-1'


@pytest.mark.parametrize('field,value', [('eligible', False), ('expiry', '2026-10-02'),
    ('premium', float('nan')), ('lot_size', 0), ('option_type', 'FUT')])
def test_incomplete_or_rejected_contract_is_not_promoted(field, value):
    source = scan()
    source['candidates'][0]['selected_contract'][field] = value
    assert build_fo_desk(source, {}, now=NOW)['candidates'] == []


def test_telegram_candidate_open_exit_are_distinct_durable_and_deduped(tmp_path):
    engine = Engine()
    result = notifier(tmp_path, engine).notify_fno(scan(), paper())
    assert result == {'candidates': 1, 'opened': 1, 'closed': 1, 'status': 0, 'reason': 'sent'}
    assert len(engine.messages) == 3
    assert 'PAPER CANDIDATE' in engine.messages[0]
    assert 'expiry 2026-10-29' in engine.messages[0]
    assert 'stop ₹40.00' in engine.messages[0] and 'target ₹70.00' in engine.messages[0]
    assert 'not win probabilities' in engine.messages[0]
    assert 'PAPER OPEN' in engine.messages[1] and 'PAPER CLOSED' in engine.messages[2]
    assert 'net P&amp;L ₹490.00' in engine.messages[2]
    assert all('No live order' in message for message in engine.messages)
    result = notifier(tmp_path, engine).notify_fno(scan(), paper())
    assert result['reason'] == 'already_sent' and len(engine.messages) == 3
    assert notifier(tmp_path, engine).state['delivery']['fno']['reason'] == 'already_sent'


def test_failed_telegram_send_is_retried_without_losing_events(tmp_path):
    engine = Engine()
    engine.fail = True
    assert notifier(tmp_path, engine).notify_fno(scan(), {})['reason'] == 'send_failed'
    engine.fail = False
    assert notifier(tmp_path, engine).notify_fno(scan(), {})['candidates'] == 1


def test_stale_candidate_alert_is_withheld_but_reason_is_sent_once(tmp_path):
    engine = Engine()
    old = scan()
    old['as_of'] = '2026-10-02'
    result = notifier(tmp_path, engine).notify_fno(old, {})
    assert result['candidates'] == 0 and result['status'] == 1
    assert 'SAVED_FNO_SCAN_IS_FROM_ANOTHER_SESSION' in engine.messages[0]
    notifier(tmp_path, engine).notify_fno(old, {})
    assert len(engine.messages) == 1


def test_retry_reads_committed_ledger_during_writer_lock_and_no_missing_ledger_creation(tmp_path, monkeypatch):
    from product.fo_paper_store import FoPaperStore
    from core import runtime_paths
    from product import fo_paper_store
    monkeypatch.setattr(runtime_paths, 'logs_dir', lambda: tmp_path)
    path = tmp_path / 'fo.sqlite3'
    monkeypatch.setattr(fo_paper_store, 'logs_path', lambda *parts: path)
    product = tmp_path / 'product'
    product.mkdir()
    (product / 'fo_directional.json').write_text(json.dumps(scan()))
    engine = Engine()
    n = notifier(tmp_path / 'alerts', engine)
    assert n.drain_fno_alerts()['candidates'] == 1
    assert not path.exists()
    with FoPaperStore(path) as writer:
        writer.replace_positions(paper()['open_positions'])
        writer.conn.execute('BEGIN IMMEDIATE')
        assert n.drain_fno_alerts()['opened'] == 1
        writer.conn.rollback()


def test_weekend_worker_completes_without_running_live_scan_or_paper_cycle(tmp_path, monkeypatch):
    from operations import market_ops as mo
    from operations.store import OperationStore
    from research.intelligence.data import nse_calendar as cal
    from product import fo_runtime, fo_paper_runtime
    from data import nfo_market, fno_universe
    monkeypatch.setenv('QT_RUNTIME_ROOT', str(tmp_path))
    sunday = datetime(2026, 10, 4, 14, 30, tzinfo=ZoneInfo('Asia/Kolkata'))
    monkeypatch.setattr(cal, '_now_ist', lambda: sunday)
    report = SimpleNamespace(source='zerodha_kite', total_instrument_rows=2,
        total_future_contracts=1, index_future_contracts=0, unique_stock_underlyings=1,
        mapped_underlyings=1, underlyings=[SimpleNamespace(symbol='TEST')], exclusions=[])
    client = SimpleNamespace(instruments=lambda exchange: [{'exchange': exchange}])
    monkeypatch.setattr(nfo_market.NfoMarketDataClient, 'from_config', lambda: client)
    monkeypatch.setattr(fno_universe, 'build_fno_universe', lambda *args, **kw: report)
    def forbidden(**kwargs):
        raise AssertionError('Weekend refresh must not run live directional or paper cycle')
    monkeypatch.setattr(fo_runtime, 'run_fo_directional_scan', forbidden)
    monkeypatch.setattr(fo_paper_runtime, 'run_fo_paper_cycle', forbidden)
    deliveries = []
    monkeypatch.setattr(TelegramNotifier, 'notify_fno', lambda self, d, p: deliveries.append((d, p)) or {'reason': 'sent'})
    store = OperationStore(tmp_path / 'jobs.db')
    queued, _ = store.enqueue('FNO_REFRESH', lane='fno')
    result = mo.MarketOperationsWorker(store)._run_fno(queued)
    saved = json.loads((tmp_path / 'logs/product/fo_directional.json').read_text())
    assert saved['reason'] == 'NSE_SESSION_CLOSED'
    assert saved['decision'] == 'NO_ELIGIBLE_TRADE'
    assert result['paper']['cycle_ran'] is False
    assert deliveries[0][0]['candidate_count'] == 0
    assert result['telegram']['reason'] == 'sent'


def test_fno_refresh_rebuilds_market_client_if_kite_session_rotates_mid_run(tmp_path, monkeypatch):
    """A long FNO_REFRESH job must not scan with the pre-login Kite token.

    NfoMarketDataClient.from_config() is built once at the top of _run_fno to
    fetch the instrument master. That pass (history adoption, bulk prefetch)
    can take minutes, during which an interactive Kite login can complete and
    write a fresh access_token. The actual quote-dependent directional scan
    must see that fresh token, not the one captured minutes earlier.
    """
    from operations import market_ops as mo
    from operations.store import OperationStore
    from research.intelligence.data import nse_calendar as cal
    from product import fo_runtime, fo_paper_runtime
    from data import nfo_market, fno_universe
    from research.autonomy.telegram_notifications import TelegramNotifier

    monkeypatch.setenv('QT_RUNTIME_ROOT', str(tmp_path))
    session_day = datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo('Asia/Kolkata'))
    monkeypatch.setattr(cal, '_now_ist', lambda: session_day)
    monkeypatch.setattr(cal, 'is_session', lambda *a, **k: True)

    report = SimpleNamespace(source='zerodha_kite', total_instrument_rows=2,
        total_future_contracts=1, index_future_contracts=0, unique_stock_underlyings=1,
        mapped_underlyings=1, underlyings=[SimpleNamespace(symbol='TEST')], exclusions=[])

    token = {'value': 'pre-login-token'}
    build_calls = []

    def fake_from_config():
        client = SimpleNamespace(instruments=lambda exchange: [{'exchange': exchange}],
                                  token=token['value'])
        build_calls.append(client.token)
        return client
    monkeypatch.setattr(nfo_market.NfoMarketDataClient, 'from_config', fake_from_config)

    def fake_build_fno_universe(*args, **kw):
        # Stands in for the minutes-long history/prefilter pass: an
        # interactive login completes and rotates the token during it.
        token['value'] = 'post-login-token'
        return report
    monkeypatch.setattr(fno_universe, 'build_fno_universe', fake_build_fno_universe)

    seen_scan_client = {}

    def fake_scan(*, report, instrument_rows, client, as_of, progress_callback=None):
        seen_scan_client['token'] = client.token
        return {'available': True, 'status': 'READY', 'as_of': as_of.isoformat(),
                'decision': 'NO_ELIGIBLE_TRADE', 'candidate_count': 0,
                'prefilter_passed': 0, 'deep_evaluated': 0, 'candidates': []}
    monkeypatch.setattr(fo_runtime, 'run_fo_directional_scan', fake_scan)
    monkeypatch.setattr(fo_paper_runtime, 'run_fo_paper_cycle',
                         lambda *a, **kw: {'available': True, 'status': 'READY'})
    monkeypatch.setattr(TelegramNotifier, 'notify_fno', lambda self, d, p: {'reason': 'sent'})

    store = OperationStore(tmp_path / 'jobs.db')
    queued, _ = store.enqueue('FNO_REFRESH', lane='fno')
    result = mo.MarketOperationsWorker(store)._run_fno(queued)

    assert build_calls == ['pre-login-token', 'post-login-token']
    assert seen_scan_client['token'] == 'post-login-token'
    assert result['directional']['status'] == 'READY'


def test_api_projects_same_desk_without_starting_scan(tmp_path, monkeypatch):
    import terminal_api as api
    from product import fo_desk
    product = tmp_path / 'product'
    product.mkdir()
    (product / 'fno_universe.json').write_text(json.dumps({'mapped_underlyings': 1}))
    monkeypatch.setattr(api, 'logs_dir', lambda: tmp_path)
    monkeypatch.setattr(api, '_fo_directional_payload', scan)
    monkeypatch.setattr(api, '_fo_paper_payload', paper)
    monkeypatch.setattr(api, '_fno_learning_impact_payload', lambda _: {})
    build = fo_desk.build_fo_desk
    monkeypatch.setattr(fo_desk, 'build_fo_desk', lambda d, p: build(d, p, now=NOW))
    projected = api._fno_payload()['desk']
    assert projected['candidate_count'] == 1
    assert projected['open_positions'][0]['trade_id'] == 'open-1'


def test_concurrent_notifiers_do_not_duplicate_paper_events(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    engine = Engine()
    original_send = engine.send
    def slow_send(text):
        time.sleep(0.02)
        return original_send(text)
    engine.send = slow_send
    notifiers = [notifier(tmp_path, engine) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: n.notify_fno(scan(), paper()), notifiers))
    assert sum(row.get('candidates', 0) for row in results) == 1
    assert len(engine.messages) == 3


def test_fno_delivery_status_survives_already_sent_retry(tmp_path, monkeypatch):
    from product import telegram_delivery
    engine = Engine()
    n = notifier(tmp_path, engine)
    n.notify_fno(scan(), paper())
    n.notify_fno(scan(), paper())
    monkeypatch.setattr(telegram_delivery, 'TelegramNotifier', lambda _: n)
    status = telegram_delivery.delivery_status(tmp_path)
    assert status['state'] == 'scan_sent'
    assert status['headline'] == 'F&O paper alerts sent'
