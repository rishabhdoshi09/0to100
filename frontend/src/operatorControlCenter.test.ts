import { describe, expect, it } from 'vitest'

import {
  OPERATOR_ACTIONS,
  operatorActionDisabled,
  operatorState,
} from './operatorControlCenter'
import type { DashboardPayload } from './types'

function dashboard(overrides: Partial<DashboardPayload> = {}): DashboardPayload {
  return {
    generated_at: '',
    market: {
      available: true,
      health: 'OK',
      summary: '',
      trade_stance: '',
      breadth: '',
      leaders: [],
      laggards: [],
      nifty_change_1d: null,
      nifty_change_5d: null,
      vix: null,
    },
    daily_wrap: [],
    scan: { available: true, universe_size: 2655, summary: {}, records: [] },
    long_term: { available: true, summary: {}, records: [], job: {} },
    paper: {
      enabled: true,
      supervisor_running: true,
      capital: 0,
      equity: 0,
      open_risk: 0,
      risk_per_trade_pct: 0.01,
      max_positions: 5,
      open_positions: [],
      closed_trades: [],
    },
    autonomy: {
      running: true,
      process_running: true,
      state: 'RUNNING',
      plain_state: 'Running',
      explanation: '',
      heartbeat_ist: '',
      new_paper_entries: true,
      recent_dialogue: [],
      jobs: {},
      broker: {
        state: 'READY',
        ready: true,
        live_data_ready: true,
        execution_ready: false,
        auth_ready: true,
        login_required: false,
        auth_status: 'READY',
        reason_code: '',
        detail: '',
        snapshot_id: '',
      },
    },
    operations: {
      available: true,
      running: true,
      worker_pid: 123,
      heartbeat: 'now',
      active_lanes: {},
      counts: {},
      active: [],
      recent: [],
      latest: {},
    },
    news: { available: true, stats: {}, articles: [], source_health: [] },
    fno: { available: true, source: 'kite', mapped_underlyings: 0, underlyings: [], exclusions: [] },
    data: {
      ready: true,
      snapshot: { ready: true, snapshot_id: 'x', latest_date: '2026-09-11', source: 'official' },
      bhavcopy: { ready: true, symbols: 3376, sessions: 1806, latest_date: '2026-09-11', csv_files: 1, cache_exists: true },
      scan_saved: true,
      scan_records: 0,
      long_term_saved: true,
      long_term_records: 0,
      blockers: [],
    },
    conviction: [],
    ...overrides,
  }
}

describe('operator control center', () => {
  it('exposes all expected safe operator controls and no live-order controls', () => {
    const controls = OPERATOR_ACTIONS.map((action) => action.control)
    expect(controls).toEqual(expect.arrayContaining([
      'RUN_SCAN_NOW',
      'REFRESH_DATA_NOW',
      'REFRESH_NEWS_NOW',
      'REFRESH_LONG_TERM_NOW',
      'REFRESH_FNO_NOW',
      'REFRESH_MARKET_REPORT_NOW',
      'RUN_CYCLE_NOW',
    ]))
    expect(controls.some((control) => /LIVE|BUY|SELL|UNLOCK/.test(control))).toBe(false)
  })

  it('reports running, refreshing and attention from current operator-plane truth', () => {
    expect(operatorState(dashboard())).toBe('RUNNING')

    const base = dashboard()
    const activeScan = {
      operation_id: 'scan-1',
      kind: 'MARKET_SCAN',
      lane: 'scan',
      status: 'RUNNING',
      requested_by: 'user',
      requested_at: 1,
      updated_at: 1,
      attempt: 1,
      stage: 'SCANNING',
      message: 'Scanning',
      progress_current: 10,
      progress_total: 2655,
    }
    expect(operatorState(dashboard({
      operations: { ...base.operations, active: [activeScan], latest: { MARKET_SCAN: activeScan } },
    }))).toBe('REFRESHING')

    expect(operatorState(dashboard({
      operations: { ...base.operations, running: false },
    }))).toBe('ATTENTION')

    expect(operatorState(dashboard({
      autonomy: { ...base.autonomy, running: false, process_running: false },
    }))).toBe('ATTENTION')
  })

  it('disables a duplicate durable operation while it is active', () => {
    const base = dashboard()
    const activeScan = {
      operation_id: 'scan-1',
      kind: 'MARKET_SCAN',
      lane: 'scan',
      status: 'RUNNING',
      requested_by: 'user',
      requested_at: 1,
      updated_at: 1,
      attempt: 1,
      stage: 'SCANNING',
      message: 'Scanning',
      progress_current: 0,
      progress_total: 2655,
    }
    const payload = dashboard({
      operations: { ...base.operations, active: [activeScan], latest: { MARKET_SCAN: activeScan } },
    })
    const action = OPERATOR_ACTIONS.find((row) => row.control === 'RUN_SCAN_NOW')!
    expect(operatorActionDisabled(payload, action)).toBe(true)
  })
})
