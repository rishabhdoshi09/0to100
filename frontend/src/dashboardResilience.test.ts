import { describe, expect, it } from 'vitest'
import { reconcileDashboard } from './dashboardResilience'
import type { DashboardPayload } from './types'

function dashboard(overrides: Partial<DashboardPayload> = {}): DashboardPayload {
  const base = {
    generated_at: '2026-09-09T12:00:00Z',
    market: {
      available: true,
      health: 'Healthy',
      summary: 'saved market',
      trade_stance: 'saved stance',
      breadth: 'positive',
      leaders: ['IT'],
      laggards: [],
      nifty_change_1d: 1,
      nifty_change_5d: 2,
      vix: 12,
    },
    scan: {
      available: true,
      scanned_at: '2026-09-09T12:00:00Z',
      universe_size: 500,
      summary: { with_any_setup: 1 },
      records: [{ symbol: 'ABC', score: 80 }],
    },
    long_term: {
      available: true,
      scanned_at: '2026-09-09T12:01:00Z',
      summary: { quality_compounder: 1 },
      records: [{ symbol: 'ABC', combined_score: 75 }],
      job: {},
    },
    paper: {
      available: true,
      enabled: true,
      supervisor_running: true,
      capital: 100000,
      equity: 101000,
      equity_curve: [100000, 101000],
      open_risk: 500,
      risk_per_trade_pct: 0.01,
      max_positions: 5,
      open_positions: [{ symbol: 'ABC' }],
      closed_trades: [{ symbol: 'XYZ', pnl: 500 }],
      refusals: [{ symbol: 'NOPE' }],
      last_cycle: { cycle_id: 'old' },
      learning: { summary: 'persisted learning' },
    },
    autonomy: {
      available: true,
      running: true,
      process_running: true,
      state: 'OBSERVING',
      plain_state: 'Observing',
      explanation: 'running',
      heartbeat_ist: '18:30:00',
      new_paper_entries: true,
      existing_exits: true,
      research_enabled: true,
      recent_dialogue: [{ message: 'old' }],
      recent_transitions: [{ from: 'WAIT', to: 'BUY' }],
      jobs: { PENDING: 1 },
      jobs_recent: [{ kind: 'MARKET_SCAN' }],
      last_cycle: { cycle_id: 'old' },
    },
    operations: {
      available: true,
      running: true,
      heartbeat: 'now',
      active_lanes: {},
      counts: {},
      active: [],
      recent: [],
      latest: {},
    },
    news: {
      available: true,
      stats: { total: 1 },
      articles: [{ title: 'saved news' }],
      source_health: [],
      latest_refresh: {},
    },
    fno: {
      available: true,
      source: 'saved',
      mapped_underlyings: 1,
      underlyings: [{ symbol: 'ABC' }],
      exclusions: [],
    },
    data: {
      ready: true,
      snapshot: { ready: true, snapshot_id: 'snap', latest_date: '2026-09-09', source: 'nse' },
      bhavcopy: { ready: true, symbols: 500, sessions: 100, latest_date: '2026-09-09', csv_files: 100, cache_exists: true },
      scan_saved: true,
      scan_records: 1,
      long_term_saved: true,
      long_term_records: 1,
      blockers: [],
    },
    conviction: [{ symbol: 'ABC', conviction_score: 80 }],
  } as unknown as DashboardPayload
  return { ...base, ...overrides }
}

describe('reconcileDashboard', () => {
  it('keeps durable reads visible while current safety state stays fail-closed', () => {
    const previous = dashboard()
    const incoming = dashboard({
      generated_at: '2026-09-09T12:10:00Z',
      market: { ...previous.market, available: false, summary: 'unavailable' },
      scan: { available: false, universe_size: 0, summary: {}, records: [] },
      long_term: { available: false, summary: {}, records: [], job: {} },
      conviction: [],
      paper: {
        available: false,
        enabled: false,
        supervisor_running: false,
        capital: 0,
        equity: 0,
        equity_curve: [],
        open_risk: 0,
        risk_per_trade_pct: 0.01,
        max_positions: 0,
        open_positions: [],
        closed_trades: [],
        refusals: [],
        last_cycle: {},
      },
      autonomy: {
        available: false,
        running: false,
        process_running: false,
        state: 'DATA_BLOCKED',
        plain_state: 'Data blocked',
        explanation: 'snapshot stale',
        heartbeat_ist: '18:40:00',
        new_paper_entries: false,
        existing_exits: false,
        research_enabled: false,
        recent_dialogue: [],
        recent_transitions: [],
        jobs: { PENDING: 3 },
        jobs_recent: [],
        last_cycle: {},
      },
      news: { available: false, stats: {}, articles: [], source_health: [], error: 'refreshing' },
      fno: { available: false, source: 'unavailable', mapped_underlyings: 0, underlyings: [], exclusions: [], error: 'refreshing' },
      data: {
        ...previous.data,
        ready: false,
        snapshot: { ...previous.data.snapshot, ready: false },
        scan_saved: false,
        scan_records: 0,
        long_term_saved: false,
        long_term_records: 0,
        blockers: ['snapshot_stale'],
      },
    })

    const result = reconcileDashboard(previous, incoming)

    expect(result.scan.records.map((row) => row.symbol)).toEqual(['ABC'])
    expect(result.scan.available).toBe(false)
    expect(result.long_term.records.map((row) => row.symbol)).toEqual(['ABC'])
    expect(result.conviction.map((row) => row.symbol)).toEqual(['ABC'])

    expect(result.paper.available).toBe(true)
    expect(result.paper.enabled).toBe(false)
    expect(result.paper.supervisor_running).toBe(false)
    expect(result.paper.capital).toBe(100000)
    expect(result.paper.open_positions.map((row) => row.symbol)).toEqual(['ABC'])
    expect(result.paper.closed_trades.map((row) => row.symbol)).toEqual(['XYZ'])

    expect(result.autonomy.running).toBe(false)
    expect(result.autonomy.process_running).toBe(false)
    expect(result.autonomy.new_paper_entries).toBe(false)
    expect(result.autonomy.state).toBe('DATA_BLOCKED')
    expect(result.autonomy.recent_transitions).toEqual([{ from: 'WAIT', to: 'BUY' }])
    expect(result.autonomy.last_cycle).toEqual({ cycle_id: 'old' })

    expect(result.news.available).toBe(false)
    expect(result.news.articles).toEqual([{ title: 'saved news' }])
    expect(result.fno.available).toBe(false)
    expect(result.fno.underlyings).toEqual([{ symbol: 'ABC' }])
    expect(result.data.ready).toBe(false)
    expect(result.data.blockers).toEqual(['snapshot_stale'])
    expect(result.data.scan_records).toBe(1)
  })

  it('uses newer durable paper and autonomy history when the incoming read has it', () => {
    const previous = dashboard()
    const incoming = dashboard({
      paper: {
        ...previous.paper,
        capital: 200000,
        equity: 202000,
        open_positions: [{ symbol: 'NEW' }] as typeof previous.paper.open_positions,
      },
      autonomy: {
        ...previous.autonomy,
        recent_transitions: [{ from: 'WAIT', to: 'AVOID' }],
      },
    })

    const result = reconcileDashboard(previous, incoming)
    expect(result.paper.capital).toBe(200000)
    expect(result.paper.open_positions.map((row) => row.symbol)).toEqual(['NEW'])
    expect(result.autonomy.recent_transitions).toEqual([{ from: 'WAIT', to: 'AVOID' }])
  })

  it('does not resurrect stale data when a healthy subsystem reports a real empty state', () => {
    const previous = dashboard()
    const incoming = dashboard({
      scan: {
        available: true,
        scanned_at: '2026-09-09T12:20:00Z',
        universe_size: 500,
        summary: { with_any_setup: 0 },
        records: [],
      },
      conviction: [],
      paper: {
        available: true,
        enabled: true,
        supervisor_running: true,
        capital: 100000,
        equity: 100000,
        equity_curve: [],
        open_risk: 0,
        risk_per_trade_pct: 0.01,
        max_positions: 5,
        open_positions: [],
        closed_trades: [],
        refusals: [],
        last_cycle: {},
      },
      autonomy: {
        ...previous.autonomy,
        available: true,
        recent_dialogue: [],
        recent_transitions: [],
        jobs_recent: [],
        last_cycle: {},
      },
      data: {
        ...previous.data,
        scan_saved: true,
        scan_records: 0,
      },
    })

    const result = reconcileDashboard(previous, incoming)
    expect(result.scan.records).toEqual([])
    expect(result.conviction).toEqual([])
    expect(result.paper.open_positions).toEqual([])
    expect(result.paper.closed_trades).toEqual([])
    expect(result.autonomy.recent_transitions).toEqual([])
    expect(result.data.scan_records).toBe(0)
  })
})
