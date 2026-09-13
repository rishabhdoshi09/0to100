import { describe, expect, it } from 'vitest'
import { deriveTruthLanes } from './truthfulStatus'
import type { DashboardPayload } from './types'

function dashboard(overrides: Record<string, unknown> = {}): DashboardPayload {
  const base = {
    generated_at: '2026-09-06T06:00:00Z',
    data: {
      ready: true,
      snapshot: { ready: true, snapshot_id: 'snap', latest_date: '2026-09-04', source: 'NSE' },
      bhavcopy: {
        ready: true,
        symbols: 1801,
        sessions: 1801,
        latest_date: '2026-09-04',
        available_session: '2026-09-04',
        expected_latest_completed_session: '2026-09-04',
        csv_files: 1801,
        cache_exists: true,
        current: true,
        source: 'NSE bhavcopy',
      },
      scan_saved: true,
      scan_records: 100,
      long_term_saved: true,
      long_term_records: 20,
      blockers: [],
    },
    scan: { available: true, scanned_at: '2026-09-06T05:30:00Z', universe_size: 1801, summary: {}, records: [] },
    long_term: { available: true, scanned_at: '2026-09-05T16:30:00Z', summary: {}, records: [], fundamentals_source: 'saved fundamentals' },
    autonomy: {
      available: true,
      running: true,
      process_running: true,
      state: 'RUNNING',
      plain_state: 'Supervisor working',
      explanation: '',
      heartbeat_ist: '2026-09-06T11:30:00+05:30',
      new_paper_entries: true,
      recent_dialogue: [],
      jobs: {},
      jobs_recent: [],
      active_failures: [],
    },
    operations: {
      available: true,
      running: true,
      heartbeat: '2026-09-06T06:00:00Z',
      active_lanes: {},
      counts: {},
      active: [],
      recent: [],
      latest: {},
    },
  } as unknown as DashboardPayload
  return Object.assign(base, overrides)
}

describe('truthful persisted-state monitor', () => {
  it('does not call saved scan fresh when official history is stale or the latest refresh failed', () => {
    const d = dashboard()
    d.data.bhavcopy.current = false
    d.data.bhavcopy.expected_latest_completed_session = '2026-09-05'
    d.operations.latest.MARKET_SCAN = {
      operation_id: 'op1', kind: 'MARKET_SCAN', lane: 'scan', status: 'FAILED', requested_by: 'test',
      requested_at: 1, updated_at: 2, attempt: 2, stage: 'FAILED', message: 'scan failed',
      progress_current: 10, progress_total: 100, error_message: 'provider timeout',
    }
    const lanes = deriveTruthLanes(d)
    expect(lanes.find((lane) => lane.id === 'prices')?.status).toBe('STALE')
    expect(lanes.find((lane) => lane.id === 'scan')?.status).toBe('STALE')
    expect(lanes.find((lane) => lane.id === 'scan')?.detail).toContain('provider timeout')
  })

  it('shows durable work as REFRESHING without erasing the last successful artifact timestamp', () => {
    const d = dashboard()
    d.operations.active.push({
      operation_id: 'op2', kind: 'MARKET_SCAN', lane: 'scan', status: 'RUNNING', requested_by: 'user',
      requested_at: 1, updated_at: 2, attempt: 1, stage: 'SCANNING', message: 'Evaluating NSE universe',
      progress_current: 800, progress_total: 1801,
    })
    const scan = deriveTruthLanes(d).find((lane) => lane.id === 'scan')
    expect(scan?.status).toBe('REFRESHING')
    expect(scan?.asOf).toBe('2026-09-06T05:30:00Z')
    expect(scan?.detail).toContain('Evaluating NSE universe')
  })

  it('surfaces recent retryable autonomy failures instead of a decorative all-green state', () => {
    const d = dashboard()
    d.autonomy.jobs_recent = [{
      job_id: 'job-1', job_type: 'OUTCOME_RESOLUTION', status: 'RETRYABLE_FAILED', attempt: 3,
      error_message: 'Official settlement incomplete',
    }]
    const jobs = deriveTruthLanes(d).find((lane) => lane.id === 'jobs')
    expect(jobs?.status).toBe('DEGRADED')
    expect(jobs?.detail).toContain('1 recent failed/retryable job')
  })
})
