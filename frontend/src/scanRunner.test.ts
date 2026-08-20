import { describe, expect, it } from 'vitest'
import {
  buildProgressLine,
  formatElapsed,
  friendlyStageLabel,
  isActiveStatus,
  isTerminalStatus,
  progressPercent,
  qualifiedResultLine,
  secondsSinceUpdate,
  staleProgressHint,
  estimateRemainingSeconds,
  formatRemaining,
  jobClock,
  deskWaitClock,
  recoWorkspaceClock,
  ideasPollMs,
  TERMINAL_STATUSES,
} from './scanRunner'
import type { OperationRecord } from './types'

const baseOperation = (overrides: Partial<OperationRecord> = {}): OperationRecord => ({
  operation_id: 'abc123',
  kind: 'MARKET_SCAN',
  lane: 'market_scan',
  status: 'RUNNING',
  requested_by: 'terminal',
  requested_at: Date.now() / 1000,
  updated_at: Date.now() / 1000,
  attempt: 1,
  stage: 'SCANNING',
  message: 'Working',
  progress_current: 0,
  progress_total: 0,
  ...overrides,
})

describe('scanRunner semantics', () => {
  it('maps backend stages to friendly retail language', () => {
    expect(friendlyStageLabel('PREPARING_HISTORY', 'RUNNING')).toBe('Preparing official NSE price history…')
    expect(friendlyStageLabel('WAITING_HISTORY', 'RUNNING')).toContain('shared NSE history')
    expect(friendlyStageLabel('SCANNING', 'RUNNING')).toBe('Scanning market candidates…')
    expect(friendlyStageLabel('STARTING', 'RUNNING')).toBe('Worker picked up the job…')
    expect(friendlyStageLabel('ACCEPTED', 'RUNNING')).toBe('Worker picked up the job…')
    expect(friendlyStageLabel('TECHNICAL_SCREEN', 'RUNNING')).toContain('technical')
    expect(friendlyStageLabel('FUNDAMENTALS', 'RUNNING')).toContain('fundamentals')
    expect(friendlyStageLabel('', 'SUCCEEDED')).toBe('Scan complete')
    expect(friendlyStageLabel('', 'FAILED')).toBe('Scan failed')
    expect(friendlyStageLabel('', 'CANCELLED')).toBe('Scan stopped')
  })

  it('makes PENDING queue state honest about worker health', () => {
    expect(friendlyStageLabel('', 'PENDING', { running: false })).toContain('OFFLINE')
    expect(friendlyStageLabel('', 'PENDING', { running: true })).toContain('ONLINE')
    expect(friendlyStageLabel('', 'PENDING', { running: true }, 20)).toContain('has not leased')
    expect(friendlyStageLabel('', 'PENDING', { running: true, activeKind: 'MARKET_SCAN' }))
      .toContain('Queued behind MARKET_SCAN')
    // Unknown worker state after a few seconds is treated as offline, not endless "waiting".
    expect(friendlyStageLabel('', 'PENDING', { running: null }, 12)).toContain('OFFLINE')
  })

  it('detects terminal and active statuses', () => {
    expect(isTerminalStatus('SUCCEEDED')).toBe(true)
    expect(isTerminalStatus('RUNNING')).toBe(false)
    expect(isActiveStatus('PENDING')).toBe(true)
    expect(isActiveStatus('SUCCEEDED')).toBe(false)
    TERMINAL_STATUSES.forEach((status) => expect(isTerminalStatus(status)).toBe(true))
  })

  it('builds progress line only with a real denominator', () => {
    expect(buildProgressLine(baseOperation({ progress_current: 487, progress_total: 1842 })))
      .toContain('487')
    expect(buildProgressLine(baseOperation({
      kind: 'LONG_TERM_SCAN',
      stage: 'TECHNICAL_SCREEN',
      message: 'Technical screen · 120/500 symbols',
      progress_current: 120,
      progress_total: 500,
    }))).toContain('120')
    expect(buildProgressLine(baseOperation({ progress_current: 10, progress_total: 0, message: 'Worker accepted' })))
      .toBe('Worker accepted')
  })

  it('formats elapsed time without raw second spam', () => {
    expect(formatElapsed(12)).toBe('12s')
    expect(formatElapsed(211)).toBe('3m 31s')
  })

  it('never invents a percentage without totals', () => {
    expect(progressPercent(baseOperation({ progress_pct: null, progress_total: 0 }))).toBeNull()
    expect(progressPercent(baseOperation({ progress_current: 50, progress_total: 200 }))).toBe(25)
  })

  it('estimates remaining time from observed scan rate', () => {
    expect(estimateRemainingSeconds(3, 10, 100, 1000)).toBeNull()
    expect(estimateRemainingSeconds(40, 25, 800, 3200)).toBe(120)
    expect(formatRemaining(125)).toBe('~2m 5s left')
    expect(formatRemaining(null, 120)).toBe('usually ~2m')
    const clock = jobClock({
      kind: 'MARKET_SCAN',
      isActive: true,
      friendlyPhase: 'Scanning market candidates…',
      progressLine: '1,200 of 3,191 stocks',
      percent: 38,
      elapsedSeconds: 50,
      current: 1200,
      total: 3191,
    })
    expect(clock.button).toMatch(/Working… 38%/)
    expect(clock.button).toMatch(/left/)
    expect(clock.line).toMatch(/elapsed/)
  })

  it('explains Pulse wait with a usual ETA', () => {
    const waiting = deskWaitClock({ kind: 'MARKET_PULSE', elapsedSeconds: 1 })
    expect(waiting.doing).toMatch(/Pulse/)
    expect(waiting.button).toMatch(/usually ~8s/)
    expect(waiting.doing).toMatch(/last scan/)
  })

  it('explains Ideas category wait with a usual ETA', () => {
    const waiting = recoWorkspaceClock({ elapsedSeconds: 1 })
    expect(waiting.doing).toMatch(/SEPA|Best Setups/)
    expect(waiting.doing).toMatch(/Wealth Builders/)
    expect(waiting.button).toMatch(/usually ~8s/)
    expect(waiting.line).toMatch(/elapsed 1s/)
    const later = recoWorkspaceClock({ elapsedSeconds: 6 })
    expect(later.button).toMatch(/left|few seconds/)
    const overtime = recoWorkspaceClock({ elapsedSeconds: 20 })
    expect(overtime.button).toMatch(/taking longer than usual/)
    expect(overtime.button).not.toMatch(/few seconds left/)
    expect(ideasPollMs({ sepa_pending: true })).toBe(4000)
    expect(ideasPollMs({ stale_ranking: true })).toBe(4000)
    expect(ideasPollMs({})).toBe(60_000)
    const duringScan = recoWorkspaceClock({
      elapsedSeconds: 2,
      scan: {
        kind: 'MARKET_SCAN',
        isActive: true,
        friendlyPhase: 'Scanning market candidates…',
        progressLine: '1,200 of 3,191 stocks',
        percent: 38,
        elapsedSeconds: 50,
        current: 1200,
        total: 3191,
      },
    })
    expect(duringScan.doing).toContain('1,200')
    expect(duringScan.button).toMatch(/Working… 38%/)
  })

  it('surfaces qualified counts from persisted results', () => {
    const op = baseOperation({
      status: 'SUCCEEDED',
      result: { summary: { qualified: 26 }, records: 26 },
    })
    expect(qualifiedResultLine(op)).toBe('26 qualified ideas found')
  })

  it('exposes heartbeat age and stale hints while running', () => {
    const now = Date.now()
    const fresh = baseOperation({
      status: 'RUNNING',
      stage: 'SCANNING',
      updated_at: now / 1000,
      progress_current: 10,
      progress_total: 100,
    })
    expect(secondsSinceUpdate(fresh, now)).toBe(0)
    expect(staleProgressHint(fresh, now)).toBeNull()

    const stuckAtZero = baseOperation({
      status: 'RUNNING',
      stage: 'SCANNING',
      updated_at: (now / 1000) - 20,
      progress_current: 0,
      progress_total: 1800,
    })
    expect(secondsSinceUpdate(stuckAtZero, now)).toBe(20)
    expect(staleProgressHint(stuckAtZero, now) || '').toMatch(/0\/1800|last engine update/i)

    const waiting = baseOperation({
      status: 'RUNNING',
      stage: 'WAITING_HISTORY',
      updated_at: (now / 1000) - 3,
    })
    expect(staleProgressHint(waiting, now) || '').toMatch(/shared history/i)
  })
})
