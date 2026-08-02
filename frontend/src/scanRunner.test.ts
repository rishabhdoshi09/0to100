import { describe, expect, it } from 'vitest'
import {
  buildProgressLine,
  formatElapsed,
  friendlyStageLabel,
  isActiveStatus,
  isTerminalStatus,
  progressPercent,
  qualifiedResultLine,
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

  it('surfaces qualified counts from persisted results', () => {
    const op = baseOperation({
      status: 'SUCCEEDED',
      result: { summary: { qualified: 26 }, records: 26 },
    })
    expect(qualifiedResultLine(op)).toBe('26 qualified ideas found')
  })
})
