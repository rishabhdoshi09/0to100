import { describe, expect, it } from 'vitest'
import {
  buildProgressLine,
  estimateEtaSeconds,
  formatEta,
  friendlyStageLabel,
  isActiveStatus,
  isTerminalStatus,
  progressPercent,
  qualifiedResultLine,
  seedKindMatches,
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
    expect(friendlyStageLabel('PREPARING_HISTORY', 'RUNNING')).toBe('Preparing market history…')
    expect(friendlyStageLabel('WARMING_HISTORY', 'RUNNING')).toBe('Warming official price cache…')
    expect(friendlyStageLabel('SCANNING', 'RUNNING')).toBe('Scanning market candidates…')
    expect(friendlyStageLabel('', 'SUCCEEDED')).toBe('Scan complete')
    expect(friendlyStageLabel('', 'FAILED')).toBe('Scan failed')
    expect(friendlyStageLabel('', 'CANCELLED')).toBe('Scan stopped')
    expect(friendlyStageLabel('', 'PENDING', 20)).toBe('Waiting for the scan worker…')
  })

  it('detects terminal and active statuses', () => {
    expect(isTerminalStatus('SUCCEEDED')).toBe(true)
    expect(isTerminalStatus('RUNNING')).toBe(false)
    expect(isActiveStatus('PENDING')).toBe(true)
    expect(isActiveStatus('SUCCEEDED')).toBe(false)
    TERMINAL_STATUSES.forEach((status) => expect(isTerminalStatus(status)).toBe(true))
  })

  it('attaches long-term refresh jobs to the long-term scan runner', () => {
    expect(seedKindMatches('LONG_TERM_REFRESH', 'LONG_TERM_SCAN')).toBe(true)
    expect(seedKindMatches('LONG_TERM_SCAN', 'LONG_TERM_SCAN')).toBe(true)
    expect(seedKindMatches('MARKET_SCAN', 'LONG_TERM_SCAN')).toBe(false)
  })

  it('builds progress line only with a real denominator', () => {
    expect(buildProgressLine(baseOperation({ progress_current: 487, progress_total: 1842 })))
      .toBe('Scanning 487 of 1,842 stocks')
    expect(buildProgressLine(baseOperation({ progress_current: 10, progress_total: 0 }))).toBeNull()
    expect(buildProgressLine(baseOperation({
      stage: 'PREPARING_HISTORY',
      progress_current: 11,
      progress_total: 500,
    }))).toBeNull()
  })

  it('never invents a percentage without totals', () => {
    expect(progressPercent(baseOperation({ progress_pct: null, progress_total: 0 }))).toBeNull()
    expect(progressPercent(baseOperation({ progress_current: 50, progress_total: 200 }))).toBe(25)
    expect(progressPercent(baseOperation({
      stage: 'PREPARING_HISTORY',
      progress_current: 11,
      progress_total: 500,
      progress_pct: 2.2,
    }))).toBeNull()
  })

  it('estimates ETA from observed scan pace and never invents one early', () => {
    expect(estimateEtaSeconds(baseOperation({ progress_current: 0, progress_total: 2000 }), 10)).toBeNull()
    expect(estimateEtaSeconds(baseOperation({ progress_current: 400, progress_total: 2000 }), 40)).toBe(160)
    expect(estimateEtaSeconds(baseOperation({
      stage: 'PREPARING_HISTORY',
      progress_current: 11,
      progress_total: 500,
    }), 60)).toBeNull()
    expect(formatEta(160)).toBe('about 3 min')
    expect(formatEta(12)).toBe('under 15s')
    expect(formatEta(null)).toBeNull()
  })

  it('surfaces qualified counts from persisted results', () => {
    const op = baseOperation({
      status: 'SUCCEEDED',
      result: { summary: { qualified: 26 }, records: 26 },
    })
    expect(qualifiedResultLine(op)).toBe('26 qualified ideas found')
  })
})
