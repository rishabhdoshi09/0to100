import { describe, expect, it } from 'vitest'
import {
  attributionWarning,
  distributionBars,
  formatPct,
  formatR,
  hasSettledEvidence,
  progressToFloor,
  stateTone,
} from './forwardEvidenceModel'
import type { ForwardEvidenceBoard } from './forwardEvidenceModel'

function board(overrides: Partial<ForwardEvidenceBoard> = {}): ForwardEvidenceBoard {
  return {
    schema_version: 1,
    state: 'NO_MARKET_EVIDENCE',
    headline: 'No paper trade has settled yet.',
    evidence_class: 'PAPER_FORWARD',
    min_sample: 30,
    settled_trades: 0,
    cells: [],
    cells_usable_for_ranking: 0,
    by_setup: [],
    by_regime: [],
    by_sector: [],
    r_distribution: {},
    unresolved: [],
    unresolved_count: 0,
    unattributable_open: 0,
    non_market_evidence: { historical_replay_cells: 0, test_fixture_cells: 0, note: '' },
    ...overrides,
  }
}

describe('forward evidence board', () => {
  it('never paints an empty board with the healthy tone', () => {
    expect(stateTone('NO_MARKET_EVIDENCE')).toBe('forward-evidence__state--none')
    expect(stateTone('MEASURED')).toBe('forward-evidence__state--measured')
    expect(stateTone('ACCUMULATING')).toBe('forward-evidence__state--accumulating')
  })

  it('treats an unknown state as no evidence rather than as healthy', () => {
    expect(stateTone('SOMETHING_NEW')).toBe('forward-evidence__state--none')
  })

  it('knows the difference between nothing settled and something settled', () => {
    expect(hasSettledEvidence(board())).toBe(false)
    expect(hasSettledEvidence(board({ settled_trades: 1 }))).toBe(true)
    expect(hasSettledEvidence(null)).toBe(false)
  })

  it('shows how far a context is from counting', () => {
    expect(progressToFloor(7, 30)).toBe('7/30 to count')
    expect(progressToFloor(30, 30)).toBe('counts toward ranking')
  })

  it('renders a missing R as a dash, not as zero', () => {
    expect(formatR(null)).toBe('—')
    expect(formatR(Number.NaN)).toBe('—')
    expect(formatR(0)).toBe('0.00R')
    expect(formatR(1.5)).toBe('+1.50R')
    expect(formatR(-0.75)).toBe('-0.75R')
  })

  it('renders a missing rate as a dash', () => {
    expect(formatPct(null)).toBe('—')
    expect(formatPct(0.6)).toBe('60%')
  })

  it('keeps every R bucket visible even when empty', () => {
    const bars = distributionBars({})
    expect(bars).toHaveLength(6)
    expect(bars.every((b) => b.count === 0 && b.share === 0)).toBe(true)
  })

  it('scales the distribution to its largest bucket', () => {
    const bars = distributionBars({ '-1R..0': 10, '0..1R': 5 })
    const worst = bars.find((b) => b.bucket === '-1R..0')
    const middling = bars.find((b) => b.bucket === '0..1R')
    expect(worst?.share).toBe(1)
    expect(middling?.share).toBe(0.5)
  })

  it('warns when open trades cannot name their decision', () => {
    expect(attributionWarning(board({ unattributable_open: 2 })))
      .toContain('2 open trades cannot name the decision')
    expect(attributionWarning(board({ unattributable_open: 1 })))
      .toContain('1 open trade cannot name')
    expect(attributionWarning(board())).toBe('')
  })
})
