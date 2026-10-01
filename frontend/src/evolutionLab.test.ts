import { describe, expect, it } from 'vitest'
import {
  formatPct,
  formatR,
  incrementalTone,
  probationProgressNote,
  sampleFloorNote,
} from './evolutionLabModel'

describe('evolution lab formatting', () => {
  it('renders a missing R as a dash, not as zero', () => {
    expect(formatR(null)).toBe('—')
    expect(formatR(undefined)).toBe('—')
    expect(formatR(Number.NaN)).toBe('—')
    expect(formatR(0)).toBe('0.00R')
    expect(formatR(1.5)).toBe('+1.50R')
    expect(formatR(-0.75)).toBe('-0.75R')
  })

  it('renders a missing rate as a dash', () => {
    expect(formatPct(null)).toBe('—')
    expect(formatPct(undefined)).toBe('—')
    expect(formatPct(0.6)).toBe('60%')
  })

  it('never paints an unproven incremental edge as positive or negative', () => {
    expect(incrementalTone(null)).toBe('neutral')
    expect(incrementalTone(undefined)).toBe('neutral')
    expect(incrementalTone(Number.NaN)).toBe('neutral')
    expect(incrementalTone(0)).toBe('neutral')
    expect(incrementalTone(0.4)).toBe('positive')
    expect(incrementalTone(-0.4)).toBe('negative')
  })

  it('flags a paired sample below the promotion floor', () => {
    expect(sampleFloorNote(7, 30)).toBe('7/30 paired observations — below the promotion floor')
    expect(sampleFloorNote(30, 30)).toBe('30 paired observations')
    expect(sampleFloorNote(45, 30)).toBe('45 paired observations')
  })

  it('reports no probation note when a policy is not in probation', () => {
    expect(probationProgressNote(null, null)).toBe('')
  })

  it('shows probation progress toward the new-evidence floor', () => {
    const probation = { started_at: '2026-09-01T00:00:00Z', minimum_additional_paired: 10 }
    expect(probationProgressNote(probation, { since_probation_paired: 3, passed: false }))
      .toBe('probation since 2026-09-01T00:00:00Z · 3/10 new paired observations since checkpoint (in progress)')
    expect(probationProgressNote(probation, { since_probation_paired: 10, passed: true }))
      .toBe('probation since 2026-09-01T00:00:00Z · 10/10 new paired observations since checkpoint (complete)')
  })
})
