import { describe, expect, it } from 'vitest'
import { paperCycleNote } from './paperCycleNote'

describe('paperCycleNote', () => {
  it('shows the recorded eligibility reasons', () => {
    const note = paperCycleNote({
      eligibility: 'BLOCKED_SAFETY',
      brain: { posture: 'STAND_ASIDE', posture_reason: 'book danger' },
      why_no_trade: {
        available: true,
        headline: 'No paper trade taken: BRAIN_STAND_ASIDE',
        decision: 'NO_TRADE',
        reasons: ['BRAIN_STAND_ASIDE'],
      },
    })
    expect(note?.headline).toContain('BRAIN_STAND_ASIDE')
    expect(note?.reasons).toEqual(['BRAIN_STAND_ASIDE'])
    expect(note?.eligibility).toBe('BLOCKED_SAFETY')
    expect(note?.posture).toBe('STAND_ASIDE')
  })

  it('stays empty when no cycle was recorded', () => {
    expect(paperCycleNote(undefined)).toBeNull()
    expect(paperCycleNote({})).toBeNull()
    expect(paperCycleNote({ why_no_trade: { available: false } })).toBeNull()
  })

  it('does not invent reasons when the blotter only has a headline', () => {
    const note = paperCycleNote({
      why_no_trade: {
        available: false,
        headline: 'No paper-autopilot cycle has been recorded yet.',
        reasons: ['NO_CYCLE_RECORDED'],
      },
    })
    expect(note?.headline).toContain('No paper-autopilot cycle')
    expect(note?.reasons).toEqual(['NO_CYCLE_RECORDED'])
    expect(note?.eligibility).toBe('')
  })
})
