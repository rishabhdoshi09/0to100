import { describe, expect, it } from 'vitest'
import { journeyTone, labKidLane, labKidTone, labStatusTone, readyLaneLabel } from './tradeDesk'

describe('trade desk copy', () => {
  it('labels Stage 2, Prime, and a complete ticket', () => {
    expect(readyLaneLabel('stage2')).toBe('Stage 2')
    expect(readyLaneLabel('prime')).toBe('Prime')
    expect(readyLaneLabel('actionable')).toBe('Ticket')
  })

  it('keeps live lock as a lock tone, not a pass', () => {
    expect(journeyTone('LOCKED')).toBe('is-lock')
    expect(journeyTone('PASS')).toBe('is-pass')
    expect(labStatusTone('READY')).toBe('is-pass')
    expect(labStatusTone('MISSING')).toBe('is-wait')
  })

  it('reads the backtest scoreboard like a report card', () => {
    expect(labKidLane('keep')).toBe('Passed')
    expect(labKidLane('skip')).toBe('Failed')
    expect(labKidLane('quiet')).toBe('Too few tries')
    expect(labKidTone('keep')).toBe('is-pass')
    expect(labKidTone('skip')).toBe('is-lock')
  })
})
