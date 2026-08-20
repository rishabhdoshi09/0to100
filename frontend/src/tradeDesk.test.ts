import { describe, expect, it } from 'vitest'
import { journeyTone, labStatusTone, readyLaneLabel } from './tradeDesk'

describe('trade desk copy', () => {
  it('labels Prime separately from a complete ticket', () => {
    expect(readyLaneLabel('prime')).toBe('Prime')
    expect(readyLaneLabel('actionable')).toBe('Ticket')
  })

  it('keeps live lock as a lock tone, not a pass', () => {
    expect(journeyTone('LOCKED')).toBe('is-lock')
    expect(journeyTone('PASS')).toBe('is-pass')
    expect(labStatusTone('READY')).toBe('is-pass')
    expect(labStatusTone('MISSING')).toBe('is-wait')
  })
})
