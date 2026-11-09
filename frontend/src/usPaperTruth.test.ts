import { describe, expect, it } from 'vitest'

import { usPaperBlockers, usPaperModeLabel } from './usPaperTruth'


describe('US paper Home-card truth', () => {
  it('never shows an unrestored saved state as a routine OFF', () => {
    const paper = {
      armed: false,
      disarmed_reason: 'US_PAPER_STATE_RECOVERY_REQUIRED',
      state_recovery_required: true,
      state_recovery_error: 'JSONDecodeError:Expecting value',
    }
    expect(usPaperModeLabel(paper)).toBe(' · paper BLOCKED: saved state must be restored')
    const blockers = usPaperBlockers(paper)
    expect(blockers).toHaveLength(1)
    expect(blockers[0]).toContain('No entries, exits or P&L settlement')
    expect(blockers[0]).toContain('JSONDecodeError')
  })

  it('does not invent a state when the projection is missing', () => {
    expect(usPaperModeLabel(undefined)).toBe(' · paper status unknown')
    expect(usPaperBlockers(undefined)).toEqual([])
  })

  it('shows armed and operator-disarmed states plainly', () => {
    expect(usPaperModeLabel({ armed: true, state_recovery_required: false })).toBe(' · paper auto ON')
    expect(usPaperModeLabel({ armed: false, disarmed_reason: 'user', state_recovery_required: false }))
      .toBe(' · paper auto OFF')
    expect(usPaperModeLabel({ armed: false, disarmed_reason: 'circuit breaker: day P&L $-3,100' }))
      .toBe(' · paper auto OFF (circuit breaker: day P&L $-3,100)')
  })

  it('flags a legacy-watermark pool for reconciliation', () => {
    const rows = usPaperBlockers({ armed: true, accounting_reconciliation_required: true })
    expect(rows).toHaveLength(1)
    expect(rows[0]).toContain('Reconcile')
  })
})
