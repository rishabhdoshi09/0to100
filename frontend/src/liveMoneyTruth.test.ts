import { describe, expect, it } from 'vitest'

import { checkSystemRows, liveMoneyStatus, liveMoneyStillLocked } from './backendControlPlane'


describe('live-money display truth', () => {
  it('does not turn missing safety evidence into Locked', () => {
    expect(liveMoneyStillLocked(undefined, undefined)).toBeNull()
    expect(liveMoneyStatus(undefined, undefined)).toBe('Unverified')
    const row = checkSystemRows(undefined, {}).find(item => item.id === 'live_money')
    expect(row?.status).toBe('Unverified')
  })

  it('renders an explicit verified locked state as Locked', () => {
    expect(liveMoneyStillLocked(true, { live_locked: true, live_lock_verified: true })).toBe(true)
    expect(liveMoneyStatus(true, { live_locked: true, live_lock_verified: true })).toBe('Locked')
  })

  it('never hides an explicit unlocked state', () => {
    expect(liveMoneyStillLocked(true, { live_locked: false, live_lock_verified: true })).toBe(false)
    expect(liveMoneyStatus(true, { live_locked: false, live_lock_verified: true })).toBe('Not locked')
  })
})
