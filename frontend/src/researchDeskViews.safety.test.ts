import { describe, expect, it } from 'vitest'

import { liveSafetyLabel } from './researchDeskViews'

describe('learning live-safety labels', () => {
  it('renders missing or unverified safety as UNVERIFIED', () => {
    expect(liveSafetyLabel()).toBe('UNVERIFIED')
    expect(liveSafetyLabel({ live_locked: true, live_lock_verified: false })).toBe('UNVERIFIED')
    expect(liveSafetyLabel({ live_locked: null, live_lock_verified: false })).toBe('UNVERIFIED')
  })

  it('renders locked only when the canonical boundary is verified locked', () => {
    expect(liveSafetyLabel({ live_locked: true, live_lock_verified: true })).toBe('VERIFIED / LOCKED')
  })

  it('renders a verified unlocked state explicitly instead of pretending safety', () => {
    expect(liveSafetyLabel({ live_locked: false, live_lock_verified: true })).toBe('VERIFIED / UNLOCKED')
  })
})
