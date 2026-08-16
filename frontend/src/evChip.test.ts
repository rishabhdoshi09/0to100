import { describe, expect, it } from 'vitest'
import { EV_MIN_N, hasGatedEv } from './evChip'

describe('hasGatedEv', () => {
  it('hides EV below the sample floor', () => {
    expect(hasGatedEv({ ev_pct: 1.2, ev_n: 12 })).toBe(false)
    expect(hasGatedEv({ ev_pct: 1.2, ev_n: EV_MIN_N - 1 })).toBe(false)
    expect(hasGatedEv({ ev_n: 80 })).toBe(false)
  })

  it('shows EV only when n >= 30 and ev_pct is present', () => {
    expect(hasGatedEv({ ev_pct: 0.8, ev_n: 30, ev_lb_pct: 0.4 })).toBe(true)
  })
})
