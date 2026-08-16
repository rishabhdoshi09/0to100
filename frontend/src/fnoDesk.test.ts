import { describe, expect, it } from 'vitest'
import { canOpenStockFromFno, defaultFnoFocus, isFnoIndex } from './fnoDesk'

describe('F&O desk routing', () => {
  it('treats index underlyings as desk-only, not company workspaces', () => {
    expect(isFnoIndex('NIFTY')).toBe(true)
    expect(isFnoIndex('banknifty')).toBe(true)
    expect(isFnoIndex('RELIANCE')).toBe(false)
    expect(canOpenStockFromFno('NIFTY')).toBe(false)
    expect(canOpenStockFromFno('RELIANCE')).toBe(true)
    expect(canOpenStockFromFno('')).toBe(false)
  })

  it('opens the F&O floor on the selected name when one is already in hand', () => {
    expect(defaultFnoFocus('')).toBe('NIFTY')
    expect(defaultFnoFocus('RELIANCE')).toBe('RELIANCE')
    expect(defaultFnoFocus('NIFTY')).toBe('NIFTY')
  })
})
