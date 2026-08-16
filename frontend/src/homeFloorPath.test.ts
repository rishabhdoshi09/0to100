import { describe, expect, it } from 'vitest'
import { FLOOR_JUMPS, pathButtonLabel, pickTodaySymbol } from './homeFloorPath'

describe('pickTodaySymbol', () => {
  it('prefers the selected name, then best, then first visible, then scan, then query', () => {
    expect(pickTodaySymbol({
      selected: 'RPEL',
      best: { symbol: 'ASAHIINDIA' },
      visible: [{ symbol: 'TCS' }],
      scan: [{ symbol: 'INFY' }],
      query: 'RELIANCE',
    })).toBe('RPEL')
    expect(pickTodaySymbol({
      best: { symbol: 'ASAHIINDIA' },
      visible: [{ symbol: 'TCS' }],
    })).toBe('ASAHIINDIA')
    expect(pickTodaySymbol({
      visible: [{ symbol: 'tcs' }],
      scan: [{ symbol: 'INFY' }],
    })).toBe('TCS')
    expect(pickTodaySymbol({ scan: [{ symbol: 'INFY' }] })).toBe('INFY')
    expect(pickTodaySymbol({ query: 'reliance' })).toBe('RELIANCE')
  })

  it('skips junk and stays empty when nothing is usable', () => {
    expect(pickTodaySymbol({ selected: 'not a symbol!!', query: '' })).toBe('')
    expect(pickTodaySymbol({})).toBe('')
  })
})

describe('path chrome', () => {
  it('names the click and keeps floor jumps on Home / Options / Data / Holdings / Health', () => {
    expect(pathButtonLabel('RPEL')).toBe("Open RPEL's floors")
    expect(pathButtonLabel('')).toBe("Open today's path")
    expect(FLOOR_JUMPS.map((item) => item.id)).toEqual(['desk', 'options', 'data', 'holdings', 'health'])
    expect(FLOOR_JUMPS.find((item) => item.id === 'options')?.intelTab).toBe('Options')
    expect(FLOOR_JUMPS.map((item) => item.page)).toEqual([
      'Home',
      'Stock Intelligence',
      'Research Data',
      'Paper Portfolio',
      'System Health',
    ])
  })
})
