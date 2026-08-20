import { describe, expect, it } from 'vitest'
import { formatPeekValue, filledPeekMetrics, mergePeekMetrics, orderPeekMetrics, peekNumber, peekUpsidePct } from './stockPeek'

describe('stock peek numbers', () => {
  it('never invents a missing metric', () => {
    expect(formatPeekValue(null, '%')).toBe('Not on file')
    expect(formatPeekValue(undefined)).toBe('Not on file')
    expect(formatPeekValue(Number.NaN)).toBe('Not on file')
    expect(formatPeekValue('')).toBe('Not on file')
  })

  it('formats ratios with their unit', () => {
    expect(formatPeekValue(41.62, 'x')).toBe('41.62x')
    expect(formatPeekValue(19, '%')).toBe('19%')
    expect(formatPeekValue(22400)).toMatch(/22,400|22400/)
  })

  it('computes upside from buy and target without waiting on a fetch', () => {
    expect(peekUpsidePct(743.9, 842.7)).toBe(13.3)
    expect(peekUpsidePct(0, 100)).toBeNull()
  })

  it('keeps a fallback metric when the primary value is missing', () => {
    const merged = mergePeekMetrics(
      [{ key: 'change_pct', label: 'Change', value: null, unit: '%' }],
      [{ key: 'change_pct', label: 'Change', value: 1.2, unit: '%' }, { key: 'rsi14', label: 'RSI', value: 69.2 }],
    )
    expect(filledPeekMetrics(merged).map((item) => item.key)).toEqual(['change_pct', 'rsi14'])
    expect(peekNumber(merged.find((item) => item.key === 'change_pct')?.value)).toBe(1.2)
  })

  it('keeps preferred PE/ROE first without dropping extras', () => {
    const ordered = orderPeekMetrics(
      [
        { key: 'sales_growth_3y', label: 'Sales 3y', value: 12, unit: '%' },
        { key: 'pe', label: 'P/E', value: 22, unit: 'x' },
        { key: 'roe', label: 'ROE', value: 18, unit: '%' },
      ],
      ['pe', 'roe', 'roce'],
    )
    expect(ordered.map((item) => item.key)).toEqual(['pe', 'roe', 'sales_growth_3y'])
  })
})
