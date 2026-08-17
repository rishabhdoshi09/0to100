import { describe, expect, it } from 'vitest'
import {
  deskSymbol,
  filingsNeedRefresh,
  sectorWaveFirstLine,
  sectorWaveVerdict,
  thesisReplacesList,
} from './deskThesis'

describe('desk thesis helpers', () => {
  it('normalizes symbols so every card opens the same way', () => {
    expect(deskSymbol(' eimcoeleco ')).toBe('EIMCOELECO')
    expect(deskSymbol('BSE')).toBe('BSE')
    expect(deskSymbol('')).toBe('')
  })

  it('answers sector wave with YES only on inflow', () => {
    expect(sectorWaveVerdict({ wave: 'INFLOW' })).toBe('YES')
    expect(sectorWaveVerdict({ wave: 'OUTFLOW' })).toBe('NO')
    expect(sectorWaveVerdict({ wave: 'MIXED' })).toBe('NO')
    expect(sectorWaveVerdict({ wave: 'NO_CLAIM' })).toBe('NO')
    expect(sectorWaveVerdict({ verdict: 'YES', wave: 'MIXED' })).toBe('YES')
  })

  it('puts the yes/no line first when discussing a sector wave', () => {
    expect(sectorWaveFirstLine({
      wave: 'INFLOW',
      verdict: 'YES',
      verdict_line: 'YES — sector money is coming in around this name.',
    }).startsWith('YES')).toBe(true)
    expect(sectorWaveFirstLine({ wave: 'NO_CLAIM' }).startsWith('NO')).toBe(true)
  })

  it('replaces the card list on a phone while a thesis is open', () => {
    expect(thesisReplacesList(true, 'BSE')).toBe(true)
    expect(thesisReplacesList(true, '')).toBe(false)
    expect(thesisReplacesList(false, 'BSE')).toBe(false)
  })

  it('refetches filings when the pack is stale, not only when coverage is thin', () => {
    expect(filingsNeedRefresh({
      filings_stale: true,
      fundamentals: { available: true, coverage_pct: 90 },
    })).toBe(true)
    expect(filingsNeedRefresh({
      filings_stale: true,
      filings_refresh_attempted: true,
      fundamentals: { available: true, coverage_pct: 90 },
    })).toBe(false)
    expect(filingsNeedRefresh({
      filings_stale: false,
      fundamentals: { available: true, coverage_pct: 90 },
    })).toBe(false)
    expect(filingsNeedRefresh({
      filings_stale: false,
      fundamentals: { available: true, coverage_pct: 10 },
    })).toBe(true)
  })
})
