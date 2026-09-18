import { describe, expect, it } from 'vitest'
import { compactDateTime, formatIst } from './format'

describe('formatIst', () => {
  it('renders unix seconds as an IST wall clock, never the raw epoch', () => {
    const text = formatIst(1789664045.360964)
    expect(text).not.toContain('1789664045')
    expect(text).toMatch(/IST$/)
    expect(text).toMatch(/\d{1,2} \w{3} \d{4} · \d{2}:\d{2} IST/)
  })

  it('renders ISO timestamps in Asia/Kolkata', () => {
    const text = formatIst('2026-09-17T16:54:00+00:00')
    expect(text).toBe('17 Sep 2026 · 22:24 IST')
  })

  it('compactDateTime uses the same IST contract', () => {
    expect(compactDateTime(1789664045)).toEqual(formatIst(1789664045))
  })
})
