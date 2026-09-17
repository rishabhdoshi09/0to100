import { describe, expect, it } from 'vitest'

import { istDateTime } from './format'

describe('istDateTime', () => {
  it('renders epoch seconds as a human IST timestamp', () => {
    const rendered = istDateTime(1789664045.360964)
    expect(rendered).toContain('2026')
    expect(rendered).toContain('22:24:05')
    expect(rendered).toContain('IST')
  })

  it('does not echo an invalid operational timestamp as a valid date', () => {
    expect(istDateTime(null)).toBe('—')
  })
})
