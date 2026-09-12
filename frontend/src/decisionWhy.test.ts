import { describe, expect, it } from 'vitest'
import {
  fieldLabel,
  gapSummary,
  isRenderable,
  presentFields,
  renderValue,
  sectionTone,
  visibleSections,
} from './decisionWhyModel'
import type { DecisionSection, DecisionWhy } from './decisionWhyModel'

function section(overrides: Partial<DecisionSection> = {}): DecisionSection {
  return {
    title: 'Setup',
    available: true,
    value: 'VCP',
    text: '',
    items: [],
    fields: {},
    ...overrides,
  }
}

describe('WHY THIS DECISION rendering rules', () => {
  it('shows an unavailable section rather than hiding it', () => {
    const empty = section({ title: 'Portfolio effect', available: false, value: null })
    expect(sectionTone(empty)).toBe('decision-why__section--unavailable')
  })

  it('marks conflicting evidence distinctly from supporting evidence', () => {
    expect(sectionTone(section({ title: 'Conflicting evidence' }))).toBe(
      'decision-why__section--conflict',
    )
    expect(sectionTone(section({ title: 'Supporting evidence' }))).toBe('')
  })

  it('marks missing evidence as a gap, not as a normal section', () => {
    expect(sectionTone(section({ title: 'Missing evidence' }))).toBe('decision-why__section--gap')
  })

  it('renders a missing number as a dash, never as zero', () => {
    expect(renderValue(null)).toBe('—')
    expect(renderValue(undefined)).toBe('—')
    expect(renderValue('')).toBe('—')
    expect(renderValue(Number.NaN)).toBe('—')
    expect(renderValue(0)).toBe('0')
  })

  it('keeps integers exact and rounds only fractions', () => {
    expect(renderValue(95)).toBe('95')
    expect(renderValue(2.5)).toBe('2.50')
  })

  it('treats a null field as absent rather than printing it', () => {
    const fields = { expected_value: null, calibrated_confidence: 0.66, note: '' }
    expect(presentFields(fields)).toEqual(['calibrated_confidence'])
  })

  it('keeps a zero field, which is a measurement', () => {
    expect(presentFields({ evidence_adjustment: 0 })).toEqual(['evidence_adjustment'])
  })

  it('humanises field keys', () => {
    expect(fieldLabel('expected_R_from_levels')).toBe('expected R from levels')
  })

  it('names the sections the desk could not fill', () => {
    const why = {
      available: true,
      symbol: 'INFY',
      unfilled_sections: ['Expected value', 'Portfolio effect'],
    } as DecisionWhy
    expect(gapSummary(why)).toBe(
      '2 sections the desk could not fill: Expected value, Portfolio effect',
    )
  })

  it('uses the singular for one gap and says nothing when there are none', () => {
    expect(
      gapSummary({ available: true, symbol: 'A', unfilled_sections: ['Setup'] } as DecisionWhy),
    ).toBe('1 section the desk could not fill: Setup')
    expect(
      gapSummary({ available: true, symbol: 'A', unfilled_sections: [] } as DecisionWhy),
    ).toBe('')
    expect(gapSummary(null)).toBe('')
  })

  it('preserves the server section order', () => {
    const why = {
      available: true,
      symbol: 'INFY',
      sections: [section({ title: 'Supporting evidence' }), section({ title: 'Risk' })],
    } as DecisionWhy
    expect(visibleSections(why).map((s) => s.title)).toEqual(['Supporting evidence', 'Risk'])
  })

  it('refuses to render a decision the desk never made', () => {
    expect(isRenderable(null)).toBe(false)
    expect(isRenderable({ available: false, symbol: 'ZZZZ' } as DecisionWhy)).toBe(false)
    expect(isRenderable({ available: true, symbol: 'INFY' } as DecisionWhy)).toBe(true)
  })
})
