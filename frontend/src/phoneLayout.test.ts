import { describe, expect, it } from 'vitest'
import {
  PHONE_MAX_WIDTH_PX,
  chartHeightForWidth,
  isPhoneLayout,
  thesisSheetClassName,
} from './phoneLayout'

describe('phone layout helpers', () => {
  it('treats 390px as a phone and 821px as a desk', () => {
    expect(isPhoneLayout(390)).toBe(true)
    expect(isPhoneLayout(PHONE_MAX_WIDTH_PX)).toBe(true)
    expect(isPhoneLayout(821)).toBe(false)
  })

  it('shortens the chart on a phone so the thesis can scroll', () => {
    expect(chartHeightForWidth(390)).toBe(200)
    expect(chartHeightForWidth(1280)).toBe(360)
  })

  it('marks a closable thesis sheet for the phone overlay', () => {
    expect(thesisSheetClassName(true)).toBe('reco-sheet thesis-sheet has-close')
    expect(thesisSheetClassName(false)).toBe('reco-sheet thesis-sheet')
  })
})
