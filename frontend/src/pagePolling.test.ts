import { describe, expect, it } from 'vitest'
import { shouldPollPage } from './pagePolling'

describe('visible desk polling', () => {
  it('keeps NSE stock Home active', () => {
    expect(shouldPollPage('Home', ['Home', 'Command Center', 'Market'])).toBe(true)
    expect(shouldPollPage('Market', ['Home', 'Command Center', 'Market'])).toBe(true)
  })
  it('pauses hidden Home while equity recommendations or F&O Desk is open', () => {
    expect(shouldPollPage('Recommendations', ['Home', 'Command Center', 'Market'])).toBe(false)
    expect(shouldPollPage('F&O Desk', ['Home', 'Command Center', 'Market'])).toBe(false)
  })
  it('pauses hidden Recommendations and reports while retaining their state', () => {
    expect(shouldPollPage('Home', ['Recommendations'])).toBe(false)
    expect(shouldPollPage('Recommendations', ['Recommendations'])).toBe(true)
    expect(shouldPollPage('Home', ['Market Reports'])).toBe(false)
  })
  it('retains standalone view behavior if activePage is not passed', () => {
    expect(shouldPollPage(undefined, ['Recommendations'])).toBe(true)
  })
})
