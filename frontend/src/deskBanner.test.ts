import { describe, expect, it } from 'vitest'
import { deskRefreshBanner } from './deskBanner'
import { API_DOWN_MESSAGE, REQUEST_TIMEOUT_MESSAGE } from './http'

describe('deskRefreshBanner', () => {
  it('keeps Home honest when a refresh times out but a snapshot exists', () => {
    const banner = deskRefreshBanner(REQUEST_TIMEOUT_MESSAGE, true)
    expect(banner.title).toBe('Desk refresh timed out')
    expect(banner.body).toContain('last readable snapshot')
  })

  it('does not call a timeout a dead API when the first Home load is still assembling', () => {
    const banner = deskRefreshBanner(REQUEST_TIMEOUT_MESSAGE, false)
    expect(banner.title).toBe('Assembling the desk…')
    expect(banner.body).toContain('not waiting forever')
  })

  it('tells the operator to start the stack when :8765 is down', () => {
    const banner = deskRefreshBanner(API_DOWN_MESSAGE, false)
    expect(banner.title).toBe('Market API is not running')
    expect(banner.body).toContain('run_quantterm_complete.sh')
  })
})
