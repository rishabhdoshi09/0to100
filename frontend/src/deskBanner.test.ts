import { describe, expect, it } from 'vitest'
import { deskRefreshBanner } from './deskBanner'
import { API_DOWN_MESSAGE, REQUEST_TIMEOUT_MESSAGE } from './http'

describe('deskRefreshBanner', () => {
  it('does not treat a stale snapshot as a live desk when the API is down', () => {
    const banner = deskRefreshBanner(API_DOWN_MESSAGE, true)
    expect(banner.title).toContain('FAILED')
    expect(banner.body).toContain('stale')
    expect(banner.body).not.toContain('last readable snapshot is fine')
  })

  it('keeps a timeout honest when a snapshot exists', () => {
    const banner = deskRefreshBanner(REQUEST_TIMEOUT_MESSAGE, true)
    expect(banner.title).toContain('DEGRADED')
    expect(banner.body).toContain('not a fresh health check')
  })

  it('does not call a timeout a dead API when the first Home load is still assembling', () => {
    const banner = deskRefreshBanner(REQUEST_TIMEOUT_MESSAGE, false)
    expect(banner.title).toContain('STARTING')
    expect(banner.body).toContain('not waiting forever')
  })

  it('tells the operator to start the stack when :8765 is down', () => {
    const banner = deskRefreshBanner(API_DOWN_MESSAGE, false)
    expect(banner.title).toContain('FAILED')
    expect(banner.body).toContain('run_quantterm_complete.sh')
  })
})
