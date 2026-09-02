import { API_DOWN_MESSAGE, REQUEST_TIMEOUT_MESSAGE } from './http'

export type DeskBanner = {
  title: string
  body: string
}

export function deskRefreshBanner(error: string, hasDesk: boolean): DeskBanner {
  const text = String(error || '')
  const timedOut = text === REQUEST_TIMEOUT_MESSAGE || text.toLowerCase().includes('timed out')
  const apiDown = text === API_DOWN_MESSAGE || text.includes('Market API is not running')

  if (hasDesk) {
    if (apiDown) {
      return {
        title: 'FAILED · Market API is not responding',
        body: 'Numbers on this page are stale from the last successful load, not a live desk. Start bash scripts/run_quantterm_complete.sh and wait for READY.',
      }
    }
    if (timedOut) {
      return {
        title: 'DEGRADED · Desk refresh timed out',
        body: 'The last snapshot is still on screen, but it is not a fresh health check. Retry. If this stays, the backend is busy or recovering.',
      }
    }
    return {
      title: 'DEGRADED · Could not refresh the desk',
      body: 'The last snapshot is not current. Retry. Do not treat these numbers as live.',
    }
  }

  if (apiDown) {
    return {
      title: 'FAILED · Market API is not running',
      body: 'Start with bash scripts/run_quantterm_complete.sh, then retry.',
    }
  }
  if (timedOut) {
    return {
      title: 'STARTING · Assembling the desk…',
      body: 'The first Home load can take a moment while scan and data lanes come up. This page is not waiting forever — retry.',
    }
  }
  return {
    title: 'STARTING · Connecting to the market API…',
    body: 'QuantTerm is starting the data lanes. Retry if this stays for more than a minute.',
  }
}
