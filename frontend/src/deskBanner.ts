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
        title: 'Market API dropped',
        body: 'Home is showing the last readable snapshot. Start the stack again if this stays.',
      }
    }
    if (timedOut) {
      return {
        title: 'Desk refresh timed out',
        body: 'Home is showing the last readable snapshot. This page is not waiting forever — retry.',
      }
    }
    return {
      title: 'Could not refresh the desk',
      body: 'Home is showing the last readable snapshot. Retry if this stays.',
    }
  }

  if (apiDown) {
    return {
      title: 'Market API is not running',
      body: 'Start with bash scripts/run_quantterm_complete.sh, then retry.',
    }
  }
  if (timedOut) {
    return {
      title: 'Assembling the desk…',
      body: 'The first Home load can take a moment while scan and data lanes come up. This page is not waiting forever — retry.',
    }
  }
  return {
    title: 'Connecting to the market API…',
    body: 'QuantTerm is starting the data lanes. Retry if this stays for more than a minute.',
  }
}
