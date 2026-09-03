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
        title: 'Desk offline · showing last saved data',
        body: 'Live refresh is unavailable. The numbers below are from the last successful load, so treat them as stale until the connection returns.',
      }
    }
    if (timedOut) {
      return {
        title: 'Refresh delayed · showing last saved data',
        body: 'The backend is busy or recovering. QuantTerm kept the last successful snapshot on screen instead of replacing it with blanks.',
      }
    }
    return {
      title: 'Live refresh interrupted · showing last saved data',
      body: 'The page is still usable for reference, but the values below are not confirmed current until the next successful refresh.',
    }
  }

  if (apiDown) {
    return {
      title: 'Desk is not connected yet',
      body: 'Start QuantTerm with bash scripts/run_quantterm_complete.sh, then retry the connection.',
    }
  }
  if (timedOut) {
    return {
      title: 'Desk is still starting…',
      body: 'The first load can take a moment while the data and scan lanes come online. Retry if it remains here.',
    }
  }
  return {
    title: 'Connecting to QuantTerm…',
    body: 'The market API and data lanes are starting. Retry if the desk does not appear within a minute.',
  }
}
