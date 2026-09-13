export type ScanProvenance = {
  available: boolean
  reason?: string
  source?: string
  schema_version?: number | null
  scan_id?: string | null
  scanned_at?: string | null
  scan_started_at?: string | null
  scan_completed_at?: string | null
  scan_duration_s?: number | null
  scan_duration_status?: string | null
  scan_duration_reason?: string | null
  market_session_date?: string | null
  price_data_as_of?: string | null
  expected_session_date?: string | null
  fundamental_data_as_of?: string | null
  news_data_as_of?: string | null
  freshness_state?: string | null
  provenance_reason?: string | null
  source_set?: string[] | null
  approved_universe?: number | null
  requested_universe?: number | null
  universe_requested?: number | null
  universe_loaded?: number | null
  universe_scanned?: number | null
  universe_failed?: number | null
  candidate_count?: number | null
  source_snapshot_id?: string | null
  coverage_state?: string | null
  scan_status?: string | null
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export function formatMarketDate(value?: string | null): string {
  const text = String(value || '').trim()
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text)
  if (!match) return 'Unavailable'
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  if (!Number.isInteger(year) || month < 1 || month > 12 || day < 1 || day > 31) {
    return 'Unavailable'
  }
  return `${String(day).padStart(2, '0')} ${MONTHS[month - 1]} ${year}`
}

export function formatExecutionTime(value?: string | null): string {
  const text = String(value || '').trim()
  if (!text) return 'Unavailable'
  const parsed = new Date(text)
  if (Number.isNaN(parsed.getTime())) return 'Unavailable'
  return new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(parsed)
}

export function scanTruthRows(scan?: ScanProvenance | null): Array<{ label: string; value: string }> {
  if (!scan?.available) {
    return [
      { label: 'Scan executed', value: 'Unavailable' },
      { label: 'Market session', value: 'Unavailable' },
      { label: 'Price data as of', value: 'Unavailable' },
    ]
  }
  return [
    {
      label: 'Scan executed',
      value: formatExecutionTime(scan.scan_completed_at || scan.scanned_at),
    },
    {
      label: 'Market session',
      value: formatMarketDate(scan.market_session_date),
    },
    {
      label: 'Price data as of',
      value: formatMarketDate(scan.price_data_as_of),
    },
  ]
}

export function scanTruthDetail(scan?: ScanProvenance | null): string {
  if (!scan) return 'Loading scan provenance…'
  if (!scan.available) return scan.reason || 'No saved scan'
  if (scan.freshness_state && scan.freshness_state !== 'CURRENT') {
    return `${scan.freshness_state}${scan.provenance_reason ? ` · ${scan.provenance_reason}` : ''}`
  }
  if (scan.scan_duration_status === 'UNAVAILABLE' && scan.scan_duration_reason) {
    return `Timing unavailable · ${scan.scan_duration_reason}`
  }
  return scan.scan_status || scan.freshness_state || 'Recorded'
}
