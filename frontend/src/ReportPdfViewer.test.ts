import { describe, expect, it } from 'vitest'
import { reportDownloadUrl } from './ReportPdfViewer'

describe('reportDownloadUrl', () => {
  it('keeps view URLs free of download=true', () => {
    expect(reportDownloadUrl('/reports/equity/GAIL')).toBe('/reports/equity/GAIL?download=true')
    expect(reportDownloadUrl('/reports/basket/long-term?limit=3')).toBe(
      '/reports/basket/long-term?limit=3&download=true',
    )
    expect(reportDownloadUrl('/reports/market/institutional?days=30')).toBe(
      '/reports/market/institutional?days=30&download=true',
    )
  })

  it('returns empty for empty input', () => {
    expect(reportDownloadUrl('')).toBe('')
  })
})
