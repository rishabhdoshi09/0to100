import { useEffect, useState } from 'react'
import { fetchQuarterlyInsights, type QuarterlyInsightsReport, type QuarterlyEvidenceFinding } from './productApi'
import './quarterlyInsights.css'

function formatCrore(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return 'Unmeasured'
  return '₹' + new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 }).format(n) + ' Cr'
}

function readable(value: string): string {
  return value.replaceAll('_', ' ').toLowerCase()
}

function FindingList({ items, empty }: { items: QuarterlyEvidenceFinding[]; empty: string }) {
  if (!items.length) return <p className="qi-empty">{empty}</p>
  return (
    <ul className="qi-findings">
      {items.map((item) => (
        <li key={item.id}>
          <strong>{item.observation}</strong>
          <small>
            {item.sources.map((source) => source.evidence_id).filter(Boolean).join(' · ') || 'No evidence ID'}
          </small>
        </li>
      ))}
    </ul>
  )
}

/** Research-only section; never calls the trading or acquisition endpoints. */
export function QuarterlyInsightsPanel({
  symbol,
  refreshKey = '',
}: {
  symbol: string
  refreshKey?: string
}) {
  const [report, setReport] = useState<QuarterlyInsightsReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    setReport(null) // Never display the previous symbol's evidence while reloading.
    setError('')
    if (!symbol) return () => { active = false }
    setLoading(true)
    void fetchQuarterlyInsights(symbol)
      .then((value) => { if (active) setReport(value) })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : 'Quarterly research unavailable')
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [symbol, refreshKey])

  if (!symbol) return null
  return (
    <section className="qi-review" aria-label="Quarterly thesis and anti-thesis">
      <header className="qi-review-header">
        <div>
          <span className="qi-eyebrow">Long-term research · point-in-time</span>
          <h3>Quarterly thesis vs. anti-thesis</h3>
          <p>Evidence from dated company filings, not a price-target recommendation.</p>
        </div>
        <span className="qi-pill">{loading ? 'Checking evidence' : report ? readable(report.status) : 'Unavailable'}</span>
      </header>

      {loading && <p className="qi-empty" role="status">Reading the dated financial warehouse…</p>}
      {error && !loading && <p className="qi-error" role="alert">Quarterly research could not be verified: {error}</p>}

      {report && !loading && (
        <>
          <div className="qi-facts">
            <div><span>Latest quarter</span><strong>{report.latest_period_end || 'Not available'}</strong></div>
            <div><span>Year-ago comparison</span><strong>{report.comparison_period_end || 'Not available'}</strong></div>
            <div><span>Publication recency</span><strong>{readable(report.publication_freshness)}</strong></div>
            <div><span>Statement basis</span><strong>{readable(report.accounting_scope)}</strong></div>
          </div>
          {report.status === 'NO_VERIFIABLE_QUARTER' ? (
            <p className="qi-empty">No safely dated, parsed quarterly financial filing is available. QuantTerm cannot establish a thesis, anti-thesis, target or winning probability.</p>
          ) : (
            <>
              <div className="qi-metrics">
                {['revenue', 'pat', 'total_debt', 'operating_cash_flow'].map((key) => {
                  const item = report.metrics[key]
                  return item ? (
                    <div key={key}>
                      <small>{item.label}</small>
                      <strong>{formatCrore(item.current)}</strong>
                      <span>{item.yoy_pct == null ? 'YoY unmeasured' : 'YoY ' + (item.yoy_pct >= 0 ? '+' : '') + item.yoy_pct + '%'}</span>
                    </div>
                  ) : null
                })}
              </div>
              <div className="qi-columns">
                <div className="qi-case qi-support">
                  <h4>Thesis evidence</h4>
                  <FindingList items={report.thesis_evidence} empty="No measured positive change. This does not mean the business is weak." />
                </div>
                <div className="qi-case qi-challenge">
                  <h4>Anti-thesis evidence</h4>
                  <FindingList items={report.anti_thesis_evidence} empty="No measured counter-signal in available data. Other risks may still be unknown." />
                </div>
              </div>
            </>
          )}
          <div className="qi-unknowns">
            <h4>What remains unverified</h4>
            <p>{report.unknowns.length ? report.unknowns.map(readable).join(' · ') : 'No listed gaps, but this is still not an investment recommendation.'}</p>
            <p>{report.price_track.reason}</p>
          </div>
          {report.source_documents.length > 0 && (
            <footer className="qi-sources">
              <strong>Source lineage</strong>
              {report.source_documents.map((source) => (
                <span key={source.evidence_id}>
                  {source.source_url.startsWith('https://')
                    ? <a href={source.source_url} target="_blank" rel="noopener noreferrer">{source.source || 'Company filing'}</a>
                    : (source.source || 'Company filing')}
                  {' · published '}{source.publication_date || 'unverified date'}
                  {' · evidence '}{source.evidence_id}
                </span>
              ))}
            </footer>
          )}
          <p className="qi-disclaimer">{report.note}</p>
        </>
      )}
    </section>
  )
}
