import './recommendations.css'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { money, pct, relativeAge, words } from './format'
import {
  fetchMarketReportsWorkspace,
  fetchRecommendationDetail,
  fetchRecommendationsWorkspace,
  type MarketReportItem,
  type RecommendationCard,
  type RecommendationDetail,
  type RecommendationKpi,
  type RecommendationsWorkspace,
  type MarketReportsWorkspace,
} from './productApi'
import type { ExperienceViewProps } from './experience'
import { LiveScanBanner } from './experience'

const CAT_ICONS: Record<string, string> = {
  wealth_builders: 'W',
  super_trends: 'S',
  momentum_breakouts: 'B',
  recovery_setups: 'R',
}

type KpiTab = 'profitability' | 'valuation' | 'margins'

function badgeClass(action: string): string {
  const a = action.toLowerCase()
  if (a.includes('buy') || a === 'open' || a === 'tracked' || a === 'win') return ''
  if (a.includes('closed') || a.includes('loss') || a.includes('void')) return 'is-closed'
  return 'is-watch'
}

function initials(name: string, symbol: string): string {
  const base = (name || symbol || '?').trim()
  const parts = base.split(/\s+/).filter(Boolean)
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase()
  return (symbol || base).slice(0, 2).toUpperCase()
}

function MetricTile({
  label,
  value,
  tone,
  badge,
}: {
  label: string
  value: string
  tone: 'entry' | 'current' | 'target'
  badge?: string
}) {
  return (
    <div className={`rw-metric-tile tone-${tone}`}>
      <span className="dot" aria-hidden="true" />
      <span className="lbl">{label}</span>
      <strong>{value}</strong>
      {badge ? <em className="badge">{badge}</em> : null}
    </div>
  )
}

function KpiList({ rows }: { rows: RecommendationKpi[] }) {
  if (!rows.length) return <p className="rw-kpi-empty">No metrics in this group.</p>
  return (
    <ul className="rw-kpi-list">
      {rows.map((row) => (
        <li key={row.key}>
          <div className="rw-kpi-name">
            <span>{row.label}</span>
            {row.hint ? <abbr title={row.hint}>i</abbr> : null}
          </div>
          <strong className={row.available ? '' : 'missing'}>{row.display}</strong>
        </li>
      ))}
    </ul>
  )
}

function PickDetailSheet({
  symbol,
  categoryId,
  onClose,
  onOpenIntel,
}: {
  symbol: string
  categoryId: string
  onClose: () => void
  onOpenIntel: (symbol: string) => void
}) {
  const [detail, setDetail] = useState<RecommendationDetail | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [mainTab, setMainTab] = useState<'performance' | 'thesis'>('performance')
  const [kpiTab, setKpiTab] = useState<KpiTab>('profitability')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchRecommendationDetail(symbol, categoryId)
      .then((payload) => {
        if (!cancelled) {
          setDetail(payload)
          setError('')
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message || 'Failed to load detail')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [symbol, categoryId])

  const kpiRows = detail?.kpis?.[kpiTab] || []
  const perf = detail?.performance
  const upsideBadge =
    perf?.upside_from_entry_pct != null
      ? `${perf.upside_from_entry_pct >= 0 ? '' : ''}${pct(perf.upside_from_entry_pct)} Upside`
      : undefined

  return createPortal(
    <div className="reco-light rw-sheet-backdrop" role="presentation" onClick={onClose}>
      <div
        className="rw-sheet"
        role="dialog"
        aria-modal="true"
        aria-label={`${symbol} recommendation detail`}
        onClick={(e) => e.stopPropagation()}
      >
        {loading ? (
          <div className="rw-sheet-loading">Loading detail…</div>
        ) : error || !detail ? (
          <div className="rw-sheet-loading">
            <strong>{error || 'Detail unavailable'}</strong>
            <button type="button" className="rw-sheet-close" onClick={onClose}>Close</button>
          </div>
        ) : (
          <>
            <header className="rw-sheet-hero">
              <div className="rw-sheet-hero-top">
                <span className={`reco-buy ${badgeClass(detail.action_badge)}`}>{detail.action_badge}</span>
                <button type="button" className="rw-sheet-x" onClick={onClose} aria-label="Close">×</button>
              </div>
              <div className="rw-sheet-identity">
                <div className="reco-avatar light">{initials(detail.company, detail.symbol)}</div>
                <div>
                  <h2>{detail.company}</h2>
                  <div className="ticker">{detail.symbol}</div>
                </div>
              </div>
              <div className="rw-sheet-tags">
                {detail.category_label ? <span className="tag cat">{detail.category_label}</span> : null}
                <span className={`tag risk ${(detail.risk_tier || '').toLowerCase()}`}>
                  {detail.risk_tier} Risk
                </span>
              </div>
            </header>

            <div className="rw-sheet-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                className={mainTab === 'performance' ? 'active' : ''}
                onClick={() => setMainTab('performance')}
              >
                Performance
              </button>
              <button
                type="button"
                role="tab"
                className={mainTab === 'thesis' ? 'active' : ''}
                onClick={() => setMainTab('thesis')}
              >
                Thesis
              </button>
            </div>

            <div className="rw-sheet-body">
              {mainTab === 'performance' ? (
                <>
                  <section className="rw-section">
                    <h3>Performance Metrics</h3>
                    <p className="sub">Entry, current, and target from this setup — never invented.</p>
                    <div className="rw-metric-grid">
                      <MetricTile
                        label="Entry Price"
                        value={perf?.entry != null ? money(perf.entry, 2) : '—'}
                        tone="entry"
                      />
                      <MetricTile
                        label="Current Price"
                        value={perf?.cmp != null ? money(perf.cmp, 2) : '—'}
                        tone="current"
                        badge={perf?.price_tag || undefined}
                      />
                      <MetricTile
                        label="Target Price"
                        value={perf?.target != null ? money(perf.target, 2) : '—'}
                        tone="target"
                        badge={upsideBadge}
                      />
                    </div>
                  </section>

                  {perf?.stop != null ? (
                    <section className="rw-section rw-stop-card">
                      <h3>Stop Loss Protection</h3>
                      <p className="sub">Downside protection level on this setup.</p>
                      <div className="rw-stop-row">
                        <span>Downside protection level</span>
                        <div>
                          <strong>{money(perf.stop, 2)}</strong>
                          {perf.downside_from_cmp_pct != null ? (
                            <small>{pct(perf.downside_from_cmp_pct)} from current</small>
                          ) : null}
                        </div>
                      </div>
                    </section>
                  ) : null}

                  <section className="rw-section">
                    <h3>Key Performance Indicators</h3>
                    <p className="sub">Profitability, valuation, and margins from verified fundamentals.</p>
                    <div className="rw-kpi-toggle" role="tablist" aria-label="KPI group">
                      {([
                        ['profitability', 'Profitability'],
                        ['valuation', 'Valuation'],
                        ['margins', 'Margins'],
                      ] as const).map(([id, label]) => (
                        <button
                          key={id}
                          type="button"
                          role="tab"
                          className={kpiTab === id ? 'active' : ''}
                          onClick={() => setKpiTab(id)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    <div className="rw-kpi-panel">
                      <div className="rw-kpi-heading">{kpiTab[0].toUpperCase() + kpiTab.slice(1)}</div>
                      <KpiList rows={kpiRows} />
                      <p className="rw-fund-note">{detail.fundamentals_note}</p>
                    </div>
                  </section>
                </>
              ) : (
                <>
                  <section className="rw-section rw-report-link">
                    <div>
                      <strong>Full research context</strong>
                      <p>Open Stock Intelligence for charts, news, and the full verified pack.</p>
                    </div>
                    <button type="button" onClick={() => onOpenIntel(detail.symbol)}>Open →</button>
                  </section>
                  <section className="rw-section">
                    <span className="rw-our-take">Our take</span>
                    <p className="rw-thesis-body">{detail.thesis.our_take}</p>
                    {detail.thesis.quality_factors.length > 0 ? (
                      <>
                        <h4>Quality factors</h4>
                        <ul>{detail.thesis.quality_factors.map((f) => <li key={f}>{f}</li>)}</ul>
                      </>
                    ) : null}
                    {detail.thesis.risk_flags.length > 0 ? (
                      <>
                        <h4>What can go wrong</h4>
                        <ul>{detail.thesis.risk_flags.map((f) => <li key={f}>{f}</li>)}</ul>
                      </>
                    ) : null}
                  </section>
                </>
              )}
              <p className="rw-sheet-disclaimer">{detail.disclaimer}</p>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}

function CardTile({
  card,
  onSelect,
}: {
  card: RecommendationCard
  onSelect: (symbol: string) => void
}) {
  const upside = card.upside_from_entry_pct
  const toTarget = card.upside_to_target_pct
  const risk = (card.risk_tier || 'Medium').toLowerCase()
  const updateNote = card.qualify_reason || card.reason || ''
  return (
    <button type="button" className="reco-pick" onClick={() => onSelect(card.symbol)}>
      <div className="reco-pick-inner">
        <div className="reco-pick-row1">
          <span className={`reco-buy ${badgeClass(card.action_badge)}`}>{card.action_badge}</span>
          <span className="reco-bookmark" aria-hidden="true">☆</span>
        </div>

        <div className="reco-identity">
          <div className="reco-avatar" aria-hidden="true">
            {initials(card.company || '', card.symbol)}
          </div>
          <div>
            <h3 className="reco-pick-name">{card.company || card.symbol}</h3>
            <div className="reco-ticker">{card.symbol}</div>
          </div>
        </div>

        <div className="reco-pick-sub">
          <span className="reco-tag">{card.category_label}</span>
          <span className={`reco-risk-chip ${risk}`}>
            <span className="reco-risk-meter" aria-hidden="true" />
            {card.risk_tier} Risk
          </span>
          {card.price_tag ? <span className="reco-tag">{card.price_tag}</span> : null}
        </div>

        <div className="reco-pick-stats">
          <div className="reco-prices">
            <span className="label">Target Price</span>
            <div className="target-val">{card.target != null ? money(card.target, 2) : '—'}</div>
            <div className="mini-row">
              <em>Entry Price</em>
              <span>{card.entry != null ? money(card.entry, 2) : '—'}</span>
            </div>
            <div className="mini-row">
              <em>Current Price</em>
              <span>{card.cmp != null ? money(card.cmp, 2) : '—'}</span>
            </div>
          </div>
          <div className="reco-gain">
            {upside != null ? (
              <>
                <strong className={upside < 0 ? 'neg' : ''}>
                  {upside >= 0 ? '↗ ' : '↘ '}
                  {pct(upside)}
                </strong>
                <small>
                  Upside from entry
                  {toTarget != null ? (
                    <>
                      <br />
                      {pct(toTarget)} from current
                    </>
                  ) : null}
                </small>
              </>
            ) : (
              <>
                <strong>—</strong>
                <small>{toTarget != null ? `${pct(toTarget)} to target` : 'Entry not set'}</small>
              </>
            )}
          </div>
        </div>

        {card.ev_lb_pct != null ? (
          <p className="reco-pick-ev">
            EV {card.ev_lb_pct >= 0 ? '+' : ''}{card.ev_lb_pct.toFixed(2)}%
            {card.p_win != null ? ` · p(win) ${card.p_win.toFixed(0)}%` : ''}
            {card.ev_n != null ? ` · n=${card.ev_n}` : ''}
          </p>
        ) : null}
      </div>
      {updateNote ? (
        <div className="reco-trade-update">
          <span>Setup note</span>
          <span>{updateNote.length > 72 ? `${updateNote.slice(0, 70)}…` : updateNote}</span>
        </div>
      ) : null}
    </button>
  )
}

function DeskShell({
  children,
  title = 'QuantTerm',
}: {
  children: ReactNode
  title?: string
}) {
  return (
    <div className="reco-light">
      <div className="reco-topbar">
        <div className="reco-topbar-brand">
          <span className="mark" aria-hidden="true">Q</span>
          <span>{title}</span>
        </div>
        <div className="avatar" aria-hidden="true">Q</div>
      </div>
      <div className="reco-desk-body">{children}</div>
    </div>
  )
}

export function RecommendationsView({
  setSelected,
  setActive,
  marketScan,
  longTermScan,
  depth,
}: ExperienceViewProps) {
  const [data, setData] = useState<RecommendationsWorkspace | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [categoryId, setCategoryId] = useState('wealth_builders')
  const [lifecycle, setLifecycle] = useState<'Active' | 'Closed'>('Active')
  const [query, setQuery] = useState('')
  const [detailSymbol, setDetailSymbol] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchRecommendationsWorkspace()
      .then((payload) => {
        if (!cancelled) {
          setData(payload)
          const firstWithCards = payload.categories.find((c) => c.count > 0)
          if (firstWithCards) setCategoryId(firstWithCards.id)
          setError('')
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message || 'Failed to load recommendations')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const category = useMemo(
    () => data?.categories.find((c) => c.id === categoryId) || data?.categories[0],
    [data, categoryId],
  )

  const cards = useMemo(() => {
    if (!data || !category) return []
    const q = query.trim().toUpperCase()
    const matchQuery = (c: RecommendationCard) => {
      if (!q) return true
      return c.symbol.includes(q) || (c.company || '').toUpperCase().includes(q)
    }
    if (lifecycle === 'Closed') {
      const closed = data.lifecycle.closed || []
      const inCat = closed.filter((c) => c.category_id === category.id)
      const pool = inCat.length > 0 ? inCat : closed
      return pool.filter(matchQuery)
    }
    return (category.cards || []).filter(matchQuery)
  }, [data, category, lifecycle, query])

  const onSelect = (symbol: string) => {
    setDetailSymbol(symbol)
  }

  const onOpenIntel = (symbol: string) => {
    setDetailSymbol(null)
    setSelected(symbol)
    setActive('Stock Intelligence')
  }

  if (loading) {
    return (
      <DeskShell>
        <div className="reco-empty"><strong>Loading recommendations…</strong></div>
      </DeskShell>
    )
  }
  if (error || !data || !category) {
    return (
      <DeskShell>
        <div className="reco-empty">
          <strong>{error || 'No recommendation data yet'}</strong>
          <p>Run a market scan and long-term refresh first.</p>
        </div>
      </DeskShell>
    )
  }

  return (
    <DeskShell>
      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
      <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />

      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Home')}>Home</button>
        <span>›</span>
        <strong>Recommendations</strong>
        <span>›</span>
        <strong>{category.label}</strong>
      </nav>

      <div className="reco-cmp-banner" role="status">
        <span className="ico" aria-hidden="true">!</span>
        <div>
          <div>CMP may be delayed — {data.cmp_note.split('.')[0]}.</div>
          {data.scan_scanned_at ? (
            <em>Last scan {relativeAge(data.scan_scanned_at)}</em>
          ) : null}
        </div>
      </div>

      <div className="reco-controls">
        <div className="reco-search-wrap">
          <input
            type="search"
            placeholder="Search stocks"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search stocks"
          />
          <button type="button" className="reco-filter-btn" aria-label="Filters" title="Filters">
            ☰
          </button>
        </div>
        <div className="reco-life-toggle" role="tablist" aria-label="Lifecycle">
          {(['Active', 'Closed'] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              role="tab"
              aria-selected={lifecycle === tab}
              className={lifecycle === tab ? 'active' : ''}
              onClick={() => setLifecycle(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="reco-cat-rail" role="tablist" aria-label="Recommendation categories">
        {data.categories.map((c) => (
          <button
            key={c.id}
            type="button"
            role="tab"
            aria-selected={c.id === category.id}
            className={c.id === category.id ? 'active' : ''}
            onClick={() => setCategoryId(c.id)}
          >
            {c.label} · {c.count}
          </button>
        ))}
      </div>

      <header className="reco-hero">
        <div className="reco-hero-icon" aria-hidden="true">
          {CAT_ICONS[category.id] || '•'}
        </div>
        <div>
          <h2>{category.label}</h2>
          <p>{category.blurb}</p>
        </div>
      </header>

      {cards.length === 0 ? (
        <div className="reco-empty">
          <strong>
            {lifecycle === 'Closed'
              ? 'No closed picks in this category yet.'
              : 'No active picks in this category yet.'}
          </strong>
          <p>
            {lifecycle === 'Active'
              ? (category.empty_detail || 'Evidence filter found no matches in the current scan.')
              : 'Closed outcomes appear after tracked picks exit or signals resolve.'}
          </p>
        </div>
      ) : (
        <div className="reco-card-stack">
          {cards.map((card) => (
            <CardTile
              key={`${card.lifecycle}-${card.symbol}-${card.setup_label}-${card.category_id}`}
              card={card}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
      <p className="reco-foot">{data.disclaimer}</p>
      {detailSymbol ? (
        <PickDetailSheet
          symbol={detailSymbol}
          categoryId={category.id}
          onClose={() => setDetailSymbol(null)}
          onOpenIntel={onOpenIntel}
        />
      ) : null}
    </DeskShell>
  )
}

function formatReportDate(value: string): string {
  if (!value) return ''
  try {
    const d = new Date(value.length <= 10 ? `${value}T12:00:00` : value)
    if (Number.isNaN(d.getTime())) return words(value)
    return d.toLocaleDateString('en-IN', {
      day: 'numeric', month: 'long', year: 'numeric',
    }).toUpperCase()
  } catch {
    return words(value)
  }
}

function BreadthGauge({
  gauge,
}: {
  gauge?: {
    available?: boolean
    score?: number | null
    label?: string
    line?: string
    verdict?: string
  } | null
}) {
  if (!gauge?.available || gauge.score == null) {
    return (
      <div className="rw-gauge-card">
        <div className="rw-gauge-head">
          <strong>Market Breadth</strong>
          <span className="hint">from scan cache</span>
        </div>
        <p style={{ margin: 0, color: 'var(--rw-muted)', fontSize: 14 }}>
          {gauge?.line || 'Breadth unavailable until a full-market scan fills the cache.'}
        </p>
      </div>
    )
  }
  const score = Math.max(0, Math.min(100, Number(gauge.score)))
  // Map 0..100 onto a semicircle needle: 180deg (left) → 0deg (right) in our CSS rotate.
  const rotation = -90 + (score / 100) * 180
  return (
    <div className="rw-gauge-card">
      <div className="rw-gauge-head">
        <strong>Market Breadth</strong>
        <span className="hint">% above 50-DMA · India</span>
      </div>
      <div className="rw-gauge-visual" aria-label={`Breadth score ${score}`}>
        <div className="rw-gauge-arc" />
        <div className="rw-gauge-needle" style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }} />
        <div className="rw-gauge-value">{score}</div>
      </div>
      <div className="rw-gauge-foot">
        <strong>{gauge.label || gauge.verdict || 'Tape'}</strong>
        <span>{gauge.line || 'Live breadth from QuantTerm market cache.'}</span>
      </div>
    </div>
  )
}

export function MarketReportsView({ setActive }: ExperienceViewProps) {
  const [data, setData] = useState<MarketReportsWorkspace | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<MarketReportItem | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchMarketReportsWorkspace()
      .then((payload) => {
        if (!cancelled) {
          setData(payload)
          setSelected(payload.reports[0] || null)
          setError('')
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message || 'Failed to load market reports')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const reports = useMemo(() => {
    if (!data) return []
    const q = query.trim().toLowerCase()
    if (!q) return data.reports
    return data.reports.filter(
      (r) =>
        r.title.toLowerCase().includes(q)
        || r.summary.toLowerCase().includes(q)
        || r.date.includes(q),
    )
  }, [data, query])

  if (loading) {
    return (
      <DeskShell>
        <div className="reco-empty"><strong>Loading market reports…</strong></div>
      </DeskShell>
    )
  }
  if (error || !data) {
    return (
      <DeskShell>
        <div className="reco-empty">
          <strong>{error || 'No reports yet'}</strong>
        </div>
      </DeskShell>
    )
  }

  const pulse = data.today_pulse || {}
  const takeaways = (pulse.takeaways as string[] | undefined) || []
  const insights = data.insights || []

  return (
    <DeskShell>
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Home')}>Home</button>
        <span>›</span>
        <strong>Market Reports</strong>
      </nav>

      <header className="rw-reports-hero">
        <h1>{data.title}</h1>
        <p>{data.blurb}</p>
      </header>

      <BreadthGauge gauge={data.breadth_gauge} />

      {insights.slice(0, 3).map((item, idx) => (
        <article className="rw-insight-card" key={`${item.title}-${idx}`}>
          <h3>{item.title === 'Headline' ? item.body.slice(0, 90) : item.body.slice(0, 110)}</h3>
          {item.title === 'Headline' ? null : <p>{item.body}</p>}
        </article>
      ))}

      <div className="reco-controls">
        <div className="reco-search-wrap">
          <input
            type="search"
            placeholder="Search reports"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search reports"
          />
          <button type="button" className="reco-filter-btn" aria-label="Filters" title="Filters">
            ☰
          </button>
        </div>
      </div>

      <ul className="rw-report-list">
        {reports.length === 0 ? (
          <li>
            <button type="button" disabled>
              <strong>No reports match this search.</strong>
            </button>
          </li>
        ) : (
          reports.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                className={`${selected?.id === r.id ? 'active' : ''} ${r.is_new ? 'is-new' : ''}`}
                onClick={() => setSelected(r)}
              >
                <strong>{r.title}</strong>
                <span className="date">{formatReportDate(r.date)}</span>
                {r.is_new ? (
                  <em className="rw-report-new">{r.badge || 'New market report'}</em>
                ) : null}
                <small>{r.summary}</small>
              </button>
            </li>
          ))
        )}
      </ul>

      {selected ? (
        <div className="rw-report-detail">
          <h2>{selected.title}</h2>
          <p className="when">{formatReportDate(selected.date)}</p>
          {takeaways.length > 0 && selected.is_new ? (
            <ul>
              {takeaways.map((t) => <li key={t}>{t}</li>)}
            </ul>
          ) : (
            <p>{selected.summary}</p>
          )}
          {Array.isArray(pulse.breakouts_today) && pulse.breakouts_today.length > 0 ? (
            <>
              <h3>Breakouts in focus</h3>
              <p>
                {pulse.breakouts_today
                  .map((b: { symbol?: string }) => b.symbol)
                  .filter(Boolean)
                  .join(', ')}
              </p>
            </>
          ) : null}
        </div>
      ) : null}

      <p className="reco-foot">{data.disclaimer}</p>
      {data.error ? <p className="reco-foot">Pulse note: {data.error}</p> : null}
    </DeskShell>
  )
}
