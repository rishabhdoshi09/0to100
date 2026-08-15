import './radar.css'
import { useEffect, useMemo, useState } from 'react'
import { Panel } from './components'
import { money, pct, relativeAge, words } from './format'
import {
  fetchMarketReportsWorkspace,
  fetchRecommendationsWorkspace,
  type MarketReportItem,
  type RecommendationCard,
  type RecommendationsWorkspace,
  type MarketReportsWorkspace,
} from './productApi'
import { EmptyState, SectionTabs } from './designSystem'
import type { ExperienceViewProps } from './experience'
import { LiveScanBanner } from './experience'

function CardTile({
  card,
  onSelect,
}: {
  card: RecommendationCard
  onSelect: (symbol: string) => void
}) {
  const upside = card.upside_from_entry_pct
  const toTarget = card.upside_to_target_pct
  const upsideCls =
    upside == null ? '' : upside >= 0 ? 'reco-upside-pos' : 'reco-upside-neg'
  return (
    <button type="button" className="reco-card" onClick={() => onSelect(card.symbol)}>
      <div className="reco-card-top">
        <span className={`reco-badge reco-badge-${card.action_badge.toLowerCase().replace(/\s+/g, '-')}`}>
          {card.action_badge}
        </span>
        <span className={`reco-risk reco-risk-${card.risk_tier.toLowerCase()}`}>
          {card.risk_tier} Risk
        </span>
      </div>
      <strong className="reco-company">{card.company || card.symbol}</strong>
      <div className="reco-meta">
        <span>{card.symbol}</span>
        <span className="reco-cat-tag">{card.category_label}</span>
      </div>
      <div className="reco-prices">
        <div>
          <span>CMP</span>
          <strong>{card.cmp != null ? money(card.cmp) : '—'}</strong>
          {card.price_tag ? <small>{card.price_tag}</small> : null}
        </div>
        <div>
          <span>Target</span>
          <strong>{card.target != null ? money(card.target) : '—'}</strong>
        </div>
        <div>
          <span>Entry</span>
          <strong>{card.entry != null ? money(card.entry) : '—'}</strong>
        </div>
      </div>
      <div className={`reco-upside ${upsideCls}`}>
        {upside != null ? (
          <>
            <span>{upside >= 0 ? '↗' : '↘'} {pct(upside)}</span>
            <small>% from entry</small>
          </>
        ) : (
          <small>Entry not set</small>
        )}
        {toTarget != null ? <em>{pct(toTarget)} to target</em> : null}
      </div>
      {card.reason ? <p className="reco-reason">{card.reason}</p> : null}
    </button>
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
    if (lifecycle === 'Closed') {
      return (data.lifecycle.closed || []).filter((c) => {
        if (categoryId && c.category_id && c.category_id !== categoryId) {
          // Show all closed if category filter would empty the strip.
          const anyInCat = data.lifecycle.closed.some((x) => x.category_id === categoryId)
          if (anyInCat) return c.category_id === categoryId
        }
        if (!q) return true
        return c.symbol.includes(q) || (c.company || '').toUpperCase().includes(q)
      })
    }
    // Active: category cards first; merge tracker actives for same category.
    const fromCat = category.cards || []
    const tracked = (data.lifecycle.active || []).filter(
      (c) => !c.category_id || c.category_id === category.id,
    )
    const seen = new Set(fromCat.map((c) => c.symbol))
    const merged = [...fromCat, ...tracked.filter((c) => !seen.has(c.symbol))]
    return merged.filter((c) => {
      if (!q) return true
      return c.symbol.includes(q) || (c.company || '').toUpperCase().includes(q)
    })
  }, [data, category, categoryId, lifecycle, query])

  const onSelect = (symbol: string) => {
    setSelected(symbol)
    setActive('Stock Intelligence')
  }

  if (loading) return <div className="large-empty"><strong>Loading recommendations…</strong></div>
  if (error) return <EmptyState title="Recommendations unavailable" detail={error} />
  if (!data || !category) return <EmptyState title="No recommendation data yet" detail="Run a market scan and long-term refresh first." />

  const catTabs = data.categories.map((c) => `${c.label} (${c.count})`)
  const activeCatLabel = `${category.label} (${category.count})`

  return (
    <div className="reco-desk">
      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
      <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />
      <Panel
        title={category.label}
        subtitle={category.blurb}
      >
        <div className="reco-cmp-note" role="status">
          <span aria-hidden="true">⚠</span>
          <span>{data.cmp_note}</span>
          {data.scan_scanned_at ? (
            <em>Scan {relativeAge(data.scan_scanned_at)}</em>
          ) : null}
        </div>

        <SectionTabs
          tabs={catTabs}
          active={activeCatLabel}
          onChange={(tab) => {
            const match = data.categories.find((c) => tab.startsWith(c.label))
            if (match) setCategoryId(match.id)
          }}
        />

        <div className="reco-toolbar">
          <input
            type="search"
            placeholder="Search stocks"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search stocks"
          />
          <div className="reco-lifecycle" role="tablist" aria-label="Lifecycle">
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
                {tab === 'Active' ? ` · ${data.lifecycle.active_count}` : ` · ${data.lifecycle.closed_count}`}
              </button>
            ))}
          </div>
        </div>

        {cards.length === 0 ? (
          <EmptyState
            title={lifecycle === 'Closed' ? 'No closed picks in this category yet.' : category.empty_detail || 'No active picks in this category yet.'}
            detail={lifecycle === 'Active' ? 'Evidence filter found no matches in the current scan.' : 'Closed outcomes appear after tracked picks exit or signals resolve.'}
          />
        ) : (
          <div className="reco-grid">
            {cards.map((card) => (
              <CardTile key={`${card.lifecycle}-${card.symbol}-${card.setup_label}`} card={card} onSelect={onSelect} />
            ))}
          </div>
        )}
        <p className="reco-disclaimer">{data.disclaimer}</p>
      </Panel>
    </div>
  )
}

export function MarketReportsView(_props: ExperienceViewProps) {
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

  if (loading) return <div className="large-empty"><strong>Loading market reports…</strong></div>
  if (error) return <EmptyState title="Market reports unavailable" detail={error} />
  if (!data) return <EmptyState title="No reports yet" />

  const pulse = data.today_pulse || {}
  const takeaways = (pulse.takeaways as string[] | undefined) || []

  return (
    <div className="reco-desk market-reports-desk">
      <Panel title={data.title} subtitle={data.blurb}>
        <div className="reco-toolbar">
          <input
            type="search"
            placeholder="Search reports"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search reports"
          />
        </div>

        <div className="market-reports-layout">
          <ul className="market-reports-list">
            {reports.length === 0 ? (
              <li className="market-reports-empty">No reports match this search.</li>
            ) : (
              reports.map((r) => (
                <li key={r.id}>
                  <button
                    type="button"
                    className={selected?.id === r.id ? 'active' : ''}
                    onClick={() => setSelected(r)}
                  >
                    <strong>{r.title}</strong>
                    <span>{words(r.date)}</span>
                    {r.is_new ? <em className="market-report-new">{r.badge || 'New market report'}</em> : null}
                    <small>{r.summary}</small>
                  </button>
                </li>
              ))
            )}
          </ul>

          <div className="market-report-detail">
            {selected ? (
              <>
                <header>
                  <h2>{selected.title}</h2>
                  <p>{words(selected.date)}</p>
                </header>
                {takeaways.length > 0 && selected.is_new ? (
                  <ul className="market-pulse-takeaways">
                    {takeaways.map((t) => <li key={t}>{t}</li>)}
                  </ul>
                ) : (
                  <p>{selected.summary}</p>
                )}
                {Array.isArray(pulse.breakouts_today) && pulse.breakouts_today.length > 0 ? (
                  <div className="market-pulse-block">
                    <h3>Breakouts in focus</h3>
                    <p>{pulse.breakouts_today.map((b: { symbol?: string }) => b.symbol).filter(Boolean).join(', ')}</p>
                  </div>
                ) : null}
              </>
            ) : (
              <EmptyState title="Select a report" detail="Choose a Market Pulse entry from the list." />
            )}
          </div>
        </div>
        <p className="reco-disclaimer">{data.disclaimer}</p>
        {data.error ? <p className="reco-disclaimer">Pulse note: {data.error}</p> : null}
      </Panel>
    </div>
  )
}
