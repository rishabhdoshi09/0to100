import './recommendations.css'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { money, pct, relativeAge, words } from './format'
import {
  fetchMarketReportsWorkspace,
  fetchRecommendationsWorkspace,
  type MarketReportItem,
  type RecommendationCard,
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
