import './recommendations.css'
import { useEffect, useMemo, useState } from 'react'
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

function toneClass(value?: string | null): string {
  const v = (value || '').toLowerCase()
  if (['positive', 'strong', 'normal', 'buy'].includes(v)) return 'is-good'
  if (['mixed', 'moderate', 'caution', 'watch', 'unproven', 'unmeasured', 'thin'].includes(v)) return 'is-mid'
  if (['negative', 'weak', 'degraded', 'high'].includes(v)) return 'is-bad'
  return 'is-mid'
}

function buyZoneLabel(card: RecommendationCard): { label: string; value: string } {
  const lo = card.buy_zone_low
  const hi = card.buy_zone_high
  if (lo != null && hi != null && lo !== hi) {
    return { label: 'Buy Zone', value: `${money(lo, 2)} – ${money(hi, 2)}` }
  }
  if (lo != null || card.entry != null) {
    return { label: 'Entry', value: money(lo ?? card.entry, 2) }
  }
  return { label: 'Entry', value: '—' }
}

function StatusChip({ label, value }: { label: string; value?: string | null }) {
  return (
    <span className={`reco-status ${toneClass(value)}`}>
      <small>{label}</small>
      <strong>{value || '—'}</strong>
    </span>
  )
}

function CardTile({
  card,
  onSelect,
}: {
  card: RecommendationCard
  onSelect: (card: RecommendationCard) => void
}) {
  const upside = card.upside_to_target_pct ?? card.upside_from_entry_pct
  const risk = (card.risk_tier || 'Medium').toLowerCase()
  const zone = buyZoneLabel(card)
  const why = (card.why_now && card.why_now[0]) || card.qualify_reason || card.reason
  return (
    <article className="reco-pick">
      <button type="button" className="reco-pick-hit" onClick={() => onSelect(card)}>
        <div className="reco-pick-row1">
          <span className={`reco-buy ${badgeClass(card.action_badge)}`}>{card.action_badge}</span>
          <span className="reco-opp">{card.opportunity_label || 'WATCH'}</span>
          <span className={`reco-risk-chip ${risk}`}>
            <span className="reco-risk-meter" aria-hidden="true" />
            {card.risk_tier} Risk
          </span>
        </div>
        <h3 className="reco-pick-name">{card.company || card.symbol}</h3>
        <div className="reco-pick-sub">
          <span>{card.symbol}</span>
          <span className="reco-tag">{card.category_label}</span>
          {card.horizon ? <span>{card.horizon}</span> : null}
          {card.price_tag ? <span>{card.price_tag}</span> : null}
        </div>
        <div className="reco-pick-kpis">
          <div>
            <span>{zone.label}</span>
            <strong>{zone.value}</strong>
          </div>
          <div>
            <span>Target</span>
            <strong>{card.target != null ? money(card.target, 2) : '—'}</strong>
          </div>
          <div>
            <span>Stop</span>
            <strong>{card.stop != null ? money(card.stop, 2) : '—'}</strong>
          </div>
          <div className="reco-gain">
            <span>Potential upside</span>
            {upside != null ? (
              <strong className={upside < 0 ? 'neg' : ''}>
                {upside >= 0 ? '↗ ' : '↘ '}
                {pct(upside)}
              </strong>
            ) : (
              <strong>—</strong>
            )}
          </div>
        </div>
        <div className="reco-status-row" aria-label="Decision status">
          <StatusChip label="Payoff" value={card.expected_payoff} />
          <StatusChip label="Evidence" value={card.evidence} />
          <StatusChip label="Health" value={card.strategy_health} />
          <StatusChip label="Market" value={card.market_support} />
        </div>
        {why ? <p className="reco-pick-note">{why}</p> : null}
      </button>
    </article>
  )
}

function EvidencePanel({ card }: { card: RecommendationCard }) {
  const panel = card.evidence_panel
  if (!panel) {
    return <p className="reco-pick-note">No evidence panel on this row.</p>
  }
  const rows: Array<[string, string]> = [
    ['Sample size', panel.sample_size != null ? String(panel.sample_size) : 'Below 30 — no claim'],
    ['Conservative EV', panel.ev_lb_pct != null ? pct(panel.ev_lb_pct) : 'Unproven'],
    ['Headline EV', panel.ev_pct != null ? pct(panel.ev_pct) : '—'],
    ['Win rate', panel.p_win != null ? `${panel.p_win}%` : '—'],
    ['Confidence', panel.confidence || '—'],
    ['Score', panel.score != null ? String(Math.round(panel.score)) : '—'],
    ['RSI', panel.rsi != null ? panel.rsi.toFixed(1) : '—'],
    ['Volume', panel.volume_ratio != null ? `${panel.volume_ratio.toFixed(1)}×` : '—'],
    ['Coverage', panel.fundamental_coverage != null ? `${panel.fundamental_coverage}%` : '—'],
    ['Source', [panel.tech_source, panel.price_tag].filter(Boolean).join(' · ') || 'Saved scan'],
    ['Signals', (panel.signals || []).join(', ') || '—'],
  ]
  return (
    <div className="reco-evidence-panel">
      <p>{panel.provenance}</p>
      <dl>
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function DecisionSheet({
  card,
  categoryLabel,
  onBack,
  onResearch,
}: {
  card: RecommendationCard
  categoryLabel: string
  onBack: () => void
  onResearch: () => void
}) {
  const [showEvidence, setShowEvidence] = useState(false)
  const zone = buyZoneLabel(card)
  const upside = card.upside_to_target_pct ?? card.upside_from_entry_pct
  const risk = (card.risk_tier || 'Medium').toLowerCase()
  return (
    <section className="reco-sheet" aria-label={`${card.symbol} decision`}>
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={onBack}>Recommendations</button>
        <span>›</span>
        <button type="button" onClick={onBack}>{categoryLabel}</button>
        <span>›</span>
        <strong>{card.symbol}</strong>
      </nav>
      <header className="reco-sheet-hero">
        <div className="reco-pick-row1">
          <span className={`reco-buy ${badgeClass(card.action_badge)}`}>{card.action_badge}</span>
          <span className="reco-opp">{card.opportunity_label || 'WATCH'}</span>
          <span className={`reco-risk-chip ${risk}`}>{card.risk_tier} Risk</span>
        </div>
        <h2>{card.company || card.symbol}</h2>
        <p>
          {card.symbol}
          {card.sector && card.sector !== '—' ? ` · ${card.sector}` : ''}
          {card.horizon ? ` · Horizon ${card.horizon}` : ''}
        </p>
      </header>
      <div className="reco-sheet-kpis">
        <div>
          <span>{zone.label}</span>
          <strong>{zone.value}</strong>
        </div>
        <div>
          <span>Target</span>
          <strong>{card.target != null ? money(card.target, 2) : '—'}</strong>
        </div>
        <div>
          <span>Stop</span>
          <strong>{card.stop != null ? money(card.stop, 2) : '—'}</strong>
        </div>
        <div>
          <span>Potential upside</span>
          <strong className={upside != null && upside < 0 ? 'neg' : ''}>
            {upside != null ? pct(upside) : '—'}
          </strong>
        </div>
      </div>
      {card.cmp != null ? (
        <p className="reco-sheet-cmp">
          Current price {money(card.cmp, 2)}
          {card.price_tag ? ` · ${card.price_tag}` : ''}
        </p>
      ) : null}
      <div className="reco-status-row reco-status-row-lg">
        <StatusChip label="Expected payoff" value={card.expected_payoff} />
        <StatusChip label="Evidence" value={card.evidence} />
        <StatusChip label="Strategy health" value={card.strategy_health} />
        <StatusChip label="Market support" value={card.market_support} />
      </div>
      {card.expected_payoff_detail ? (
        <p className="reco-pick-note">{card.expected_payoff_detail}</p>
      ) : null}
      <div className="reco-sheet-cols">
        <div>
          <h3>Why now</h3>
          {(card.why_now && card.why_now.length > 0) ? (
            <ul>{card.why_now.map((item) => <li key={item}>{item}</li>)}</ul>
          ) : (
            <p>No plain-language confirms on this snapshot.</p>
          )}
        </div>
        <div>
          <h3>What changes our mind</h3>
          {(card.what_changes_mind && card.what_changes_mind.length > 0) ? (
            <ul>{card.what_changes_mind.map((item) => <li key={item}>{item}</li>)}</ul>
          ) : (
            <p>Invalidation levels are not set on this row.</p>
          )}
        </div>
      </div>
      {card.next_step ? <p className="reco-next"><strong>Next step.</strong> {card.next_step}</p> : null}
      <div className="reco-sheet-actions">
        <button
          type="button"
          className="reco-ghost"
          aria-expanded={showEvidence}
          onClick={() => setShowEvidence((open) => !open)}
        >
          {showEvidence ? 'Hide evidence' : 'See evidence'}
        </button>
        <button type="button" className="reco-primary" onClick={onResearch}>
          Full research
        </button>
      </div>
      {showEvidence ? <EvidencePanel card={card} /> : null}
    </section>
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
  const [selectedCard, setSelectedCard] = useState<RecommendationCard | null>(null)

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
    // Active: only this category's research cards (never bleed prior category).
    return (category.cards || []).filter(matchQuery)
  }, [data, category, lifecycle, query])

  const onSelect = (card: RecommendationCard) => {
    setSelectedCard(card)
  }

  const openResearch = (symbol: string) => {
    setSelected(symbol)
    setActive('Stock Intelligence')
  }

  if (loading) {
    return (
      <div className="reco-light">
        <div className="reco-empty"><strong>Loading recommendations…</strong></div>
      </div>
    )
  }
  if (error || !data || !category) {
    return (
      <div className="reco-light">
        <div className="reco-empty">
          <strong>{error || 'No recommendation data yet'}</strong>
          <p>Run a market scan and long-term refresh first.</p>
        </div>
      </div>
    )
  }

  if (selectedCard) {
    return (
      <div className="reco-light">
        <DecisionSheet
          card={selectedCard}
          categoryLabel={category.label}
          onBack={() => setSelectedCard(null)}
          onResearch={() => openResearch(selectedCard.symbol)}
        />
        <p className="reco-foot">{data.disclaimer}</p>
      </div>
    )
  }

  return (
    <div className="reco-light">
      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
      <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />

      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Home')}>Home</button>
        <span>›</span>
        <button type="button" onClick={() => setLifecycle('Active')}>Recommendations</button>
        <span>›</span>
        <strong>{category.label}</strong>
      </nav>

      <header className="reco-hero">
        <div className="reco-hero-icon" aria-hidden="true">
          {CAT_ICONS[category.id] || '•'}
        </div>
        <div>
          <h2>{category.label}</h2>
          <p>{category.blurb}</p>
        </div>
      </header>

      {data.desk ? (
        <div className="reco-desk-strip" aria-label="Market and strategy snapshot">
          <StatusChip label="Market support" value={data.desk.market_support} />
          <StatusChip label="Strategy health" value={data.desk.strategy_health} />
          <p>{data.desk.market_support_detail || data.desk.strategy_health_detail}</p>
        </div>
      ) : null}

      <div className="reco-cat-rail" role="tablist" aria-label="Recommendation categories">
        {data.categories.map((c) => (
          <button
            key={c.id}
            type="button"
            role="tab"
            aria-selected={c.id === category.id}
            className={c.id === category.id ? 'active' : ''}
            onClick={() => {
              setCategoryId(c.id)
              setSelectedCard(null)
            }}
          >
            {c.label} · {c.count}
          </button>
        ))}
      </div>

      <div className="reco-cmp-banner" role="status">
        <span className="ico" aria-hidden="true">!</span>
        <div>
          <div>{data.cmp_note}</div>
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
    </div>
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
      <div className="reco-light">
        <div className="reco-empty"><strong>Loading market reports…</strong></div>
      </div>
    )
  }
  if (error || !data) {
    return (
      <div className="reco-light">
        <div className="reco-empty">
          <strong>{error || 'No reports yet'}</strong>
        </div>
      </div>
    )
  }

  const pulse = data.today_pulse || {}
  const takeaways = (pulse.takeaways as string[] | undefined) || []

  return (
    <div className="reco-light market-reports-desk">
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Home')}>Home</button>
        <span>›</span>
        <strong>Market Reports</strong>
      </nav>

      <header className="rw-reports-hero">
        <h1>{data.title}</h1>
        <p>{data.blurb}</p>
      </header>

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
    </div>
  )
}
