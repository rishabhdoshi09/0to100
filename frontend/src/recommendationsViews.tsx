import './recommendations.css'
import { useEffect, useMemo, useState } from 'react'
import { money, pct, relativeAge, words } from './format'
import { BuyThesisSheet } from './BuyThesisSheet'
import { deskSymbol, thesisReplacesList } from './deskThesis'
import { usePhoneLayout } from './phoneLayout'
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
import { EvChip } from './evChip'
import { DeskWait, toDeskWaitScan } from './DeskWait'
import { SepaScoreChip } from './SepaMonitor'
import type { DisplayDepth } from './productLanguage'
import type { ScanRunnerHandle } from './scanRunner'

const CAT_ICONS: Record<string, string> = {
  best_setups: '7',
  wealth_builders: 'W',
  super_trends: 'S',
  momentum_breakouts: 'B',
  recovery_setups: 'R',
}

function badgeClass(action: string): string {
  const a = action.toLowerCase()
  if (a.includes('buy') || a === 'open' || a === 'tracked' || a === 'win' || a === 'strong') return ''
  if (a.includes('closed') || a.includes('loss') || a.includes('void') || a === 'weak') return 'is-closed'
  return 'is-watch'
}

function CardTile({
  card,
  selected,
  onSelect,
}: {
  card: RecommendationCard
  selected?: string
  onSelect: (symbol: string) => void
}) {
  const buy = card.entry ?? card.cmp ?? null
  const upside = card.upside_from_buy_pct
    ?? (buy != null && card.target != null && buy > 0
      ? ((Number(card.target) - Number(buy)) / Number(buy)) * 100
      : null)
  const risk = (card.risk_tier || 'Medium').toLowerCase()
  const symbol = deskSymbol(card.symbol)
  const openThesis = () => { if (symbol) onSelect(symbol) }
  return (
    <article
      role="button"
      tabIndex={0}
      data-symbol={symbol}
      className={`reco-pick${deskSymbol(selected) === symbol ? ' is-active' : ''}`}
      onClick={openThesis}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          openThesis()
        }
      }}
    >
      <div className="reco-pick-row1">
        <span className={`reco-buy ${badgeClass(card.action_badge)}`}>{card.action_badge}</span>
        <span className={`reco-risk-chip ${risk}`}>
          <span className="reco-risk-meter" aria-hidden="true" />
          {card.risk_tier} Risk
        </span>
      </div>
      <h3 className="reco-pick-name">{card.company || card.symbol}</h3>
      <div className="reco-pick-sub">
        <span>{card.symbol}</span>
        <span className="reco-tag">{card.category_label}</span>
        {card.cmp != null ? <span>CMP {money(card.cmp, 2)}</span> : null}
        {card.price_tag ? <span>{card.price_tag}</span> : null}
      </div>
      <div className="reco-pick-kpis reco-numbers-light">
        <div>
          <span>Buy</span>
          <strong>{buy != null ? money(buy, 2) : '—'}</strong>
        </div>
        <div>
          <span>Stop</span>
          <strong>{card.stop != null ? money(card.stop, 2) : '—'}</strong>
        </div>
        <div>
          <span>Target</span>
          <strong>{card.target != null ? money(card.target, 2) : '—'}</strong>
        </div>
        <div className="reco-gain">
          <strong className={upside != null && upside < 0 ? 'neg' : ''}>
            {upside != null ? `${upside >= 0 ? '↗ ' : '↘ '}${pct(upside)}` : '—'}
          </strong>
          <small>% upside from buy</small>
        </div>
      </div>
      <EvChip row={card} />
      <SepaScoreChip
        score={card.sepa_score}
        max={card.sepa_max}
        passed={card.sepa_passed}
        total={card.sepa_total}
        verdict={card.sepa_verdict}
        headline={card.sepa_headline}
      />
      {(card.qualify_reason || card.reason) ? (
        <p className="reco-pick-note">{card.qualify_reason || card.reason}</p>
      ) : null}
      {card.evidence_tags && card.evidence_tags.length > 0 ? (
        <div className="reco-pick-tags" aria-label="Evidence tags">
          {card.evidence_tags.slice(0, 4).map((tag) => (
            <span key={tag} className="reco-evidence-tag">{tag.replace(/_/g, ' ')}</span>
          ))}
        </div>
      ) : null}
      <span className="reco-pick-open">Read thesis</span>
    </article>
  )
}

function RecoWaitPanel({
  marketScan,
  longTermScan,
  depth,
}: {
  marketScan: ScanRunnerHandle
  longTermScan: ScanRunnerHandle
  depth: DisplayDepth
}) {
  const activeScan = marketScan.isActive ? marketScan : (longTermScan.isActive ? longTermScan : null)
  return (
    <div className="reco-light">
      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
      <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />
      <DeskWait kind="RECO_WORKSPACE" scan={toDeskWaitScan(activeScan)} />
    </div>
  )
}

export function RecommendationsView({
  dashboard,
  selected,
  setSelected,
  bars,
  setActive,
  marketScan,
  longTermScan,
  depth,
  onCompare,
  onWatchlist,
}: ExperienceViewProps & {
  onCompare?: (symbol: string) => void
  onWatchlist?: (symbol: string) => void
}) {
  const [data, setData] = useState<RecommendationsWorkspace | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [categoryId, setCategoryId] = useState('best_setups')
  const [lifecycle, setLifecycle] = useState<'Active' | 'Closed'>('Active')
  const [query, setQuery] = useState('')
  const phone = usePhoneLayout()

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

  const onSelect = (symbol: string) => {
    const clean = deskSymbol(symbol)
    if (clean) setSelected(clean)
  }

  const selectedRow = dashboard.scan.records.find((row) => deskSymbol(row.symbol) === deskSymbol(selected))
    || dashboard.long_term.records.find((row) => deskSymbol(row.symbol) === deskSymbol(selected))
    || dashboard.conviction.find((row) => deskSymbol(row.symbol) === deskSymbol(selected))

  if (loading) {
    return (
      <RecoWaitPanel
        marketScan={marketScan}
        longTermScan={longTermScan}
        depth={depth}
      />
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

  const thesisSheet = selected ? (
    <BuyThesisSheet
      symbol={deskSymbol(selected)}
      bars={bars}
      row={selectedRow as Record<string, unknown> | null}
      onClose={() => setSelected('')}
      onOpenResearch={() => setActive('Stock Intelligence')}
      onCompare={() => onCompare?.(selected)}
      onWatchlist={() => onWatchlist?.(selected)}
    />
  ) : null

  if (thesisReplacesList(phone, selected) && thesisSheet) {
    return <div className="reco-light reco-thesis-only">{thesisSheet}</div>
  }

  return (
    <div className="reco-light">
      <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
      <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />

      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={() => setActive('Recommendations')}>Ideas</button>
        <span>›</span>
        <button type="button" onClick={() => setLifecycle('Active')}>Categories</button>
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
          <button type="button" className="reco-filter-btn" onClick={() => setActive('Market Scanner')}>
            Table
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
          {cards.map((card, idx) => (
            <CardTile
              key={`${card.lifecycle}-${deskSymbol(card.symbol)}-${card.setup_label}-${card.category_id}-${idx}`}
              card={card}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
      {thesisSheet || (
        <p className="reco-foot">Tap a name for the buy thesis.</p>
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

export function MarketReportsView({
  setActive, setSelected: setStock, marketScan, longTermScan, depth,
}: ExperienceViewProps) {
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
    const activeScan = marketScan.isActive ? marketScan : (longTermScan.isActive ? longTermScan : null)
    return (
      <div className="reco-light">
        <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
        <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />
        <DeskWait kind="MARKET_PULSE" scan={toDeskWaitScan(activeScan)} />
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
        <button type="button" onClick={() => setActive('News & Events')}>Context</button>
        <span>›</span>
        <strong>Pulse</strong>
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
          <button type="button" className="reco-filter-btn" onClick={() => setActive('Education')}>
            Learn
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
              <div className="news-symbols">
                {pulse.breakouts_today
                  .map((b: { symbol?: string }) => b.symbol)
                  .filter((symbol): symbol is string => Boolean(symbol))
                  .map((symbol) => (
                    <button
                      type="button"
                      key={symbol}
                      onClick={() => {
                        setStock(symbol)
                        setActive('Stock Intelligence')
                      }}
                    >
                      {symbol}
                    </button>
                  ))}
              </div>
            </>
          ) : null}
        </div>
      ) : null}

      <p className="reco-foot">{data.disclaimer}</p>
      {data.error ? <p className="reco-foot">Pulse note: {data.error}</p> : null}
    </div>
  )
}
