import './recommendations.css'
import { useEffect, useMemo, useState } from 'react'
import { money, pct, relativeAge, words } from './format'
import {
  fetchMarketReportsWorkspace,
  fetchRecommendationsWorkspace,
  type DeskNote,
  type DeskNoteCompany,
  type MarketReportItem,
  type RecommendationCard,
  type RecommendationCase,
  type RecommendationsWorkspace,
  type MarketReportsWorkspace,
} from './productApi'
import type { ExperienceViewProps } from './experience'
import { LiveScanBanner } from './experience'
import { keepRicher, markInvestigate, recall } from './sessionMemory'

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

function CaseMemoryBox({ memory }: { memory?: RecommendationCase | null }) {
  if (!memory) return null
  const n = memory.n_similar ?? 0
  const verdict = (memory.verdict || 'unmeasured').replace(/_/g, ' ')
  const invalidation = (memory.invalidation || []).filter(Boolean)
  const similar = memory.similar
  const edge = memory.edge
  const quality = memory.setup_quality
  return (
    <aside className={`reco-case is-${memory.verdict || 'unmeasured'}`} aria-label="Case memory">
      <span>Case memory · {n} similar · {verdict}{memory.stance ? ` · ${memory.stance}` : ''}</span>
      <p>{memory.memory_line || memory.idea}</p>
      {similar?.found && similar.line ? <p>{similar.line}</p> : null}
      {edge && edge.profile && edge.profile !== 'UNKNOWN' ? <p>{edge.line}</p> : null}
      {quality?.score != null ? (
        <p className="reco-case-invalid">{quality.label || 'Setup Quality'}: {quality.score}/100 — not a win probability.</p>
      ) : null}
      {invalidation.length > 0 ? (
        <p className="reco-case-invalid">What proves it wrong: {invalidation[0]}</p>
      ) : null}
      {memory.proven ? null : (
        <em>{n > 0 ? 'Not proven yet — fewer than 30 comparable outcomes.' : 'Not remembered yet. Tonight’s check writes the first outcome.'}</em>
      )}
    </aside>
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
  const points = (card.key_points && card.key_points.length > 0)
    ? card.key_points
    : (card.why_now || []).filter(Boolean)
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
        <div className="reco-pick-together">
          <div className="reco-pick-identity">
            <h3 className="reco-pick-name">{card.company || card.symbol}</h3>
            <div className="reco-pick-sub">
              <span>{card.symbol}</span>
              <span className="reco-tag">{card.category_label}</span>
              {card.horizon ? <span>{card.horizon}</span> : null}
              {card.price_tag ? <span>{card.price_tag}</span> : null}
            </div>
            {card.setup_label ? <p className="reco-pick-setup">{card.setup_label}</p> : null}
          </div>
          {points.length > 0 ? (
            <div className="reco-key-points">
              <span>Key points</span>
              <ul>
                {points.slice(0, 5).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : (
            why ? <p className="reco-pick-note">{why}</p> : <p className="reco-pick-note">No key points on this snapshot.</p>
          )}
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
          <StatusChip label="Setup Quality" value={card.setup_quality != null ? `${Math.round(card.setup_quality)}/100` : '—'} />
          <StatusChip label="Payoff" value={card.expected_payoff} />
          <StatusChip label="Evidence" value={card.evidence} />
          <StatusChip label="Health" value={card.strategy_health} />
          <StatusChip label="Market" value={card.market_support} />
        </div>
        <CaseMemoryBox memory={card.case} />
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
  onInvestigate,
}: {
  card: RecommendationCard
  categoryLabel: string
  onBack: () => void
  onResearch: () => void
  onInvestigate: () => void
}) {
  const [showEvidence, setShowEvidence] = useState(false)
  const zone = buyZoneLabel(card)
  const upside = card.upside_to_target_pct ?? card.upside_from_entry_pct
  const risk = (card.risk_tier || 'Medium').toLowerCase()
  const points = (card.key_points && card.key_points.length > 0)
    ? card.key_points
    : (card.why_now || [])
  return (
    <section className="reco-sheet" aria-label={`${card.symbol} decision`}>
      <nav className="reco-crumb" aria-label="Breadcrumb">
        <button type="button" onClick={onBack}>Recommendations</button>
        <span>›</span>
        <button type="button" onClick={onBack}>{categoryLabel}</button>
        <span>›</span>
        <strong>{card.symbol}</strong>
      </nav>
      <header className="reco-sheet-hero reco-pick-together">
        <div className="reco-pick-identity">
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
        </div>
        {points.length > 0 ? (
          <div className="reco-key-points">
            <span>Key points</span>
            <ul>
              {points.slice(0, 5).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </header>
      <CaseMemoryBox memory={card.case} />
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
        <StatusChip label="Setup Quality" value={card.setup_quality != null ? `${Math.round(card.setup_quality)}/100` : '—'} />
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
          <h3>Key points</h3>
          {points.length > 0 ? (
            <ul>{points.map((item) => <li key={item}>{item}</li>)}</ul>
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
        <button type="button" className="reco-primary" onClick={onInvestigate}>
          Investigate
        </button>
        <button type="button" className="reco-ghost" onClick={onResearch}>
          Full research
        </button>
      </div>
      {showEvidence ? <EvidencePanel card={card} /> : null}
    </section>
  )
}

export function RecommendationsView({
  dashboard,
  setSelected,
  setActive,
  marketScan,
  longTermScan,
  depth,
}: ExperienceViewProps) {
  const [data, setData] = useState<RecommendationsWorkspace | null>(() => recall<RecommendationsWorkspace>('reco-workspace') ?? null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(() => !recall('reco-workspace'))
  const [categoryId, setCategoryId] = useState('wealth_builders')
  const [lifecycle, setLifecycle] = useState<'Active' | 'Closed'>('Active')
  const [query, setQuery] = useState('')
  const [selectedCard, setSelectedCard] = useState<RecommendationCard | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!recall('reco-workspace')) setLoading(true)
    fetchRecommendationsWorkspace()
      .then((payload) => {
        if (!cancelled) {
          const kept = keepRicher('reco-workspace', payload, (row) => !(row.categories || []).some((c) => (c.count || 0) > 0 || (c.cards || []).length > 0))
          setData(kept)
          const firstWithCards = kept.categories.find((c) => c.count > 0)
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
  }, [dashboard.scan.scanned_at, dashboard.long_term.scanned_at, marketScan.succeeded, longTermScan.succeeded])

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

  if (loading && !data) {
    return (
      <div className="reco-light">
        <div className="reco-empty"><strong>Loading recommendations…</strong></div>
      </div>
    )
  }
  if (error || !data || !category) {
    return (
      <div className="reco-light">
        <LiveScanBanner scan={marketScan} depth={depth} label="Market scan" />
        <LiveScanBanner scan={longTermScan} depth={depth} label="Long-term scan" />
        <div className="reco-empty">
          <strong>{error || 'No recommendation data yet'}</strong>
          <p>Run a market scan and long-term refresh. Sidebar navigation only opens this page — it does not start those jobs.</p>
          <div className="reco-hero-actions">
            <button type="button" className="reco-primary" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
              {marketScan.isBusy ? 'Scanning market…' : 'Scan market'}
            </button>
            <button type="button" className="reco-ghost" disabled={longTermScan.isBusy} onClick={() => void longTermScan.start()}>
              {longTermScan.isBusy ? 'Refreshing long-term…' : 'Refresh long-term'}
            </button>
          </div>
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
          onInvestigate={() => {
            markInvestigate(selectedCard.symbol)
            openResearch(selectedCard.symbol)
          }}
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
        <div className="reco-hero-actions">
          <button type="button" className="reco-primary" disabled={marketScan.isBusy} onClick={() => void marketScan.start()}>
            {marketScan.isBusy
              ? `Scanning…${marketScan.percent != null ? ` ${marketScan.percent}%` : ''}${marketScan.etaLine ? ` · ${marketScan.etaLine}` : ''}`
              : 'Scan market'}
          </button>
          <button type="button" className="reco-ghost" disabled={longTermScan.isBusy} onClick={() => void longTermScan.start()}>
            {longTermScan.isBusy ? 'Refreshing long-term…' : 'Refresh long-term'}
          </button>
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

function DeskNoteMagazine({
  note,
  onSymbol,
}: {
  note: DeskNote
  onSymbol: (symbol: string) => void
}) {
  const wrap = note.wrap || []
  const explainers = note.explainers || []
  const desks = note.desks || []
  const sourced = note.wrap_sourced ?? wrap.filter((b) => b.available).length
  const empty = note.wrap_empty ?? wrap.filter((b) => !b.available).length

  return (
    <section className="desk-note" aria-label="Today’s market wrap">
      <header className="desk-note-hero">
        <p className="desk-kicker">Desk note · sourced, not invented</p>
        <h2>{note.title || 'Today’s market wrap'}</h2>
        <p>{note.blurb}</p>
        <p className="desk-tally">
          {sourced} sourced wrap line{sourced === 1 ? '' : 's'}
          {empty ? ` · ${empty} empty slot${empty === 1 ? '' : 's'}` : ''}
        </p>
      </header>

      <ol className="desk-wrap">
        {wrap.map((bullet) => (
          <li key={bullet.id} className={bullet.available ? '' : 'is-empty'}>
            <article>
              <span className="desk-label">{bullet.label}</span>
              {bullet.available ? (
                <>
                  <h3>{bullet.headline}</h3>
                  {bullet.summary ? <p>{bullet.summary}</p> : null}
                  <div className="desk-meta">
                    {bullet.source ? <span>{bullet.source}</span> : null}
                    {bullet.official ? <em>Official</em> : null}
                    {bullet.symbols.map((sym) => (
                      <button key={sym} type="button" onClick={() => onSymbol(sym)}>{sym}</button>
                    ))}
                    {bullet.url ? (
                      <a href={bullet.url} target="_blank" rel="noreferrer">Open source</a>
                    ) : null}
                  </div>
                </>
              ) : (
                <p className="desk-empty">{bullet.empty_detail || 'No sourced headline yet.'}</p>
              )}
            </article>
          </li>
        ))}
      </ol>

      {explainers.length > 0 ? (
        <div className="desk-explainers">
          {explainers.map((item) => (
            <article key={item.id}>
              <span className="desk-label">Concept</span>
              <h3>{item.title}</h3>
              <p>{item.teach_point}</p>
              {item.why_it_matters ? <p className="desk-why">{item.why_it_matters}</p> : null}
            </article>
          ))}
        </div>
      ) : null}

      {desks.length > 0 ? (
        <>
          <h3 className="desk-section">Company desks · watch questions, not a buy list</h3>
          <div className="desk-desks">
            {desks.map((desk) => (
              <DeskTile key={desk.symbol} desk={desk} onSymbol={onSymbol} />
            ))}
          </div>
        </>
      ) : null}

      {note.memory ? (
        <aside className="desk-theme reco-case-morning">
          <span className="desk-label">{note.memory.title || 'What QuantTerm remembers'}</span>
          <p>{note.memory.blurb}</p>
          {(note.memory.setups || []).length > 0 ? (
            <ul>
              {(note.memory.setups || []).map((item) => (
                <li key={item.setup}>{item.memory_line}</li>
              ))}
            </ul>
          ) : (
            <p className="desk-empty">No settled cases yet — tonight’s check is how memory starts.</p>
          )}
        </aside>
      ) : null}

      {note.decision_memory ? (
        <aside className="desk-theme reco-case-morning" aria-label="Decision memory">
          <span className="desk-label">{note.decision_memory.title || 'Decision Memory'}</span>
          <p>{note.decision_memory.blurb}</p>
          {note.decision_memory.shadow?.line ? <p>{note.decision_memory.shadow.line}</p> : null}
          {note.decision_memory.trust?.line ? <p>{note.decision_memory.trust.line}</p> : null}
          {(note.decision_memory.shadow?.gates || []).length > 0 ? (
            <ul>
              {(note.decision_memory.shadow?.gates || []).map((g) => (
                <li key={g.gate}>{g.line}</li>
              ))}
            </ul>
          ) : null}
        </aside>
      ) : null}

      {note.theme ? (
        <aside className="desk-theme">
          <span className="desk-label">Common theme</span>
          <h3>{note.theme.title}</h3>
          <p>{note.theme.body}</p>
        </aside>
      ) : null}

      {note.disclaimer ? <p className="reco-foot">{note.disclaimer}</p> : null}
      {note.error ? <p className="reco-foot">Desk note: {note.error}</p> : null}
    </section>
  )
}

function DeskTile({
  desk,
  onSymbol,
}: {
  desk: DeskNoteCompany
  onSymbol: (symbol: string) => void
}) {
  return (
    <article className={`desk-tile ${desk.available ? '' : 'is-empty'}`}>
      <header>
        <button type="button" className="desk-sym" onClick={() => onSymbol(desk.symbol)}>
          {desk.symbol}
        </button>
        <strong>{desk.name}</strong>
        {desk.is_recommendation ? null : <em>Not a pick</em>}
      </header>
      <p className="desk-lens">{desk.lens}</p>
      {desk.available ? (
        <>
          {desk.source_headline ? <p className="desk-src">{desk.source_headline}</p> : null}
          {desk.scan_status ? (
            <p className="desk-scan">Scan {desk.scan_status}{desk.scan_reason ? ` · ${desk.scan_reason}` : ''}</p>
          ) : null}
        </>
      ) : (
        <p className="desk-empty">{desk.empty_detail}</p>
      )}
      {desk.watch.length > 0 ? (
        <>
          <h4>Watch</h4>
          <ul>{desk.watch.map((item) => <li key={item}>{item}</li>)}</ul>
        </>
      ) : null}
      {desk.risks.length > 0 ? (
        <>
          <h4>Risks</h4>
          <ul>{desk.risks.map((item) => <li key={item}>{item}</li>)}</ul>
        </>
      ) : null}
      <div className="desk-meta">
        {desk.source ? <span>{desk.source}</span> : null}
        {desk.url ? <a href={desk.url} target="_blank" rel="noreferrer">Open source</a> : null}
        <button type="button" onClick={() => onSymbol(desk.symbol)}>Stock Intelligence</button>
      </div>
    </article>
  )
}

export function MarketReportsView({ dashboard, setActive, setSelected, marketScan, runControl }: ExperienceViewProps) {
  const [data, setData] = useState<MarketReportsWorkspace | null>(() => recall<MarketReportsWorkspace>('market-reports') ?? null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(() => !recall('market-reports'))
  const [query, setQuery] = useState('')
  const [selected, setSelectedReport] = useState<MarketReportItem | null>(null)
  const [newsBusy, setNewsBusy] = useState(false)
  const newsStamp = dashboard.operations?.latest?.NEWS_REFRESH?.updated_at
    || dashboard.news?.latest_refresh?.updated_at

  useEffect(() => {
    let cancelled = false
    if (!recall('market-reports')) setLoading(true)
    fetchMarketReportsWorkspace()
      .then((payload) => {
        if (!cancelled) {
          const kept = keepRicher('market-reports', payload, (row) => !(row.reports || []).length)
          setData(kept)
          setSelectedReport(kept.reports[0] || payload.reports[0] || null)
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
  }, [dashboard.scan.scanned_at, marketScan.succeeded, newsStamp])

  const refreshNews = async () => {
    setNewsBusy(true)
    try {
      await runControl('REFRESH_NEWS_NOW')
    } finally {
      setNewsBusy(false)
    }
  }

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

  const openSymbol = (symbol: string) => {
    setSelected(symbol)
    setActive('Stock Intelligence')
  }

  if (loading && !data) {
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
          <p>Refresh news to build Market Reports. Opening this page does not fetch anything by itself.</p>
          <div className="reco-hero-actions">
            <button type="button" className="reco-primary" disabled={newsBusy} onClick={() => void refreshNews()}>
              {newsBusy ? 'Refreshing news…' : 'Refresh news and filings'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  const pulse = data.today_pulse || {}
  const selectedTakeaways = selected?.takeaways?.length
    ? selected.takeaways
    : (selected?.is_new ? ((pulse.takeaways as string[] | undefined) || []) : [])
  const selectedBreakouts = selected?.breakouts_today?.length
    ? selected.breakouts_today
    : (selected?.is_new && Array.isArray(pulse.breakouts_today)
      ? (pulse.breakouts_today as { symbol?: string }[]).map((b) => b.symbol).filter(Boolean) as string[]
      : [])
  const selectedGainers = selected?.gainers?.length
    ? selected.gainers
    : (selected?.is_new && Array.isArray(pulse.gainers) ? pulse.gainers as { symbol: string; price?: number; chg_pct?: number }[] : [])
  const selectedLosers = selected?.losers?.length
    ? selected.losers
    : (selected?.is_new && Array.isArray(pulse.losers) ? pulse.losers as { symbol: string; price?: number; chg_pct?: number }[] : [])
  const selectedIndices = selected?.snapshot?.indices?.length
    ? selected.snapshot.indices
    : (selected?.is_new && Array.isArray((pulse.snapshot as { indices?: unknown[] } | undefined)?.indices)
      ? ((pulse.snapshot as { indices: Array<{ name: string; price?: number; chg_pct?: number }> }).indices)
      : [])

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
        {data.as_of_ist ? (
          <p className="reco-sheet-cmp">As of {data.as_of_ist} IST — latest session only.</p>
        ) : null}
        <div className="reco-hero-actions">
          <button type="button" className="reco-primary" disabled={newsBusy} onClick={() => void refreshNews()}>
            {newsBusy ? 'Refreshing news…' : 'Refresh news and filings'}
          </button>
        </div>
      </header>

      {data.desk_note ? (
        <DeskNoteMagazine note={data.desk_note} onSymbol={openSymbol} />
      ) : null}

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
                onClick={() => setSelectedReport(r)}
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
          {selectedTakeaways.length > 0 ? (
            <ul>
              {selectedTakeaways.map((t) => <li key={t}>{t}</li>)}
            </ul>
          ) : (
            <p>{selected.summary}</p>
          )}
          {selectedIndices.length > 0 ? (
            <>
              <h3>Latest session</h3>
              <ul>
                {selectedIndices.map((idx) => (
                  <li key={idx.name}>
                    {idx.name}
                    {idx.chg_pct != null ? ` ${idx.chg_pct >= 0 ? '▲' : '▼'} ${idx.chg_pct.toFixed(2)}%` : ''}
                    {idx.price != null ? ` at ${money(idx.price, 0)}` : ''}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
          {selectedGainers.length > 0 ? (
            <>
              <h3>Top gainers</h3>
              <p>{selectedGainers.map((g) => `${g.symbol}${g.chg_pct != null ? ` ${g.chg_pct >= 0 ? '+' : ''}${g.chg_pct.toFixed(1)}%` : ''}`).join(', ')}</p>
            </>
          ) : null}
          {selectedLosers.length > 0 ? (
            <>
              <h3>Top losers</h3>
              <p>{selectedLosers.map((g) => `${g.symbol}${g.chg_pct != null ? ` ${g.chg_pct.toFixed(1)}%` : ''}`).join(', ')}</p>
            </>
          ) : null}
          {selectedBreakouts.length > 0 ? (
            <>
              <h3>Breakouts in focus</h3>
              <p>{selectedBreakouts.join(', ')}</p>
            </>
          ) : null}
        </div>
      ) : null}

      <p className="reco-foot">{data.disclaimer}</p>
      {data.error ? <p className="reco-foot">Pulse note: {data.error}</p> : null}
    </div>
  )
}
