import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ControlName } from './types'
import { compactDateTime } from './format'
import { MetricCard, Panel } from './components'
import {
  fetchEducation,
  type EducationCard,
  type EducationFeed,
  type EducationLens,
} from './productApi'
import { DeskWait } from './DeskWait'

type Props = {
  runControl: (control: ControlName) => Promise<void>
  setSelected?: (symbol: string) => void
  setActive?: (page: string) => void
}

const LENS_LABELS: Record<EducationLens | 'ALL', string> = {
  ALL: 'All',
  MACRO: 'Macro',
  MICRO: 'Micro',
  POLICY: 'Policy',
  DERIVATIVES: 'F&O',
  CONCEPT: 'Concepts',
}

const KIND_LABEL: Record<string, string> = {
  NEWS_LESSON: 'News lesson',
  MACRO_THEME: 'Macro theme',
  CONCEPT: 'Concept',
}

function lensTone(lens: string): string {
  if (lens === 'MACRO') return 'edu-lens-macro'
  if (lens === 'MICRO') return 'edu-lens-micro'
  if (lens === 'POLICY') return 'edu-lens-policy'
  if (lens === 'DERIVATIVES') return 'edu-lens-fno'
  return 'edu-lens-concept'
}

function EducationCardRow({
  card,
  openSymbol,
  openFno,
}: {
  card: EducationCard
  openSymbol: (symbol: string) => void
  openFno: (symbol: string) => void
}) {
  return (
    <article className="edu-card">
      <header>
        <div>
          <span className={`edu-lens ${lensTone(card.lens)}`}>{card.lens}</span>
          <span>{KIND_LABEL[card.kind] || card.kind}</span>
          {card.official ? <span className="edu-official">Official</span> : null}
          {card.level && card.level !== 'current_events' ? <span>{card.level}</span> : null}
        </div>
        {card.published_at ? <time>{compactDateTime(card.published_at)}</time> : null}
      </header>
      <h3>{card.title}</h3>
      <p className="edu-teach">{card.teach_point}</p>
      {card.why_it_matters && card.why_it_matters !== card.teach_point ? (
        <p className="edu-why">{card.why_it_matters}</p>
      ) : null}
      <div className="edu-meta">
        {card.source ? <span>{card.source}</span> : null}
        {card.impact_score > 0 ? <span>Impact {card.impact_score}</span> : null}
        {card.corroboration_count > 1 ? <span>{card.corroboration_count} sources</span> : null}
        {card.direction && card.direction !== 'unclear' ? <span>{card.direction}</span> : null}
      </div>
      {(card.symbols.length > 0 || card.fno_symbols.length > 0) && (
        <div className="edu-symbols">
          {card.symbols.map((symbol) => (
            <button key={symbol} type="button" onClick={() => openSymbol(symbol)}>{symbol}</button>
          ))}
          {card.fno_symbols.map((symbol) => (
            <button key={`fno-${symbol}`} type="button" onClick={() => openFno(symbol)}>F&O {symbol}</button>
          ))}
        </div>
      )}
      {card.url ? (
        <a href={card.url} target="_blank" rel="noreferrer">Open original source</a>
      ) : null}
    </article>
  )
}

export function EducationView({ runControl, setSelected, setActive }: Props) {
  const [feed, setFeed] = useState<EducationFeed | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [lens, setLens] = useState<EducationLens | 'ALL'>('ALL')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const payload = await fetchEducation()
      setFeed(payload)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Education API unavailable')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const cards = useMemo(() => {
    const rows = feed?.cards || []
    if (lens === 'ALL') return rows
    return rows.filter((card) => card.lens === lens)
  }, [feed, lens])

  const openSymbol = (symbol: string) => {
    setSelected?.(symbol)
    setActive?.('Stock Intelligence')
  }
  const openFno = (symbol: string) => {
    setSelected?.(symbol)
    setActive?.('F&O Desk')
  }

  const summary = feed?.summary
  const byLens = summary?.by_lens || {}

  return (
    <section className="workspace-view">
      <div className="feature-purpose">
        <strong>What this page is for</strong>
        <p>
          Market education that crunches curated news, macro themes and fixed concept teach-ins
          into learnable cards (micro + macro). Never invents articles — and never a buy/sell signal.
        </p>
      </div>
      <div className="inline-actions">
        <button
          type="button"
          onClick={() => {
            void runControl('REFRESH_NEWS_NOW').then(() => void load())
          }}
        >
          Refresh news, then reload lessons
        </button>
        <button type="button" onClick={() => void load()}>
          {loading ? 'Loading…' : 'Reload education feed'}
        </button>
        <button type="button" onClick={() => setActive?.('News & Events')}>
          Open the news list
        </button>
        <button type="button" onClick={() => setActive?.('Market Overview')}>
          See market weather
        </button>
      </div>
      {error ? <div className="large-empty">{error}</div> : null}
      <div className="view-metrics">
        <MetricCard
          label="NEWS LESSONS"
          value={String(summary?.news_lessons ?? '—')}
          detail={`${summary?.articles_considered ?? 0} articles considered`}
        />
        <MetricCard
          label="MACRO THEMES"
          value={String(summary?.macro_themes ?? '—')}
          detail="Corroborated keyword themes"
          tone="purple"
        />
        <MetricCard
          label="CONCEPTS"
          value={String(summary?.concepts ?? '—')}
          detail="Evergreen share-market teach-ins"
          tone="green"
        />
        <MetricCard
          label="LENSES"
          value={`${byLens.MACRO || 0}M · ${byLens.MICRO || 0}μ`}
          detail={`Policy ${byLens.POLICY || 0} · F&O ${byLens.DERIVATIVES || 0} · Concepts ${byLens.CONCEPT || 0}`}
        />
      </div>
      {feed?.honesty ? <p className="edu-honesty">{feed.honesty}</p> : null}
      <div className="mode-tabs">
        {(Object.keys(LENS_LABELS) as Array<EducationLens | 'ALL'>).map((key) => (
          <button
            type="button"
            key={key}
            className={lens === key ? 'active' : ''}
            onClick={() => setLens(key)}
          >
            {LENS_LABELS[key]}
            {key !== 'ALL' && byLens[key] != null ? ` (${byLens[key]})` : ''}
          </button>
        ))}
      </div>
      <Panel
        title={`EDUCATION FEED · ${cards.length}`}
        subtitle="Macro weather, micro company context, policy, F&O positioning literacy, and concepts"
      >
        <div className="edu-feed">
          {loading && !feed ? <DeskWait kind="EDUCATION" /> : null}
          {!loading && cards.length === 0 ? (
            <div className="large-empty">
              {feed?.empty_hint
                || 'No cards for this lens yet. Refresh news, then reload Learn.'}
            </div>
          ) : null}
          {cards.map((card) => (
            <EducationCardRow key={card.id} card={card} openSymbol={openSymbol} openFno={openFno} />
          ))}
        </div>
      </Panel>
    </section>
  )
}
