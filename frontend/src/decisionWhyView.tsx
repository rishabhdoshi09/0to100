import { useEffect, useState } from 'react'
import { fetchDecisionWhy } from './productApi'
import { DecisionWhyPanel } from './decisionWhy'
import type { DecisionWhy } from './decisionWhyModel'
import { EmptyState } from './designSystem'
import './decisionWhy.css'

/**
 * The page behind a symbol: why the desk decided what it decided.
 *
 * A load failure is shown as a load failure. It is never rendered as "no
 * evidence", because those are opposite claims and a trader acting on the
 * second one would be acting on a network error.
 */
export function DecisionWhyView({
  symbol,
  onSelect,
  suggestions = [],
}: {
  symbol: string
  onSelect?: (symbol: string) => void
  suggestions?: string[]
}) {
  const [why, setWhy] = useState<DecisionWhy | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [requested, setRequested] = useState((symbol || '').trim().toUpperCase())
  const [query, setQuery] = useState((symbol || '').trim().toUpperCase())

  useEffect(() => {
    const next = (symbol || '').trim().toUpperCase()
    if (!next) return
    setRequested(next)
    setQuery(next)
  }, [symbol])

  useEffect(() => {
    const wanted = requested.trim()
    if (!wanted) {
      setWhy(null)
      setError('')
      return
    }
    let cancelled = false
    setLoading(true)
    setError('')
    fetchDecisionWhy(wanted)
      .then((payload) => {
        if (!cancelled) setWhy(payload)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setWhy(null)
          setError(err instanceof Error ? err.message : String(err))
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [requested])

  const inspect = (raw: string) => {
    const clean = raw.trim().toUpperCase()
    if (!/^[A-Z0-9&.-]{1,32}$/.test(clean)) {
      setError('Enter a valid NSE symbol such as RELIANCE, TCS or HDFCBANK.')
      return
    }
    setError('')
    setQuery(clean)
    setRequested(clean)
    onSelect?.(clean)
  }

  const toolbar = (
    <div className="decision-why__toolbar" aria-label="Decision inspection controls">
      <form
        className="decision-why__search"
        onSubmit={(event) => {
          event.preventDefault()
          inspect(query)
        }}
      >
        <input
          aria-label="NSE symbol for decision explanation"
          placeholder="Enter NSE symbol"
          value={query}
          onChange={(event) => setQuery(event.target.value.toUpperCase())}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading…' : 'Explain decision'}
        </button>
      </form>
      {suggestions.length > 0 ? (
        <div className="decision-why__suggestions" aria-label="Recent symbols">
          <span>Recent</span>
          {suggestions.slice(0, 6).map((item) => (
            <button key={item} type="button" onClick={() => inspect(item)}>{item}</button>
          ))}
        </div>
      ) : null}
    </div>
  )

  if (!requested) {
    return (
      <section className="decision-why decision-why--console">
        {toolbar}
        <EmptyState
          title="Why this decision"
          detail="Enter any NSE symbol or choose a recent scanned name. QuantTerm will show the persisted decision record; an unavailable record stays unavailable."
        />
      </section>
    )
  }
  if (loading && !why) {
    return (
      <section className="decision-why decision-why--console">
        {toolbar}
        <EmptyState title={`Why ${requested}`} detail="Loading the persisted decision record…" />
      </section>
    )
  }
  if (error) {
    return (
      <section className="decision-why decision-why--console">
        {toolbar}
        <EmptyState
          title={`Why ${requested}`}
          detail={`Could not load the decision record: ${error}`}
        />
      </section>
    )
  }
  return (
    <section className="decision-why decision-why--console">
      {toolbar}
      <DecisionWhyPanel why={why} />
    </section>
  )
}
