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
export function DecisionWhyView({ symbol }: { symbol: string }) {
  const [why, setWhy] = useState<DecisionWhy | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const wanted = (symbol || '').trim()
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
  }, [symbol])

  if (!symbol) {
    return <EmptyState title="Why this decision" detail="Pick a symbol to see its decision." />
  }
  if (loading && !why) {
    return <EmptyState title={`Why ${symbol}`} detail="Loading the decision record…" />
  }
  if (error) {
    return (
      <EmptyState
        title={`Why ${symbol}`}
        detail={`Could not load the decision record: ${error}`}
      />
    )
  }
  return <DecisionWhyPanel why={why} />
}
