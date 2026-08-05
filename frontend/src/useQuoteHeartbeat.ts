import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchQuoteHeartbeat, type QuoteHeartbeatPayload, type QuoteTick } from './api'

const LOW_POWER = import.meta.env.VITE_QT_LOW_POWER === '1'

export function useQuoteHeartbeat(symbols: string[], opts?: { enabled?: boolean }) {
  const enabled = opts?.enabled !== false
  const [payload, setPayload] = useState<QuoteHeartbeatPayload | null>(null)
  const [error, setError] = useState('')
  const key = useMemo(() => {
    const clean = [...new Set(symbols.map((s) => s.trim().toUpperCase()).filter(Boolean))]
    clean.sort()
    return clean.join(',')
  }, [symbols])
  const keyRef = useRef(key)
  keyRef.current = key

  useEffect(() => {
    if (!enabled || !key) {
      return
    }
    let cancelled = false
    const pollMs = LOW_POWER ? 10_000 : 4_000

    const tick = async () => {
      try {
        const next = await fetchQuoteHeartbeat(keyRef.current.split(',').filter(Boolean))
        if (!cancelled) {
          setPayload(next)
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Live quotes unavailable')
        }
      }
    }

    void tick()
    const timer = window.setInterval(() => void tick(), pollMs)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [enabled, key])

  const bySymbol = payload?.quotes || {}
  const get = (symbol: string): QuoteTick | undefined => bySymbol[symbol.toUpperCase()]

  return {
    payload,
    error,
    get,
    streaming: Boolean(payload?.streaming),
    sessionOpen: Boolean(payload?.session_open),
    honesty: payload?.honesty || '',
  }
}
