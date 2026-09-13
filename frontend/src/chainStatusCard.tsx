import { useEffect, useState } from 'react'
import { fetchChainStatus, type ChainStatus } from './productApi'

/**
 * An intentionally gated desk and a broken desk both render as an empty
 * screen. This card is the difference: it names the first link that stopped,
 * why, and what clears it -- so "nothing is happening" is never a mystery.
 */
export function ChainStatusCard() {
  const [chain, setChain] = useState<ChainStatus | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let alive = true
    const load = () => {
      fetchChainStatus()
        .then((data) => { if (alive) { setChain(data); setError('') } })
        .catch((err: unknown) => {
          if (alive) setError(err instanceof Error ? err.message : 'chain status unavailable')
        })
    }
    load()
    // Coarse: this walks real subsystems, so it must not be polled hard.
    const timer = window.setInterval(load, 60_000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [])

  if (error) {
    return (
      <div className="chain-card">
        <div className="chain-head"><strong>PIPELINE</strong><span className="chain-unknown">UNAVAILABLE</span></div>
        <p className="chain-why">{error}</p>
      </div>
    )
  }
  if (!chain) {
    return (
      <div className="chain-card">
        <div className="chain-head"><strong>PIPELINE</strong><span className="chain-unknown">CHECKING…</span></div>
      </div>
    )
  }

  const tone: Record<string, string> = {
    FLOWING: 'chain-ok', BLOCKED: 'chain-stop', WAITING: 'chain-wait', UNKNOWN: 'chain-unknown',
  }

  return (
    <div className="chain-card">
      <div className="chain-head">
        <strong>PIPELINE</strong>
        <span className={chain.chain_complete ? 'chain-ok' : 'chain-wait'}>
          {chain.flowing}/{chain.total} FLOWING
        </span>
      </div>
      <ol className="chain-links">
        {chain.links.map((l) => (
          <li key={l.link} title={l.unblock || l.detail}>
            <span className={`chain-dot ${tone[l.state]}`} aria-hidden="true" />
            <span className="chain-name">{l.link}</span>
            <span className="chain-detail">{l.detail}</span>
          </li>
        ))}
      </ol>
      {chain.first_stop ? (
        <div className="chain-stop-box">
          <strong>First stop — {chain.first_stop}</strong>
          <p>{chain.first_stop_detail}</p>
          <p className="chain-action">{chain.first_stop_unblock}</p>
        </div>
      ) : (
        <p className="chain-why">Every link is flowing.</p>
      )}
    </div>
  )
}
