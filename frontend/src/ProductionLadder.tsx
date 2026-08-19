import { useEffect, useState } from 'react'
import { fetchProductionLadder, type ProductionLadder } from './productApi'
import './productionLadder.css'

export function ProductionLadderBoard() {
  const [data, setData] = useState<ProductionLadder | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let alive = true
    fetchProductionLadder()
      .then((payload) => { if (alive) setData(payload) })
      .catch((err: Error) => { if (alive) setError(err.message || 'Ladder unread') })
    return () => { alive = false }
  }, [])
  if (error) {
    return (
      <section className="prod-ladder" aria-label="Production ladder">
        <p className="prod-ladder-note">{error}</p>
      </section>
    )
  }
  if (!data) {
    return (
      <section className="prod-ladder" aria-label="Production ladder">
        <p className="prod-ladder-note">Reading subsystems…</p>
      </section>
    )
  }
  const live = data.live_unlocked
  return (
    <section className={`prod-ladder is-${data.rung.id.toLowerCase()}`} aria-label="Production ladder">
      <header>
        <p className="prod-kicker">Orchestra</p>
        <h3>{data.rung.label}</h3>
        <em>{data.thesis}</em>
      </header>
      <div className="prod-rung-meta">
        <article>
          <span>Paper closed</span>
          <strong>{data.paper_closed}/{data.paper_e4_n}</strong>
        </article>
        <article>
          <span>Alpha</span>
          <strong>{data.alpha.label}</strong>
        </article>
        <article>
          <span>Live</span>
          <strong className={live ? 'pos' : 'neg'}>{live ? 'UNLOCKED' : 'LOCKED'}</strong>
        </article>
      </div>
      {data.rung.next ? <p className="prod-ladder-note">{data.rung.next}</p> : null}
      {(data.live_blockers || []).length > 0 ? (
        <ul className="prod-blockers">
          {data.live_blockers.map((item) => <li key={item}>{item}</li>)}
        </ul>
      ) : null}
      <div className="prod-nodes">
        {data.subsystems.map((node) => (
          <article key={node.id} className={`is-${node.status.toLowerCase()}`}>
            <b>{node.id}</b>
            <p>{node.job}</p>
            <em>{node.status}{node.may_order ? (node.mode === 'LIVE' ? ' · orders if unlocked' : ' · paper orders') : ' · no orders'}</em>
          </article>
        ))}
      </div>
      <ol className="prod-handshake" aria-label="Subsystem handshake">
        {data.handshake.map((edge) => (
          <li key={`${edge.from}-${edge.to}`}>
            <b>{edge.from}</b>
            <span>→ {edge.to}</span>
            <em>{edge.payload}</em>
          </li>
        ))}
      </ol>
      <ul className="prod-rules">
        {(data.rules || []).map((rule) => <li key={rule}>{rule}</li>)}
      </ul>
    </section>
  )
}
