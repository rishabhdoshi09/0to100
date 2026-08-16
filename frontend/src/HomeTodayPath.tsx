import { useState } from 'react'
import { money } from './format'
import { FLOOR_JUMPS, pathButtonLabel, type FloorId } from './homeFloorPath'
import { fetchMarketOptions, fetchOptionsEodHistory, watchOptionsEod } from './api'
import {
  fetchDataCoverage,
  fetchDecisionJournal,
  fetchPortfolioIntel,
  fetchPreTrade,
  type DecisionJournalPayload,
  type PortfolioIntelPayload,
  type PreTrade,
} from './productApi'
import type { OptionsChainPayload, OptionsEodHistoryPayload } from './types'

type EvidenceSummary = {
  symbol?: string
  coverage_pct?: number
}

export type TodayFloors = {
  symbol: string
  desk: PreTrade | null
  chain: OptionsChainPayload | null
  history: OptionsEodHistoryPayload | null
  watch: { accepted?: boolean; message?: string; capture_list?: string[] } | null
  evidence: EvidenceSummary | null
  coverageReady: boolean
  journal: DecisionJournalPayload | null
  intel: PortfolioIntelPayload | null
}

async function fetchEvidence(symbol: string): Promise<EvidenceSummary | null> {
  const response = await fetch(`/evidence/${encodeURIComponent(symbol)}`, {
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) throw new Error('Evidence unavailable')
  return response.json() as Promise<EvidenceSummary>
}

export async function loadTodayFloors(symbol: string): Promise<TodayFloors> {
  const [desk, chain, history, watch, evidence, coverage, journal, intel] = await Promise.all([
    fetchPreTrade(symbol).catch(() => null),
    fetchMarketOptions(symbol, false).catch(() => null),
    fetchOptionsEodHistory(symbol, 14).catch(() => null),
    watchOptionsEod(symbol).catch(() => null),
    fetchEvidence(symbol).catch(() => null),
    fetchDataCoverage(symbol).then(() => true).catch(() => false),
    fetchDecisionJournal().catch(() => null),
    fetchPortfolioIntel().catch(() => null),
  ])
  return {
    symbol,
    desk,
    chain,
    history,
    watch,
    evidence,
    coverageReady: coverage,
    journal,
    intel,
  }
}

function floorCopy(id: FloorId, floors: TodayFloors): { title: string; detail: string } {
  if (id === 'desk') {
    const buy = floors.desk?.plan?.entry ?? floors.desk?.scan?.entry
    const verdict = floors.desk?.verdict || 'No pre-trade yet'
    return {
      title: verdict,
      detail: buy != null ? `Buy ${money(buy, 2)} · not an order` : 'Desk numbers load from the last scan',
    }
  }
  if (id === 'options') {
    if (floors.chain?.available) {
      return { title: `PCR ${floors.chain.pcr ?? '—'}`, detail: 'Nearest-expiry context · no Greeks' }
    }
    const queued = floors.watch?.accepted
      ? floors.watch.message || `${floors.symbol} queued for EOD`
      : 'EOD queue not updated'
    return {
      title: floors.chain?.backoff ? 'Retry is live' : 'Chain unavailable',
      detail: floors.chain?.message || queued,
    }
  }
  if (id === 'data') {
    const pct = floors.evidence?.coverage_pct
    return {
      title: pct == null ? (floors.coverageReady ? 'Files reachable' : 'Evidence offline') : `${pct}% coverage`,
      detail: pct == null
        ? 'Open Data to see which files are missing'
        : 'Same-origin evidence · upload if a number is missing',
    }
  }
  if (id === 'holdings') {
    return {
      title: floors.intel?.swap ? `Review ${floors.intel.swap.out} → ${floors.intel.swap.in}` : 'No swap claim',
      detail: floors.intel?.message || 'Advice only — never rotates',
    }
  }
  return {
    title: floors.journal?.thin === false ? (floors.journal.message || 'Journal has a claim') : 'No claim yet',
    detail: floors.journal?.message || 'Taken and rejected need ≥10 resolved outcomes',
  }
}

export function HomeTodayPath({
  symbol,
  busy,
  floors,
  error,
  onOpen,
  onJump,
}: {
  symbol: string
  busy: boolean
  floors: TodayFloors | null
  error: string
  onOpen: () => void
  onJump: (page: string, intelTab?: string) => void
}) {
  return (
    <section className="home-path" aria-label="Today's path">
      <header className="home-path-head">
        <div>
          <p>Today's path</p>
          <h3>{symbol ? `${symbol} across today's floors` : 'Pick a name, then one click'}</h3>
          <em>One click wires Desk, Options and Data on this page. Jumps stay optional. Nothing here places an order.</em>
        </div>
        <button
          type="button"
          className="reco-primary"
          disabled={!symbol || busy}
          onClick={onOpen}
        >
          {busy ? 'Opening floors…' : pathButtonLabel(symbol)}
        </button>
      </header>
      {error ? <p className="home-path-error">{error}</p> : null}
      {floors && floors.symbol === symbol ? (
        <div className="home-path-grid">
          {FLOOR_JUMPS.map((floor) => {
            const copy = floorCopy(floor.id, floors)
            return (
              <button
                key={floor.id}
                type="button"
                className="home-path-tile"
                onClick={() => onJump(floor.page, floor.intelTab)}
              >
                <small>{floor.label}</small>
                <strong>{copy.title}</strong>
                <span>{copy.detail}</span>
              </button>
            )
          })}
        </div>
      ) : (
        <p className="home-path-hint">
          {symbol
            ? `Click to load Desk numbers, queue Options EOD, and read Data coverage for ${symbol}.`
            : 'Scan now or search a name in the top bar first.'}
        </p>
      )}
    </section>
  )
}

export function useTodayFloors() {
  const [floors, setFloors] = useState<TodayFloors | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const open = async (symbol: string) => {
    if (!symbol) {
      setError('Scan now or search a name first.')
      return
    }
    setBusy(true)
    setError('')
    try {
      setFloors(await loadTodayFloors(symbol))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not open today\'s floors')
    } finally {
      setBusy(false)
    }
  }

  return { floors, busy, error, open }
}
