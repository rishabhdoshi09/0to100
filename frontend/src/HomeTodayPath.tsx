import { useState } from 'react'
import {
  FLOOR_JUMPS,
  PATH_BUTTON_LABEL,
  dataFloorCopy,
  deskFloorCopy,
  optionsFloorCopy,
  type FloorContext,
  type FloorId,
} from './homeFloorPath'
import {
  fetchDataCoverage,
  fetchDecisionJournal,
  fetchPortfolioIntel,
  type DecisionJournalPayload,
  type PortfolioIntelPayload,
} from './productApi'

export type TodayFloors = {
  context: FloorContext
  audited: number | null
  coverageReady: boolean
  journal: DecisionJournalPayload | null
  intel: PortfolioIntelPayload | null
}

export async function loadTodayFloors(context: FloorContext): Promise<TodayFloors> {
  const [coverage, journal, intel] = await Promise.all([
    fetchDataCoverage().catch(() => null),
    fetchDecisionJournal().catch(() => null),
    fetchPortfolioIntel().catch(() => null),
  ])
  const audited = coverage?.audited != null ? Number(coverage.audited) : null
  return {
    context,
    audited: Number.isFinite(audited) ? audited : null,
    coverageReady: Boolean(coverage),
    journal,
    intel,
  }
}

function floorCopy(id: FloorId, floors: TodayFloors): { title: string; detail: string } {
  if (id === 'desk') return deskFloorCopy(floors.context)
  if (id === 'options') return optionsFloorCopy(floors.context)
  if (id === 'data') return dataFloorCopy(floors.audited, floors.coverageReady)
  if (id === 'holdings') {
    return {
      title: floors.intel?.swap ? 'Opportunity-cost note' : 'No swap claim',
      detail: floors.intel?.message || 'Advice only — never rotates, never picks a stock',
    }
  }
  return {
    title: floors.journal?.thin === false ? 'Journal has a claim' : 'No claim yet',
    detail: floors.journal?.message || 'Taken and rejected need ≥10 resolved outcomes',
  }
}

export function HomeTodayPath({
  busy,
  floors,
  error,
  onOpen,
  onJump,
}: {
  busy: boolean
  floors: TodayFloors | null
  error: string
  onOpen: () => void
  onJump: (page: string) => void
}) {
  return (
    <section className="home-path" aria-label="Today's path">
      <header className="home-path-head">
        <div>
          <p>Today's path</p>
          <h3>Wire today's floors</h3>
          <em>
            One click reads Desk, Options, Data, Holdings and Health as floors — not as one stock.
            Jumps stay optional. Nothing here places an order or picks a name.
          </em>
        </div>
        <button type="button" className="reco-primary" disabled={busy} onClick={onOpen}>
          {busy ? 'Opening floors…' : PATH_BUTTON_LABEL}
        </button>
      </header>
      {error ? <p className="home-path-error">{error}</p> : null}
      {floors ? (
        <div className="home-path-grid">
          {FLOOR_JUMPS.map((floor) => {
            const copy = floorCopy(floor.id, floors)
            return (
              <button
                key={floor.id}
                type="button"
                className="home-path-tile"
                onClick={() => onJump(floor.page)}
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
          Click to load the floors. Search stays yours if you later want one stock.
        </p>
      )}
    </section>
  )
}

export function useTodayFloors() {
  const [floors, setFloors] = useState<TodayFloors | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const open = async (context: FloorContext) => {
    setBusy(true)
    setError('')
    try {
      setFloors(await loadTodayFloors(context))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not open today\'s floors')
    } finally {
      setBusy(false)
    }
  }

  return { floors, busy, error, open }
}
