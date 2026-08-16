import { useState } from 'react'
import {
  FLOOR_JUMPS,
  dataFloorCopy,
  deskFloorCopy,
  optionsFloorCopy,
  type FloorContext,
  type FloorId,
  type JobClock,
  type NextStep,
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
  step,
  progress,
  clock,
  onOpen,
  onJump,
}: {
  busy: boolean
  floors: TodayFloors | null
  error: string
  step: NextStep
  progress: string
  clock?: JobClock | null
  onOpen: () => void
  onJump: (page: string) => void
}) {
  const working = busy || step.id === 'working'
  const buttonLabel = working ? (clock?.button || 'Working…') : step.label
  const why = working ? (clock?.line || progress || step.why) : step.why
  return (
    <section className="home-path" aria-label="Start here">
      <header className="home-path-head">
        <div>
          <p>Start here</p>
          <h3>One click. The system knows the next job.</h3>
          <em>
            System khud jaanta hai agla kaam kya hai — desk bharo, names dhoondo, ya picture refresh.
            You click. Results stay on Home. No ticker required.
          </em>
        </div>
        <button type="button" className="reco-primary" disabled={working} onClick={onOpen}>
          {buttonLabel}
        </button>
      </header>
      {working && clock?.percent != null ? (
        <div className="home-path-meter" role="progressbar" aria-valuenow={clock.percent} aria-valuemin={0} aria-valuemax={100}>
          <span style={{ width: `${Math.max(4, Math.min(100, clock.percent))}%` }} />
        </div>
      ) : null}
      <p className="home-path-why">{why}</p>
      {step.resultHint ? <p className="home-path-hint">{working ? 'Stay on this page. The bar and the time left update every second.' : step.resultHint}</p> : null}
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
          Click once. The floors fill themselves. Search stays yours if you later want one stock.
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
      setError(reason instanceof Error ? reason.message : 'Could not read today\'s floors')
    } finally {
      setBusy(false)
    }
  }

  return { floors, busy, error, open }
}
