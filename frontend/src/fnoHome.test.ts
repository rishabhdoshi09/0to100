import { describe, expect, it } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { FnoHomePanel } from './fnoHome'
import type { DashboardPayload } from './types'

function render(desk: DashboardPayload['fno']['desk'], active: unknown[] = []) {
  const dashboard = { fno: { desk }, operations: { active } } as DashboardPayload
  return renderToStaticMarkup(createElement(FnoHomePanel, { dashboard, setSelected: () => {}, setActive: () => {}, runControl: async () => {} }))
}
const desk: NonNullable<DashboardPayload['fno']['desk']> = {
  status: 'READY', reason: 'PAPER_CANDIDATES', as_of: '2026-10-05', candidate_count: 1,
  candidates: [{ symbol: 'TEST', direction: 'LONG', decision: 'PAPER_OPTION_CANDIDATE', setup: { score: 82 },
    selected_contract: { symbol: 'TEST26OCT100CE', option_type: 'CE', strike: 100, expiry: '2026-10-29', premium: 50, score: 86,
      trade_plan: { entry: 50, stop: 40, target: 70 } } }],
  paper_available: true, blockers: [], open_positions: [], recent_closed_trades: [],
}

describe('F&O on Home', () => {
  it('renders contract and complete plan as a candidate awaiting execution', () => {
    const html = render(desk)
    expect(html).toContain('data-testid="home-fno-desk"')
    for (const text of ['TEST26OCT100CE', 'expiry 2026-10-29', 'entry ₹50', 'stop ₹40', 'target ₹70',
      'quality scores, not win probabilities', 'Candidate only', 'No open F&amp;O paper position']) expect(html).toContain(text)
  })
  it('keeps real open positions visible when the scan is blocked and shows the blocker', () => {
    const html = render({ ...desk, status: 'BLOCKED', reason: 'NIFTY_HISTORY_UNAVAILABLE', candidates: [],
      open_positions: [{ trade_id: 'p1', underlying: 'TEST', option_symbol: 'TEST26OCT100CE', entry_price: 50, stop_price: 40, target_price: 70 }] })
    expect(html).toContain('No current eligible F&amp;O candidate')
    expect(html).toContain('NIFTY HISTORY UNAVAILABLE')
    expect(html).toContain('PAPER OPEN')
    expect(html).not.toContain('Candidate only')
  })
  it('shows current scan progress and disables duplicate refresh', () => {
    const html = render(undefined, [{ kind: 'FNO_REFRESH', status: 'RUNNING', stage: 'FNO_PREFILTER', message: 'Checking 25/200' }])
    expect(html).toContain('Checking 25/200')
    expect(html).toContain('disabled=""')
    expect(html).toContain('Paper ledger unavailable')
  })
})
