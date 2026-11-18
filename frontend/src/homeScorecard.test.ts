import { describe, expect, it } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { LearningScorecardBlock } from './homeScorecard'
import type { LearningScorecardSummary } from './productApi'

const render = (card: LearningScorecardSummary | null) =>
  renderToStaticMarkup(createElement(LearningScorecardBlock, { card }))

describe('LearningScorecardBlock', () => {
  it('renders nothing without a payload', () => {
    expect(render(null)).toBe('')
  })

  it('says the scorecard is not computed yet and that live is locked', () => {
    const html = render({ available: false, headline: 'Scorecard not computed yet: it runs after the next nightly learning cycle.', live_trading: 'LOCKED' })
    expect(html).toContain('not computed yet')
    expect(html).toContain('Live trading LOCKED')
  })

  it('shows every unmet gate with its first blocker and the calibration gap', () => {
    const html = render({
      available: true,
      headline: 'No proven edge yet: tracked BUYs -0.0984R per session over 16 sessions.',
      calibration: { verdict: 'NO_USEFUL_DISCRIMINATION', auc: 0.511, top_bucket_score_as_prob: 0.9123, top_bucket_realized_win_rate: 0.199 },
      paper_trades_settled: 0,
      decisions_changed_by_learning: 0,
      stage: 0,
      verdicts: { 'RESEARCH READY': 'NOT READY', 'AUTONOMOUS LIVE ELIGIBLE': 'NOT ELIGIBLE' },
      blockers: { 'RESEARCH READY': ['tracked BUYs: net edge INSUFFICIENT EVIDENCE over 16 sessions'] },
      live_trading: 'LOCKED',
    })
    expect(html).toContain('No proven edge yet')
    expect(html).toContain('no useful discrimination')
    expect(html).toContain('top scores imply 91% but won 20%')
    expect(html).toContain('RESEARCH READY: NOT READY — tracked BUYs: net edge INSUFFICIENT EVIDENCE')
    expect(html).toContain('AUTONOMOUS LIVE ELIGIBLE: NOT ELIGIBLE')
    expect(html).toContain('home-os-blocker')
    expect(html).toContain('stage 0 of 5 · live trading LOCKED')
  })
})
