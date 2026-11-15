import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { BuyOnlyPanel } from './marketRadarViews'

// Home · BUY only (2026-11-15): every canonical BUY and nothing else.
describe('Home BUY-only section', () => {
  it('lists BUYs with levels and the measured edge behind them', () => {
    const html = renderToStaticMarkup(createElement(BuyOnlyPanel, {
      section: {
        rows: [{
          symbol: 'INFY', cmp: 1500, entry: 1505, stop: 1450, target: 1615, reward_r: 2,
          edge_r: 0.12, edge_lower_r: 0.03, edge_source: 'combination', why: 'Committee passed.',
        }],
        count: 1, verified: true, withheld_count: 4, session: '2026-11-13',
        rule: 'BUY only: proven edge',
      },
    }))
    expect(html).toContain('BUY ONLY')
    expect(html).toContain('1 BUY now')
    expect(html).toContain('INFY')
    expect(html).toContain('Measured edge +0.12R/trade')
    expect(html).toContain('95% floor +0.03R')
    expect(html).toContain('this exact setup')
    expect(html).toContain('reward 2.0R')
  })

  it('says why when there is no BUY instead of showing near-misses', () => {
    const html = renderToStaticMarkup(createElement(BuyOnlyPanel, {
      section: {
        rows: [], count: 0, verified: true, withheld_count: 19,
        empty_reason: 'No BUY today. 19 setup(s) looked technically ready but none has a proven edge.',
      },
    }))
    expect(html).toContain('No BUY right now')
    expect(html).toContain('19 setup(s)')
    expect(html).not.toContain('home-os-inspect-link')
  })

  it('is rendered first on Home, before the operating-system card', () => {
    const home = readFileSync(new URL('./marketRadarViews.tsx', import.meta.url), 'utf8')
    const start = home.indexOf('<section className="radar-home">')
    const panel = home.indexOf('<BuyOnlyPanel', start)
    const osCard = home.indexOf('<HomeOsCard', start)
    expect(panel).toBeGreaterThan(start)
    expect(panel).toBeLessThan(osCard)
  })

  it('renders nothing until the backend provides the section', () => {
    expect(renderToStaticMarkup(createElement(BuyOnlyPanel, { section: null }))).toBe('')
  })
})
