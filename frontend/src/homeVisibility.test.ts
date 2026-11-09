import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('./home-calm.css', import.meta.url), 'utf8')
const home = readFileSync(new URL('./marketRadarViews.tsx', import.meta.url), 'utf8')

// Regression: a calm-layout update must not hide the very NSE stock research
// the operator came to the Home page to inspect.
function selectorsHiddenByDisplayNone(styles: string): string[] {
  return [...styles.matchAll(/([^{}]+)\{\s*display:\s*none\s*;?\s*\}/g)]
    .flatMap((match) => match[1].split(',').map((selector) => selector.trim()))
}

describe('NSE equity-first Home visibility', () => {
  it('keeps scan actions, research rankings, scanner lanes and risk preview visible', () => {
    const hidden = selectorsHiddenByDisplayNone(css)
    for (const selector of [
      '.radar-home > .radar-hero',
      '.radar-home > .radar-bob-hero',
      '.radar-home > .radar-top-stocks',
      '.radar-home > .radar-best-breakout',
      '.radar-home > .radar-three-lanes',
      '.radar-home > .radar-workspace',
    ]) {
      expect(hidden, selector).not.toContain(selector)
    }
    // Secondary internal surfaces may stay collapsed to retain a calm layout.
    expect(hidden).toContain('.radar-home > .radar-rank-legend')
  })

  it('places NSE equity research ahead of the F&O panel without promoting research to BUY', () => {
    expect(home.indexOf('<HomeOsCard')).toBeGreaterThan(0)
    expect(home.indexOf('<div className="radar-three-lanes">')).toBeGreaterThan(home.indexOf('<HomeOsCard'))
    expect(home.indexOf('<FnoHomePanel')).toBeGreaterThan(home.indexOf('<div className="radar-three-lanes">'))
    expect(home).toContain('RESEARCH SHORTLIST · NOT TRADES')
    expect(home).toContain('Research ranking · not a trade')
    expect(home).toContain('verifiedHomeStockScan(')
  })
})
