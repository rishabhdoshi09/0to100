import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('light theme contract', () => {
  it('keeps the product shell on a light color-scheme', () => {
    const styles = readFileSync(new URL('./styles.css', import.meta.url), 'utf8')
    expect(styles).toContain('color-scheme: light')
    expect(styles).toContain('--bg: #f4f7f5')
    expect(styles).toContain('--panel: #ffffff')
    expect(styles).toContain('--text: #14201a')
  })

  it('does not leave the workspace or topbar on a dark navy canvas', () => {
    const polish = readFileSync(new URL('./operator-polish.css', import.meta.url), 'utf8')
    expect(polish).not.toMatch(/\.workspace\s*\{[^}]*rgba\(7, 12, 23/)
    expect(polish).not.toMatch(/\.topbar\s*\{[^}]*rgba\(7, 13, 24/)
    expect(polish).toMatch(/\.workspace[\s\S]*#f8faf9/)
  })

    it('does not keep dark-navy hover leftovers on Home system lanes', () => {
    const radar = readFileSync(new URL('./radar.css', import.meta.url), 'utf8')
    expect(radar).not.toContain('rgba(20, 34, 54, .92)')
    expect(radar).not.toContain('rgba(24, 42, 66, .96)')
    expect(radar).not.toContain('rgba(8, 14, 27, .72)')
  })

  it('does not leave dark navy canvases on shared surfaces', () => {
    const views = readFileSync(new URL('./views.css', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./styles.css', import.meta.url), 'utf8')
    expect(views).not.toContain('#07101d')
    expect(views).not.toContain('#08111e')
    expect(styles).not.toContain('rgba(13, 20, 33')
    expect(views).toContain('color: var(--text, #14201a)')
  })

  it('loads the partner overhaul last and keeps it light-only', () => {
    const main = readFileSync(new URL('./main.tsx', import.meta.url), 'utf8')
    const partner = readFileSync(new URL('./partner-overhaul.css', import.meta.url), 'utf8')
    expect(main.lastIndexOf("import './partner-overhaul.css'"))
      .toBeGreaterThan(main.lastIndexOf("import './final-polish.css'"))
    expect(partner).toContain('color-scheme: light')
    expect(partner).toContain('--qt-app-bg: #f3f6f4')
    expect(partner).toContain('--qt-app-surface: #ffffff')
    expect(partner).not.toMatch(/#07101d|#08111e|rgba\(7,\s*12,\s*23|rgba\(8,\s*15,\s*28/)
    expect(partner).toContain('body:has(.terminal-root.reco-desk)::before')
    expect(partner).toContain('display: none !important')
  })
  it('keeps price charts on the professional light palette instead of legacy neon colors', () => {
    const chart = readFileSync(new URL('./PriceChart.tsx', import.meta.url), 'utf8')
    expect(chart).not.toContain('#31e981')
    expect(chart).not.toContain('#ff667f')
    expect(chart).not.toContain('#24d6ff')
    expect(chart).toContain('#1b6b45')
    expect(chart).toContain('#b42318')
  })

})
