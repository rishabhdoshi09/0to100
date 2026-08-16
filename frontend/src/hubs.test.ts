import { describe, expect, it } from 'vitest'
import {
  contextTabOf,
  homeTabOf,
  hubOf,
  ideasTabOf,
  routeForHub,
  systemTabOf,
} from './hubs'

describe('hub routing', () => {
  it('maps complementary pages onto six sidebar hubs', () => {
    expect(hubOf('Home')).toBe('Home')
    expect(hubOf('Market Overview')).toBe('Home')
    expect(hubOf('Market Internals')).toBe('Home')
    expect(hubOf('Recommendations')).toBe('Ideas')
    expect(hubOf('Market Scanner')).toBe('Ideas')
    expect(hubOf('Long-Term Picks')).toBe('Ideas')
    expect(hubOf('F&O Desk')).toBe('Ideas')
    expect(hubOf('Market Reports')).toBe('Context')
    expect(hubOf('News & Events')).toBe('Context')
    expect(hubOf('Education')).toBe('Context')
    expect(hubOf('Watchlist')).toBe('Watchlist')
    expect(hubOf('Paper Portfolio')).toBe('Holdings')
    expect(hubOf('System Health')).toBe('System')
    expect(hubOf('Research Data')).toBe('System')
    expect(hubOf('Stock Intelligence')).toBe('')
    expect(hubOf('Compare')).toBe('')
  })

  it('opens each hub on its default building-block page', () => {
    expect(routeForHub('Home')).toBe('Home')
    expect(routeForHub('Ideas')).toBe('Recommendations')
    expect(routeForHub('Context')).toBe('News & Events')
    expect(routeForHub('Watchlist')).toBe('Watchlist')
    expect(routeForHub('Holdings')).toBe('Paper Portfolio')
    expect(routeForHub('System')).toBe('System Health')
  })

  it('keeps inner tabs aligned with the merged pages', () => {
    expect(homeTabOf('Home')).toBe('Desk')
    expect(homeTabOf('Market Overview')).toBe('Internals')
    expect(ideasTabOf('Recommendations')).toBe('Categories')
    expect(ideasTabOf('Scanner')).toBe('Table')
    expect(ideasTabOf('Long-Term')).toBe('Long-term')
    expect(ideasTabOf('F&O Desk')).toBe('F&O')
    expect(contextTabOf('Market Reports')).toBe('Pulse')
    expect(contextTabOf('News & Events')).toBe('News')
    expect(contextTabOf('Education')).toBe('Learn')
    expect(systemTabOf('System Health')).toBe('Health')
    expect(systemTabOf('Research Data')).toBe('Data')
  })
})
