import { describe, expect, it } from 'vitest'
import { dashboardWrapLines, isFreshWrapArticle, magazineWrapLines, WRAP_NEWS_MAX_AGE_MS } from './dailyWrap'

const NOW = Date.parse('2026-09-04T08:30:00+05:30')

function article(id: string, publishedAt?: string) {
  return {
    article_id: id,
    headline: `Headline ${id}`,
    summary: `Summary ${id}`,
    source: 'Test source',
    published_at: publishedAt,
    impact_score: 90,
  }
}

describe('daily wrap freshness', () => {
  it('keeps recent post-close context but rejects old and undated news', () => {
    expect(isFreshWrapArticle(article('fresh', '2026-09-03T18:00:00+05:30'), NOW)).toBe(true)
    expect(isFreshWrapArticle(article('old', '2026-09-03T10:00:00+05:30'), NOW)).toBe(false)
    expect(isFreshWrapArticle(article('undated'), NOW)).toBe(false)
    expect(WRAP_NEWS_MAX_AGE_MS).toBe(18 * 60 * 60 * 1000)
  })

  it('never promotes stale dashboard headlines into the latest wrap', () => {
    const lines = dashboardWrapLines({
      market: { available: false },
      news: {
        articles: [
          article('fresh', '2026-09-04T07:45:00+05:30'),
          article('old', '2026-09-02T12:00:00+05:30'),
          article('undated'),
        ],
      },
    }, NOW)
    expect(lines.map((line) => line.id)).toEqual(['fresh'])
  })

  it('filters saved API wrap lines against the current news timestamps', () => {
    const lines = magazineWrapLines([
      { id: 'session_indices', text: 'Official session', official: true },
      { id: 'fresh', text: 'Fresh API headline' },
      { id: 'old', text: 'Old API headline' },
    ], {
      news: {
        articles: [
          article('fresh', '2026-09-04T07:45:00+05:30'),
          article('old', '2026-09-02T12:00:00+05:30'),
        ],
      },
    }, NOW)
    expect(lines.map((line) => line.id)).toEqual(['session_indices', 'fresh'])
  })
})
