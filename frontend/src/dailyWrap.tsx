export type DailyWrapLine = {
  id?: string
  text: string
  source?: string
  official?: boolean
  url?: string
  symbols?: string[]
  published_at?: string
}

type WrapArticle = {
  article_id?: string
  headline?: string
  summary?: string
  why_it_matters?: string
  source?: string
  official?: boolean
  url?: string
  mentioned_symbols?: string[]
  impact_score?: number
  published_at?: string
}

type WrapDashboard = {
  market?: {
    available?: boolean
    summary?: string
    nifty_change_1d?: number | null
    nifty_price?: number | null
    leaders?: string[]
    laggards?: string[]
  }
  news?: {
    articles?: WrapArticle[]
  }
}

// "Today" on a market desk must not silently mean "best headline from the last week".
// An 18-hour rolling window keeps prior-evening/post-close context visible for the
// next morning while dropping yesterday's old intraday/news inventory by the time
// it is no longer actionable context. Undated news is excluded rather than guessed.
export const WRAP_NEWS_MAX_AGE_MS = 18 * 60 * 60 * 1000

const GLOBAL_NEEDLES = [
  'us inflation', 'fed chair', 'federal reserve', 'treasury yield',
  'us futures', 'bond yield', 's&p', 'nasdaq', 'us markets', 'nvidia',
  'wall street', 'dow jones',
]
const FILING_NEEDLES = [
  'pursuant to the provisions of regulation',
  'listing obligations and disclosure',
]

function prettySector(name: string): string {
  const raw = String(name || '').trim()
  if (!raw) return ''
  if (['IT', 'FMCG', 'NBFC'].includes(raw.toUpperCase())) return raw.toUpper-Case?.() || raw.toUpperCase()
  return raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase()
}

function joinNames(names: string[]): string {
  const items = names.map((n) => n.trim()).filter(Boolean)
  if (!items.length) return ''
  if (items.length === 1) return items[0]
  if (items.length === 2) return `${rawSafe(items[0])} and ${rawSafe(items[1])}`
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

function rawSafe(value: string): string {
  return value
}

function articleBlob(article: WrapArticle | undefined): string {
  return [article?.headline, article?.summary, article?.why_it_matters, article?.source]
    .map((part) => String(part || '').toString().toLowerCase())
    .join(' ')
}

function articleKind(article: WrapArticle | undefined): 'skip' | 'global' | 'stock' | 'other' {
  const blob = articleBlob(article)
  if (FILING_NEEDLES.some((token) => blob.includes(token))) return 'skip'
  if (GLOBAL_NEEDLES.some((token) => blob.includes(token))) return 'global'
  if ((article?.mentioned_symbols || []).some((sym) => String(sym || '').trim())) return 'stock'
  return 'other'
}

function publishedMs(value: string | undefined): number | null {
  const text = String(value || '').trim()
  if (!text) return null
  const parsed = Date.parse(text)
  return Number.isFinite(parsed) ? parsed : null
}

export function isFreshWrapArticle(
  article: WrapArticle | undefined,
  nowMs: number = Date.now(),
): boolean {
  const published = publishedMs(article?.published_at)
  if (published == null) return false
  const age = nowMs - published
  // Small positive clock skew is tolerated; far-future timestamps are rejected.
  return age >= -(5 * 60 * 1000) && age <= WRAP_NEWS_MAX_AGE_MS
}

function isSessionLine(line: DailyWrapLine): boolean {
  return line.id === 'session_indices' || line.id === 'session_regime'
}

function freshApiLines(
  apiLines: DailyWrapLine[],
  dashboard: WrapDashboard | undefined | null,
  nowMs: number,
): DailyWrapLine[] {
  const articles = dashboard?.news?.articles || []
  const byId = new Map<string, WrapArticle>()
  const byUrl = new Map<string, WrapArticle>()
  for (const article of articles) {
    const id = String(article.article_id || '').trim()
    const url = String(article.url || '').trim()
    if (id) byId.set(id, article)
    if (url) byUrl.set(url, article)
  }
  return apiLines.filter((line) => {
    if (isSessionLine(line)) return true
    const directPublished = publishedMs(line.published_at)
    if (directPublished != null) {
      const age = nowMs - directPublished
      return age >= -(5 * 60 * 1000) && age <= WRAP_NEWS_MAX_AGE_MS
    }
    const id = String(line.id || '').trim()
    const url = String(line.url || '').trim()
    const article = (id ? byId.get(id) : undefined) || (url ? byUrl.get(url) : undefined)
    return isFreshWrapArticle(article, nowMs)
  })
}

export function DailyWrapList({
  lines,
  onSymbol,
}: {
  lines: DailyWrapLine[]
  onSymbol?: (symbol: string) => void
}) {
  if (!lines.length) return null
  return (
    <section className="daily-wrap" aria-label="Latest market wrap">
      <p className="desk-kicker">Fresh market context</p>
      <h2>Latest market wrap</h2>
      <p className="daily-wrap-freshness">Last official NSE session plus sourced news published in the last 18 hours. Old or undated headlines are not promoted here.</p>
      <ol>
        {lines.map((line, index) => (
          <li key={line.id || `${index}-${line.text.slice(0, 24)}`}>
            <div>
              <p>{line.text}</p>
              <div className="desk-meta">
                {line.source ? <span>{line.source}</span> : null}
                {line.official ? <em>Official</em> : null}
                {(line.symbols || []).map((sym) => (
                  onSymbol ? (
                    <button key={sym} type="button" onClick={() => onSymbol(sym)}>{sym}</button>
                  ) : <span key={sym}>{sym}</span>
                ))}
                {line.url ? (
                  <a href={line.url} target="_blank" rel="noreferrer">Open source</a>
                ) : null}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

export function isLegacyScanWrap(lines: DailyWrapLine[] | undefined | null): boolean {
  return (lines || []).some((line) => (
    line.id === 'session_scan'
    || /last market scan has/i.test(line.text || '')
  ))
}

export function magazineWrapLines(
  apiLines: DailyWrapLine[] | undefined | null,
  dashboard: WrapDashboard | undefined | null,
  nowMs: number = Date.now(),
): DailyWrapLine[] {
  const api = apiLines || []
  if (api.length && !isLegacyScanWrap(api)) {
    const fresh = freshApiLines(api, dashboard, nowMs)
    if (fresh.length) return fresh.slice(0, 5)
  }
  return dashboardWrapLines(dashboard, nowMs)
}

export function dashboardWrapLines(
  dashboard: WrapDashboard | undefined | null,
  nowMs: number = Date.now(),
): DailyWrapLine[] {
  const lines: DailyWrapLine[] = []
  const market = dashboard?.market
  if (market?.available && (market.summary || market.nifty_change_1d != null)) {
    const chg = market.nifty_change_1d
    let streak = 'ended little changed'
    if (chg != null) {
      if (chg < -0.05) streak = 'ended lower'
      else if (chg > 0.05) streak = 'ended higher'
    }
    let head = `Indian markets ${streak}`
    if (chg != null) {
      const verb = chg < -0.05
        ? `falling ${Math.abs(chg).toFixed(1)}%`
        : chg > 0.05
          ? `rising ${chg.toFixed(1)}%`
          : `little changed (${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%)`
      let nifty = `with the Nifty ${verb}`
      if (market.nifty_price && market.nifty_price > 0) {
        nifty += ` to ${Math.round(market.nifty_price).toLocaleString('en-IN')}`
      }
      head = `Indian markets ${streak}, ${nifty}`
    }
    const leaders = (market.leaders || []).slice(0, 3).map(prettySector).filter(Boolean)
    const laggards = (market.laggards || []).slice(0, 3).map(prettySector).filter(Boolean)
    let sector = ''
    if (leaders.length) {
      sector = `${joinNames(leaders)} stocks led`
      if (laggards.length) sector += `, while ${joinNames(laggards)} lagged`
    } else if (laggards.length) {
      sector = `${joinNames(laggards)} lagged`
    }
    lines.push({
      id: 'session_indices',
      text: [head + '.', sector ? `${sector}.` : ''].filter(Boolean).join(' '),
      source: 'Last official NSE session',
      official: true,
    })
  }

  const articles = [...(dashboard?.news?.articles || [])]
    .filter((article) => String(article.headline || '').trim())
    .filter((article) => isFreshWrapArticle(article, nowMs))
    .sort((a, b) => (b.impact_score || 0) - (a.impact_score || 0))
  const stock: DailyWrapLine[] = []
  const other: DailyWrapLine[] = []
  const global: DailyWrapLine[] = []
  for (const article of articles) {
    const kind = articleKind(article)
    if (kind === 'skip') continue
    const headline = String(article.headline || '').trim()
    const summary = String(article.summary || article.why_it_matters || '').trim()
    const item: DailyWrapLine = {
      id: article.article_id || headline,
      text: summary ? `${headline} ${summary}` : headline,
      source: article.source || 'Sourced news',
      official: Boolean(article.official),
      url: article.url,
      symbols: (article.mentioned_symbols || []).slice(0, 6),
      published_at: article.published_at,
    }
    if (kind === 'stock') stock.push(item)
    else if (kind === 'global') global.push(item)
    else other.push(item)
  }
  const picked = [...stock.slice(0, 3)]
  const remain = Math.max(0, 4 - picked.length)
  picked.push(...other.slice(0, remain))
  picked.push(...global.slice(0, 1))
  lines.push(...picked)
  return lines.slice(0, 5)
}
