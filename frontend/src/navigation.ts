export type NavRow = readonly [icon: string, route: string, label: string]

export type NavGroup = {
  label: string
  items: readonly NavRow[]
}

/**
 * Product navigation has five operator workspaces.
 *
 * Internal route names intentionally stay stable while the UI uses clearer
 * workspace labels. This lets us simplify the product without breaking saved
 * navigation state, deep links, or existing view components.
 */
export const WORKSPACE_NAV: readonly NavRow[] = [
  ['⌂', 'Home', 'Today'],
  ['▣', 'Recommendations', 'Opportunities'],
  ['✎', 'Learning', 'Research'],
  ['▣', 'Paper Portfolio', 'Portfolio'],
  ['⚙', 'System Health', 'System'],
] as const

/**
 * Secondary tools remain available, but they are not peers of the five
 * workspaces. A tool answers a focused question inside a workspace.
 */
export const TOOL_GROUPS: readonly NavGroup[] = [
  {
    label: 'Market tools',
    items: [
      ['◎', 'Market Scanner', 'Scanner'],
      ['◉', 'Stock Intelligence', 'Company'],
      ['▤', 'Market Reports', 'Reports'],
      ['★', 'Watchlist', 'Watchlist'],
      ['⇔', 'Compare', 'Compare'],
    ],
  },
  {
    label: 'Research tools',
    items: [
      ['?', 'Why This Decision', 'Decision'],
      ['∑', 'Forward Evidence', 'Evidence'],
      ['⌬', 'Strategies', 'Strategies'],
      ['🧪', 'Backtest', 'Backtest'],
      ['▤', 'Research Data', 'Data'],
      ['◎', 'Coverage', 'Coverage'],
    ],
  },
] as const

/**
 * Historical route names are normalized at the navigation boundary only.
 * Business code must never depend on display labels.
 */
export type PageMeta = {
  title: string
  subtitle: string
}

export const PAGE_META: Readonly<Record<string, PageMeta>> = {
  Home: {
    title: 'Today',
    subtitle: 'What matters today — market state, current opportunities, portfolio activity and system status.',
  },
  Recommendations: {
    title: 'Opportunities',
    subtitle: 'Best eligible setups from the production thesis. No qualifying trade is a valid result.',
  },
  Learning: {
    title: 'Research',
    subtitle: 'Evidence, outcomes, experiments and thesis learning — with historical and forward lanes kept separate.',
  },
  'Paper Portfolio': {
    title: 'Portfolio',
    subtitle: 'Paper positions, realized outcomes and portfolio risk in one place.',
  },
  'System Health': {
    title: 'System',
    subtitle: 'Runtime, data, jobs and recovery controls. Each health lane remains independently truthful.',
  },
  'Market Scanner': {
    title: 'Market Scanner',
    subtitle: 'One saved scan, four lanes — Breakouts, Momentum, SEPA Best Setups and long-term.',
  },
  'Market Reports': {
    title: 'Market Reports',
    subtitle: 'Daily Market Pulse archive — trends, sector movers and breakout context from live system state.',
  },
  'Stock Intelligence': {
    title: 'Company Intelligence',
    subtitle: 'Company workspace — business framework, quality, cash flow, thesis breakers and missing evidence.',
  },
  Strategies: {
    title: 'Strategies',
    subtitle: 'Production recommendation methods with explicit strategy id, version and parity.',
  },
  Coverage: {
    title: 'Coverage',
    subtitle: 'Requested vs checked vs qualified vs missing from the last whole-market scan.',
  },
  'Long-Term Picks': {
    title: 'Long-Term Picks',
    subtitle: 'Quality overlay from the same market scan — Refresh funds only reloads Screener.',
  },
  Compare: {
    title: 'Compare',
    subtitle: 'Side-by-side comparison across market, growth, quality and technical dimensions.',
  },
  Watchlist: {
    title: 'Watchlist',
    subtitle: 'Names you are tracking with latest scan context.',
  },
  'Market Overview': {
    title: 'Market Overview',
    subtitle: 'Regime, breadth, volatility and sector leadership.',
  },
  'News & Events': {
    title: 'News & Events',
    subtitle: 'Dated market context with source health.',
  },
  Education: {
    title: 'Education',
    subtitle: 'Crunched news + macro/micro teach-ins for the share market — never invented blogs, never a signal.',
  },
  'Research Data': {
    title: 'Research Data',
    subtitle: 'Verified snapshots, data platform jobs, and evidence uploads.',
  },
  Backtest: {
    title: 'Backtest',
    subtitle: 'Production-connected backtests only. Unproven hash stays BACKTEST PARITY: UNVERIFIED.',
  },
  'F&O Desk': {
    title: 'F&O Desk',
    subtitle: 'Mapped futures, plus an acquired nearest-expiry OI / IV / PCR snapshot when present.',
  },
  'Why This Decision': {
    title: 'Why This Decision',
    subtitle: 'The persisted reasons, evidence and blockers behind the selected production decision.',
  },
  'Forward Evidence': {
    title: 'Forward Evidence',
    subtitle: 'Real forward-paper evidence stays separate from historical replay and counterfactual outcomes.',
  },
}

export const ROUTE_ALIAS: Readonly<Record<string, string>> = {
  'Command Center': 'Home',
  Market: 'Home',
  Today: 'Home',
  Desk: 'Home',

  Opportunities: 'Recommendations',

  Research: 'Learning',

  Portfolio: 'Paper Portfolio',

  System: 'System Health',
  Automation: 'System Health',
  Health: 'System Health',

  Scanner: 'Market Scanner',
  Setups: 'Market Scanner',
  Reports: 'Market Reports',
  'Long-Term': 'Long-Term Picks',
  'Market Internals': 'Market Overview',

  'Stock Investigator': 'Stock Intelligence',
  'Company Intelligence': 'Stock Intelligence',

  Backtests: 'Backtest',
  Data: 'Research Data',
  Why: 'Why This Decision',
  Evidence: 'Forward Evidence',
}

export function canonicalNavRoute(route: string): string {
  return ROUTE_ALIAS[route] || route
}

export function pageMeta(route: string): PageMeta {
  const canonical = canonicalNavRoute(route)
  return PAGE_META[canonical] || {
    title: canonical,
    subtitle: '',
  }
}

const TOOL_ROUTES = new Set(
  TOOL_GROUPS.flatMap((group) => group.items.map(([, route]) => route)),
)

export function isToolRoute(route: string): boolean {
  return TOOL_ROUTES.has(canonicalNavRoute(route))
}
