import type { ReactNode } from 'react'
import { SectionTabs } from './designSystem'

export const HOME_TABS = ['Desk', 'Internals'] as const
export const IDEAS_TABS = ['Categories', 'Table', 'Long-term', 'F&O'] as const
export const TRADE_TABS = ['Ready', 'Lab', 'Journey'] as const
export const CONTEXT_TABS = ['Pulse', 'News', 'Learn'] as const
export const SYSTEM_TABS = ['Health', 'Data'] as const

const HOME_ROUTE: Record<(typeof HOME_TABS)[number], string> = {
  Desk: 'Home',
  Internals: 'Market Overview',
}

const IDEAS_ROUTE: Record<(typeof IDEAS_TABS)[number], string> = {
  Categories: 'Recommendations',
  Table: 'Market Scanner',
  'Long-term': 'Long-Term Picks',
  'F&O': 'F&O Desk',
}

const TRADE_ROUTE: Record<(typeof TRADE_TABS)[number], string> = {
  Ready: 'Ready Trades',
  Lab: 'Backtest Lab',
  Journey: 'Live Journey',
}

const CONTEXT_ROUTE: Record<(typeof CONTEXT_TABS)[number], string> = {
  Pulse: 'Market Reports',
  News: 'News & Events',
  Learn: 'Education',
}

const SYSTEM_ROUTE: Record<(typeof SYSTEM_TABS)[number], string> = {
  Health: 'System Health',
  Data: 'Research Data',
}

export type NavHub = 'Home' | 'Ideas' | 'Trade' | 'Context' | 'Watchlist' | 'Holdings' | 'System' | ''

export function hubOf(active: string): NavHub {
  if (['Home', 'Command Center', 'Market Overview', 'Market Internals'].includes(active)) return 'Home'
  if (['Recommendations', 'Market Scanner', 'Scanner', 'Long-Term Picks', 'Long-Term', 'F&O Desk'].includes(active)) return 'Ideas'
  if (['Ready Trades', 'Backtest Lab', 'Live Journey'].includes(active)) return 'Trade'
  if (['Market Reports', 'News & Events', 'Education'].includes(active)) return 'Context'
  if (active === 'Watchlist') return 'Watchlist'
  if (['Paper Portfolio', 'Portfolio'].includes(active)) return 'Holdings'
  if (['System Health', 'Automation', 'Research Data'].includes(active)) return 'System'
  return ''
}

export function homeTabOf(active: string): (typeof HOME_TABS)[number] {
  if (active === 'Market Overview' || active === 'Market Internals') return 'Internals'
  return 'Desk'
}

export function ideasTabOf(active: string): (typeof IDEAS_TABS)[number] {
  if (active === 'Market Scanner' || active === 'Scanner') return 'Table'
  if (active === 'Long-Term Picks' || active === 'Long-Term') return 'Long-term'
  if (active === 'F&O Desk') return 'F&O'
  return 'Categories'
}

export function tradeTabOf(active: string): (typeof TRADE_TABS)[number] {
  if (active === 'Backtest Lab') return 'Lab'
  if (active === 'Live Journey') return 'Journey'
  return 'Ready'
}

export function contextTabOf(active: string): (typeof CONTEXT_TABS)[number] {
  if (active === 'News & Events') return 'News'
  if (active === 'Education') return 'Learn'
  return 'Pulse'
}

export function systemTabOf(active: string): (typeof SYSTEM_TABS)[number] {
  if (active === 'Research Data') return 'Data'
  return 'Health'
}

export function routeForHub(hub: Exclude<NavHub, ''>): string {
  if (hub === 'Home') return 'Home'
  if (hub === 'Ideas') return 'Recommendations'
  if (hub === 'Trade') return 'Ready Trades'
  if (hub === 'Context') return 'News & Events'
  if (hub === 'Watchlist') return 'Watchlist'
  if (hub === 'Holdings') return 'Paper Portfolio'
  return 'System Health'
}

function HubShell({
  blurb,
  tabs,
  active,
  onChange,
  children,
}: {
  blurb: string
  tabs: readonly string[]
  active: string
  onChange: (tab: string) => void
  children: ReactNode
}) {
  return (
    <div className="hub-page">
      <p className="hub-blurb">{blurb}</p>
      <SectionTabs tabs={[...tabs]} active={active} onChange={onChange} />
      {children}
    </div>
  )
}

export function HomeHub({
  active,
  setActive,
  children,
}: {
  active: string
  setActive: (page: string) => void
  children: ReactNode
}) {
  const tab = homeTabOf(active)
  return (
    <HubShell
      blurb="Sit down, click once. The system picks the next job and shows results here. Internals is the weather the desk sits in."
      tabs={HOME_TABS}
      active={tab}
      onChange={(next) => setActive(HOME_ROUTE[next as (typeof HOME_TABS)[number]])}
    >
      {children}
    </HubShell>
  )
}

export function IdeasHub({
  active,
  setActive,
  children,
}: {
  active: string
  setActive: (page: string) => void
  children: ReactNode
}) {
  const tab = ideasTabOf(active)
  return (
    <HubShell
      blurb="Research monitor, not the order desk. Best Setups is Minervini SEPA on official NSE history. Trade → Ready overlays that ranking onto complete tickets — it is not a Prime-only waiting room."
      tabs={IDEAS_TABS}
      active={tab}
      onChange={(next) => setActive(IDEAS_ROUTE[next as (typeof IDEAS_TABS)[number]])}
    >
      {children}
    </HubShell>
  )
}

export function TradeHub({
  active,
  setActive,
  children,
}: {
  active: string
  setActive: (page: string) => void
  children: ReactNode
}) {
  const tab = tradeTabOf(active)
  return (
    <HubShell
      blurb="The order path. Ready ranks Stage-2 SEPA plus complete scan tickets (ATQ). Lab is the walk-forward backtest as four jobs. Journey is paper autopilot until live is earned — this desk never arms live."
      tabs={TRADE_TABS}
      active={tab}
      onChange={(next) => setActive(TRADE_ROUTE[next as (typeof TRADE_TABS)[number]])}
    >
      {children}
    </HubShell>
  )
}

export function ContextHub({
  active,
  setActive,
  children,
}: {
  active: string
  setActive: (page: string) => void
  children: ReactNode
}) {
  const tab = contextTabOf(active)
  return (
    <HubShell
      blurb="What the tape and the news are saying — and what that means. Pulse is the daily digest. News is the source list. Learn is the same flow, taught. Not a third news feed."
      tabs={CONTEXT_TABS}
      active={tab}
      onChange={(next) => setActive(CONTEXT_ROUTE[next as (typeof CONTEXT_TABS)[number]])}
    >
      {children}
    </HubShell>
  )
}

export function SystemHub({
  active,
  setActive,
  children,
}: {
  active: string
  setActive: (page: string) => void
  children: ReactNode
}) {
  const tab = systemTabOf(active)
  return (
    <HubShell
      blurb="Keep the machine honest. Health is the orchestra: tape → scan → evidence → brain → risk → paper. Live is earned, not toggled. Data is whether the files behind a stock are fresh."
      tabs={SYSTEM_TABS}
      active={tab}
      onChange={(next) => setActive(SYSTEM_ROUTE[next as (typeof SYSTEM_TABS)[number]])}
    >
      {children}
    </HubShell>
  )
}

export function wrapInHub(
  active: string,
  setActive: (page: string) => void,
  children: ReactNode,
): ReactNode {
  const hub = hubOf(active)
  if (hub === 'Home') return <HomeHub active={active} setActive={setActive}>{children}</HomeHub>
  if (hub === 'Ideas') return <IdeasHub active={active} setActive={setActive}>{children}</IdeasHub>
  if (hub === 'Trade') return <TradeHub active={active} setActive={setActive}>{children}</TradeHub>
  if (hub === 'Context') return <ContextHub active={active} setActive={setActive}>{children}</ContextHub>
  if (hub === 'System') return <SystemHub active={active} setActive={setActive}>{children}</SystemHub>
  return children
}
