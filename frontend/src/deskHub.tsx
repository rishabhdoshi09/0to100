import { useState } from 'react'
import { CompareView, WatchlistView } from './marketRadarViews'
import { NewsView } from './marketViews'
import { ProductStockIntelligenceView } from './productViews'
import { ResearchDataView } from './researchData'
import { AutomationView, MarketInternalsView, type ViewProps } from './views'

const TABS = ['Market', 'News', 'Data', 'Stock', 'Compare', 'Watchlist', 'System'] as const

export function DeskHub(props: ViewProps & {
  compareSymbols: string[]
  setCompareSymbols: (symbols: string[]) => void
  onCompare: (symbol: string) => void
  onWatchlist: (symbol: string) => void
  depth: 'simple' | 'professional'
}) {
  const [tab, setTab] = useState<(typeof TABS)[number]>('Market')
  return (
    <section className="workspace-view">
      <div className="reco-tabs">
        {TABS.map((item) => (
          <button key={item} type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>
        ))}
      </div>
      {tab === 'Market' && <MarketInternalsView {...props} />}
      {tab === 'News' && <NewsView {...props} />}
      {tab === 'Data' && <ResearchDataView symbol={props.selected} />}
      {tab === 'Stock' && (
        <ProductStockIntelligenceView {...props} depth={props.depth} onCompare={props.onCompare} onWatchlist={props.onWatchlist} />
      )}
      {tab === 'Compare' && (
        <CompareView
          symbols={props.compareSymbols}
          setSymbols={props.setCompareSymbols}
          setActive={props.setActive}
          setSelected={props.setSelected}
        />
      )}
      {tab === 'Watchlist' && (
        <WatchlistView setActive={props.setActive} setSelected={props.setSelected} onCompare={props.onCompare} />
      )}
      {tab === 'System' && <AutomationView {...props} />}
    </section>
  )
}
