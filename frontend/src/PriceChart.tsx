import { useEffect, useRef } from 'react'
import { ColorType, createChart, type IChartApi } from 'lightweight-charts'
import type { ChartBar } from './types'

type Props = {
  symbol: string
  bars: ChartBar[]
}

export function PriceChart({ symbol, bars }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const chart = createChart(container, {
      width: container.clientWidth,
      height: 360,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#66758a',
        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, .22)' },
        horzLines: { color: 'rgba(148, 163, 184, .22)' },
      },
      rightPriceScale: { borderColor: '#cbd5e1' },
      timeScale: { borderColor: '#cbd5e1', timeVisible: true },
      crosshair: {
        vertLine: { color: '#087ea4', labelBackgroundColor: '#087ea4' },
        horzLine: { color: '#087ea4', labelBackgroundColor: '#087ea4' },
      },
    })

    const candles = chart.addCandlestickSeries({
      upColor: '#17795e',
      downColor: '#b42318',
      borderUpColor: '#17795e',
      borderDownColor: '#b42318',
      wickUpColor: '#17795e',
      wickDownColor: '#b42318',
    })
    candles.setData(bars.map(({ time, open, high, low, close }) => ({
      time,
      open,
      high,
      low,
      close,
    })))

    const volume = chart.addHistogramSeries({
      color: '#087ea4',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    volume.setData(bars.map((bar) => ({
      time: bar.time,
      value: bar.volume,
      color: bar.close >= bar.open ? 'rgba(23,121,94,.35)' : 'rgba(180,35,24,.30)',
    })))

    chart.timeScale().fitContent()
    chartRef.current = chart

    const resize = new ResizeObserver(([entry]) => {
      chart.applyOptions({ width: entry.contentRect.width })
    })
    resize.observe(container)

    return () => {
      resize.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [bars, symbol])

  return <div className="chart-canvas" ref={containerRef} aria-label={`${symbol} price chart`} />
}
