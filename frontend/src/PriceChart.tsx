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
        textColor: '#5c6b63',
        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(216, 226, 219, .9)' },
        horzLines: { color: 'rgba(216, 226, 219, .9)' },
      },
      rightPriceScale: { borderColor: '#d8e2db' },
      timeScale: { borderColor: '#d8e2db', timeVisible: true },
      crosshair: {
        vertLine: { color: '#1b6b45', labelBackgroundColor: '#e7f5ee' },
        horzLine: { color: '#1b6b45', labelBackgroundColor: '#e7f5ee' },
      },
    })

    const candles = chart.addCandlestickSeries({
      upColor: '#31e981',
      downColor: '#ff667f',
      borderUpColor: '#31e981',
      borderDownColor: '#ff667f',
      wickUpColor: '#31e981',
      wickDownColor: '#ff667f',
    })
    candles.setData(bars.map(({ time, open, high, low, close }) => ({
      time,
      open,
      high,
      low,
      close,
    })))

    const volume = chart.addHistogramSeries({
      color: '#24d6ff',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    volume.setData(bars.map((bar) => ({
      time: bar.time,
      value: bar.volume,
      color: bar.close >= bar.open ? 'rgba(49,233,129,.45)' : 'rgba(255,102,127,.45)',
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
