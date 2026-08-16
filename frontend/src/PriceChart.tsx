import { useEffect, useRef } from 'react'
import { ColorType, createChart, type IChartApi } from 'lightweight-charts'
import { chartHeightForWidth } from './phoneLayout'
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
      height: chartHeightForWidth(window.innerWidth),
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#8390a8',
        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(46, 62, 91, .38)' },
        horzLines: { color: 'rgba(46, 62, 91, .38)' },
      },
      rightPriceScale: { borderColor: '#24314b' },
      timeScale: { borderColor: '#24314b', timeVisible: true },
      crosshair: {
        vertLine: { color: '#24d6ff', labelBackgroundColor: '#0d2031' },
        horzLine: { color: '#24d6ff', labelBackgroundColor: '#0d2031' },
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
      chart.applyOptions({
        width: entry.contentRect.width,
        height: chartHeightForWidth(window.innerWidth),
      })
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
