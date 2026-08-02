import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchOperation, sendControl } from './api'
import type { ControlName, OperationRecord } from './types'

export type ScanKind = 'MARKET_SCAN' | 'LONG_TERM_SCAN'

const KIND_CONTROL: Record<ScanKind, ControlName> = {
  MARKET_SCAN: 'RUN_SCAN_NOW',
  LONG_TERM_SCAN: 'RUN_LONG_TERM_SCAN_NOW',
}

export const TERMINAL_STATUSES = new Set(['SUCCEEDED', 'FAILED', 'BLOCKED', 'CANCELLED'])

export function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status)
}

export function isActiveStatus(status: string): boolean {
  return status === 'PENDING' || status === 'RUNNING'
}

const STAGE_LABELS: Record<string, string> = {
  PENDING: 'Queued — waiting for the market-ops worker…',
  ACCEPTED: 'Worker picked up the job…',
  STARTING: 'Worker picked up the job…',
  PREPARING_HISTORY: 'Preparing official NSE price history…',
  HISTORY_READY: 'Official history ready…',
  LOADING_UNIVERSE: 'Loading the NSE universe…',
  SCANNING: 'Scanning market candidates…',
  RANKING: 'Ranking qualified ideas…',
  SAVING: 'Saving the latest results…',
  TECHNICAL_SCREEN: 'Screening long-term technicals across the universe…',
  FUNDAMENTALS: 'Scoring current fundamentals on shortlisted names…',
  FETCHING_SOURCES: 'Fetching news sources…',
  RECOVERED: 'Recovering interrupted job…',
}

export function friendlyStageLabel(stage: string, status: string): string {
  if (status === 'SUCCEEDED') return 'Scan complete'
  if (status === 'FAILED') return 'Scan failed'
  if (status === 'CANCELLED') return 'Scan stopped'
  if (status === 'BLOCKED') return 'Scan blocked'
  const key = String(stage || '').trim().toUpperCase()
  if (key && STAGE_LABELS[key]) return STAGE_LABELS[key]
  if (!key && status === 'PENDING') return STAGE_LABELS.PENDING
  if (!key && status === 'RUNNING') return 'Working — progress updates every few seconds…'
  if (key) return key.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (c) => c.toUpperCase())
  return 'Scanning…'
}

export function buildProgressLine(operation: OperationRecord | null): string | null {
  if (!operation) return null
  const total = Number(operation.progress_total || 0)
  const current = Number(operation.progress_current || 0)
  const message = String(operation.message || '').trim()
  if (total > 0) {
    const noun = String(operation.kind || '').includes('LONG_TERM') ? 'names' : 'stocks'
    const counts = `${current.toLocaleString('en-IN')} of ${total.toLocaleString('en-IN')} ${noun}`
    if (message && (/\d+\s*\/\s*\d+/.test(message) || /of\s+[\d,]+/i.test(message))) {
      return message
    }
    return message ? `${message} · ${counts}` : counts
  }
  return message || null
}

export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(Number(seconds) || 0))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  if (m < 60) return `${m}m ${rem}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

export function qualifiedResultLine(operation: OperationRecord | null): string | null {
  if (!operation?.result) return null
  const result = operation.result
  const summary = result.summary as Record<string, unknown> | undefined
  if (summary && summary.qualified != null) {
    return `${Number(summary.qualified).toLocaleString('en-IN')} qualified ideas found`
  }
  if (typeof result.records === 'number' && result.records > 0) {
    return `${result.records.toLocaleString('en-IN')} ideas saved`
  }
  return null
}

export function progressPercent(operation: OperationRecord | null): number | null {
  if (!operation) return null
  if (operation.progress_pct != null && Number.isFinite(operation.progress_pct)) {
    return Math.max(0, Math.min(100, Number(operation.progress_pct)))
  }
  const total = Number(operation.progress_total || 0)
  const current = Number(operation.progress_current || 0)
  if (total > 0) return Math.round((current / total) * 100)
  return null
}

export type ScanRunnerHandle = {
  kind: ScanKind
  operation: OperationRecord | null
  isActive: boolean
  isBusy: boolean
  friendlyPhase: string
  progressLine: string | null
  qualifiedLine: string | null
  percent: number | null
  elapsedSeconds: number
  notice: string | null
  failed: boolean
  succeeded: boolean
  start: () => Promise<void>
  retry: () => Promise<void>
  dismissNotice: () => void
}

type ScanRunnerOptions = {
  onComplete?: () => void
  seedOperation?: OperationRecord | null
}

export function useScanRunner(kind: ScanKind, options: ScanRunnerOptions = {}): ScanRunnerHandle {
  const { onComplete, seedOperation } = options
  const [operation, setOperation] = useState<OperationRecord | null>(null)
  const [isBusy, setIsBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const mountedRef = useRef(true)
  const pollRef = useRef<number | null>(null)
  const trackedIdRef = useRef<string | null>(null)
  const completedIdRef = useRef<string | null>(null)
  const startedAtRef = useRef<number | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  const clearPoll = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const handleTerminal = useCallback((op: OperationRecord) => {
    if (completedIdRef.current === op.operation_id) return
    completedIdRef.current = op.operation_id
    clearPoll()
    trackedIdRef.current = null
    setIsBusy(false)
    startedAtRef.current = null
    if (op.status === 'SUCCEEDED') {
      setNotice('Scan complete — refreshing results…')
      onComplete?.()
      window.setTimeout(() => {
        if (mountedRef.current) setNotice(null)
      }, 5000)
    } else {
      const detail = op.error_message || op.message || `Scan ${String(op.status).toLowerCase()}`
      setNotice(detail)
    }
  }, [clearPoll, onComplete])

  const pollOnce = useCallback(async (operationId: string) => {
    try {
      const op = await fetchOperation(operationId)
      if (!mountedRef.current) return
      setOperation(op)
      if (isTerminalStatus(op.status)) handleTerminal(op)
    } catch {
      // transient network errors while polling — keep trying until terminal or unmount
    }
  }, [handleTerminal])

  const beginPolling = useCallback((operationId: string) => {
    trackedIdRef.current = operationId
    startedAtRef.current = Date.now()
    clearPoll()
    pollRef.current = window.setInterval(() => void pollOnce(operationId), 1000)
    void pollOnce(operationId)
  }, [clearPoll, pollOnce])

  const attachOperation = useCallback((op: OperationRecord) => {
    setOperation(op)
    if (isActiveStatus(op.status)) {
      setIsBusy(true)
      if (trackedIdRef.current !== op.operation_id) beginPolling(op.operation_id)
    } else if (isTerminalStatus(op.status)) {
      setIsBusy(false)
    }
  }, [beginPolling])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      clearPoll()
    }
  }, [clearPoll])

  useEffect(() => {
    const seed = seedOperation
    if (!seed || seed.kind !== kind) return
    if (trackedIdRef.current === seed.operation_id) {
      setOperation(seed)
      return
    }
    if (isActiveStatus(seed.status)) attachOperation(seed)
  }, [attachOperation, kind, seedOperation?.operation_id, seedOperation?.status, seedOperation?.progress_current])

  useEffect(() => {
    const active = Boolean(
      isBusy || (operation && isActiveStatus(operation.status)),
    )
    if (!active) return
    const timer = window.setInterval(() => {
      if (!startedAtRef.current) return
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAtRef.current) / 1000)))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [isBusy, operation?.operation_id, operation?.status])

  const start = useCallback(async () => {
    setIsBusy(true)
    setNotice(null)
    completedIdRef.current = null
    try {
      const result = await sendControl(KIND_CONTROL[kind])
      if (!result.accepted) {
        setIsBusy(false)
        setNotice('Scan request was not accepted by the backend')
        return
      }
      if (!result.operation_id) {
        setIsBusy(false)
        setNotice('Scan queued without an operation id — check System Health')
        return
      }
      const op = await fetchOperation(result.operation_id)
      if (!mountedRef.current) return
      setOperation(op)
      if (isActiveStatus(op.status)) {
        beginPolling(op.operation_id)
      } else if (isTerminalStatus(op.status)) {
        handleTerminal(op)
      } else {
        setIsBusy(false)
      }
    } catch (reason) {
      setIsBusy(false)
      setNotice(reason instanceof Error ? reason.message : 'Scan could not start')
    }
  }, [beginPolling, handleTerminal, kind])

  const retry = useCallback(async () => {
    await start()
  }, [start])

  const dismissNotice = useCallback(() => setNotice(null), [])

  const friendlyPhase = useMemo(
    () => friendlyStageLabel(operation?.stage || '', operation?.status || (isBusy ? 'PENDING' : '')),
    [isBusy, operation?.stage, operation?.status],
  )

  const progressLine = useMemo(() => buildProgressLine(operation), [operation])
  const qualifiedLine = useMemo(() => qualifiedResultLine(operation), [operation])
  const percent = useMemo(() => progressPercent(operation), [operation])
  const detailLine = useMemo(() => {
    if (progressLine) return progressLine
    const message = String(operation?.message || '').trim()
    if (message && message.toLowerCase() !== friendlyPhase.toLowerCase()) return message
    return null
  }, [friendlyPhase, operation?.message, progressLine])

  const isActive = Boolean(
    isBusy || (operation && isActiveStatus(operation.status)),
  )

  return {
    kind,
    operation,
    isActive,
    isBusy,
    friendlyPhase,
    progressLine: detailLine,
    qualifiedLine,
    percent,
    elapsedSeconds,
    notice,
    failed: operation?.status === 'FAILED' || operation?.status === 'BLOCKED',
    succeeded: operation?.status === 'SUCCEEDED',
    start,
    retry,
    dismissNotice,
  }
}
