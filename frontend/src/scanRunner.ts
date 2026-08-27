import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchOperation, sendControl } from './api'
import type { ControlName, OperationRecord } from './types'

export type ScanKind = 'MARKET_SCAN' | 'LONG_TERM_SCAN' | 'LONG_TERM_REFRESH'

const KIND_CONTROL: Record<ScanKind, ControlName> = {
  MARKET_SCAN: 'RUN_SCAN_NOW',
  LONG_TERM_SCAN: 'RUN_SCAN_NOW',
  LONG_TERM_REFRESH: 'REFRESH_LONG_TERM_NOW',
}

export const SCAN_POLL_MS = 300
export const TERMINAL_STATUSES = new Set(['SUCCEEDED', 'FAILED', 'BLOCKED', 'CANCELLED'])

export function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status)
}

export function isActiveStatus(status: string): boolean {
  return status === 'PENDING' || status === 'RUNNING'
}

export function seedKindMatches(seedKind: string, runnerKind: ScanKind): boolean {
  if (seedKind === runnerKind) return true
  if (runnerKind === 'LONG_TERM_SCAN' && (seedKind === 'LONG_TERM_REFRESH' || seedKind === 'MARKET_SCAN')) return true
  if (runnerKind === 'LONG_TERM_REFRESH' && (seedKind === 'LONG_TERM_SCAN' || seedKind === 'LONG_TERM_REFRESH')) return true
  if (runnerKind === 'MARKET_SCAN' && seedKind === 'LONG_TERM_SCAN') return true
  return false
}

const STAGE_LABELS: Record<string, string> = {
  PENDING: 'Starting the scan…',
  PREPARING_HISTORY: 'Preparing market history…',
  WAITING_FOR_HISTORY: 'Waiting for official prices…',
  WARMING_HISTORY: 'Warming official price cache…',
  HISTORY_READY: 'Market history ready…',
  LOADING_UNIVERSE: 'Loading the NSE universe…',
  SCANNING: 'Scanning market candidates…',
  RANKING: 'Ranking qualified ideas…',
  SAVING: 'Saving the latest results…',
  TECHNICAL_SCREEN: 'Screening long-term candidates…',
  LONG_TERM_OVERLAY: 'Applying long-term overlay from the same scan…',
  FETCHING_SOURCES: 'Fetching news sources…',
  RECOVERED: 'Recovering interrupted job…',
}

const STOCK_PROGRESS_STAGES = new Set(['SCANNING', 'RANKING', 'SAVING'])

export function isStockScanStage(stage: string): boolean {
  return STOCK_PROGRESS_STAGES.has(String(stage || '').trim().toUpperCase())
}

export function friendlyStageLabel(stage: string, status: string, elapsedSeconds = 0): string {
  if (status === 'SUCCEEDED') return 'Scan complete'
  if (status === 'FAILED') return 'Scan failed'
  if (status === 'CANCELLED') return 'Scan stopped'
  if (status === 'BLOCKED') return 'Scan blocked'
  if (status === 'PENDING' && elapsedSeconds >= 15) return 'Waiting for the scan worker…'
  const key = String(stage || '').trim().toUpperCase()
  if (key && STAGE_LABELS[key]) return STAGE_LABELS[key]
  if (!key && status === 'PENDING') return 'Starting the scan…'
  if (!key && status === 'RUNNING') return 'Working on the scan…'
  if (key) return key.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (c) => c.toUpperCase())
  return 'Scanning…'
}

export function buildProgressLine(operation: OperationRecord | null): string | null {
  if (!operation) return null
  const stage = String(operation.stage || '').trim().toUpperCase()
  if (!isStockScanStage(stage)) return null
  const total = Number(operation.progress_total || 0)
  const current = Number(operation.progress_current || 0)
  if (total >= 100) {
    return `Scanning ${current.toLocaleString('en-IN')} of ${total.toLocaleString('en-IN')} stocks`
  }
  return null
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

export function formatEta(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null
  if (seconds < 15) return 'under 15s'
  if (seconds < 60) return `about ${Math.round(seconds / 5) * 5}s`
  const minutes = Math.max(1, Math.round(seconds / 60))
  return minutes === 1 ? 'about 1 min' : `about ${minutes} min`
}

export function estimateEtaSeconds(
  operation: OperationRecord | null,
  elapsedSeconds: number,
): number | null {
  if (!operation) return null
  if (!isStockScanStage(String(operation.stage || ''))) return null
  const total = Number(operation.progress_total || 0)
  const current = Number(operation.progress_current || 0)
  if (total < 100 || current <= 0 || elapsedSeconds <= 0) return null
  const rate = current / elapsedSeconds
  if (rate <= 0) return null
  return Math.max(0, Math.round((total - current) / rate))
}

export function progressPercent(operation: OperationRecord | null): number | null {
  if (!operation) return null
  if (!isStockScanStage(String(operation.stage || ''))) return null
  if (operation.progress_pct != null && Number.isFinite(operation.progress_pct)) {
    return Math.max(0, Math.min(100, Number(operation.progress_pct)))
  }
  const total = Number(operation.progress_total || 0)
  const current = Number(operation.progress_current || 0)
  if (total >= 100) return Math.round((current / total) * 100)
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
  etaLine: string | null
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
  const scanPaceIdRef = useRef<string | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  const clearPoll = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const handleTerminal = useCallback((op: OperationRecord) => {
    setIsBusy(false)
    startedAtRef.current = null
    scanPaceIdRef.current = null
    if (completedIdRef.current === op.operation_id) return
    completedIdRef.current = op.operation_id
    clearPoll()
    trackedIdRef.current = null
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
      if (isStockScanStage(op.stage) && scanPaceIdRef.current !== op.operation_id) {
        scanPaceIdRef.current = op.operation_id
        startedAtRef.current = Date.now()
        setElapsedSeconds(0)
      }
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
    pollRef.current = window.setInterval(() => void pollOnce(operationId), SCAN_POLL_MS)
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
    if (!seed || !seedKindMatches(seed.kind, kind)) return
    if (trackedIdRef.current === seed.operation_id) {
      setOperation(seed)
      if (isTerminalStatus(seed.status)) handleTerminal(seed)
      return
    }
    if (isActiveStatus(seed.status)) attachOperation(seed)
  }, [attachOperation, handleTerminal, kind, seedOperation?.operation_id, seedOperation?.status, seedOperation?.progress_current])

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
    scanPaceIdRef.current = null
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
    () => friendlyStageLabel(
      operation?.stage || '',
      operation?.status || (isBusy ? 'PENDING' : ''),
      elapsedSeconds,
    ),
    [elapsedSeconds, isBusy, operation?.stage, operation?.status],
  )

  const progressLine = useMemo(() => buildProgressLine(operation), [operation])
  const qualifiedLine = useMemo(() => qualifiedResultLine(operation), [operation])
  const percent = useMemo(() => progressPercent(operation), [operation])
  const etaLine = useMemo(
    () => formatEta(estimateEtaSeconds(operation, elapsedSeconds)),
    [elapsedSeconds, operation],
  )

  const isActive = Boolean(
    isBusy || (operation && isActiveStatus(operation.status)),
  )

  return {
    kind,
    operation,
    isActive,
    isBusy,
    friendlyPhase,
    progressLine,
    qualifiedLine,
    percent,
    etaLine,
    elapsedSeconds,
    notice,
    failed: operation?.status === 'FAILED' || operation?.status === 'BLOCKED',
    succeeded: operation?.status === 'SUCCEEDED',
    start,
    retry,
    dismissNotice,
  }
}
