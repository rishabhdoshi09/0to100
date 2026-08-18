import { useEffect, useState } from 'react'
import {
  deskWaitClock,
  formatElapsed,
  type DeskWaitKind,
  type DeskWaitScan,
  type ScanRunnerHandle,
} from './scanRunner'

export function toDeskWaitScan(scan: ScanRunnerHandle | null | undefined): DeskWaitScan | null {
  if (!scan?.isActive) return null
  return {
    kind: scan.kind,
    isActive: scan.isActive,
    friendlyPhase: scan.friendlyPhase,
    progressLine: scan.progressLine,
    percent: scan.percent,
    elapsedSeconds: scan.elapsedSeconds,
    current: scan.operation?.progress_current ?? undefined,
    total: scan.operation?.progress_total ?? undefined,
  }
}

export function DeskWait({
  kind,
  scan = null,
  className = '',
}: {
  kind: DeskWaitKind
  scan?: DeskWaitScan | null
  className?: string
}) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => setElapsed((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [])
  const clock = deskWaitClock({ kind, elapsedSeconds: elapsed, scan })
  const elapsedLabel = scan?.isActive
    ? `Scan elapsed ${formatElapsed(scan.elapsedSeconds)}`
    : elapsed > 0
      ? `elapsed ${formatElapsed(elapsed)}`
      : 'Starting…'
  return (
    <div className={`desk-wait ${className}`.trim()} role="status" aria-live="polite">
      <strong>{clock.button}</strong>
      <p>{clock.doing || clock.line}</p>
      <small>{elapsedLabel}</small>
      {clock.percent != null ? (
        <div className="live-scan-progress" aria-label={`${clock.percent}%`}>
          <b style={{ width: `${Math.max(4, clock.percent)}%` }} />
        </div>
      ) : (
        <div className="live-scan-progress live-scan-progress-pulse" aria-hidden="true">
          <b className="pulse-bar" />
        </div>
      )}
    </div>
  )
}
