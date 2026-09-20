import type { DashboardPayload } from "./types"

export type OperatorState = "RUNNING" | "REFRESHING" | "ATTENTION"

const ACTIVE_STATUSES = new Set(["PENDING", "RUNNING"])

/**
 * Compact runtime truth used by the sidebar summary.
 * Manual controls live in the System workspace; this module does not own actions.
 */
export function operatorState(dashboard: DashboardPayload): OperatorState {
  const operationsOnline = dashboard.operations.running === true
  const autonomyOnline = Boolean(dashboard.autonomy.running || dashboard.autonomy.process_running)
  if (!operationsOnline || !autonomyOnline) return "ATTENTION"
  if ((dashboard.operations.active || []).some((row) => ACTIVE_STATUSES.has(String(row.status || "").toUpperCase()))) {
    return "REFRESHING"
  }
  return "RUNNING"
}
