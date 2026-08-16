import { pct } from './format'

export type EvFields = {
  ev_pct?: number | null
  ev_lb_pct?: number | null
  ev_n?: number | null
  ev_conf?: string
  p_win?: number | null
}

export const EV_MIN_N = 30

export function hasGatedEv(row: EvFields | null | undefined): row is EvFields & { ev_pct: number; ev_n: number } {
  return row != null && row.ev_pct != null && Number(row.ev_n || 0) >= EV_MIN_N
}

export function EvChip({ row }: { row: EvFields | null | undefined }) {
  if (!hasGatedEv(row)) return null
  const conf = row.ev_conf ? ` · ${row.ev_conf}` : ''
  const win = row.p_win != null ? ` · p(win) ${row.p_win}%` : ''
  return (
    <span className="reco-ev-chip" title={`Conservative EV uses the Wilson lower bound when present. n=${row.ev_n}${conf}${win}`}>
      EV {pct(row.ev_lb_pct ?? row.ev_pct)} · n {row.ev_n}
    </span>
  )
}
