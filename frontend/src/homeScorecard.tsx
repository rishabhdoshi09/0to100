import type { LearningScorecardSummary } from './productApi'

const pct = (value: number | null | undefined) => (value == null ? null : Math.round(value * 100))

/**
 * Home: measured learning + graduation, straight from the nightly scorecard.
 * Never a green badge over unproven evidence: every gate that is not READY is
 * shown with its first blocker, and live trading is always stated.
 */
export function LearningScorecardBlock({ card }: { card: LearningScorecardSummary | null | undefined }) {
  if (!card) return null
  const cal = card.calibration || {}
  const implied = pct(cal.top_bucket_score_as_prob)
  const realized = pct(cal.top_bucket_realized_win_rate)
  return (
    <div className="home-os-past" data-testid="home-learning-scorecard">
      <span>SCORECARD &amp; GRADUATION</span>
      <strong>{card.headline || 'Scorecard not computed yet.'}</strong>
      {card.available ? (
        <>
          <small>
            Setup score calibration: {(cal.verdict || 'INSUFFICIENT EVIDENCE').replaceAll('_', ' ').toLowerCase()}
            {cal.auc != null ? ` · AUC ${cal.auc}` : ''}
            {implied != null && realized != null ? ` · top scores imply ${implied}% but won ${realized}%` : ''}
          </small>
          <small>
            Settled paper trades {card.paper_trades_settled ?? 0}
            {' '}· decisions changed by learning {card.decisions_changed_by_learning ?? 0}
            {' '}· stage {card.stage ?? 0} of 5 · live trading {card.live_trading || 'LOCKED'}
          </small>
          {Object.entries(card.verdicts || {}).map(([gate, verdict]) => {
            const first = (card.blockers?.[gate] || [])[0]
            return (
              <small key={gate} className={verdict === 'READY' ? '' : 'home-os-blocker'}>
                {gate}: {verdict}
                {first ? ` — ${first}` : ''}
              </small>
            )
          })}
        </>
      ) : (
        <small>Live trading {card.live_trading || 'LOCKED'}.</small>
      )}
    </div>
  )
}
