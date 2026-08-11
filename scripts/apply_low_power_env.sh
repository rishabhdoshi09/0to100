# Source this file to throttle QuantTerm CPU on older / fanless Macs.
# Usage: source scripts/apply_low_power_env.sh
#
# Same trading stack as complete: autonomy, market-ops, market-scan bootstrap,
# autopilot feed, sniper, Telegram, report API. Only background CPU hogs are
# reduced (idle full-universe backtest, optional US bootstrap) and polls slow.

export QT_LOW_POWER=1
export QT_DISABLE_IDLE_BACKTEST="${QT_DISABLE_IDLE_BACKTEST:-1}"
export QT_DISABLE_US_BOOTSTRAP="${QT_DISABLE_US_BOOTSTRAP:-1}"
# Never disable India market / long-term bootstrap — autopilot needs those scans.
unset QT_DISABLE_AUTO_MARKET_SCAN 2>/dev/null || true
unset QT_DISABLE_AUTO_LONG_TERM 2>/dev/null || true
export QT_SCAN_WORKERS="${QT_SCAN_WORKERS:-2}"
export QT_STACK_WATCH_SLEEP_S="${QT_STACK_WATCH_SLEEP_S:-45}"
export QT_STACK_HTTP_PROBE_EVERY="${QT_STACK_HTTP_PROBE_EVERY:-8}"
export QT_SNIPER_POLL_SECONDS="${QT_SNIPER_POLL_SECONDS:-45}"
export QT_AUTONOMY_INTERVAL_S="${QT_AUTONOMY_INTERVAL_S:-45}"
export VITE_QT_LOW_POWER=1

echo "[LOW POWER] QuantTerm eco profile on — lighter CPU for other Mac work."
echo "[LOW POWER] Same stack as complete: market scan + autopilot feed still auto-start."
echo "[LOW POWER] Idle backtest / US bootstrap OFF. Scan workers=${QT_SCAN_WORKERS} · autonomy=${QT_AUTONOMY_INTERVAL_S}s"
