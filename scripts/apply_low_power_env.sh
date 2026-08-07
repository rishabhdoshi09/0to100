# Source this file to throttle QuantTerm for old / fanless Macs.
# Usage: source scripts/apply_low_power_env.sh
#
# Keeps Telegram sniper + manual Scan now. Disables background CPU hogs
# (idle full-universe backtest, US/long-term auto bootstrap) and slows polls.

export QT_LOW_POWER=1
export QT_DISABLE_IDLE_BACKTEST=1
export QT_DISABLE_US_BOOTSTRAP=1
export QT_DISABLE_AUTO_LONG_TERM=1
export QT_DISABLE_AUTO_MARKET_SCAN=1
export QT_SCAN_WORKERS="${QT_SCAN_WORKERS:-2}"
export QT_STACK_WATCH_SLEEP_S="${QT_STACK_WATCH_SLEEP_S:-45}"
export QT_STACK_HTTP_PROBE_EVERY="${QT_STACK_HTTP_PROBE_EVERY:-8}"
export QT_SNIPER_POLL_SECONDS="${QT_SNIPER_POLL_SECONDS:-45}"
export QT_AUTONOMY_INTERVAL_S="${QT_AUTONOMY_INTERVAL_S:-45}"
export VITE_QT_LOW_POWER=1

echo "[LOW POWER] QuantTerm eco profile on — lighter CPU for other Mac work."
echo "[LOW POWER] Idle backtest / auto US+long-term+market bootstrap OFF. Scan now is manual."
echo "[LOW POWER] Scan workers=${QT_SCAN_WORKERS} · watch=${QT_STACK_WATCH_SLEEP_S}s · autonomy=${QT_AUTONOMY_INTERVAL_S}s"
