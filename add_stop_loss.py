import re

with open('backtest/backtester.py', 'r') as f:
    content = f.read()

# Find line "self._portfolio.update_prices(bar_closes)" and add stop-check after it
stop_loss_code = '''
            # --- Stop-loss / take-profit (ATR-based) ---
            for sym in list(self._portfolio.get_open_positions().keys()):
                pos_data = self._portfolio.get_open_positions()[sym]
                entry = pos_data['avg_entry_price']
                current = bar_closes.get(sym, entry)
                features = self.feature_store.get_features_at_time(sym, bar_time, lookback_days=20)
                atr = features.get('atr') if features else None
                if atr is None or atr == 0:
                    continue
                pnl_pct = (current - entry) / entry
                if pos_data['quantity'] > 0:  # long
                    if pnl_pct < -2 * atr / entry:
                        self._portfolio.close_position(sym, current, order_id=f"SL-{sym}", transaction_cost_rate=0.0, timestamp=bar_time)
                        self._risk.record_pnl(0)  # placeholder
                    elif pnl_pct > 4 * atr / entry:
                        self._portfolio.close_position(sym, current, order_id=f"TP-{sym}", transaction_cost_rate=0.0, timestamp=bar_time)
                else:  # short
                    if pnl_pct > 2 * atr / entry:
                        self._portfolio.close_position(sym, current, order_id=f"SL-{sym}", transaction_cost_rate=0.0, timestamp=bar_time)
                    elif pnl_pct < -4 * atr / entry:
                        self._portfolio.close_position(sym, current, order_id=f"TP-{sym}", transaction_cost_rate=0.0, timestamp=bar_time)
'''

# Insert after the line "self._portfolio.update_prices(bar_closes)"
if 'self._portfolio.update_prices(bar_closes)' in content:
    content = content.replace(
        'self._portfolio.update_prices(bar_closes)',
        'self._portfolio.update_prices(bar_closes)\n' + stop_loss_code
    )
    with open('backtest/backtester.py', 'w') as f:
        f.write(content)
    print("Stop-loss code added.")
else:
    print("Line not found. Manual edit required.")
