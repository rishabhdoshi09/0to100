# Replace the signal loop in backtest/backtester.py with this ranking approach

# Inside run(), after computing composite for all symbols:
scores = []
for symbol, ...:
    composite = self.composite_signal.compute(features, llm_context)
    scores.append((symbol, composite['direction'] * composite['confidence'] / 100))  # signed score

# Sort by score (high = long candidate, low = short candidate)
scores.sort(key=lambda x: x[1], reverse=True)
longs = scores[:2]   # top 2
shorts = scores[-2:] # bottom 2

# Submit orders
for sym, score in longs:
    if score > 0.15:   # lower threshold
        signal = TradingSignal(sym, "BUY", min(score*100, 95), ...)
        risk_decision = self._risk.evaluate(...)
        if risk_decision.approved:
            self._broker.submit_order(sym, "BUY", quantity, ...)

for sym, score in shorts:
    if score < -0.15:
        signal = TradingSignal(sym, "SELL", ...)
        ...
