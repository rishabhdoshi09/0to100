# QuantTerm retail UI audit

## Evidence-based findings

- The legacy `app.py` exposed five engineering-oriented top-level choices: Control, Pulse, Markets, Autopilot and JARVIS, plus a large More Tools list.
- The router contained 25+ page branches and `ui/` contained dozens of specialist modules.
- Momentum existed in backend scanners but had no clear first-class retail page.
- F&O discovery used a small hard-coded list in the legacy options-flow surface. The replacement derives every current individual-stock future underlying from the Kite instrument master.
- Backtesting existed in AlgoLab, Tools and research surfaces but was not a visible top-level user action.
- Control Room was an operations/research observatory, not a plain automatic-paper-trading page.

## Product correction

The default `app.py` is now retail-first. The prior terminal is retained unchanged as `legacy_app.py`.

Default navigation:

- Everyday: Home, Momentum Stocks, Automatic Paper Trading, Portfolio, Market
- Learn and Test: Backtest, What We’ve Learned, Reports
- System: Data and Zerodha, Alerts, Settings, Help
- Advanced: Research Laboratory

The new product layer is a read-only projection over existing backend state. It does not duplicate trading, evidence, portfolio, risk, scanner, execution or backtest logic.
