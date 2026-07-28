# Behaviour matrix (what the system does in each state)

One shared source of truth: `core.simple_language.BEHAVIOUR_MATRIX` (used by the UI, this doc and the tests, so messaging can never contradict permissions).

| State | Mode | Data | You can | You cannot | Why | Next |
|---|---|---|---|---|---|---|
| research_data_available | RESEARCH | available | run a historical test; read past results | place any order | Research only studies history. It can never place an order. | Open the Research Lab and read a result. |
| research_data_unavailable | RESEARCH | unavailable | read the honest 'data unavailable' explanation | judge the test; place any order | Without the historical data the test cannot honestly be judged. | Install the data (see Data Health). |
| paper_market_open | PAPER | available | take a paper trade; arm/disarm autopilot | place a REAL order | Paper practice uses pretend money. No real money moves. | Watch, or read why a setup qualified. |
| paper_market_closed | PAPER | available | review past paper trades; do the walkthrough | enter a new trade now; place a REAL order | The market is closed, so no new entries happen. | Do the PAPER walkthrough or review results. |
| paper_safety_stop | PAPER | available | review today's trades | new paper trades today; place a REAL order | Today's loss limit was reached; new trades are blocked to protect capital. | Stop for today. It resets next trading day. |
| paper_trade_limit | PAPER | available | review today's trades | more trades today; place a REAL order | The daily trade limit stops over-trading. | Come back tomorrow. |
| live_migration_lock | LIVE | any | read why live is locked | arm live; place a REAL order | Live real-money trading is held behind a temporary lock. An environment variable alone does NOT make a strategy safe or eligible. | Keep practising in PAPER. Live needs formal evidence + deployment sign-off. |
| telegram_paper_action | PAPER | available | accept a paper trade from Telegram | place a REAL order from Telegram — ever | Telegram can only ever trigger PAPER practice, never a real order. | Tap it to record a paper trade, or ignore it. |
| stale_market_data | PAPER | stale | refresh data | trust the shown prices as live | Old data can be wrong. It is labelled as stale, never shown as fresh. | Refresh before relying on any price. |
| broker_mismatch | LIVE | any | read the mismatch warning | proceed as if records agree | If the system's records and the broker's records disagree, it stops and warns rather than guessing. | Resolve the mismatch before anything live (operator task). |
| no_valid_setup | PAPER | available | wait | force a trade | Nothing met all the rules. 'No trade' is a correct outcome. | No action needed. |
| eligible_candidate | PAPER | available | read why it qualified; take the paper trade | place a REAL order | A possible setup passed every rule. It is still only a possibility. | Read the entry, stop and maximum loss before anything. |
| rejected_candidate | PAPER | available | read why it was skipped | override the skip | A possible setup failed a rule. The reason is always recorded. | Read the reason to learn the rules. |

**No row anywhere offers a real-money order** — proven by the test `test_no_row_ever_offers_a_real_order`.
