# Running the Historical Gauntlet

The gauntlet answers one question: **does a strategy have a statistically
significant, durable, benchmark-beating edge after realistic costs?** It produces
exactly one verdict per strategy — `PASS` / `FAIL` / `INCONCLUSIVE` — and a
committee-grade report. It refuses to run on data it cannot trust.

> This must run in an environment with NSE archive access. It will **abort** —
> correctly — anywhere the data is missing.

## 1. Acquire the data

```bash
# price history (logs/bhav/) and index+VIX history (logs/indices/)
python -c "from data.bhavcopy_store import build_store; build_store()"
python -c "from data.index_store  import build_index_store; build_index_store()"
```

Then supply the two tables the archives don't hand you in one file each:

- `logs/ca_events.json` — NSE Corporate Actions for the test window:
  ```json
  [{"symbol": "RELIANCE", "ex_date": "2024-10-28", "factor": 2.0, "type": "bonus"}]
  ```
  `factor` = the multiple the share count rose by (1:1 bonus → 2.0, 1→5 split → 5.0).
- `logs/universe_history.json` — point-in-time membership incl. delisted names:
  ```json
  [{"symbol": "DHFL", "listed": "2010-01-01", "delisted": "2019-11-20"}]
  ```

The window **must span a real drawdown** or the ≥2-regime criterion can't be met.

## 2. Validate integrity (the gate)

```bash
python -c "from data.bhavcopy_store import reload_corporate_actions as r; print('CA symbols:', r())"
python -c "from core.data_integrity import verify_ca_adjustment as v; print(v())"
```

`verify_ca_adjustment()` must report `passed: True` (a CA table is loaded **and**
the phantom-gap rate has collapsed to ≈0). If it fails, the gauntlet will abort.

## 3. Run

```bash
python -m gauntlet            # markdown report to stdout; exit 2 on ABORT
python -m gauntlet --json     # machine-readable report
python -m gauntlet --factors  # also test factor-neutral alpha (needs factor data)
```

Reports and experiment stamps are written to `logs/gauntlet/`.

## 4. Interpret — strictly by the pre-registered criteria

A strategy is **PASS** only when the harness `PROMOTE`s it **and** it survives the
Benjamini-Hochberg FDR correction across strategies. The verdict is bound to a
frozen config hash (`gauntlet/freeze.py`) and an experiment id
(`gauntlet/registry.py`) so it is reproducible.

- **On FAIL** — do **not** tweak a parameter and rerun. Write the EXP post-mortem
  in `docs/RESEARCH_LOG.md` first (why it failed; data / execution / strategy;
  per-signal contribution; reject or pre-register a genuinely new hypothesis).
- **On PASS** — freeze the strategy and **begin forward paper trading
  immediately**. Lost out-of-sample days are unrecoverable.
- **On INCONCLUSIVE** — keep tracking; make no claim.

Record the outcome as the next `EXP-NNN` entry in the Research Log either way —
negative evidence is as valuable as positive.
