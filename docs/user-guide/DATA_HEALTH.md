# Data health

Everything depends on good data. Bad or missing data makes every result untrustworthy.

## What "healthy" means
- The data is **present**.
- The data is **fresh** (not old / stale).

If either is not true, the system says so honestly. It will **never** show a green
"Ready" while data is missing or stale.

## Missing is NOT zero
If a number is missing, it is **unknown** — not zero. Treating missing as zero would
quietly corrupt every result, so the system refuses to do that.

## When historical research data is not installed
You may see: **INCONCLUSIVE — DATA UNAVAILABLE**. This means:
- **What happened:** the historical data needed for the test is not installed.
- **What it means:** the system cannot honestly decide pass or fail.
- **What still works:** moving around the app, paper features (where live prices exist),
  past research records, the documentation, and reviewing settings.
- **What to do next:** ask the operator to install / locate / validate the dataset. The
  exact file paths and diagnostics are in **Advanced Mode**.

👉 Next: `TROUBLESHOOTING.md`
