# Historical Data Setup — a simple guide

QuantTerm can only test an idea honestly if you give it **real past market data**. This
page (More Tools → **🗂️ Historical Data Setup**) walks you through it. It only manages
data — **it can never place a trade.**

## 1. Where to get the data (permitted sources)

- **NSE official archives** are the intended source: daily equity files
  (`sec_bhavdata_full_DDMMYYYY.csv`) and index files (`ind_close_all_DDMMYYYY.csv`).
- On a computer with internet, QuantTerm can fetch these itself
  (`build_store()` / `build_index_store()`), or you can bring files you already have.
- For the best results also add a corporate-action file (`ca_events.json`) and a
  listing/delisting history (`universe_history.json`). Without them the test still runs,
  but it will refuse to hand you a "PASS" (see readiness below).
- Do **not** use screen-scraped or third-party "adjusted" data — it is unreliable and
  can quietly bias the result.

## 2. Upload a ZIP

Make a `.zip` containing any of these:

```
bhav/                 ← daily equity CSVs (one file per trading day)
index/                ← Nifty index CSVs
ca_events.json        ← corporate actions (optional)
universe_history.json ← listing/delisting history (optional)
```

Choose **Upload a ZIP package**, pick the file, and click **Check this ZIP**. Unsafe or
unrecognised files are skipped automatically for your safety.

## 3. Use an existing folder

If your data already lives in a folder on this computer, choose **Use an existing
folder**, type the full path, and click **Check this folder**. Nothing is copied until
you press Save.

## 4. Understand readiness

You'll see one clear result:

- 🟢 **Research ready** — good to go.
- 🟠 **Usable for limited analysis** — the test can run, but because something is
  missing (e.g. corporate actions or listing history) it will not issue a "PASS".
- 🔴 **Experiment cannot be run** — data is missing or looks corrupted. The reasons are
  listed in plain language. This gate **cannot be bypassed.**

The panel also shows what was found (price data, benchmark, delivery, dates, number of
stocks and rows) and whether prices are adjusted.

## 5. Save, then run EXP-006

- Click **💾 Save data**. If a dataset already exists you choose: create a new one,
  replace the old one, or cancel — nothing is ever overwritten silently.
- After saving you get an **exact data version** (a snapshot id) so the run is
  reproducible.
- When readiness is green or amber, click **▶️ Run EXP-006 Historical Test**. This runs
  the **unchanged** frozen research test. It never changes the strategy's rules.

## 6. Where results are saved

Each run is written to a **new, permanent folder** under
`docs/overhaul/experiments/EXP-006/runs/<run-id>/` (previous runs are never overwritten).
You'll see the verdict — **PASS / FAIL / INCONCLUSIVE** — with a plain-language meaning,
and the technical details are one click away.

Remember: a PASS is *promising, not a promise*; INCONCLUSIVE is not a PASS; and
"data unavailable" does not mean the strategy failed — it means the data couldn't judge it.
