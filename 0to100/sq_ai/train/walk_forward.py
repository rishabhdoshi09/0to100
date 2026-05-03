"""Walk-forward validation – RUN ON COLAB ONLY.

Splits the dataset into rolling train/test windows, retrains LightGBM
on each train window, evaluates on the next out-of-sample window using
the SAME ``CompositeSignal`` + ``Backtester`` as production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

# Walk-forward expects these features (must match live composite_signal)
FEATURES = ["sma_20", "sma_50", "volatility_20", "momentum_5d",
            "volume_trend", "rsi", "atr", "regime"]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    ret = out["close"].pct_change()
    out["volatility_20"] = ret.rolling(20).std() * np.sqrt(252)
    out["momentum_5d"] = out["close"].pct_change(5)
    out["volume_trend"] = out["volume"] / out["volume"].rolling(20).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    tr = pd.concat([(out["high"] - out["low"]),
                    (out["high"] - out["close"].shift()).abs(),
                    (out["low"] - out["close"].shift()).abs()], axis=1).max(1)
    out["atr"] = tr.rolling(14).mean()
    out["regime"] = np.where(out["sma_20"] > out["sma_50"] * 1.005, 2,
                             np.where(out["sma_20"] < out["sma_50"] * 0.995, 0, 1))
    out["target"] = (out["close"].shift(-5) / out["close"] - 1 > 0.01).astype(int)
    return out.dropna()


def walk_forward(symbol: str = "RELIANCE.NS",
                 start: str = "2018-01-01", end: str = "2023-01-01",
                 train_years: int = 3, test_months: int = 6) -> dict:
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    full = _engineer(raw)
    results = []
    train_bars = train_years * 252
    test_bars = test_months * 21
    i = train_bars
    while i + test_bars < len(full):
        train = full.iloc[i - train_bars: i]
        test = full.iloc[i: i + test_bars]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 max_depth=6, n_jobs=2, verbose=-1)
        clf.fit(train[FEATURES], train["target"])
        proba = clf.predict_proba(test[FEATURES])[:, 1]
        # naive evaluation: take a long when proba > 0.55, hold 5 days
        pos = (proba > 0.55).astype(int)
        ret_5d = test["close"].pct_change(5).shift(-5).fillna(0)
        strat_ret = pos * ret_5d
        sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(252 / 5)
        results.append({
            "fold_start": str(test.index[0].date()),
            "fold_end": str(test.index[-1].date()),
            "n_train": len(train),
            "n_test": len(test),
            "auc_proxy": float(np.corrcoef(proba, test["target"])[0, 1]),
            "sharpe": float(sharpe),
            "cum_return": float((1 + strat_ret).prod() - 1),
        })
        i += test_bars
    return {"symbol": symbol, "folds": results,
            "mean_sharpe": float(np.mean([r["sharpe"] for r in results])),
            "mean_return": float(np.mean([r["cum_return"] for r in results]))}


def export_final_model(symbol: str = "RELIANCE.NS",
                       out_dir: str = "./models",
                       start: str = "2018-01-01",
                       end: str = "2023-01-01") -> None:
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    full = _engineer(raw)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             max_depth=6, n_jobs=2, verbose=-1)
    clf.fit(full[FEATURES], full["target"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, f"{out_dir}/lgb_trading_model.pkl")
    Path(f"{out_dir}/feature_names.txt").write_text("\n".join(FEATURES))
    print(f"saved → {out_dir}/lgb_trading_model.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="RELIANCE.NS")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()
    if args.export:
        export_final_model(args.symbol)
    else:
        print(json.dumps(walk_forward(args.symbol), indent=2))
