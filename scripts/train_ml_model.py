#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/apple/0to100/sq_ai')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from data.historical import HistoricalDataFetcher
from data.kite_client import KiteClient
from data.instruments import InstrumentManager
from feature_store.store import PointInTimeFeatureStore

def main():
    kite = KiteClient()
    instruments = InstrumentManager()
    fetcher = HistoricalDataFetcher(kite, instruments)
    
    symbols = ["NIFTY 50", "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "WIPRO", "LT", "BAJFINANCE"]
    
    feature_store = PointInTimeFeatureStore()
    all_dfs = {}
    
    for sym in symbols:
        df = fetcher.fetch(sym, "2020-01-01", "2023-12-31", interval="day")
        if not df.empty:
            df = df.reset_index().rename(columns={'index': 'timestamp'})
            all_dfs[sym] = df
            feature_store.load_historical_data(sym, df)
            print(f"Loaded {sym}: {len(df)} bars")
    
    regime_df = all_dfs.get("NIFTY 50")
    if regime_df is None:
        print("WARNING: NIFTY 50 missing, regime=1 always")
    
    def get_regime(timestamp):
        if regime_df is None:
            return 1
        past = regime_df[regime_df['timestamp'] <= timestamp]
        if len(past) < 200:
            return 1
        sma200 = past['close'].tail(200).mean()
        current = past['close'].iloc[-1]
        return 1 if current > sma200 else 0
    
    X_rows = []
    y_rows = []
    # These features are guaranteed to exist from your diagnostic
    features_list = ['sma_20', 'sma_50', 'volatility_20', 'momentum_5d', 'volume_trend', 'rsi', 'atr']
    
    for sym, df in all_dfs.items():
        if sym == "NIFTY 50":
            continue
        print(f"Processing {sym}...")
        cnt = 0
        for i in range(60, len(df) - 1):  # start after 60 bars for enough history
            features = feature_store.get_features_at_time(sym, df.iloc[i]['timestamp'], lookback_days=60)
            if not features:
                continue
            regime = get_regime(df.iloc[i]['timestamp'])
            
            # Build row
            row = {f: features.get(f) for f in features_list}
            if any(v is None for v in row.values()):
                continue
            row['regime'] = regime
            
            # Label: 1 if next day positive
            ret = (df.iloc[i+1]['close'] / df.iloc[i]['close']) - 1
            label = 1 if ret > 0 else 0
            
            X_rows.append(row)
            y_rows.append(label)
            cnt += 1
        print(f"  -> {cnt} rows from {sym}")
    
    if len(X_rows) == 0:
        print("No training data generated!")
        return
    
    df_train = pd.DataFrame(X_rows)
    print(f"Total training samples: {len(df_train)}")
    
    split = int(len(df_train) * 0.8)
    X_train, X_test = df_train.iloc[:split], df_train.iloc[split:]
    y_train, y_test = y_rows[:split], y_rows[split:]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.3f}")
    
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lgb_trading_model.pkl")
    with open("models/feature_names.txt", "w") as f:
        f.write("\n".join(features_list + ['regime']))
    print("Model saved to models/lgb_trading_model.pkl")

if __name__ == "__main__":
    main()
