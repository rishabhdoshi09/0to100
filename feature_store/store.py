"""
Point-in-time feature store - NO LEAKAGE, NO LOOKAHEAD
Handles DataFrames with DatetimeIndex (from Kite)
"""

import pandas as pd
from sortedcontainers import SortedList
from datetime import datetime
from typing import Dict, List, Optional

class PointInTimeFeatureStore:
    """Every feature computed uses ONLY data available at that timestamp."""
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = data_dir
        self._cache = {}
        self._timestamp_index = SortedList()
        self._historical_bars = {}
        
    def load_historical_data(self, symbol: str, df: pd.DataFrame):
        """
        Load data ONCE at backtest start.
        df has DatetimeIndex, columns: open, high, low, close, volume
        """
        df = df.copy()
        
        # Handle DatetimeIndex (Kite format) - don't add duplicate column
        if isinstance(df.index, pd.DatetimeIndex):
            # Use index as timestamp, don't create separate column
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            raise ValueError(f"No datetime index or timestamp column for {symbol}")
        
        df = df.sort_values('timestamp')
        self._historical_bars[symbol] = df
        
        # Index all timestamps
        for ts in df['timestamp']:
            self._timestamp_index.add(ts)
    
    def get_features_at_time(self, symbol: str, current_time: datetime, lookback_days: int = 20) -> Dict:
        """Get features using ONLY data <= current_time."""
        cache_key = (symbol, current_time)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self._historical_bars.get(symbol)
        if df is None:
            return {}
        
        # CRITICAL: Filter to past data only
        past_data = df[df['timestamp'] <= current_time].copy()
        
        if len(past_data) < lookback_days:
            return {}
        
        # Compute features
        features = {
            'sma_20': past_data['close'].tail(20).mean(),
            'sma_50': past_data['close'].tail(50).mean() if len(past_data) >= 50 else None,
            'volatility_20': past_data['close'].tail(20).pct_change().std(),
            'momentum_5d': (past_data['close'].iloc[-1] / past_data['close'].iloc[-5] - 1) if len(past_data) >= 5 else None,
            'volume_trend': past_data['volume'].tail(10).mean() / past_data['volume'].tail(30).mean() if len(past_data) >= 30 else None,
            'rsi': self._compute_rsi(past_data['close'], 14),
            'atr': self._compute_atr(past_data, 14),
        }
        
        self._cache[cache_key] = features
        return features
    
    def _compute_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return None
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
    
    def _compute_atr(self, df, period=14):
        if len(df) < period + 1:
            return None
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None
    
    def clear_cache(self):
        self._cache.clear()
