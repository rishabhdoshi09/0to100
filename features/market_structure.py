import pandas as pd

def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    highs = df['high']
    lows = df['low']
    window = 2 * lookback + 1
    swing_high = (highs == highs.rolling(window, center=True).max())
    swing_low = (lows == lows.rolling(window, center=True).min())
    swing_high.iloc[:lookback] = False
    swing_high.iloc[-lookback:] = False
    swing_low.iloc[:lookback] = False
    swing_low.iloc[-lookback:] = False
    return pd.DataFrame({'swing_high': swing_high, 'swing_low': swing_low})

def is_recent_swing_breakout(df: pd.DataFrame, lookback: int = 3):
    swings = find_swing_points(df, lookback)
    last_swing_high = df[swings['swing_high']]['high'].iloc[-1] if swings['swing_high'].any() else None
    last_swing_low = df[swings['swing_low']]['low'].iloc[-1] if swings['swing_low'].any() else None
    close = df['close'].iloc[-1]
    if last_swing_high and close > last_swing_high:
        return 'break_high'
    if last_swing_low and close < last_swing_low:
        return 'break_low'
    return False
