"""
Volume Profile indicator for market microstructure analysis.
Identifies high/low volume nodes and value area boundaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class VolumeProfile:
    """
    Volume Profile indicator calculating POC, Value Area, and volume nodes.
    
    Volume Profile shows trading activity at specific price levels rather than over time.
    - POC (Point of Control): Price level with highest volume
    - High Volume Nodes (HVN): Zones with concentrated trading activity
    - Low Volume Nodes (LVN): Zones with sparse trading (gaps)
    - Value Area (VA): Range containing ~70% of all volume
    """
    
    def __init__(self, num_bins: int = 50, value_area_pct: float = 0.7):
        """
        Initialize Volume Profile calculator.
        
        Args:
            num_bins: Number of price bins for the histogram
            value_area_pct: Percentage of volume for value area (typically 0.7)
        """
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self._last_profile: Optional[Dict] = None
    
    def compute(self, df: pd.DataFrame) -> Dict:
        """
        Compute volume profile from OHLCV data.
        
        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns
            
        Returns:
            Dictionary with POC, VAH, VAL, HVNs, LVNs
        """
        if df is None or len(df) < 2:
            return self._empty_profile()
        
        price_high = df['high'].max()
        price_low = df['low'].min()
        
        if price_high <= price_low:
            return self._empty_profile()
        
        # Create price bins
        bin_size = (price_high - price_low) / self.num_bins
        bins = np.linspace(price_low, price_high, self.num_bins + 1)
        
        # Initialize volume per bin
        volume_profile = np.zeros(self.num_bins)
        
        # Distribute volume across bins for each candle
        for _, row in df.iterrows():
            low = row['low']
            high = row['high']
            volume = row['volume']
            
            # Find bins this candle touches
            start_bin = max(0, int((low - price_low) / bin_size))
            end_bin = min(self.num_bins - 1, int((high - price_low) / bin_size))
            
            if start_bin == end_bin:
                volume_profile[start_bin] += volume
            else:
                # Distribute evenly across touched bins
                bins_touched = end_bin - start_bin + 1
                vol_per_bin = volume / bins_touched
                for b in range(start_bin, end_bin + 1):
                    volume_profile[b] += vol_per_bin
        
        # Find POC (bin with max volume)
        poc_idx = np.argmax(volume_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Calculate Value Area
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.value_area_pct
        
        # Sort bins by volume (descending)
        sorted_indices = np.argsort(volume_profile)[::-1]
        
        cum_volume = 0
        value_area_bins = []
        for idx in sorted_indices:
            cum_volume += volume_profile[idx]
            value_area_bins.append(idx)
            if cum_volume >= target_volume:
                break
        
        # Get price levels for value area
        value_area_min = min(bins[va_idx] for va_idx in value_area_bins)
        value_area_max = max(bins[va_idx + 1] for va_idx in value_area_bins)
        
        # Identify HVN (High Volume Nodes) and LVN (Low Volume Nodes)
        avg_volume = volume_profile.mean()
        std_volume = volume_profile.std()
        
        hvns = []
        lvns = []
        
        for i, vol in enumerate(volume_profile):
            price_level = (bins[i] + bins[i + 1]) / 2
            if vol > avg_volume + std_volume:
                hvns.append(price_level)
            elif vol < avg_volume * 0.3:  # Very low volume threshold
                lvns.append(price_level)
        
        self._last_profile = {
            'poc': float(poc_price),
            'vah': float(value_area_max),
            'val': float(value_area_min),
            'hvns': hvns,
            'lvns': lvns,
            'value_area_width': float(value_area_max - value_area_min),
            'total_volume': float(total_volume),
            'volume_profile': volume_profile.tolist(),
            'bins': bins.tolist()
        }
        
        return self._last_profile
    
    def _empty_profile(self) -> Dict:
        """Return empty profile when insufficient data"""
        return {
            'poc': 0.0,
            'vah': 0.0,
            'val': 0.0,
            'hvns': [],
            'lvns': [],
            'value_area_width': 0.0,
            'total_volume': 0.0,
            'volume_profile': [],
            'bins': []
        }
    
    def get_signal(self, profile: Dict, current_price: float) -> Tuple[float, str]:
        """
        Generate trading signal based on volume profile.
        
        Args:
            profile: Volume profile from compute() method
            current_price: Current market price
            
        Returns:
            Tuple of (signal_strength, signal_type)
            signal_strength: -1 to 1, positive = bullish, negative = bearish
            signal_type: 'HVN', 'LVN', 'VA_LONG', 'VA_SHORT', or 'NONE'
        """
        if profile['total_volume'] == 0:
            return (0.0, 'NONE')
        
        signal = 0.0
        signal_type = 'NONE'
        threshold_pct = 0.005  # 0.5% threshold for "at" level
        
        # Check if price is at High Volume Node (HVN) - support/resistance
        for hvn in profile['hvns']:
            if abs(current_price - hvn) / current_price < threshold_pct:
                # At HVN: fade the move (mean reversion)
                signal = 0.3 if current_price > hvn else -0.3
                signal_type = 'HVN'
                break
        
        # Check if price is at Low Volume Node (LVN) - breakout zone
        # LVN suggests price can move quickly through this zone
        for lvn in profile['lvns']:
            if abs(current_price - lvn) / current_price < threshold_pct:
                signal_type = 'LVN'
                # Signal direction depends on which side of LVN we're on
                if current_price > lvn:
                    signal = 0.2  # Bullish breakout potential
                else:
                    signal = -0.2  # Bearish breakdown potential
                break
        
        # Value Area boundaries - contrarian signals
        if signal == 0.0:
            if current_price < profile['val']:
                signal = 0.25  # Below value area - buy (mean reversion)
                signal_type = 'VA_LONG'
            elif current_price > profile['vah']:
                signal = -0.25  # Above value area - sell (mean reversion)
                signal_type = 'VA_SHORT'
        
        return (float(signal), signal_type)
    
    def get_poc_position(self, profile: Dict, current_price: float) -> str:
        """
        Determine where current price is relative to POC.
        
        Returns:
            'above', 'below', or 'at'
        """
        if profile['total_volume'] == 0:
            return 'unknown'
        
        if current_price > profile['poc'] * 1.01:
            return 'above'
        elif current_price < profile['poc'] * 0.99:
            return 'below'
        else:
            return 'at'
    
    def get_value_area_ratio(self, profile: Dict, current_price: float) -> float:
        """
        Calculate where price sits within value area (0 to 1).
        0 = at VAL, 1 = at VAH, 0.5 = middle of value area.
        """
        if profile['value_area_width'] == 0:
            return 0.5
        
        ratio = (current_price - profile['val']) / profile['value_area_width']
        return max(0.0, min(1.0, float(ratio)))


# Simple test function
def test_volume_profile():
    """Quick test of volume profile functionality"""
    import numpy as np
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    # Compute volume profile
    vp = VolumeProfile(num_bins=50, value_area_pct=0.7)
    profile = vp.compute(df)
    
    print("Volume Profile Test Results:")
    print(f"  POC: {profile['poc']:.2f}")
    print(f"  VAH: {profile['vah']:.2f}")
    print(f"  VAL: {profile['val']:.2f}")
    print(f"  Value Area Width: {profile['value_area_width']:.2f}")
    print(f"  HVNs: {len(profile['hvns'])} nodes at {[f'{h:.2f}' for h in profile['hvns'][:3]]}")
    print(f"  LVNs: {len(profile['lvns'])} nodes")
    
    # Test signal generation
    current_price = df['close'].iloc[-1]
    signal, signal_type = vp.get_signal(profile, current_price)
    print(f"\nCurrent Price: {current_price:.2f}")
    print(f"Signal: {signal:.2f} ({signal_type})")
    print(f"POC Position: {vp.get_poc_position(profile, current_price)}")
    print(f"Value Area Ratio: {vp.get_value_area_ratio(profile, current_price):.2f}")
    
    return profile


if __name__ == "__main__":
    test_volume_profile()
