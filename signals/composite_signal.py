import os
import joblib
import pandas as pd
from typing import Dict, Optional, Any
import numpy as np
from features.volume_profile import VolumeProfile

class CompositeSignal:
    def __init__(self, model_path: str = "models/lgb_trading_model.pkl"):
        self._ml_model = None
        self._feature_names = None
        self._debug_printed = False
        self.vp = VolumeProfile(num_bins=50, value_area_pct=0.7)
        if os.path.exists(model_path):
            self.load_ml_model(model_path)
    
    def compute(self, features: Dict, llm_signal: Optional[Any] = None, regime: int = 1, price_data: pd.DataFrame = None) -> Dict:
        """Compute composite signal from all sources"""
        # Get individual signals
        factor_score = self._compute_factor_signal(features)
        ml_score = self._compute_ml_signal(features)
        regime_score = self._compute_regime_signal(features)
        llm_score = self._parse_llm_context(llm_signal) if llm_signal is not None else 0.5
        volume_score = self._compute_volume_signal(features, price_data)
        
        # Weighted combination - MODIFIED: higher ML weight, lower factor weight
        weights = {'factor': 0.1, 'ml': 0.6, 'regime': 0.1, 'llm': 0.1, 'volume': 0.1}
        combined = (weights['factor'] * factor_score + 
                   weights['ml'] * ml_score + 
                   weights['regime'] * regime_score + 
                   weights['llm'] * llm_score + 
                   weights['volume'] * volume_score)
        
        # Clip to [-1, 1] range
        combined = max(-1.0, min(1.0, combined))
        
        # Determine direction - MODIFIED: lower threshold to 0.05
        if combined > 0.05:
            direction = 1
        elif combined < -0.05:
            direction = -1
        else:
            direction = 0
        
        # Confidence is absolute value of signal, scaled to 0-100
        confidence = abs(combined) * 100
        
        return {
            'signal': combined,
            'direction': direction,
            'confidence': confidence,
            'attribution': {
                'factor': factor_score,
                'ml': ml_score,
                'regime': regime_score,
                'llm': llm_score,
                'volume': volume_score,
                'combined': combined
            }
        }
    
    def _compute_factor_signal(self, features: Dict) -> float:
        """Simple technical factor scoring"""
        score = 0
        count = 0
        
        # RSI oversold/overbought
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            score += 0.5
            count += 1
        elif rsi > 70:
            score -= 0.5
            count += 1
        
        # Z-score
        zscore = features.get('zscore_20', 0)
        if zscore < -1.5:
            score += 0.3
            count += 1
        elif zscore > 1.5:
            score -= 0.3
            count += 1
        
        # Momentum
        momentum = features.get('momentum_5d_pct', 0)
        if momentum > 0.02:
            score += 0.2
            count += 1
        elif momentum < -0.02:
            score -= 0.2
            count += 1
        
        return score / max(count, 1) if count > 0 else 0.0
    
    def _compute_ml_signal(self, features: Dict) -> float:
        if self._ml_model is None:
            return 0.5
        
        # Map IndicatorEngine output names to model expected names
        feature_order = self._feature_names or ['sma_20', 'sma_50', 'volatility_20', 'momentum_5d', 'volume_trend', 'rsi', 'atr', 'regime']
        
        X = []
        missing_count = 0
        
        for model_feature in feature_order:
            # Try direct or mapped feature names
            if model_feature == 'volatility_20':
                val = features.get('volatility_20', features.get('vol_20d_ann', 0.02))
            elif model_feature == 'momentum_5d':
                val = features.get('momentum_5d', features.get('momentum_5d_pct', 0.0))
            elif model_feature == 'volume_trend':
                val = features.get('volume_trend', features.get('volume_ratio', 1.0))
            elif model_feature == 'rsi':
                val = features.get('rsi', features.get('rsi_14', 50.0))
            elif model_feature == 'atr':
                val = features.get('atr', features.get('atr_14', 0.0))
            else:
                val = features.get(model_feature, 0.0)
            
            if val is None or pd.isna(val):
                val = 0.0
                missing_count += 1
            X.append(val)
        
        # Debug print once
        if not self._debug_printed:
            print(f"ML DEBUG: Sample features: {list(zip(feature_order[:4], X[:4]))}")
            self._debug_printed = True
        
        if missing_count > 2:
            return 0.5
        
        try:
            proba = self._ml_model.predict_proba([X])[0][1]
            signal = (proba - 0.5) * 2
            return max(-1.0, min(1.0, signal))
        except Exception as e:
            print(f"ML prediction error: {e}")
            return 0.5
    
    def _compute_regime_signal(self, features: Dict) -> float:
        sma20 = features.get('sma_20')
        sma50 = features.get('sma_50')
        if sma20 and sma50:
            if sma20 > sma50 * 1.005:  # MODIFIED: 0.5% instead of 2%
                return 0.7
            elif sma20 < sma50 * 0.995:  # MODIFIED: 0.5% instead of 2%
                return -0.5
        return 0.0
    
    def _parse_llm_context(self, llm_signal: Optional[Dict]) -> float:
        if llm_signal is None:
            return 0.5
        sentiment = llm_signal.get('sentiment_score', 0)
        return (sentiment + 1) / 2
    
    
    def _compute_volume_signal(self, features: Dict, price_data: pd.DataFrame = None) -> float:
        """Generate signal from volume profile"""
        if price_data is None or len(price_data) < 20:
            return 0.0
        
        try:
            profile = self.vp.compute(price_data)
            current_price = price_data['close'].iloc[-1]
            signal, signal_type = self.vp.get_signal(profile, current_price)
            
            # Cache profile for debugging
            self._last_vp_profile = profile
            self._last_vp_signal_type = signal_type
            
            return signal
        except Exception as e:
            return 0.0

    def load_ml_model(self, model_path: str) -> bool:
        if os.path.exists(model_path):
            self._ml_model = joblib.load(model_path)
            if hasattr(self._ml_model, 'feature_names_in_'):
                self._feature_names = list(self._ml_model.feature_names_in_)
            print(f"Loaded ML model from {model_path}")
            return True
        return False
