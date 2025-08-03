#!/usr/bin/env python3
"""
OPTIMIZED 0DTE Strategy - Enhanced Signal Quality

This version focuses on SIGNAL QUALITY over quantity to achieve 65%+ win rate.
Enhanced filters include:
- Market microstructure analysis
- Intraday momentum patterns  
- Volume profile confirmation
- Volatility regime filtering
- Support/resistance levels
- Option flow indicators

Target: 65%+ win rate, then scale to $300/day

Author: Alpaca Improved Team
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import base strategy
import sys
import os
sys.path.append(os.path.dirname(__file__))
from aggressive_0dte_strategy import (
    Aggressive0DTEStrategy, Enhanced0DTESignal, SignalStrength, 
    MarketRegime, MultiIndicatorAnalyzer
)


class MarketMicrostructure(Enum):
    """Market microstructure states."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    NEUTRAL = "neutral"


class IntradayRegime(Enum):
    """Intraday market regimes."""
    OPENING_MOMENTUM = "opening_momentum"     # 9:30-10:30 AM
    MIDDAY_CHOP = "midday_chop"             # 10:30 AM - 2:00 PM  
    AFTERNOON_TREND = "afternoon_trend"      # 2:00-3:30 PM
    POWER_HOUR = "power_hour"               # 3:30-4:00 PM
    OVERNIGHT = "overnight"


@dataclass
class EnhancedSignalMetrics:
    """Enhanced signal quality metrics."""
    
    # Base metrics
    base_signal: Enhanced0DTESignal
    
    # Market microstructure
    accumulation_score: float  # 0-100
    volume_profile_score: float  # 0-100
    order_flow_score: float  # 0-100
    
    # Intraday patterns
    intraday_regime: IntradayRegime
    momentum_persistence: float  # 0-100
    reversal_probability: float  # 0-100
    
    # Support/Resistance
    distance_to_support: float
    distance_to_resistance: float
    sr_strength: float  # 0-100
    
    # Advanced filters
    vix_regime_score: float  # 0-100
    sector_rotation_score: float  # 0-100
    options_flow_score: float  # 0-100
    
    # Final quality score
    composite_score: float  # 0-100
    trade_probability: float  # Expected win probability


class OptimizedSignalAnalyzer:
    """Advanced signal analyzer focused on quality."""
    
    def __init__(self):
        self.base_analyzer = MultiIndicatorAnalyzer()
        
        # Enhanced parameters for quality
        self.min_volume_percentile = 60  # Require above-average volume
        self.min_trend_strength = 40     # Stronger trend requirement
        self.min_momentum_persistence = 50  # Momentum must persist
        
        # Support/Resistance levels
        self.lookback_periods = [20, 50, 100]  # Multi-timeframe S/R
        self.sr_touch_sensitivity = 0.5  # % distance to consider "touching" S/R
        
        # Intraday timing filters
        self.avoid_lunch_hours = True  # Skip 11:30-13:30 
        self.prefer_momentum_hours = True  # Favor 9:30-10:30, 14:00-15:30
        
    def analyze_signal_quality(self, 
                             df: pd.DataFrame, 
                             signal: Enhanced0DTESignal,
                             current_idx: int) -> EnhancedSignalMetrics:
        """Comprehensive signal quality analysis."""
        
        # Get current and historical data
        current_row = df.iloc[current_idx]
        historical_data = df.iloc[max(0, current_idx-100):current_idx+1]
        
        # 1. Market Microstructure Analysis
        accumulation_score = self._analyze_accumulation(historical_data)
        volume_profile_score = self._analyze_volume_profile(historical_data)
        order_flow_score = self._estimate_order_flow(historical_data)
        
        # 2. Intraday Pattern Analysis
        intraday_regime = self._detect_intraday_regime(signal.timestamp)
        momentum_persistence = self._analyze_momentum_persistence(historical_data)
        reversal_probability = self._calculate_reversal_probability(historical_data, signal)
        
        # 3. Support/Resistance Analysis
        support_level, resistance_level, sr_strength = self._find_support_resistance(historical_data)
        distance_to_support = ((signal.price - support_level) / signal.price) * 100
        distance_to_resistance = ((resistance_level - signal.price) / signal.price) * 100
        
        # 4. Advanced Market Filters
        vix_regime_score = self._analyze_vix_regime(current_row)
        sector_rotation_score = self._analyze_sector_rotation(historical_data)
        options_flow_score = self._estimate_options_flow(historical_data)
        
        # 5. Calculate Composite Quality Score
        composite_score = self._calculate_composite_score(
            accumulation_score, volume_profile_score, order_flow_score,
            momentum_persistence, sr_strength, vix_regime_score,
            intraday_regime, signal
        )
        
        # 6. Estimate Trade Probability
        trade_probability = self._estimate_trade_probability(composite_score, signal)
        
        return EnhancedSignalMetrics(
            base_signal=signal,
            accumulation_score=accumulation_score,
            volume_profile_score=volume_profile_score,
            order_flow_score=order_flow_score,
            intraday_regime=intraday_regime,
            momentum_persistence=momentum_persistence,
            reversal_probability=reversal_probability,
            distance_to_support=distance_to_support,
            distance_to_resistance=distance_to_resistance,
            sr_strength=sr_strength,
            vix_regime_score=vix_regime_score,
            sector_rotation_score=sector_rotation_score,
            options_flow_score=options_flow_score,
            composite_score=composite_score,
            trade_probability=trade_probability
        )
    
    def _analyze_accumulation(self, df: pd.DataFrame) -> float:
        """Analyze accumulation/distribution patterns."""
        if len(df) < 20:
            return 50
        
        # Price vs Volume analysis
        price_change = df['close'].pct_change()
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        
        # Accumulation: Price up + Volume up = bullish
        # Distribution: Price down + Volume up = bearish
        
        accumulation_signals = []
        for i in range(1, len(df)):
            price_dir = 1 if price_change.iloc[i] > 0 else -1
            volume_strength = volume_ratio.iloc[i]
            
            if volume_strength > 1.2:  # Above average volume
                if price_dir > 0:
                    accumulation_signals.append(1)  # Accumulation
                else:
                    accumulation_signals.append(-1)  # Distribution
            else:
                accumulation_signals.append(0)  # Neutral
        
        if not accumulation_signals:
            return 50
        
        # Recent accumulation bias (last 10 periods)
        recent_signals = accumulation_signals[-10:]
        accumulation_score = (sum(recent_signals) / len(recent_signals)) * 50 + 50
        
        return max(0, min(100, accumulation_score))
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> float:
        """Analyze volume profile strength."""
        if len(df) < 10:
            return 50
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_percentile = (df['volume'].rank(pct=True).iloc[-1]) * 100
        if pd.isna(volume_percentile):
            volume_percentile = 50
        
        # Volume surge analysis
        volume_surge = current_volume / avg_volume if avg_volume > 0 and not pd.isna(avg_volume) else 1
        
        # Score based on volume characteristics
        score = 0
        
        # High volume percentile (40 points)
        score += min(40, volume_percentile * 0.4)
        
        # Volume surge bonus (30 points)
        if volume_surge > 1.5:
            score += min(30, (volume_surge - 1) * 20)
        
        # Consistency bonus (30 points)
        recent_volumes = df['volume'].tail(5)
        volume_mean = recent_volumes.mean()
        volume_std = recent_volumes.std()
        if volume_mean > 0 and not pd.isna(volume_std) and not pd.isna(volume_mean):
            volume_consistency = max(0, 1 - (volume_std / volume_mean))
            score += volume_consistency * 30
        
        return max(0, min(100, score))
    
    def _estimate_order_flow(self, df: pd.DataFrame) -> float:
        """Estimate institutional order flow."""
        if len(df) < 10:
            return 50
        
        # Simplified order flow estimation using price and volume
        # Real implementation would use level 2 data
        
        # Calculate money flow
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive/negative flow
        price_change = df['close'].diff()
        positive_flow = money_flow.where(price_change > 0, 0).rolling(10).sum()
        negative_flow = money_flow.where(price_change < 0, 0).rolling(10).sum()
        
        # Money Flow Index
        total_flow = positive_flow + abs(negative_flow)
        mfi = (positive_flow / total_flow * 100).iloc[-1] if total_flow.iloc[-1] > 0 else 50
        
        return max(0, min(100, mfi))
    
    def _detect_intraday_regime(self, timestamp: datetime) -> IntradayRegime:
        """Detect intraday market regime."""
        hour = timestamp.hour
        minute = timestamp.minute
        time_decimal = hour + minute / 60
        
        if 9.5 <= time_decimal < 10.5:
            return IntradayRegime.OPENING_MOMENTUM
        elif 10.5 <= time_decimal < 14.0:
            return IntradayRegime.MIDDAY_CHOP
        elif 14.0 <= time_decimal < 15.5:
            return IntradayRegime.AFTERNOON_TREND
        elif 15.5 <= time_decimal < 16.0:
            return IntradayRegime.POWER_HOUR
        else:
            return IntradayRegime.OVERNIGHT
    
    def _analyze_momentum_persistence(self, df: pd.DataFrame) -> float:
        """Analyze momentum persistence patterns."""
        if len(df) < 20:
            return 50
        
        # Calculate multiple momentum indicators
        price_momentum = df['close'].pct_change(5).iloc[-1] * 100
        volume_momentum = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()).iloc[-1]
        
        # RSI momentum
        rsi = self._calculate_rsi(df['close'], 14).iloc[-1]
        rsi_momentum = abs(rsi - 50) / 50 * 100  # Distance from neutral
        
        # MACD momentum
        macd_line, macd_signal = self._calculate_macd(df['close'])
        macd_momentum = (macd_line.iloc[-1] - macd_signal.iloc[-1]) * 100
        
        # Combine momentum factors
        momentum_factors = [
            abs(price_momentum) * 2,  # Weight price momentum heavily
            (volume_momentum - 1) * 50,  # Volume confirmation
            rsi_momentum,  # RSI strength
            abs(macd_momentum) * 10  # MACD confirmation
        ]
        
        persistence_score = np.mean([max(0, min(100, factor)) for factor in momentum_factors])
        
        return persistence_score
    
    def _calculate_reversal_probability(self, df: pd.DataFrame, signal: Enhanced0DTESignal) -> float:
        """Calculate probability of signal being a reversal."""
        if len(df) < 30:
            return 50
        
        # Look for reversal patterns
        reversal_signals = 0
        total_signals = 0
        
        # 1. Divergence analysis
        price_trend = df['close'].rolling(10).mean().diff(5).iloc[-1]
        rsi_trend = self._calculate_rsi(df['close'], 14).diff(5).iloc[-1]
        
        if (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0):
            reversal_signals += 1
        total_signals += 1
        
        # 2. Overbought/Oversold extremes
        rsi_current = self._calculate_rsi(df['close'], 14).iloc[-1]
        if (signal.signal_type == "BULLISH" and rsi_current > 70) or \
           (signal.signal_type == "BEARISH" and rsi_current < 30):
            reversal_signals += 1
        total_signals += 1
        
        # 3. Volume exhaustion
        volume_trend = df['volume'].rolling(5).mean().diff(3).iloc[-1]
        if volume_trend < 0:  # Declining volume suggests exhaustion
            reversal_signals += 1
        total_signals += 1
        
        reversal_probability = (reversal_signals / total_signals) * 100
        return reversal_probability
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Find key support and resistance levels."""
        if len(df) < 50:
            return df['low'].iloc[-1], df['high'].iloc[-1], 50
        
        # Find recent highs and lows
        highs = df['high'].rolling(10, center=True).max()
        lows = df['low'].rolling(10, center=True).min()
        
        # Identify significant levels (touched multiple times)
        price_range = df['high'].max() - df['low'].min()
        tolerance = price_range * 0.01  # 1% tolerance
        
        # Find resistance (recent highs)
        recent_highs = highs.dropna().tail(20)
        resistance_candidates = []
        
        for high in recent_highs:
            touches = sum(abs(highs - high) <= tolerance)
            if touches >= 2:
                resistance_candidates.append((high, touches))
        
        # Find support (recent lows)  
        recent_lows = lows.dropna().tail(20)
        support_candidates = []
        
        for low in recent_lows:
            touches = sum(abs(lows - low) <= tolerance)
            if touches >= 2:
                support_candidates.append((low, touches))
        
        # Select strongest levels
        if resistance_candidates:
            resistance = max(resistance_candidates, key=lambda x: x[1])[0]
        else:
            resistance = df['high'].tail(20).max()
            
        if support_candidates:
            support = min(support_candidates, key=lambda x: x[0])[0]
        else:
            support = df['low'].tail(20).min()
        
        # Calculate strength based on number of touches
        max_touches = max(
            len([r for r in resistance_candidates if r[0] == resistance]),
            len([s for s in support_candidates if s[0] == support])
        )
        
        strength = min(100, max_touches * 25)  # More touches = stronger level
        
        return support, resistance, strength
    
    def _analyze_vix_regime(self, current_row: pd.Series) -> float:
        """Analyze VIX regime (simplified without actual VIX data)."""
        # Estimate volatility regime from price action
        # In real implementation, would use actual VIX data
        
        # Use ATR percentile as proxy for volatility regime
        atr_percentile = current_row.get('atr_percentile', 50)
        
        # Optimal volatility regimes for different strategies
        if 30 <= atr_percentile <= 70:
            return 100  # Goldilocks zone
        elif atr_percentile > 80:
            return 30   # Too volatile
        elif atr_percentile < 20:
            return 40   # Too quiet
        else:
            return 70   # Acceptable
    
    def _analyze_sector_rotation(self, df: pd.DataFrame) -> float:
        """Analyze sector rotation patterns (simplified)."""
        # Simplified sector analysis using SPY momentum
        # Real implementation would use sector ETFs
        
        short_momentum = df['close'].pct_change(5).iloc[-1]
        medium_momentum = df['close'].pct_change(20).iloc[-1] 
        
        # Consistent momentum across timeframes suggests strong sector move
        if short_momentum * medium_momentum > 0:
            consistency_score = min(100, abs(short_momentum) * 1000)
        else:
            consistency_score = 30  # Conflicting signals
        
        return consistency_score
    
    def _estimate_options_flow(self, df: pd.DataFrame) -> float:
        """Estimate options flow sentiment (simplified)."""
        # Simplified options flow estimation
        # Real implementation would use options volume data
        
        # Use volume and price action as proxy
        volume_surge = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        price_momentum = abs(df['close'].pct_change().iloc[-1]) * 100
        
        # Strong volume + momentum suggests options activity
        flow_strength = min(100, (volume_surge * price_momentum * 10))
        
        return flow_strength
    
    def _calculate_composite_score(self, 
                                 accumulation_score: float,
                                 volume_profile_score: float,
                                 order_flow_score: float,
                                 momentum_persistence: float,
                                 sr_strength: float,
                                 vix_regime_score: float,
                                 intraday_regime: IntradayRegime,
                                 signal: Enhanced0DTESignal) -> float:
        """Calculate weighted composite quality score."""
        
        # Base signal strength (30% weight)
        base_score = signal.strength.value * 20  # Convert to 0-100 scale
        
        # Technical factors (40% weight)
        technical_score = np.mean([
            accumulation_score,
            volume_profile_score,
            order_flow_score,
            momentum_persistence
        ])
        
        # Market structure (20% weight)
        structure_score = np.mean([sr_strength, vix_regime_score])
        
        # Intraday timing bonus/penalty (10% weight)
        timing_multiplier = self._get_timing_multiplier(intraday_regime)
        
        # Weighted composite
        composite = (
            base_score * 0.30 +
            technical_score * 0.40 +
            structure_score * 0.20
        ) * timing_multiplier
        
        return max(0, min(100, composite))
    
    def _get_timing_multiplier(self, regime: IntradayRegime) -> float:
        """Get timing multiplier based on intraday regime."""
        multipliers = {
            IntradayRegime.OPENING_MOMENTUM: 1.15,  # Favor opening momentum
            IntradayRegime.AFTERNOON_TREND: 1.10,   # Favor afternoon trends
            IntradayRegime.POWER_HOUR: 1.05,        # Slight favor for power hour
            IntradayRegime.MIDDAY_CHOP: 0.85,       # Avoid midday chop
            IntradayRegime.OVERNIGHT: 0.80          # Avoid overnight gaps
        }
        return multipliers.get(regime, 1.0)
    
    def _estimate_trade_probability(self, composite_score: float, signal: Enhanced0DTESignal) -> float:
        """Estimate win probability based on composite score."""
        
        # Base probability from composite score
        base_probability = composite_score / 100 * 0.85  # Max 85% probability
        
        # Adjust for signal strength
        strength_adjustment = {
            SignalStrength.WEAK: 0.85,
            SignalStrength.MODERATE: 0.95,
            SignalStrength.STRONG: 1.05,
            SignalStrength.VERY_STRONG: 1.15,
            SignalStrength.EXTREME: 1.25
        }
        
        adjusted_probability = base_probability * strength_adjustment.get(signal.strength, 1.0)
        
        # Add market regime adjustment
        regime_adjustment = {
            MarketRegime.TRENDING_UP: 1.1,
            MarketRegime.TRENDING_DOWN: 1.1,
            MarketRegime.HIGH_VOLATILITY: 1.05,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.9
        }
        
        final_probability = adjusted_probability * regime_adjustment.get(signal.market_regime, 1.0)
        
        return max(0.3, min(0.85, final_probability))  # Clamp between 30-85%
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        
        return macd_line, macd_signal


class OptimizedStrategy(Aggressive0DTEStrategy):
    """Optimized 0DTE strategy focused on signal quality."""
    
    def __init__(self, target_daily_profit: float = 300, account_size: float = 25000):
        super().__init__(target_daily_profit, account_size)
        
        # Enhanced quality filters
        self.min_composite_score = 70  # Require 70+ composite score
        self.min_trade_probability = 0.65  # Require 65%+ win probability
        self.max_reversal_probability = 40  # Avoid high reversal risk
        
        # Quality analyzer
        self.quality_analyzer = OptimizedSignalAnalyzer()
        
        # Reduce quantity for quality
        self.max_trades_per_day = 5  # Fewer, higher quality trades
        
    def generate_optimized_signals(self, df: pd.DataFrame) -> List[EnhancedSignalMetrics]:
        """Generate optimized signals with enhanced quality analysis."""
        
        # Start with base signals
        base_signals = self.generate_0dte_signals(df)
        
        # Enhance each signal with quality analysis
        enhanced_signals = []
        
        for signal in base_signals:
            # Find the signal index in the dataframe
            signal_idx = None
            for i, timestamp in enumerate(df.index):
                if abs((timestamp - signal.timestamp).total_seconds()) < 3600:  # Within 1 hour
                    signal_idx = i
                    break
            
            if signal_idx is None or signal_idx < 100:  # Need enough historical data
                continue
            
            # Perform quality analysis
            quality_metrics = self.quality_analyzer.analyze_signal_quality(df, signal, signal_idx)
            
            # Apply quality filters
            if (quality_metrics.composite_score >= self.min_composite_score and
                quality_metrics.trade_probability >= self.min_trade_probability and
                quality_metrics.reversal_probability <= self.max_reversal_probability):
                
                enhanced_signals.append(quality_metrics)
        
        # Sort by quality score (best first)
        enhanced_signals.sort(key=lambda x: x.composite_score, reverse=True)
        
        return enhanced_signals
    
    def get_optimization_summary(self) -> Dict:
        """Get optimization strategy summary."""
        return {
            "strategy_name": "Optimized 0DTE Quality Strategy",
            "focus": "Signal Quality over Quantity",
            "target_win_rate": "65%+",
            "min_composite_score": self.min_composite_score,
            "min_trade_probability": self.min_trade_probability,
            "max_reversal_probability": self.max_reversal_probability,
            "max_trades_per_day": self.max_trades_per_day,
            "quality_filters": [
                "Market Microstructure Analysis",
                "Intraday Regime Detection", 
                "Volume Profile Confirmation",
                "Support/Resistance Levels",
                "Momentum Persistence",
                "Options Flow Estimation",
                "Reversal Risk Assessment"
            ]
        }


# Example usage
if __name__ == "__main__":
    print("🎯 OPTIMIZED 0DTE STRATEGY - QUALITY FOCUS")
    print("🎯 Target: 65%+ Win Rate → Scale to $300/day")
    print("=" * 50)
    
    strategy = OptimizedStrategy()
    summary = strategy.get_optimization_summary()
    
    print(f"📊 Strategy: {summary['strategy_name']}")
    print(f"🎯 Focus: {summary['focus']}")
    print(f"📈 Target Win Rate: {summary['target_win_rate']}")
    print(f"🔧 Min Composite Score: {summary['min_composite_score']}")
    print(f"📊 Min Trade Probability: {summary['min_trade_probability']*100:.0f}%")
    print(f"⚠️ Max Reversal Risk: {summary['max_reversal_probability']}%")
    print(f"📈 Max Trades/Day: {summary['max_trades_per_day']}")
    
    print(f"\n🔧 Enhanced Quality Filters:")
    for i, filter_name in enumerate(summary['quality_filters'], 1):
        print(f"  {i}. {filter_name}")
    
    print(f"\n💡 Strategy Philosophy:")
    print(f"✅ Quality over quantity - fewer trades, higher win rate")
    print(f"🎯 65%+ win rate target with sophisticated filtering")
    print(f"📊 Multi-factor composite scoring system")
    print(f"⚡ Optimized for 0DTE time constraints")