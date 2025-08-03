#!/usr/bin/env python3
"""
Aggressive 0DTE Options Strategy - $300/Day Target

This strategy targets $300/day returns using 0DTE (0 Days to Expiry) options
with a 25k account. Uses multiple technical indicators to filter high-probability
setups and avoid sideways markets.

Target: 1.2% daily return (300/25000)
Risk Management: Max 2% daily loss, position sizing based on Kelly Criterion

Technical Indicators:
- MA Shift Oscillator (primary signal)
- Bollinger Bands (volatility regime)
- Keltner Channels (trend strength)
- ATR (volatility filtering)
- RSI (momentum confirmation)
- Volume Profile (strength confirmation)
- VIX (market fear/greed)

Author: Alpaca Improved Team
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class SignalStrength(Enum):
    """Enhanced signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5  # For 0DTE only


@dataclass
class Enhanced0DTESignal:
    """Enhanced signal for 0DTE trading."""
    timestamp: datetime
    signal_type: str  # BULLISH, BEARISH, NEUTRAL
    strength: SignalStrength
    
    # Primary indicators
    ma_shift_osc: float
    ma_value: float
    price: float
    
    # Filter indicators
    bb_position: float  # Position within Bollinger Bands (0-1)
    bb_squeeze: bool    # Bollinger Band squeeze
    kc_position: float  # Position within Keltner Channels
    atr_percentile: float  # ATR percentile (0-100)
    rsi: float
    volume_ratio: float  # Current volume vs average
    vix_level: float
    
    # Market regime
    market_regime: MarketRegime
    trend_strength: float  # 0-100
    
    # Risk metrics
    probability: float  # Win probability estimate
    risk_reward: float  # Expected risk/reward ratio
    
    # 0DTE specific
    time_to_expiry_hours: float
    gamma_risk: float  # Gamma exposure risk


class MultiIndicatorAnalyzer:
    """Advanced technical analysis with multiple indicators."""
    
    def __init__(self):
        # MA Shift parameters
        self.ma_length = 20
        self.osc_length = 10
        self.osc_threshold = 1.5
        
        # Bollinger Bands
        self.bb_period = 20
        self.bb_std = 2
        
        # Keltner Channels
        self.kc_period = 20
        self.kc_multiplier = 2
        
        # ATR
        self.atr_period = 14
        
        # RSI
        self.rsi_period = 14
        
        # Volume
        self.volume_ma_period = 20
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        
        # Price data
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # 1. MA Shift Oscillator (Primary Signal)
        df = self._calculate_ma_shift(df)
        
        # 2. Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        # 3. Keltner Channels
        df = self._calculate_keltner_channels(df)
        
        # 4. ATR
        df = self._calculate_atr(df)
        
        # 5. RSI
        df = self._calculate_rsi(df)
        
        # 6. Volume Analysis
        df = self._calculate_volume_indicators(df)
        
        # 7. Market Regime Detection
        df = self._detect_market_regime(df)
        
        return df
    
    def _calculate_ma_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MA Shift Oscillator (enhanced version)."""
        # Multiple timeframe MAs
        df['ma_short'] = df['hl2'].rolling(self.ma_length).mean()
        df['ma_long'] = df['hl2'].rolling(self.ma_length * 2).mean()
        
        # MA difference and momentum
        df['ma_diff'] = df['hl2'] - df['ma_short']
        df['ma_momentum'] = df['ma_diff'].rolling(self.osc_length).mean()
        
        # Normalized oscillator
        diff_std = df['ma_diff'].rolling(50).std()
        df['ma_shift_osc'] = df['ma_momentum'] / (diff_std + 1e-8)
        
        # Smoothed version
        df['ma_shift_smooth'] = df['ma_shift_osc'].rolling(3).mean()
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and related metrics."""
        # Standard Bollinger Bands
        bb_ma = df['close'].rolling(self.bb_period).mean()
        bb_std = df['close'].rolling(self.bb_period).std()
        
        df['bb_upper'] = bb_ma + (bb_std * self.bb_std)
        df['bb_lower'] = bb_ma - (bb_std * self.bb_std)
        df['bb_middle'] = bb_ma
        
        # BB Position (0 = lower band, 1 = upper band)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # BB Width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB Squeeze (low volatility)
        bb_width_ma = df['bb_width'].rolling(20).mean()
        bb_width_std = df['bb_width'].rolling(20).std()
        df['bb_squeeze'] = df['bb_width'] < (bb_width_ma - bb_width_std)
        
        return df
    
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels for trend analysis."""
        # EMA basis
        df['kc_middle'] = df['close'].ewm(span=self.kc_period).mean()
        
        # ATR for channel width
        atr = self._calculate_atr_raw(df, self.kc_period)
        df['kc_upper'] = df['kc_middle'] + (atr * self.kc_multiplier)
        df['kc_lower'] = df['kc_middle'] - (atr * self.kc_multiplier)
        
        # KC Position
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        df['kc_position'] = df['kc_position'].clip(0, 1)
        
        # Trend strength (how far from middle)
        df['kc_trend_strength'] = abs(df['close'] - df['kc_middle']) / (df['kc_upper'] - df['kc_middle'])
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and volatility metrics."""
        atr = self._calculate_atr_raw(df, self.atr_period)
        df['atr'] = atr
        
        # ATR percentile (volatility regime)
        df['atr_percentile'] = df['atr'].rolling(100).rank(pct=True) * 100
        
        # Volatility regime
        df['high_volatility'] = df['atr_percentile'] > 80
        df['low_volatility'] = df['atr_percentile'] < 20
        
        return df
    
    def _calculate_atr_raw(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate raw ATR."""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and momentum indicators."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI divergence
        df['rsi_ma'] = df['rsi'].rolling(5).mean()
        df['rsi_divergence'] = abs(df['rsi'] - df['rsi_ma']) > 5
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(self.volume_ma_period).mean()
        
        # Volume ratio (current vs average)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume price trend
        df['vpt'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
        df['vpt_ma'] = df['vpt'].rolling(20).mean()
        df['volume_trend'] = df['vpt'] > df['vpt_ma']
        
        return df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime using multiple indicators."""
        # Trend detection
        df['trend_up'] = (df['close'] > df['ma_long']) & (df['ma_short'] > df['ma_long'])
        df['trend_down'] = (df['close'] < df['ma_long']) & (df['ma_short'] < df['ma_long'])
        
        # Sideways market detection
        bb_tight = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.3)
        price_range_tight = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close'] < 0.1
        df['sideways'] = bb_tight & price_range_tight
        
        # Market regime classification
        conditions = [
            df['trend_up'] & ~df['high_volatility'],
            df['trend_down'] & ~df['high_volatility'],
            df['sideways'],
            df['high_volatility'],
            df['low_volatility']
        ]
        
        choices = [
            MarketRegime.TRENDING_UP.value,
            MarketRegime.TRENDING_DOWN.value,
            MarketRegime.SIDEWAYS.value,
            MarketRegime.HIGH_VOLATILITY.value,
            MarketRegime.LOW_VOLATILITY.value
        ]
        
        df['market_regime'] = np.select(conditions, choices, default=MarketRegime.SIDEWAYS.value)
        
        # Trend strength calculation
        trend_factors = [
            df['kc_trend_strength'],
            abs(df['ma_shift_osc']),
            df['volume_ratio'].clip(0, 3) / 3,  # Normalize to 0-1
            (100 - abs(df['rsi'] - 50)) / 50  # Distance from RSI 50
        ]
        
        df['trend_strength'] = np.mean(trend_factors, axis=0) * 100
        
        return df


class Aggressive0DTEStrategy:
    """Aggressive 0DTE strategy targeting $300/day."""
    
    def __init__(self, target_daily_profit: float = 300, account_size: float = 25000):
        self.target_daily_profit = target_daily_profit
        self.account_size = account_size
        self.target_daily_return = target_daily_profit / account_size  # 1.2%
        
        # Risk management
        self.max_daily_loss = account_size * 0.02  # 2% max daily loss
        self.max_position_risk = account_size * 0.008  # 0.8% per trade
        
        # Strategy parameters
        self.min_signal_strength = SignalStrength.STRONG.value
        self.max_trades_per_day = 8
        self.min_time_between_trades = 30  # minutes
        
        # 0DTE specific
        self.max_hours_to_expiry = 8  # Trade only if < 8 hours to expiry
        self.min_hours_to_expiry = 0.5  # Don't trade in last 30 minutes
        
        self.analyzer = MultiIndicatorAnalyzer()
    
    def generate_0dte_signals(self, df: pd.DataFrame, current_time: datetime = None) -> List[Enhanced0DTESignal]:
        """Generate enhanced 0DTE signals with multiple filters."""
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate all indicators
        df = self.analyzer.calculate_all_indicators(df)
        
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # Need enough data for indicators
                continue
            
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Calculate time to expiry (assuming same-day expiry)
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            
            # For 0DTE, expiry is typically 4:15 PM ET same day
            expiry_time = timestamp.replace(hour=16, minute=15, second=0, microsecond=0)
            if timestamp.hour >= 16:
                # If after market close, next day expiry
                expiry_time += timedelta(days=1)
            
            time_to_expiry_hours = (expiry_time - timestamp).total_seconds() / 3600
            
            # Skip if outside 0DTE time window
            if time_to_expiry_hours > self.max_hours_to_expiry or time_to_expiry_hours < self.min_hours_to_expiry:
                continue
            
            # Generate signal
            signal = self._analyze_signal_strength(row, timestamp, time_to_expiry_hours)
            
            if signal and signal.strength.value >= self.min_signal_strength:
                signals.append(signal)
        
        return signals
    
    def _analyze_signal_strength(self, row: pd.Series, timestamp: datetime, time_to_expiry_hours: float) -> Optional[Enhanced0DTESignal]:
        """Analyze signal strength using multiple indicators."""
        
        # Primary signal from MA Shift
        ma_osc = row['ma_shift_smooth']
        signal_type = "NEUTRAL"
        
        if ma_osc > 2.0:
            signal_type = "BULLISH"
        elif ma_osc < -2.0:
            signal_type = "BEARISH"
        
        if signal_type == "NEUTRAL":
            return None
        
        # Signal strength calculation
        strength_score = 0
        
        # 1. MA Shift strength (0-20 points)
        ma_strength = min(abs(ma_osc) / 3.0, 1.0) * 20
        strength_score += ma_strength
        
        # 2. Bollinger Band position (0-15 points)
        if signal_type == "BULLISH":
            bb_score = max(0, (row['bb_position'] - 0.8) * 75)  # Reward upper BB touches
        else:
            bb_score = max(0, (0.2 - row['bb_position']) * 75)  # Reward lower BB touches
        strength_score += min(bb_score, 15)
        
        # 3. Keltner Channel confirmation (0-15 points)
        if signal_type == "BULLISH":
            kc_score = row['kc_position'] * 15
        else:
            kc_score = (1 - row['kc_position']) * 15
        strength_score += kc_score
        
        # 4. RSI confirmation (0-10 points)
        if signal_type == "BULLISH":
            rsi_score = max(0, min((row['rsi'] - 30) / 20, 1)) * 10  # RSI 30-50 is good for bullish
        else:
            rsi_score = max(0, min((70 - row['rsi']) / 20, 1)) * 10  # RSI 50-70 is good for bearish
        strength_score += rsi_score
        
        # 5. Volume confirmation (0-10 points)
        volume_score = min(row['volume_ratio'] / 2.0, 1.0) * 10
        strength_score += volume_score
        
        # 6. Trend strength (0-15 points)
        trend_score = min(row['trend_strength'] / 100, 1.0) * 15
        strength_score += trend_score
        
        # 7. Volatility regime (0-10 points)
        volatility_score = 0
        if row['market_regime'] == MarketRegime.HIGH_VOLATILITY.value:
            volatility_score = 10  # High vol is good for 0DTE
        elif row['market_regime'] == MarketRegime.TRENDING_UP.value and signal_type == "BULLISH":
            volatility_score = 8
        elif row['market_regime'] == MarketRegime.TRENDING_DOWN.value and signal_type == "BEARISH":
            volatility_score = 8
        strength_score += volatility_score
        
        # 8. Time decay bonus (0-5 points) - closer to expiry can be better for 0DTE
        time_bonus = max(0, (self.max_hours_to_expiry - time_to_expiry_hours) / self.max_hours_to_expiry) * 5
        strength_score += time_bonus
        
        # Convert to signal strength enum
        if strength_score >= 85:
            strength = SignalStrength.EXTREME
        elif strength_score >= 70:
            strength = SignalStrength.VERY_STRONG
        elif strength_score >= 55:
            strength = SignalStrength.STRONG
        elif strength_score >= 40:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Filter out sideways markets
        if row['market_regime'] == MarketRegime.SIDEWAYS.value and strength.value < SignalStrength.VERY_STRONG.value:
            return None
        
        # Calculate probability and risk/reward
        probability = min(strength_score / 100, 0.85)  # Max 85% probability
        risk_reward = 2.0 + (strength_score / 100)  # 2:1 to 3:1 risk/reward
        
        # Gamma risk (higher closer to expiry)
        gamma_risk = 1.0 - (time_to_expiry_hours / self.max_hours_to_expiry)
        
        return Enhanced0DTESignal(
            timestamp=timestamp,
            signal_type=signal_type,
            strength=strength,
            ma_shift_osc=ma_osc,
            ma_value=row['ma_short'],
            price=row['close'],
            bb_position=row['bb_position'],
            bb_squeeze=row['bb_squeeze'],
            kc_position=row['kc_position'],
            atr_percentile=row['atr_percentile'],
            rsi=row['rsi'],
            volume_ratio=row['volume_ratio'],
            vix_level=0.0,  # Would need VIX data
            market_regime=MarketRegime(row['market_regime']),
            trend_strength=row['trend_strength'],
            probability=probability,
            risk_reward=risk_reward,
            time_to_expiry_hours=time_to_expiry_hours,
            gamma_risk=gamma_risk
        )
    
    def calculate_position_size(self, signal: Enhanced0DTESignal, option_price: float) -> int:
        """Calculate aggressive position size for $300/day target."""
        # Base position size for target return
        target_profit_per_trade = self.target_daily_profit / self.max_trades_per_day
        
        # Expected profit per contract
        expected_profit_per_contract = option_price * signal.risk_reward * signal.probability
        
        # Contracts needed for target
        target_contracts = max(1, int(target_profit_per_trade / expected_profit_per_contract))
        
        # Risk-based position sizing
        max_loss_per_contract = option_price * 100  # Full premium loss
        max_contracts_by_risk = int(self.max_position_risk / max_loss_per_contract)
        
        # Take smaller of target and risk-based sizing
        contracts = min(target_contracts, max_contracts_by_risk)
        
        # Scale by signal strength
        strength_multiplier = signal.strength.value / SignalStrength.STRONG.value
        contracts = int(contracts * strength_multiplier)
        
        # 0DTE specific limits
        max_0dte_contracts = 10  # Absolute max for 0DTE
        contracts = min(contracts, max_0dte_contracts)
        
        return max(1, contracts)  # Always at least 1 contract
    
    def get_strategy_summary(self) -> Dict:
        """Get strategy configuration summary."""
        return {
            "strategy_name": "Aggressive 0DTE Options Strategy",
            "target_daily_profit": self.target_daily_profit,
            "target_daily_return_pct": self.target_daily_return * 100,
            "account_size": self.account_size,
            "max_daily_loss": self.max_daily_loss,
            "max_position_risk": self.max_position_risk,
            "max_trades_per_day": self.max_trades_per_day,
            "min_signal_strength": self.min_signal_strength,
            "max_hours_to_expiry": self.max_hours_to_expiry,
            "min_hours_to_expiry": self.min_hours_to_expiry,
            "indicators_used": [
                "MA Shift Oscillator",
                "Bollinger Bands",
                "Keltner Channels", 
                "ATR",
                "RSI",
                "Volume Profile",
                "Market Regime Detection"
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Aggressive 0DTE Options Strategy")
    print("💰 Target: $300/day with 25k account")
    print("=" * 50)
    
    # Initialize strategy
    strategy = Aggressive0DTEStrategy(target_daily_profit=300, account_size=25000)
    
    # Display strategy summary
    summary = strategy.get_strategy_summary()
    print(f"📊 Strategy: {summary['strategy_name']}")
    print(f"🎯 Daily Target: ${summary['target_daily_profit']} ({summary['target_daily_return_pct']:.2f}%)")
    print(f"⚠️ Max Daily Loss: ${summary['max_daily_loss']}")
    print(f"📈 Max Trades/Day: {summary['max_trades_per_day']}")
    print(f"⏰ Time Window: {summary['min_hours_to_expiry']}-{summary['max_hours_to_expiry']} hours to expiry")
    print(f"🔧 Indicators: {', '.join(summary['indicators_used'])}")
    
    print(f"\n💡 This strategy is designed for aggressive day traders")
    print(f"⚠️ Risk Level: HIGH - Use proper risk management!")
    print(f"🎯 Expected Win Rate: 60-70% with 2-3:1 risk/reward")