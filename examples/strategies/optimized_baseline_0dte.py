#!/usr/bin/env python3
"""
Optimized Baseline 0DTE Strategy - Smart Exit Management

Takes the PROVEN baseline logic (57% win rate) and optimizes it with:
1. NO EXPIRY EXITS - Close all positions 30 min before market close
2. Smart profit taking - Scale out of winners
3. Tight stop losses - Cut losses quickly  
4. Conservative scaling - 1.2x to 1.8x gradual scaling
5. Better timing - Avoid choppy periods

Focus: Optimize the 57% win rate strategy rather than over-scale it.

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

# Import base components
import sys
import os
sys.path.append(os.path.dirname(__file__))
from aggressive_0dte_strategy import SignalStrength, MarketRegime


class ExitTiming(Enum):
    """Exit timing strategies."""
    QUICK_PROFIT = "quick_profit"      # Take profits in 30-60 min
    SCALE_OUT = "scale_out"           # Scale out of winners
    TIGHT_STOP = "tight_stop"         # 30-40% stop loss
    TIME_STOP = "time_stop"           # Close before expiry
    PROFIT_TARGET = "profit_target"    # Hit profit target


class OptimizationMode(Enum):
    """Optimization focus areas."""
    WIN_RATE = "win_rate"             # Focus on higher win rate
    PROFIT_FACTOR = "profit_factor"   # Focus on profit per trade
    TRADE_FREQUENCY = "frequency"     # Focus on more trades
    RISK_ADJUSTED = "risk_adjusted"   # Focus on risk-adjusted returns


@dataclass
class OptimizedSignal:
    """Enhanced signal with optimization metrics."""
    timestamp: datetime
    signal_type: str
    strength: SignalStrength
    price: float
    ma_value: float
    momentum: float
    volume_surge: bool
    
    # Timing analysis
    time_to_expiry_hours: float
    market_session: str  # "opening", "midday", "afternoon", "power"
    avoid_chop: bool
    
    # Optimization factors
    win_probability: float
    expected_return: float
    risk_score: float
    optimization_score: float


class OptimizedBaselineStrategy:
    """Optimized version of the proven baseline strategy."""
    
    def __init__(self, 
                 target_daily_profit: float = 250,  # Scaled target
                 account_size: float = 25000,
                 optimization_mode: OptimizationMode = OptimizationMode.WIN_RATE):
        
        self.target_daily_profit = target_daily_profit
        self.account_size = account_size
        self.optimization_mode = optimization_mode
        
        # Proven baseline parameters (keep what works!)
        self.ma_length = 20
        self.momentum_length = 5
        self.momentum_threshold = 1.0
        
        # OPTIMIZED exit management (this is the key!)
        self.profit_target_1 = 1.4    # First target: 40% profit
        self.profit_target_2 = 1.8    # Second target: 80% profit  
        self.stop_loss = 0.65         # Stop loss: 35% loss
        self.time_exit_minutes = 30   # Close 30 min before expiry
        
        # Conservative scaling for $250/day target
        self.position_scale_factor = 1.21  # Scale for $250/day (was $207)
        self.max_position_percent = 0.10   # 10% max per trade (slight increase)
        
        # Timing optimization
        self.avoid_lunch_chop = True      # Skip 12:00-13:30
        self.prefer_momentum_sessions = True  # Focus on 9:30-11:00, 14:00-15:30
        
        # Risk management
        self.max_daily_loss = account_size * 0.02  # 2% daily stop
        self.max_trades_per_day = 8
        
        # Performance tracking
        self.daily_pnl = 0
        self.trades_today = 0
        self.current_day = None
        
    def calculate_optimized_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators with optimization enhancements."""
        df = df.copy()
        
        # Core indicators (keep proven logic)
        df['hl2'] = (df['high'] + df['low']) / 2
        df['ma'] = df['hl2'].rolling(self.ma_length).mean()
        df['momentum'] = df['hl2'].rolling(self.momentum_length).mean() - df['ma']
        
        # Volume analysis (enhanced)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_surge'] = df['volume'] > (df['volume_ma'] * 1.3)  # Stronger requirement
        df['volume_percentile'] = df['volume'].rolling(50).rank(pct=True)
        
        # Market session analysis
        df['market_session'] = df.index.to_series().apply(self._get_market_session)
        df['avoid_chop'] = df['market_session'] == 'midday'
        
        # Momentum quality
        df['momentum_consistency'] = df['momentum'].rolling(3).std()
        df['trend_strength'] = abs(df['hl2'] - df['ma']) / df['ma']
        
        return df
    
    def _get_market_session(self, timestamp: pd.Timestamp) -> str:
        """Identify market session for timing optimization."""
        hour = timestamp.hour
        minute = timestamp.minute
        time_decimal = hour + minute / 60
        
        if 9.5 <= time_decimal < 11.0:
            return "opening"    # High momentum period
        elif 11.0 <= time_decimal < 13.5:
            return "midday"     # Choppy period  
        elif 13.5 <= time_decimal < 15.5:
            return "afternoon"  # Trend continuation
        elif 15.5 <= time_decimal < 16.0:
            return "power"      # Power hour
        else:
            return "closed"
    
    def generate_optimized_signals(self, df: pd.DataFrame) -> List[OptimizedSignal]:
        """Generate signals with optimization logic."""
        
        df = self.calculate_optimized_indicators(df)
        signals = []
        
        for i in range(len(df)):
            if i < self.ma_length + 10:
                continue
            
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Skip non-market hours
            if row['market_session'] == 'closed':
                continue
            
            # Skip choppy periods if optimization enabled
            if self.avoid_lunch_chop and row['avoid_chop']:
                continue
            
            # Calculate time to expiry
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            
            expiry_time = timestamp.replace(hour=15, minute=30)  # Stop trading 30 min early!
            if timestamp.hour >= 15 and timestamp.minute >= 30:
                continue  # Too close to expiry
                
            time_to_expiry_hours = (expiry_time - timestamp).total_seconds() / 3600
            
            # Skip if too little time (need at least 1 hour)
            if time_to_expiry_hours < 1.0:
                continue
            
            # Generate signal using PROVEN logic
            momentum = row['momentum']
            signal_type = "NEUTRAL"
            
            if momentum > self.momentum_threshold:
                signal_type = "BULLISH"
            elif momentum < -self.momentum_threshold:
                signal_type = "BEARISH"
            
            if signal_type == "NEUTRAL":
                continue
            
            # Enhanced strength calculation
            strength = self._calculate_enhanced_strength(row, time_to_expiry_hours)
            
            # Only trade moderate+ signals
            if strength.value < SignalStrength.MODERATE.value:
                continue
            
            # Calculate optimization metrics
            win_prob = self._estimate_win_probability(row, strength, time_to_expiry_hours)
            expected_return = self._estimate_expected_return(row, strength)
            risk_score = self._calculate_risk_score(row, time_to_expiry_hours)
            optimization_score = self._calculate_optimization_score(win_prob, expected_return, risk_score)
            
            signal = OptimizedSignal(
                timestamp=timestamp,
                signal_type=signal_type,
                strength=strength,
                price=row['hl2'],
                ma_value=row['ma'],
                momentum=momentum,
                volume_surge=row['volume_surge'],
                time_to_expiry_hours=time_to_expiry_hours,
                market_session=row['market_session'],
                avoid_chop=row['avoid_chop'],
                win_probability=win_prob,
                expected_return=expected_return,
                risk_score=risk_score,
                optimization_score=optimization_score
            )
            
            signals.append(signal)
        
        # Sort by optimization score (best first)
        signals.sort(key=lambda x: x.optimization_score, reverse=True)
        
        return signals
    
    def _calculate_enhanced_strength(self, row: pd.Series, time_to_expiry: float) -> SignalStrength:
        """Calculate signal strength with enhancements."""
        
        base_momentum = min(abs(row['momentum']) / 3.0, 1.0)
        
        # Volume confirmation
        volume_boost = 0
        if row['volume_surge']:
            volume_boost += 0.2
        if row['volume_percentile'] > 0.8:
            volume_boost += 0.15
        
        # Momentum consistency
        consistency_boost = 0
        if row['momentum_consistency'] < 1.0:  # Low volatility = good
            consistency_boost = 0.1
        
        # Time factor (more time = higher confidence)
        time_boost = min(time_to_expiry / 4, 0.15)
        
        # Market session bonus
        session_boost = 0
        if row['market_session'] in ['opening', 'afternoon']:
            session_boost = 0.1
        elif row['market_session'] == 'power':
            session_boost = 0.05
        
        total_strength = base_momentum + volume_boost + consistency_boost + time_boost + session_boost
        
        # Convert to enum
        if total_strength >= 1.2:
            return SignalStrength.VERY_STRONG
        elif total_strength >= 0.9:
            return SignalStrength.STRONG
        elif total_strength >= 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _estimate_win_probability(self, row: pd.Series, strength: SignalStrength, time_to_expiry: float) -> float:
        """Estimate win probability with optimization factors."""
        
        # Base probability from proven baseline (57%)
        base_prob = 0.57
        
        # Strength adjustment
        strength_multiplier = {
            SignalStrength.MODERATE: 0.95,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.12,
            SignalStrength.EXTREME: 1.25
        }
        
        prob = base_prob * strength_multiplier.get(strength, 1.0)
        
        # Volume confirmation bonus
        if row['volume_surge']:
            prob *= 1.05
        
        # Time factor (more time = higher probability)
        if time_to_expiry > 3:
            prob *= 1.08
        elif time_to_expiry > 2:
            prob *= 1.04
        
        # Market session adjustment
        session_multiplier = {
            'opening': 1.05,
            'midday': 0.90,
            'afternoon': 1.03,
            'power': 1.02
        }
        prob *= session_multiplier.get(row['market_session'], 1.0)
        
        return min(prob, 0.80)  # Cap at 80%
    
    def _estimate_expected_return(self, row: pd.Series, strength: SignalStrength) -> float:
        """Estimate expected return per trade."""
        
        # Base expected return (from baseline performance)
        base_return = 0.15  # 15% expected return
        
        # Adjust for strength
        strength_multiplier = {
            SignalStrength.MODERATE: 0.8,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.EXTREME: 1.6
        }
        
        return base_return * strength_multiplier.get(strength, 1.0)
    
    def _calculate_risk_score(self, row: pd.Series, time_to_expiry: float) -> float:
        """Calculate risk score (lower = better)."""
        
        risk = 0.3  # Base risk
        
        # Time risk (less time = more risk)
        if time_to_expiry < 1.5:
            risk += 0.2
        elif time_to_expiry < 2.5:
            risk += 0.1
        
        # Market session risk
        if row['market_session'] == 'midday':
            risk += 0.15
        elif row['market_session'] == 'power':
            risk += 0.1
        
        # Volume risk
        if not row['volume_surge']:
            risk += 0.1
        
        return min(risk, 0.8)
    
    def _calculate_optimization_score(self, win_prob: float, expected_return: float, risk_score: float) -> float:
        """Calculate composite optimization score."""
        
        if self.optimization_mode == OptimizationMode.WIN_RATE:
            return win_prob * 0.7 + expected_return * 0.2 + (1 - risk_score) * 0.1
        elif self.optimization_mode == OptimizationMode.PROFIT_FACTOR:
            return expected_return * 0.6 + win_prob * 0.3 + (1 - risk_score) * 0.1
        elif self.optimization_mode == OptimizationMode.RISK_ADJUSTED:
            return (win_prob * expected_return) / (risk_score + 0.1)
        else:  # TRADE_FREQUENCY
            return win_prob * 0.5 + expected_return * 0.3 + (1 - risk_score) * 0.2
    
    def calculate_optimized_position_size(self, signal: OptimizedSignal, option_price: float, current_cash: float) -> int:
        """Calculate position size with optimization."""
        
        # Base position (conservative)
        base_position_value = current_cash * 0.05
        
        # Scale based on optimization score
        scale_factor = self.position_scale_factor * signal.optimization_score
        scaled_position_value = base_position_value * scale_factor
        
        # Cap at maximum
        max_position_value = current_cash * self.max_position_percent
        final_position_value = min(scaled_position_value, max_position_value)
        
        # Calculate contracts
        contracts = max(1, int(final_position_value / (option_price * 100)))
        
        # Conservative limits
        return min(contracts, 8)
    
    def should_trade_optimized(self, signal: OptimizedSignal) -> bool:
        """Enhanced trade filtering."""
        
        # Daily management
        current_day = signal.timestamp.date()
        if self.current_day != current_day:
            self.current_day = current_day
            self.daily_pnl = 0
            self.trades_today = 0
        
        # Daily limits
        if self.trades_today >= self.max_trades_per_day:
            return False
        
        if self.daily_pnl <= -self.max_daily_loss:
            return False
        
        # Optimization score threshold (LOWERED for testing)
        min_scores = {
            OptimizationMode.WIN_RATE: 0.50,        # Lowered from 0.65
            OptimizationMode.PROFIT_FACTOR: 0.45,   # Lowered from 0.60
            OptimizationMode.TRADE_FREQUENCY: 0.40, # Lowered from 0.55
            OptimizationMode.RISK_ADJUSTED: 0.55    # Lowered from 0.70
        }
        
        if signal.optimization_score < min_scores[self.optimization_mode]:
            return False
        
        return True
    
    def get_exit_strategy(self, signal: OptimizedSignal, entry_price: float, current_time: datetime) -> Dict:
        """Get optimized exit strategy."""
        
        time_to_expiry = (signal.timestamp.replace(hour=16, minute=0) - current_time).total_seconds() / 3600
        
        # NEVER let expire! Close 30 minutes before market close
        max_exit_time = signal.timestamp.replace(hour=15, minute=30)
        
        return {
            'profit_target_1': entry_price * self.profit_target_1,
            'profit_target_2': entry_price * self.profit_target_2,
            'stop_loss': entry_price * self.stop_loss,
            'time_exit': max_exit_time,
            'scale_out': signal.strength == SignalStrength.VERY_STRONG
        }
    
    def get_strategy_summary(self) -> Dict:
        """Get optimized strategy summary."""
        return {
            "strategy_name": "Optimized Baseline 0DTE Strategy",
            "optimization_mode": self.optimization_mode.value,
            "target_daily_profit": self.target_daily_profit,
            "position_scale_factor": self.position_scale_factor,
            "max_position_percent": f"{self.max_position_percent*100:.1f}%",
            "key_optimizations": [
                "NO EXPIRY EXITS - Close 30 min before market close",
                "Smart profit taking - Scale out of winners", 
                "Tight stop losses - 35% stop loss",
                "Timing optimization - Avoid choppy periods",
                "Enhanced signal strength calculation",
                "Conservative position scaling"
            ],
            "exit_management": {
                "profit_target_1": f"{(self.profit_target_1-1)*100:.0f}%",
                "profit_target_2": f"{(self.profit_target_2-1)*100:.0f}%", 
                "stop_loss": f"{(1-self.stop_loss)*100:.0f}%",
                "time_exit": f"{self.time_exit_minutes} min before close"
            },
            "baseline_improvements": [
                "Eliminate 49% expiry losses",
                "Better timing (avoid lunch chop)",
                "Enhanced volume analysis", 
                "Market session optimization",
                "Conservative scaling approach"
            ]
        }


# Example usage
if __name__ == "__main__":
    print("✅ OPTIMIZED BASELINE 0DTE STRATEGY")
    print("🎯 Fix Exit Management + Smart Scaling")
    print("=" * 50)
    
    strategy = OptimizedBaselineStrategy()
    summary = strategy.get_strategy_summary()
    
    print(f"📊 Strategy: {summary['strategy_name']}")
    print(f"🎯 Target: ${summary['target_daily_profit']}/day")
    print(f"⚡ Optimization: {summary['optimization_mode']}")
    
    print(f"\n🔧 Key Optimizations:")
    for i, opt in enumerate(summary['key_optimizations'], 1):
        print(f"   {i}. {opt}")
    
    print(f"\n🚪 Exit Management:")
    exit_mgmt = summary['exit_management']
    print(f"   🎯 Profit Target 1: {exit_mgmt['profit_target_1']}")
    print(f"   🎯 Profit Target 2: {exit_mgmt['profit_target_2']}")
    print(f"   🛑 Stop Loss: {exit_mgmt['stop_loss']}")
    print(f"   ⏰ Time Exit: {exit_mgmt['time_exit']}")
    
    print(f"\n📈 Expected Improvements:")
    for improvement in summary['baseline_improvements']:
        print(f"   ✅ {improvement}")
    
    print(f"\n💡 The key insight: Fix exit management first, then scale!")
    print(f"🎯 No more 49% expiry losses = Major profit improvement")