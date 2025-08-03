#!/usr/bin/env python3
"""
MA Shift Multi-Indicator Options Strategy - Alpaca Improved

This strategy combines the Moving Average Shift oscillator with Keltner Channels,
Bollinger Bands, and ATR filtering to create a powerful options trading system.

Based on ChartPrime's Moving Average Shift indicator, enhanced with:
- Keltner Channels for trend confirmation
- Bollinger Bands for volatility-based entries
- ATR filtering to avoid sideways markets
- ML-ready feature engineering for future enhancement

Author: Alpaca Improved Team
License: MIT
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpaca.data.timeframe import TimeFrame
from strategies.base.options_strategy import BaseOptionsStrategy, OptionsStrategyConfig
from strategies.signals.technical_signals import TechnicalSignalGenerator
from data.extractors.alpaca_extractor import AlpacaDataExtractor
from data.extractors.options_chain_extractor import OptionsChainExtractor, OptionType
from utils.logger import get_logger


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class MAShiftSignal:
    """Moving Average Shift signal data."""
    timestamp: datetime
    signal_type: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: SignalStrength
    ma_shift_osc: float
    ma_value: float
    price: float
    keltner_position: float  # -1 to 1 (below to above)
    bb_position: float      # -1 to 1 (below to above)
    atr_normalized: float   # ATR / Price for volatility measure
    trend_confidence: float # 0 to 1
    volatility_regime: str  # 'LOW', 'NORMAL', 'HIGH'


class MAShiftIndicators:
    """
    Multi-indicator system combining MA Shift with Keltner, Bollinger, and ATR.
    
    This class replicates the Pine Script logic and adds complementary indicators
    for robust signal generation and market regime detection.
    """
    
    def __init__(
        self,
        ma_length: int = 40,
        ma_type: str = "SMA",
        osc_length: int = 15,
        osc_threshold: float = 0.5,
        keltner_length: int = 20,
        keltner_multiplier: float = 2.0,
        bb_length: int = 20,
        bb_std: float = 2.0,
        atr_length: int = 14
    ):
        """
        Initialize the multi-indicator system.
        
        Args:
            ma_length: Moving average length
            ma_type: Type of moving average ('SMA', 'EMA', 'WMA')
            osc_length: Oscillator smoothing length
            osc_threshold: Oscillator signal threshold
            keltner_length: Keltner Channels length
            keltner_multiplier: Keltner Channels ATR multiplier
            bb_length: Bollinger Bands length
            bb_std: Bollinger Bands standard deviation
            atr_length: ATR calculation length
        """
        self.ma_length = ma_length
        self.ma_type = ma_type
        self.osc_length = osc_length
        self.osc_threshold = osc_threshold
        self.keltner_length = keltner_length
        self.keltner_multiplier = keltner_multiplier
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.atr_length = atr_length
        
        self.logger = get_logger(self.__class__.__name__)
    
    def calculate_moving_average(self, data: pd.Series, length: int, ma_type: str) -> pd.Series:
        """Calculate moving average based on type."""
        if ma_type == "SMA":
            return data.rolling(window=length).mean()
        elif ma_type == "EMA":
            return data.ewm(span=length).mean()
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return data.rolling(window=length).apply(
                lambda x: np.average(x, weights=weights), raw=True
            )
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
    
    def calculate_hull_ma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        
        wma_half = self.calculate_moving_average(data, half_length, "WMA")
        wma_full = self.calculate_moving_average(data, length, "WMA")
        
        hull_data = 2 * wma_half - wma_full
        return self.calculate_moving_average(hull_data, sqrt_length, "WMA")
    
    def calculate_percentile_rank(self, data: pd.Series, window: int = 1000) -> pd.Series:
        """Calculate percentile rank over rolling window."""
        return data.rolling(window=min(window, len(data))).rank(pct=True) * 100
    
    def calculate_ma_shift_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Moving Average Shift oscillator (Pine Script translation).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with MA Shift oscillator and related signals
        """
        # Calculate HL2 (typical price)
        df['hl2'] = (df['high'] + df['low']) / 2
        
        # Calculate Moving Average
        df['ma'] = self.calculate_moving_average(df['hl2'], self.ma_length, self.ma_type)
        
        # Calculate difference and percentile rank
        df['diff'] = df['hl2'] - df['ma']
        df['perc_r'] = self.calculate_percentile_rank(df['diff'], 1000)
        
        # Calculate oscillator using Hull MA of change
        df['diff_normalized'] = df['diff'] / (df['perc_r'] / 100 + 1e-10)  # Avoid division by zero
        df['diff_change'] = df['diff_normalized'].diff(self.osc_length)
        df['ma_shift_osc'] = self.calculate_hull_ma(df['diff_change'], 10)
        
        # Generate signals
        df['osc_prev2'] = df['ma_shift_osc'].shift(2)
        df['signal_up'] = (
            (df['ma_shift_osc'] > df['osc_prev2']) & 
            (df['ma_shift_osc'] < -self.osc_threshold)
        )
        df['signal_dn'] = (
            (df['ma_shift_osc'] < df['osc_prev2']) & 
            (df['ma_shift_osc'] > self.osc_threshold)
        )
        
        return df
    
    def calculate_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels."""
        # Calculate ATR for Keltner
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.keltner_length).mean()
        
        # Keltner Channels
        df['keltner_mid'] = self.calculate_moving_average(df['close'], self.keltner_length, "EMA")
        df['keltner_upper'] = df['keltner_mid'] + (self.keltner_multiplier * df['atr'])
        df['keltner_lower'] = df['keltner_mid'] - (self.keltner_multiplier * df['atr'])
        
        # Position within Keltner Channels (-1 to 1)
        df['keltner_position'] = (
            (df['close'] - df['keltner_mid']) / 
            (df['keltner_upper'] - df['keltner_mid'])
        ).clip(-1, 1)
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df['bb_mid'] = df['close'].rolling(window=self.bb_length).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_length).std()
        df['bb_upper'] = df['bb_mid'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (self.bb_std * df['bb_std'])
        
        # Position within Bollinger Bands (-1 to 1)
        df['bb_position'] = (
            (df['close'] - df['bb_mid']) / 
            (df['bb_upper'] - df['bb_mid'])
        ).clip(-1, 1)
        
        # Bollinger Band squeeze detection
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)
        
        return df
    
    def calculate_atr_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-based market regime filter."""
        # ATR for volatility measurement
        df['atr_volatility'] = df['tr'].rolling(window=self.atr_length).mean()
        df['atr_normalized'] = df['atr_volatility'] / df['close']
        
        # Volatility regime classification
        atr_percentile = df['atr_normalized'].rolling(window=100).rank(pct=True)
        df['volatility_regime'] = pd.cut(
            atr_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['LOW', 'NORMAL', 'HIGH']
        )
        
        # Trend strength using ADX-like calculation
        dm_plus = np.where(df['high'].diff() > df['low'].diff().abs(), df['high'].diff(), 0)
        dm_minus = np.where(df['low'].diff().abs() > df['high'].diff(), df['low'].diff().abs(), 0)
        
        df['dm_plus'] = pd.Series(dm_plus).rolling(14).mean()
        df['dm_minus'] = pd.Series(dm_minus).rolling(14).mean()
        df['trend_strength'] = abs(df['dm_plus'] - df['dm_minus']) / (df['dm_plus'] + df['dm_minus'] + 1e-10)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[MAShiftSignal]:
        """
        Generate comprehensive trading signals combining all indicators.
        
        Args:
            df: DataFrame with all calculated indicators
            
        Returns:
            List of MAShiftSignal objects
        """
        signals = []
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['ma_shift_osc']):
                continue
            
            row = df.iloc[i]
            timestamp = df.index[i] if hasattr(df.index[i], 'to_pydatetime') else row.name
            
            # Determine signal type and strength
            signal_type = "NEUTRAL"
            strength = SignalStrength.WEAK
            
            # MA Shift primary signals
            if row['signal_up']:
                signal_type = "BULLISH"
                strength = self._calculate_signal_strength(row, "BULLISH")
            elif row['signal_dn']:
                signal_type = "BEARISH"
                strength = self._calculate_signal_strength(row, "BEARISH")
            
            # Create signal object
            signal = MAShiftSignal(
                timestamp=timestamp,
                signal_type=signal_type,
                strength=strength,
                ma_shift_osc=row['ma_shift_osc'],
                ma_value=row['ma'],
                price=row['close'],
                keltner_position=row['keltner_position'],
                bb_position=row['bb_position'],
                atr_normalized=row['atr_normalized'],
                trend_confidence=row['trend_strength'],
                volatility_regime=row['volatility_regime']
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_signal_strength(self, row: pd.Series, signal_type: str) -> SignalStrength:
        """Calculate signal strength based on multiple factors."""
        strength_score = 0
        
        # MA Shift oscillator magnitude
        osc_magnitude = abs(row['ma_shift_osc'])
        if osc_magnitude > self.osc_threshold * 2:
            strength_score += 2
        elif osc_magnitude > self.osc_threshold:
            strength_score += 1
        
        # Keltner Channel position confirmation
        if signal_type == "BULLISH" and row['keltner_position'] > 0:
            strength_score += 1
        elif signal_type == "BEARISH" and row['keltner_position'] < 0:
            strength_score += 1
        
        # Bollinger Band position confirmation
        if signal_type == "BULLISH" and row['bb_position'] < -0.5:  # Oversold
            strength_score += 1
        elif signal_type == "BEARISH" and row['bb_position'] > 0.5:  # Overbought
            strength_score += 1
        
        # Trend strength
        if row['trend_strength'] > 0.3:
            strength_score += 1
        
        # Volatility regime (avoid low volatility/sideways markets)
        if row['volatility_regime'] in ['NORMAL', 'HIGH']:
            strength_score += 1
        
        # Convert to enum
        if strength_score >= 5:
            return SignalStrength.VERY_STRONG
        elif strength_score >= 3:
            return SignalStrength.STRONG
        elif strength_score >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


class MAShiftOptionsStrategy(BaseOptionsStrategy):
    """
    MA Shift Multi-Indicator Options Strategy.
    
    This strategy uses the Moving Average Shift oscillator combined with
    Keltner Channels, Bollinger Bands, and ATR filtering to trade SPY options.
    
    Strategy Logic:
    1. Use MA Shift oscillator for primary signals
    2. Confirm with Keltner Channels for trend
    3. Use Bollinger Bands for volatility-based entries
    4. Filter out low-volatility/sideways markets with ATR
    5. Generate ML-ready features for future enhancement
    """
    
    def __init__(self, config: OptionsStrategyConfig):
        """Initialize the MA Shift options strategy."""
        super().__init__(config)
        
        # Strategy-specific parameters
        self.indicators = MAShiftIndicators(
            ma_length=config.parameters.get('ma_length', 40),
            ma_type=config.parameters.get('ma_type', 'SMA'),
            osc_length=config.parameters.get('osc_length', 15),
            osc_threshold=config.parameters.get('osc_threshold', 0.5),
            keltner_length=config.parameters.get('keltner_length', 20),
            keltner_multiplier=config.parameters.get('keltner_multiplier', 2.0),
            bb_length=config.parameters.get('bb_length', 20),
            bb_std=config.parameters.get('bb_std', 2.0),
            atr_length=config.parameters.get('atr_length', 14)
        )
        
        # Minimum signal strength for trading
        self.min_signal_strength = SignalStrength(
            config.parameters.get('min_signal_strength', 2)
        )
        
        # Options selection parameters
        self.target_dte = config.parameters.get('target_dte', 30)  # Days to expiration
        self.target_delta = config.parameters.get('target_delta', 0.3)  # Target delta
        self.max_position_size = config.parameters.get('max_position_size', 0.1)  # 10% of portfolio
        
        self.logger.info(f"MA Shift Options Strategy initialized with {config.symbol}")
    
    def analyze_market_data(self, data: pd.DataFrame) -> List[MAShiftSignal]:
        """
        Analyze market data and generate signals.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            List of trading signals
        """
        try:
            # Calculate all indicators
            data_with_indicators = self.indicators.calculate_ma_shift_oscillator(data.copy())
            data_with_indicators = self.indicators.calculate_keltner_channels(data_with_indicators)
            data_with_indicators = self.indicators.calculate_bollinger_bands(data_with_indicators)
            data_with_indicators = self.indicators.calculate_atr_filter(data_with_indicators)
            
            # Generate signals
            signals = self.indicators.generate_signals(data_with_indicators)
            
            # Filter by minimum strength
            filtered_signals = [
                signal for signal in signals 
                if signal.strength.value >= self.min_signal_strength.value
            ]
            
            self.logger.info(f"Generated {len(filtered_signals)} signals from {len(signals)} total")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data: {e}")
            return []
    
    def generate_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML-ready feature vectors for future enhancement.
        
        This prepares the data for Phase 6 ML integration by creating
        a comprehensive feature set from all indicators.
        
        Args:
            data: OHLCV DataFrame with indicators
            
        Returns:
            DataFrame with ML features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['price'] = data['close']
        features['price_change'] = data['close'].pct_change()
        features['price_volatility'] = data['close'].pct_change().rolling(20).std()
        
        # MA Shift features
        features['ma_shift_osc'] = data['ma_shift_osc']
        features['ma_shift_osc_momentum'] = data['ma_shift_osc'].diff()
        features['ma_distance'] = (data['close'] - data['ma']) / data['ma']
        
        # Keltner Channel features
        features['keltner_position'] = data['keltner_position']
        features['keltner_width'] = (data['keltner_upper'] - data['keltner_lower']) / data['keltner_mid']
        features['keltner_squeeze'] = (features['keltner_width'] < features['keltner_width'].rolling(20).quantile(0.2)).astype(int)
        
        # Bollinger Band features
        features['bb_position'] = data['bb_position']
        features['bb_width'] = data['bb_width']
        features['bb_squeeze'] = data['bb_squeeze'].astype(int)
        
        # ATR and volatility features
        features['atr_normalized'] = data['atr_normalized']
        features['volatility_regime'] = pd.Categorical(data['volatility_regime']).codes
        features['trend_strength'] = data['trend_strength']
        
        # Volume features
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
        
        # Time-based features
        features['hour'] = data.index.hour if hasattr(data.index, 'hour') else 10
        features['day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 1
        
        return features.dropna()
    
    def select_options_contracts(
        self, 
        signal: MAShiftSignal, 
        options_chain: List
    ) -> Optional[dict]:
        """
        Select appropriate options contracts based on signal.
        
        Args:
            signal: Trading signal
            options_chain: Available options contracts
            
        Returns:
            Selected contract details or None
        """
        if not options_chain:
            return None
        
        try:
            # Filter by expiration (target DTE)
            target_expiration = signal.timestamp + timedelta(days=self.target_dte)
            suitable_contracts = [
                contract for contract in options_chain
                if abs((contract.expiration_date - target_expiration).days) <= 7
            ]
            
            if not suitable_contracts:
                return None
            
            # Select contract type based on signal
            if signal.signal_type == "BULLISH":
                # Look for call options
                calls = [c for c in suitable_contracts if c.option_type == OptionType.CALL]
                if calls:
                    # Find contract closest to target delta
                    best_contract = min(
                        calls, 
                        key=lambda x: abs((x.delta or 0) - self.target_delta)
                    )
                    return {
                        'contract': best_contract,
                        'action': 'BUY',
                        'strategy_type': 'LONG_CALL',
                        'signal_strength': signal.strength.value
                    }
            
            elif signal.signal_type == "BEARISH":
                # Look for put options
                puts = [c for c in suitable_contracts if c.option_type == OptionType.PUT]
                if puts:
                    # Find contract closest to target delta (negative for puts)
                    best_contract = min(
                        puts, 
                        key=lambda x: abs((x.delta or 0) + self.target_delta)
                    )
                    return {
                        'contract': best_contract,
                        'action': 'BUY',
                        'strategy_type': 'LONG_PUT',
                        'signal_strength': signal.strength.value
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error selecting options contracts: {e}")
            return None
    
    def calculate_position_size(self, signal: MAShiftSignal, contract_price: float) -> int:
        """
        Calculate position size based on signal strength and risk management.
        
        Args:
            signal: Trading signal
            contract_price: Option contract price
            
        Returns:
            Number of contracts to trade
        """
        try:
            # Base position size as percentage of portfolio
            base_size = self.max_position_size
            
            # Adjust based on signal strength
            strength_multiplier = {
                SignalStrength.WEAK: 0.5,
                SignalStrength.MODERATE: 0.75,
                SignalStrength.STRONG: 1.0,
                SignalStrength.VERY_STRONG: 1.25
            }
            
            adjusted_size = base_size * strength_multiplier[signal.strength]
            
            # Calculate number of contracts
            portfolio_value = self.get_portfolio_value()  # Implement in base class
            position_value = portfolio_value * adjusted_size
            contracts = max(1, int(position_value / (contract_price * 100)))  # 100 shares per contract
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1


def main():
    """Example usage of the MA Shift Options Strategy."""
    from alpaca.data.timeframe import TimeFrame
    import os
    
    # Load credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    # Initialize data extractors
    data_extractor = AlpacaDataExtractor(api_key, secret_key)
    options_extractor = OptionsChainExtractor(api_key, secret_key)
    
    # Test the strategy with SPY data
    print("🔍 Fetching SPY data for strategy testing...")
    
    # Get historical data
    spy_data = data_extractor.get_bars('SPY', TimeFrame.Day, limit=100)
    if spy_data.empty:
        print("❌ Failed to fetch SPY data")
        return
    
    print(f"✅ Retrieved {len(spy_data)} bars of SPY data")
    
    # Initialize strategy
    config = OptionsStrategyConfig(
        name="MA Shift Multi-Indicator",
        symbol="SPY",
        parameters={
            'ma_length': 40,
            'ma_type': 'SMA',
            'osc_threshold': 0.5,
            'min_signal_strength': 2,
            'target_dte': 30,
            'target_delta': 0.3
        }
    )
    
    strategy = MAShiftOptionsStrategy(config)
    
    # Analyze data and generate signals
    print("📊 Analyzing market data and generating signals...")
    signals = strategy.analyze_market_data(spy_data)
    
    print(f"🎯 Generated {len(signals)} trading signals")
    
    # Display recent signals
    recent_signals = [s for s in signals if s.signal_type != "NEUTRAL"][-5:]
    
    for signal in recent_signals:
        print(f"""
📅 {signal.timestamp.strftime('%Y-%m-%d')}
🎯 Signal: {signal.signal_type} ({signal.strength.name})
📈 Price: ${signal.price:.2f}
📊 MA Shift Osc: {signal.ma_shift_osc:.3f}
🔄 Keltner Pos: {signal.keltner_position:.2f}
📋 BB Position: {signal.bb_position:.2f}
💨 Volatility: {signal.volatility_regime}
        """)
    
    # Generate ML features for future enhancement
    print("🤖 Generating ML features for future enhancement...")
    ml_features = strategy.generate_ml_features(spy_data)
    print(f"✅ Generated {len(ml_features.columns)} ML features")
    print(f"📊 Feature columns: {list(ml_features.columns)}")
    
    print("\n🎉 MA Shift Options Strategy demo completed!")
    print("🚀 Ready for backtesting and paper trading integration!")


if __name__ == "__main__":
    main()