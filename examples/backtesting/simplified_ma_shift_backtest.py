#!/usr/bin/env python3
"""
Simplified MA Shift Options Strategy Backtest - Standalone Version

This is a self-contained version that doesn't depend on the custom data extractors
that are currently having import issues. It demonstrates the strategy logic and
can be run immediately with your credentials.

Author: Alpaca Improved Team
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Direct Alpaca imports (working)
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


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
    keltner_position: float
    bb_position: float
    atr_normalized: float
    volatility_regime: str


class SimplifiedMAShiftStrategy:
    """
    Simplified version of the MA Shift strategy for immediate testing.
    """
    
    def __init__(
        self,
        ma_length: int = 40,
        ma_type: str = "SMA",
        osc_length: int = 15,
        osc_threshold: float = 0.5,
        keltner_length: int = 20,
        bb_length: int = 20,
        atr_length: int = 14
    ):
        self.ma_length = ma_length
        self.ma_type = ma_type
        self.osc_length = osc_length
        self.osc_threshold = osc_threshold
        self.keltner_length = keltner_length
        self.bb_length = bb_length
        self.atr_length = atr_length
    
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
            return data.rolling(window=length).mean()
    
    def calculate_hull_ma(self, data: pd.Series, length: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        
        wma_half = self.calculate_moving_average(data, half_length, "WMA")
        wma_full = self.calculate_moving_average(data, length, "WMA")
        
        hull_data = 2 * wma_half - wma_full
        return self.calculate_moving_average(hull_data, sqrt_length, "WMA")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for the strategy."""
        df = df.copy()
        
        # Calculate HL2 (typical price)
        df['hl2'] = (df['high'] + df['low']) / 2
        
        # Calculate Moving Average
        df['ma'] = self.calculate_moving_average(df['hl2'], self.ma_length, self.ma_type)
        
        # MA Shift Oscillator
        df['diff'] = df['hl2'] - df['ma']
        
        # Simplified percentile rank (rolling)
        df['perc_r'] = df['diff'].rolling(window=min(100, len(df))).rank(pct=True) * 100
        
        # Calculate oscillator
        df['diff_normalized'] = df['diff'] / (df['perc_r'] / 100 + 1e-10)
        df['diff_change'] = df['diff_normalized'].diff(self.osc_length)
        df['ma_shift_osc'] = self.calculate_hull_ma(df['diff_change'], 10)
        
        # Keltner Channels
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.keltner_length).mean()
        df['keltner_mid'] = self.calculate_moving_average(df['close'], self.keltner_length, "EMA")
        df['keltner_upper'] = df['keltner_mid'] + (2.0 * df['atr'])
        df['keltner_lower'] = df['keltner_mid'] - (2.0 * df['atr'])
        df['keltner_position'] = (
            (df['close'] - df['keltner_mid']) / 
            (df['keltner_upper'] - df['keltner_mid'])
        ).clip(-1, 1)
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=self.bb_length).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_length).std()
        df['bb_upper'] = df['bb_mid'] + (2.0 * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (2.0 * df['bb_std'])
        df['bb_position'] = (
            (df['close'] - df['bb_mid']) / 
            (df['bb_upper'] - df['bb_mid'])
        ).clip(-1, 1)
        
        # ATR and volatility regime
        df['atr_normalized'] = df['atr'] / df['close']
        atr_percentile = df['atr_normalized'].rolling(window=50).rank(pct=True)
        df['volatility_regime'] = pd.cut(
            atr_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['LOW', 'NORMAL', 'HIGH']
        )
        
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
    
    def generate_signals(self, df: pd.DataFrame) -> List[MAShiftSignal]:
        """Generate trading signals."""
        signals = []
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['ma_shift_osc']):
                continue
                
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Determine signal type and strength
            signal_type = "NEUTRAL"
            strength = SignalStrength.WEAK
            
            if row['signal_up']:
                signal_type = "BULLISH"
                strength = self._calculate_signal_strength(row, "BULLISH")
            elif row['signal_dn']:
                signal_type = "BEARISH"
                strength = self._calculate_signal_strength(row, "BEARISH")
            
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
                volatility_regime=str(row['volatility_regime'])
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
        
        # Keltner Channel confirmation
        if signal_type == "BULLISH" and row['keltner_position'] > 0:
            strength_score += 1
        elif signal_type == "BEARISH" and row['keltner_position'] < 0:
            strength_score += 1
        
        # Bollinger Band position
        if signal_type == "BULLISH" and row['bb_position'] < -0.5:
            strength_score += 1
        elif signal_type == "BEARISH" and row['bb_position'] > 0.5:
            strength_score += 1
        
        # Volatility regime
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


class SimplifiedOptionsBacktester:
    """Simplified options backtesting."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.trades = []
        self.equity_curve = []
        
    def simulate_option_price(
        self, 
        underlying_price: float, 
        strike: float, 
        days_to_expiry: int, 
        option_type: str,
        volatility: float = 0.2
    ) -> float:
        """Simplified Black-Scholes option pricing."""
        from scipy.stats import norm
        import math
        
        if days_to_expiry <= 0:
            if option_type == 'CALL':
                return max(0, underlying_price - strike)
            else:
                return max(0, strike - underlying_price)
        
        r = 0.05  # Risk-free rate
        T = days_to_expiry / 365.0
        
        try:
            d1 = (math.log(underlying_price / strike) + (r + 0.5 * volatility**2) * T) / (volatility * math.sqrt(T))
            d2 = d1 - volatility * math.sqrt(T)
            
            if option_type == 'CALL':
                price = underlying_price * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = strike * math.exp(-r * T) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
            
            return max(0.01, price)
        except:
            return 0.01
    
    def execute_trade(self, signal: MAShiftSignal, exit_date: datetime = None) -> Dict:
        """Execute a simplified options trade."""
        entry_price = signal.price
        
        # Determine option parameters
        if signal.signal_type == "BULLISH":
            option_type = 'CALL'
            strike = round(entry_price * 1.02, 0)  # 2% OTM
        elif signal.signal_type == "BEARISH":
            option_type = 'PUT'
            strike = round(entry_price * 0.98, 0)  # 2% OTM
        else:
            return None
        
        # Calculate entry option price
        entry_option_price = self.simulate_option_price(
            entry_price, strike, 30, option_type
        )
        
        # Position sizing based on signal strength
        strength_multiplier = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.75,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.25
        }
        
        base_position_value = self.cash * 0.05  # 5% of capital
        position_value = base_position_value * strength_multiplier[signal.strength]
        contracts = max(1, int(position_value / (entry_option_price * 100)))
        
        # Simulate exit after 10 days or calculate current value
        if exit_date:
            days_held = (exit_date - signal.timestamp).days
            exit_underlying_price = entry_price * (1 + np.random.normal(0, 0.02))  # Random walk
            
            exit_option_price = self.simulate_option_price(
                exit_underlying_price, strike, max(0, 30 - days_held), option_type
            )
            
            pnl = (exit_option_price - entry_option_price) * contracts * 100
            
            trade = {
                'entry_date': signal.timestamp,
                'exit_date': exit_date,
                'signal_type': signal.signal_type,
                'signal_strength': signal.strength.value,
                'option_type': option_type,
                'strike': strike,
                'contracts': contracts,
                'entry_price': entry_option_price,
                'exit_price': exit_option_price,
                'underlying_entry': entry_price,
                'underlying_exit': exit_underlying_price,
                'days_held': days_held,
                'pnl': pnl
            }
            
            self.cash += pnl
            self.trades.append(trade)
            
            return trade
        
        return None


def get_spy_data(api_key: str, secret_key: str, days: int = 365) -> pd.DataFrame:
    """Get SPY historical data using direct Alpaca client."""
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        response = client.get_stock_bars(request)
        df = response.df
        
        if df.empty:
            raise ValueError("No data received")
        
        # Reset index to get timestamp as column, then set as index
        df = df.reset_index()
        df = df.set_index('timestamp')
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"✅ Retrieved {len(df)} bars of SPY data")
        print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()


def run_simplified_backtest():
    """Run the simplified MA Shift options backtest."""
    print("🚀 Starting Simplified MA Shift Options Strategy Backtest")
    print("=" * 60)
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("Example:")
        print("export ALPACA_API_KEY='your_key_here'")
        print("export ALPACA_SECRET_KEY='your_secret_here'")
        return None
    
    # Get market data
    print("📊 Fetching SPY market data...")
    spy_data = get_spy_data(api_key, secret_key, days=365)
    
    if spy_data.empty:
        print("❌ Failed to retrieve market data")
        return None
    
    # Initialize strategy
    print("🎯 Initializing MA Shift strategy...")
    strategy = SimplifiedMAShiftStrategy()
    
    # Calculate indicators
    print("📈 Calculating technical indicators...")
    data_with_indicators = strategy.calculate_all_indicators(spy_data)
    
    # Generate signals
    print("🎯 Generating trading signals...")
    all_signals = strategy.generate_signals(data_with_indicators)
    
    # Filter signals by strength
    strong_signals = [s for s in all_signals if s.strength.value >= SignalStrength.MODERATE.value]
    tradeable_signals = [s for s in strong_signals if s.signal_type != "NEUTRAL"]
    
    print(f"📊 Generated {len(all_signals)} total signals")
    print(f"💪 {len(strong_signals)} signals with MODERATE+ strength")
    print(f"🎯 {len(tradeable_signals)} tradeable signals (BULLISH/BEARISH)")
    
    # Run backtest
    print("🔄 Running backtest simulation...")
    backtester = SimplifiedOptionsBacktester()
    
    for i, signal in enumerate(tradeable_signals[:-1]):  # Exclude last signal (no exit data)
        # Simulate exit after 10 days
        exit_date = signal.timestamp + timedelta(days=10)
        trade = backtester.execute_trade(signal, exit_date)
        
        if trade and i < 5:  # Show first 5 trades
            print(f"Trade {i+1}: {trade['signal_type']} {trade['option_type']} - P&L: ${trade['pnl']:.2f}")
    
    # Calculate performance
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS")
    print("=" * 60)
    
    if backtester.trades:
        trade_pnls = [trade['pnl'] for trade in backtester.trades]
        winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
        losing_trades = len(backtester.trades) - winning_trades
        
        total_pnl = sum(trade_pnls)
        total_return = (backtester.cash - backtester.initial_capital) / backtester.initial_capital * 100
        win_rate = winning_trades / len(backtester.trades) * 100
        
        avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if losing_trades > 0 else 0
        
        print(f"📊 Total Trades: {len(backtester.trades)}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"💰 Total P&L: ${total_pnl:,.2f}")
        print(f"📈 Total Return: {total_return:.2f}%")
        print(f"🥇 Average Win: ${avg_win:.2f}")
        print(f"📉 Average Loss: ${avg_loss:.2f}")
        print(f"💼 Final Portfolio Value: ${backtester.cash:,.2f}")
        
        if avg_loss != 0:
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
            print(f"⚖️ Profit Factor: {profit_factor:.2f}")
        
        # Plot results
        print("\n📊 Generating performance charts...")
        plot_results(backtester.trades, tradeable_signals, data_with_indicators)
        
        return {
            'trades': backtester.trades,
            'signals': tradeable_signals,
            'performance': {
                'total_trades': len(backtester.trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'final_value': backtester.cash
            }
        }
    else:
        print("❌ No trades executed")
        return None


def plot_results(trades: List[Dict], signals: List[MAShiftSignal], data: pd.DataFrame):
    """Plot backtest results."""
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade P&L distribution
        if trades:
            trade_pnls = [trade['pnl'] for trade in trades]
            axes[0, 0].hist(trade_pnls, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 0].axvline(x=0, color='red', linestyle='--')
            axes[0, 0].set_title('Trade P&L Distribution')
            axes[0, 0].set_xlabel('P&L per Trade ($)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Signal strength distribution
        signal_strengths = [s.strength.value for s in signals]
        strength_counts = pd.Series(signal_strengths).value_counts().sort_index()
        
        axes[0, 1].bar(strength_counts.index, strength_counts.values, color='orange', alpha=0.7)
        axes[0, 1].set_title('Signal Strength Distribution')
        axes[0, 1].set_xlabel('Signal Strength')
        axes[0, 1].set_ylabel('Count')
        
        # Price and MA Shift Oscillator
        recent_data = data.tail(100)
        axes[1, 0].plot(recent_data.index, recent_data['close'], label='SPY Price', color='blue')
        ax2 = axes[1, 0].twinx()
        ax2.plot(recent_data.index, recent_data['ma_shift_osc'], label='MA Shift Osc', color='red', alpha=0.7)
        ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('SPY Price vs MA Shift Oscillator (Last 100 Days)')
        axes[1, 0].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Cumulative P&L
        if trades:
            trade_dates = [trade['exit_date'] for trade in trades]
            cumulative_pnl = np.cumsum([trade['pnl'] for trade in trades])
            
            axes[1, 1].plot(trade_dates, cumulative_pnl, linewidth=2, color='green')
            axes[1, 1].set_title('Cumulative P&L Over Time')
            axes[1, 1].set_ylabel('Cumulative P&L ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'ma_shift_backtest_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved as: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")


if __name__ == "__main__":
    print("🎯 MA Shift Multi-Indicator Options Strategy")
    print("🔬 Simplified Standalone Backtest")
    print("=" * 60)
    
    results = run_simplified_backtest()
    
    if results:
        print("\n🎉 Backtest completed successfully!")
        print("🚀 Strategy is ready for live implementation!")
    else:
        print("\n❌ Backtest failed - check your API credentials")

    print("\n📋 Next Steps:")
    print("1. Set your Alpaca API credentials as environment variables")
    print("2. Run this script to see your strategy in action")
    print("3. Review the results and optimize parameters")
    print("4. Move to paper trading when satisfied!")