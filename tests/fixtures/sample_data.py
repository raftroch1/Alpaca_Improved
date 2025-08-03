"""
Sample data fixtures for testing data extractors.

This module provides pre-generated sample data that can be used in tests
without making actual API calls.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any


class SampleDataGenerator:
    """Generate realistic sample data for testing."""
    
    @staticmethod
    def generate_spy_daily_data(days: int = 30, start_price: float = 450.0) -> pd.DataFrame:
        """
        Generate realistic SPY daily data.
        
        Args:
            days: Number of days to generate
            start_price: Starting price
            
        Returns:
            DataFrame with OHLCV data
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        data = []
        current_price = start_price
        
        for i, date in enumerate(dates):
            # Random walk with slight upward bias
            daily_return = (i % 5 - 2) * 0.005  # -1% to +1.5% daily moves
            current_price *= (1 + daily_return)
            
            # Generate OHLC
            open_price = current_price
            close_price = current_price * (1 + (i % 3 - 1) * 0.002)
            high_price = max(open_price, close_price) * (1 + abs(i % 4) * 0.001)
            low_price = min(open_price, close_price) * (1 - abs(i % 3) * 0.001)
            volume = 50000000 + (i * 1000000) + ((i % 7) * 5000000)
            
            data.append({
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    @staticmethod
    def generate_options_chain(
        symbol: str = 'SPY',
        current_price: float = 450.0,
        expiration_days: int = 30,
        strike_range: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic options chain data.
        
        Args:
            symbol: Underlying symbol
            current_price: Current stock price
            expiration_days: Days until expiration
            strike_range: Price range around current price
            
        Returns:
            List of option contract dictionaries
        """
        expiration = datetime.now() + timedelta(days=expiration_days)
        exp_str = expiration.strftime('%y%m%d')
        
        options = []
        
        # Generate strikes from current_price - strike_range to current_price + strike_range
        for strike in range(
            int(current_price - strike_range),
            int(current_price + strike_range) + 1,
            5  # $5 strike intervals
        ):
            strike = float(strike)
            
            # Calculate realistic option prices
            moneyness = current_price - strike
            time_value = max(0.1, 5.0 * (expiration_days / 365))
            
            # Call option
            call_intrinsic = max(0, moneyness)
            call_price = call_intrinsic + time_value
            
            call_option = {
                'symbol': symbol,
                'option_symbol': f'{symbol}{exp_str}C{int(strike*1000):08d}',
                'strike_price': strike,
                'expiration_date': expiration,
                'option_type': 'C',
                'bid': round(call_price - 0.05, 2),
                'ask': round(call_price + 0.05, 2),
                'last_price': round(call_price, 2),
                'volume': max(10, int(1000 - abs(moneyness) * 10)),
                'open_interest': max(100, int(10000 - abs(moneyness) * 100)),
                'implied_volatility': round(0.15 + abs(moneyness) * 0.001, 4),
                'delta': round(max(0.01, min(0.99, 0.5 + moneyness * 0.01)), 4),
                'gamma': round(0.02 - abs(moneyness) * 0.0001, 4),
                'theta': round(-0.05 - expiration_days * 0.0001, 4),
                'vega': round(0.15 - abs(moneyness) * 0.001, 4),
                'rho': round(0.08 + strike * 0.0001, 4)
            }
            options.append(call_option)
            
            # Put option
            put_intrinsic = max(0, -moneyness)
            put_price = put_intrinsic + time_value
            
            put_option = {
                'symbol': symbol,
                'option_symbol': f'{symbol}{exp_str}P{int(strike*1000):08d}',
                'strike_price': strike,
                'expiration_date': expiration,
                'option_type': 'P',
                'bid': round(put_price - 0.05, 2),
                'ask': round(put_price + 0.05, 2),
                'last_price': round(put_price, 2),
                'volume': max(10, int(800 - abs(moneyness) * 8)),
                'open_interest': max(100, int(8000 - abs(moneyness) * 80)),
                'implied_volatility': round(0.17 + abs(moneyness) * 0.001, 4),
                'delta': round(max(-0.99, min(-0.01, -0.5 + moneyness * 0.01)), 4),
                'gamma': round(0.02 - abs(moneyness) * 0.0001, 4),
                'theta': round(-0.04 - expiration_days * 0.0001, 4),
                'vega': round(0.15 - abs(moneyness) * 0.001, 4),
                'rho': round(-0.06 - strike * 0.0001, 4)
            }
            options.append(put_option)
        
        return options
    
    @staticmethod
    def generate_multi_symbol_data(symbols: List[str], days: int = 20) -> pd.DataFrame:
        """
        Generate data for multiple symbols.
        
        Args:
            symbols: List of symbol names
            days: Number of days to generate
            
        Returns:
            DataFrame with multi-symbol data
        """
        all_data = []
        base_prices = {
            'SPY': 450.0,
            'QQQ': 380.0,
            'IWM': 220.0,
            'VTI': 240.0,
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2800.0,
            'TSLA': 200.0
        }
        
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100.0)
            symbol_data = SampleDataGenerator.generate_spy_daily_data(days, base_price)
            symbol_data['symbol'] = symbol
            symbol_data = symbol_data.reset_index()
            all_data.append(symbol_data)
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Set multi-index (symbol, timestamp)
        combined = combined.set_index(['symbol', 'timestamp'])
        
        return combined
    
    @staticmethod
    def generate_intraday_data(
        symbol: str = 'SPY',
        hours: int = 24,
        frequency_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Generate intraday (hourly/minute) data.
        
        Args:
            symbol: Symbol to generate data for
            hours: Number of hours to generate
            frequency_minutes: Data frequency in minutes
            
        Returns:
            DataFrame with intraday data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        timestamps = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{frequency_minutes}min'
        )
        
        data = []
        base_price = 450.0
        
        for i, timestamp in enumerate(timestamps):
            # Intraday volatility is lower
            change = (i % 10 - 5) * 0.001  # Smaller moves
            price = base_price * (1 + change)
            
            # Generate OHLC for the period
            open_price = price
            close_price = price * (1 + (i % 3 - 1) * 0.0005)
            high_price = max(open_price, close_price) * (1 + (i % 2) * 0.0002)
            low_price = min(open_price, close_price) * (1 - (i % 2) * 0.0002)
            
            # Volume varies by time of day
            hour = timestamp.hour
            if 9 <= hour <= 16:  # Market hours
                volume = 2000000 + (i * 10000)
            else:  # After hours
                volume = 200000 + (i * 1000)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    @staticmethod
    def generate_quote_data(symbol: str = 'SPY', count: int = 10) -> pd.DataFrame:
        """
        Generate realistic quote (bid/ask) data.
        
        Args:
            symbol: Symbol to generate quotes for
            count: Number of quotes to generate
            
        Returns:
            DataFrame with quote data
        """
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(minutes=count),
            end=datetime.now(),
            freq='1min'
        )
        
        data = []
        base_price = 450.0
        
        for i, timestamp in enumerate(timestamps):
            mid_price = base_price + (i % 5 - 2) * 0.1
            spread = 0.01  # 1 cent spread
            
            data.append({
                'timestamp': timestamp,
                'bid': round(mid_price - spread/2, 2),
                'ask': round(mid_price + spread/2, 2),
                'bid_size': 100 + (i % 10) * 10,
                'ask_size': 100 + ((i+1) % 10) * 10
            })
        
        return pd.DataFrame(data).set_index('timestamp') 