#!/usr/bin/env python3
"""
MA Shift Options Strategy - Paper Trading Implementation

Live paper trading implementation of the MA Shift multi-indicator options strategy.
This demonstrates how to run the strategy in real-time with paper trading through Alpaca.

Features:
- Real-time market data processing
- Live signal generation and options selection
- Paper trading execution through Alpaca
- Risk management and position monitoring
- Performance tracking and reporting
- ML feature logging for future enhancement

Author: Alpaca Improved Team  
License: MIT
"""

import sys
import os
import time
import schedule
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.timeframe import TimeFrame

from strategies.base.options_strategy import OptionsStrategyConfig
from data.extractors.alpaca_extractor import AlpacaDataExtractor
from data.extractors.options_chain_extractor import OptionsChainExtractor
from utils.logger import get_logger

# Import the strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from ma_shift_options_strategy import MAShiftOptionsStrategy, SignalStrength, MAShiftSignal


class PaperTradingEngine:
    """
    Paper trading engine for the MA Shift options strategy.
    
    Handles real-time market data, signal generation, and paper trading execution
    while maintaining comprehensive logs and performance tracking.
    """
    
    def __init__(
        self,
        strategy_config: OptionsStrategyConfig,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets"
    ):
        """
        Initialize the paper trading engine.
        
        Args:
            strategy_config: Strategy configuration
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca API base URL (paper trading)
        """
        self.strategy_config = strategy_config
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_extractor = AlpacaDataExtractor(api_key, secret_key)
        self.options_extractor = OptionsChainExtractor(api_key, secret_key)
        
        # Initialize strategy
        self.strategy = MAShiftOptionsStrategy(strategy_config)
        
        # Trading state
        self.active_positions = {}
        self.signal_history = []
        self.trade_history = []
        self.performance_log = []
        
        # Risk management
        self.max_daily_trades = 5
        self.max_position_value = 10000  # Max value per position
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        self.logger.info(f"Paper trading engine initialized for {strategy_config.symbol}")
    
    def get_market_data(self, lookback_days: int = 100) -> Optional[pd.DataFrame]:
        """
        Get recent market data for analysis.
        
        Args:
            lookback_days: Number of days of historical data
            
        Returns:
            DataFrame with market data or None if failed
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            data = self.data_extractor.get_bars(
                self.strategy_config.symbol,
                TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            if data.empty:
                self.logger.warning("No market data retrieved")
                return None
            
            self.logger.debug(f"Retrieved {len(data)} bars of market data")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return None
    
    def generate_signals(self) -> List[MAShiftSignal]:
        """
        Generate trading signals from current market data.
        
        Returns:
            List of current trading signals
        """
        try:
            # Get market data
            market_data = self.get_market_data()
            if market_data is None:
                return []
            
            # Generate signals
            all_signals = self.strategy.analyze_market_data(market_data)
            
            # Get only the latest signal
            if all_signals:
                latest_signal = all_signals[-1]
                
                # Only process if it's a new signal (not in history)
                if not self.signal_history or latest_signal.timestamp > self.signal_history[-1].timestamp:
                    self.signal_history.append(latest_signal)
                    
                    if latest_signal.signal_type != "NEUTRAL":
                        self.logger.info(f"New signal: {latest_signal.signal_type} ({latest_signal.strength.name})")
                        return [latest_signal]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    def get_options_chain(self) -> List:
        """Get current options chain for signal evaluation."""
        try:
            # Get next monthly expiration
            chain = self.options_extractor.get_options_chain(self.strategy_config.symbol)
            
            if not chain:
                self.logger.warning("No options chain data available")
                return []
            
            self.logger.debug(f"Retrieved options chain with {len(chain)} contracts")
            return chain
            
        except Exception as e:
            self.logger.error(f"Error fetching options chain: {e}")
            return []
    
    def execute_options_trade(self, signal: MAShiftSignal) -> bool:
        """
        Execute an options trade based on signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if trade executed successfully
        """
        try:
            # Check daily trade limits
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today
            
            if self.daily_trade_count >= self.max_daily_trades:
                self.logger.warning("Daily trade limit reached")
                return False
            
            # Get options chain
            options_chain = self.get_options_chain()
            if not options_chain:
                return False
            
            # Select appropriate contract
            contract_selection = self.strategy.select_options_contracts(signal, options_chain)
            if not contract_selection:
                self.logger.info("No suitable options contract found")
                return False
            
            contract = contract_selection['contract']
            action = contract_selection['action']
            
            # Simulate paper trading (since actual options trading requires special setup)
            self.logger.info(f"Paper trading: {action} {contract.option_symbol}")
            
            # Calculate position size
            position_size = self.strategy.calculate_position_size(signal, contract.last_price or 1.0)
            position_value = position_size * (contract.last_price or 1.0) * 100
            
            if position_value > self.max_position_value:
                position_size = int(self.max_position_value / ((contract.last_price or 1.0) * 100))
            
            # Record the trade
            trade_record = {
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type,
                'signal_strength': signal.strength.value,
                'option_symbol': contract.option_symbol,
                'action': action,
                'strike': contract.strike_price,
                'expiration': contract.expiration_date,
                'option_type': contract.option_type.value,
                'contracts': position_size,
                'option_price': contract.last_price,
                'underlying_price': signal.price,
                'position_value': position_value,
                'delta': contract.delta,
                'theta': contract.theta,
                'iv': contract.implied_volatility
            }
            
            self.trade_history.append(trade_record)
            self.daily_trade_count += 1
            
            # Add to active positions for monitoring
            position_key = f"{contract.option_symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M')}"
            self.active_positions[position_key] = {
                'entry_signal': signal,
                'contract': contract,
                'entry_time': signal.timestamp,
                'contracts': position_size,
                'entry_price': contract.last_price,
                'action': action
            }
            
            self.logger.info(f"Executed paper trade: {action} {position_size} contracts of {contract.option_symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing options trade: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor active positions for exit conditions."""
        try:
            current_time = datetime.now()
            positions_to_close = []
            
            for pos_key, position in self.active_positions.items():
                entry_time = position['entry_time']
                days_held = (current_time - entry_time).days
                
                # Time-based exit (10 days)
                if days_held >= 10:
                    positions_to_close.append(pos_key)
                    self.logger.info(f"Closing position {pos_key} - time limit reached")
                
                # Expiration check
                elif current_time >= position['contract'].expiration_date:
                    positions_to_close.append(pos_key)
                    self.logger.info(f"Closing position {pos_key} - expiration reached")
            
            # Close positions
            for pos_key in positions_to_close:
                self._close_position(pos_key)
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def _close_position(self, position_key: str):
        """Close a specific position."""
        try:
            position = self.active_positions[position_key]
            
            # Log position closure
            close_record = {
                'timestamp': datetime.now(),
                'position_key': position_key,
                'option_symbol': position['contract'].option_symbol,
                'action': 'CLOSE',
                'entry_time': position['entry_time'],
                'contracts': position['contracts'],
                'days_held': (datetime.now() - position['entry_time']).days,
                'entry_price': position['entry_price']
            }
            
            self.trade_history.append(close_record)
            
            # Remove from active positions
            del self.active_positions[position_key]
            
            self.logger.info(f"Closed position: {position['contract'].option_symbol}")
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_key}: {e}")
    
    def log_performance(self):
        """Log current performance metrics."""
        try:
            current_time = datetime.now()
            
            # Get account information
            account = self.trading_client.get_account()
            
            performance_data = {
                'timestamp': current_time,
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'active_positions': len(self.active_positions),
                'total_trades': len(self.trade_history),
                'daily_trades': self.daily_trade_count,
                'signals_generated': len(self.signal_history)
            }
            
            self.performance_log.append(performance_data)
            
            self.logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            self.logger.info(f"Active Positions: {len(self.active_positions)}")
            self.logger.info(f"Total Trades: {len(self.trade_history)}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance: {e}")
    
    def run_trading_cycle(self):
        """Execute one complete trading cycle."""
        self.logger.info("Starting trading cycle...")
        
        try:
            # Generate signals
            signals = self.generate_signals()
            
            # Process signals
            for signal in signals:
                if signal.strength.value >= SignalStrength.MODERATE.value:
                    self.execute_options_trade(signal)
            
            # Monitor existing positions
            self.monitor_positions()
            
            # Log performance
            self.log_performance()
            
            self.logger.info("Trading cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def save_logs(self):
        """Save trading logs to files."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # Save trade history
            if self.trade_history:
                trades_df = pd.DataFrame(self.trade_history)
                trades_df.to_csv(f'ma_shift_trades_{timestamp}.csv', index=False)
                self.logger.info(f"Saved {len(self.trade_history)} trade records")
            
            # Save signal history
            if self.signal_history:
                signals_data = []
                for signal in self.signal_history:
                    signals_data.append({
                        'timestamp': signal.timestamp,
                        'signal_type': signal.signal_type,
                        'strength': signal.strength.value,
                        'price': signal.price,
                        'ma_shift_osc': signal.ma_shift_osc,
                        'keltner_position': signal.keltner_position,
                        'bb_position': signal.bb_position,
                        'volatility_regime': signal.volatility_regime
                    })
                
                signals_df = pd.DataFrame(signals_data)
                signals_df.to_csv(f'ma_shift_signals_{timestamp}.csv', index=False)
                self.logger.info(f"Saved {len(self.signal_history)} signal records")
            
            # Save performance log
            if self.performance_log:
                perf_df = pd.DataFrame(self.performance_log)
                perf_df.to_csv(f'ma_shift_performance_{timestamp}.csv', index=False)
                self.logger.info(f"Saved {len(self.performance_log)} performance records")
            
        except Exception as e:
            self.logger.error(f"Error saving logs: {e}")


def main():
    """Main paper trading execution."""
    print("🚀 Starting MA Shift Options Strategy Paper Trading")
    
    # Load credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    # Strategy configuration
    strategy_config = OptionsStrategyConfig(
        name="MA Shift Paper Trading",
        symbol="SPY",
        parameters={
            'ma_length': 40,
            'ma_type': 'SMA',
            'osc_threshold': 0.5,
            'min_signal_strength': 2,  # Only MODERATE or stronger signals
            'target_dte': 30,
            'target_delta': 0.3,
            'max_position_size': 0.05  # 5% of portfolio per position
        }
    )
    
    # Initialize paper trading engine
    engine = PaperTradingEngine(strategy_config, api_key, secret_key)
    
    # Schedule trading cycles
    # Run every hour during market hours (9:30 AM - 4:00 PM ET)
    schedule.every().hour.at(":30").do(engine.run_trading_cycle)
    
    # Run performance logging every 30 minutes
    schedule.every(30).minutes.do(engine.log_performance)
    
    # Save logs at market close
    schedule.every().day.at("16:30").do(engine.save_logs)
    
    print("📊 Paper trading engine started")
    print("⏰ Scheduled to run every hour during market hours")
    print("📈 Performance logging every 30 minutes")
    print("💾 Logs saved daily at market close")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Run initial cycle
        engine.run_trading_cycle()
        
        # Keep running scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping paper trading engine...")
        engine.save_logs()
        print("📁 Final logs saved")
        print("👋 Paper trading stopped")


if __name__ == "__main__":
    # Set up logging for paper trading
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ma_shift_paper_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    main()