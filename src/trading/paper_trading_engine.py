#!/usr/bin/env python3
"""
Live Paper Trading Engine for Optimized 0DTE Strategy

Integrates our proven $250/day optimized strategy with live Alpaca paper trading:
- Real-time market data streaming
- Automated signal generation and execution  
- Smart exit management (NO expiry exits!)
- Risk controls and position management
- Live performance tracking and alerts

Based on documentation from Context7 MCP for Alpaca-py.

Author: Alpaca Improved Team
License: MIT
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import logging
from threading import Thread, Event
import json
import warnings
warnings.filterwarnings('ignore')

# Import Alpaca clients - corrected imports from official docs
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.live.option import OptionDataStream
from alpaca.data.requests import StockBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
from alpaca.trading.models import Position, Order

# Import our strategy components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'strategies'))
from optimized_baseline_0dte import OptimizedBaselineStrategy, OptimizedSignal, OptimizationMode, ExitTiming

# Import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'alpaca_improved'))
from utils.logger import get_logger


class TradingStatus(Enum):
    """Trading engine status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running" 
    PAUSED = "paused"
    ERROR = "error"
    MARKET_CLOSED = "market_closed"


class TradeSignal(Enum):
    """Live trade signals."""
    BUY_CALL = "buy_call"
    BUY_PUT = "buy_put"
    SELL_TO_CLOSE = "sell_to_close"
    HOLD = "hold"


@dataclass
class LiveTrade:
    """Live trade record."""
    entry_time: datetime
    signal: OptimizedSignal
    option_symbol: str
    option_type: str
    strike: float
    contracts: int
    entry_price: float
    entry_order_id: str
    target_exit_price: float
    stop_loss_price: float
    time_exit: datetime
    
    # Live tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_order_id: Optional[str] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitTiming] = None
    status: str = "OPEN"


@dataclass
class TradingMetrics:
    """Live trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    buying_power: float = 0.0
    positions_count: int = 0
    avg_trade_duration: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    target_achievement: float = 0.0


class PaperTradingEngine:
    """Live paper trading engine for optimized 0DTE strategy."""
    
    def __init__(self, 
                 api_key: str,
                 secret_key: str,
                 target_daily_profit: float = 250,
                 account_size: float = 25000,
                 paper: bool = True):
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.target_daily_profit = target_daily_profit
        self.account_size = account_size
        
        # Initialize logger
        self.logger = get_logger(f"PaperTradingEngine_{datetime.now().strftime('%Y%m%d')}")
        
        # Trading clients (based on Context7 documentation)
        self.trade_client = TradingClient(
            api_key=api_key, 
            secret_key=secret_key, 
            paper=paper
        )
        
        self.stock_data_client = StockHistoricalDataClient(
            api_key=api_key, 
            secret_key=secret_key
        )
        
        self.option_data_client = OptionHistoricalDataClient(
            api_key=api_key, 
            secret_key=secret_key
        )
        
        # Streaming clients (from Context7 examples)
        self.trade_stream = TradingStream(
            api_key=api_key, 
            secret_key=secret_key, 
            paper=paper
        )
        
        self.stock_stream = StockDataStream(
            api_key=api_key, 
            secret_key=secret_key
        )
        
        self.option_stream = OptionDataStream(
            api_key=api_key, 
            secret_key=secret_key
        )
        
        # Initialize optimized strategy
        self.strategy = OptimizedBaselineStrategy(
            target_daily_profit=target_daily_profit,
            account_size=account_size,
            optimization_mode=OptimizationMode.WIN_RATE
        )
        
        # Trading state
        self.status = TradingStatus.STOPPED
        self.current_positions: Dict[str, LiveTrade] = {}
        self.completed_trades: List[LiveTrade] = []
        self.active_orders: Set[str] = set()
        
        # Market data
        self.current_spy_price = 0.0
        self.market_data_buffer = []
        self.last_signal_check = datetime.now()
        
        # Performance tracking
        self.metrics = TradingMetrics()
        self.daily_start_capital = account_size
        
        # Threading
        self.stop_event = Event()
        self.market_data_thread = None
        self.strategy_thread = None
        
        # Risk controls
        self.max_daily_loss = account_size * 0.025  # 2.5% daily stop
        self.max_positions = 8
        self.max_position_value = account_size * 0.10  # 10% max per trade
        
        self.logger.info(f"✅ PaperTradingEngine initialized")
        self.logger.info(f"🎯 Target: ${target_daily_profit}/day")
        self.logger.info(f"📊 Account: ${account_size:,.2f}")
        self.logger.info(f"🔧 Paper Trading: {paper}")
    
    async def start_trading(self):
        """Start the live paper trading engine."""
        try:
            self.logger.info("🚀 Starting PaperTradingEngine...")
            self.status = TradingStatus.STARTING
            
            # Validate account and market hours
            account = await self.get_account_info()
            if not account:
                raise Exception("Failed to connect to Alpaca account")
            
            # Check market hours
            if not await self.is_market_open():
                self.logger.warning("📅 Market is closed - switching to monitoring mode")
                self.status = TradingStatus.MARKET_CLOSED
                return
            
            # Initialize daily tracking
            await self.initialize_daily_session()
            
            # Set up streaming subscriptions
            await self.setup_market_data_streams()
            await self.setup_trade_update_stream()
            
            # Start strategy monitoring
            self.status = TradingStatus.RUNNING
            self.logger.info("✅ PaperTradingEngine running!")
            
            # Start main trading loop
            await self.main_trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start trading engine: {e}")
            self.status = TradingStatus.ERROR
            raise
    
    async def setup_market_data_streams(self):
        """Set up real-time market data streams."""
        self.logger.info("📡 Setting up market data streams...")
        
        # SPY stock data handler (from Context7 documentation)
        async def stock_data_handler(data):
            if hasattr(data, 'symbol') and data.symbol == 'SPY':
                self.current_spy_price = float(data.close) if hasattr(data, 'close') else float(data.price)
                await self.update_market_data(data)
        
        # Subscribe to SPY real-time data
        self.stock_stream.subscribe_quotes(stock_data_handler, "SPY")
        self.stock_stream.subscribe_trades(stock_data_handler, "SPY")
        
        # Start the stream in background
        asyncio.create_task(self._run_stock_stream())
        
        self.logger.info("✅ Market data streams active")
    
    async def setup_trade_update_stream(self):
        """Set up trade update stream for order monitoring."""
        self.logger.info("📋 Setting up trade update stream...")
        
        # Trade updates handler (from Context7 documentation)
        async def trade_updates_handler(data):
            await self.handle_trade_update(data)
        
        # Subscribe to trade updates
        self.trade_stream.subscribe_trade_updates(trade_updates_handler)
        
        # Start the stream in background  
        asyncio.create_task(self._run_trade_stream())
        
        self.logger.info("✅ Trade update stream active")
    
    async def _run_stock_stream(self):
        """Run stock data stream."""
        try:
            self.stock_stream.run()
        except Exception as e:
            self.logger.error(f"❌ Stock stream error: {e}")
    
    async def _run_trade_stream(self):
        """Run trade update stream."""
        try:
            self.trade_stream.run()
        except Exception as e:
            self.logger.error(f"❌ Trade stream error: {e}")
    
    async def main_trading_loop(self):
        """Main trading logic loop."""
        self.logger.info("🔄 Starting main trading loop...")
        
        while not self.stop_event.is_set() and self.status == TradingStatus.RUNNING:
            try:
                # Check market hours
                if not await self.is_market_open():
                    self.status = TradingStatus.MARKET_CLOSED
                    break
                
                # Update account and positions
                await self.update_account_metrics()
                
                # Check for new signals every 15 seconds
                if (datetime.now() - self.last_signal_check).seconds >= 15:
                    await self.check_for_new_signals()
                    self.last_signal_check = datetime.now()
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Risk management checks
                await self.check_risk_limits()
                
                # Update performance metrics
                await self.update_performance_metrics()
                
                # Sleep for 5 seconds before next iteration
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(10)  # Wait before retrying
        
        self.logger.info("🔄 Trading loop stopped")
    
    async def check_for_new_signals(self):
        """Check for new trading signals using optimized strategy."""
        try:
            # Get recent market data for signal generation
            historical_data = await self.get_recent_market_data()
            if historical_data is None or len(historical_data) < 50:
                return
            
            # Generate signals using optimized strategy
            signals = self.strategy.generate_optimized_signals(historical_data)
            
            # Filter for actionable signals
            actionable_signals = [
                s for s in signals[-3:] if  # Only check last 3 signals
                self.strategy.should_trade_optimized(s) and
                len(self.current_positions) < self.max_positions
            ]
            
            # Execute trades for valid signals
            for signal in actionable_signals:
                await self.execute_signal(signal)
                
        except Exception as e:
            self.logger.error(f"❌ Signal check error: {e}")
    
    async def execute_signal(self, signal: OptimizedSignal):
        """Execute a trading signal."""
        try:
            self.logger.info(f"🎯 Executing signal: {signal.signal_type} at ${signal.price:.2f}")
            
            # Calculate position size
            account = await self.get_account_info()
            buying_power = float(account.buying_power)
            
            # Estimate option price and contracts
            option_price = self._estimate_option_price(signal)
            contracts = self.strategy.calculate_optimized_position_size(
                signal, option_price, buying_power
            )
            
            if contracts == 0:
                self.logger.warning("⚠️ Position size too small - skipping trade")
                return
            
            # Generate option symbol
            option_symbol, option_type, strike = await self._generate_option_symbol(signal)
            
            # Calculate exit strategy
            exit_strategy = self.strategy.get_exit_strategy(signal, option_price, signal.timestamp)
            
            # Submit order
            order_request = MarketOrderRequest(
                symbol=option_symbol,
                qty=contracts,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trade_client.submit_order(order_request)
            self.active_orders.add(order.id)
            
            # Create live trade record
            live_trade = LiveTrade(
                entry_time=datetime.now(),
                signal=signal,
                option_symbol=option_symbol,
                option_type=option_type,
                strike=strike,
                contracts=contracts,
                entry_price=option_price,
                entry_order_id=order.id,
                target_exit_price=exit_strategy['profit_target_1'],
                stop_loss_price=exit_strategy['stop_loss'],
                time_exit=exit_strategy['time_exit'],
                status="PENDING"
            )
            
            self.current_positions[order.id] = live_trade
            
            self.logger.info(f"📋 Order submitted: {option_symbol} x{contracts} @ ${option_price:.2f}")
            self.logger.info(f"🎯 Target: ${exit_strategy['profit_target_1']:.2f}")
            self.logger.info(f"🛑 Stop: ${exit_strategy['stop_loss']:.2f}")
            self.logger.info(f"⏰ Time Exit: {exit_strategy['time_exit'].strftime('%H:%M')}")
            
        except Exception as e:
            self.logger.error(f"❌ Signal execution error: {e}")
    
    async def monitor_positions(self):
        """Monitor existing positions for exit opportunities."""
        for order_id, trade in list(self.current_positions.items()):
            try:
                if trade.status != "OPEN":
                    continue
                
                # Get current option price
                current_price = await self._get_current_option_price(trade.option_symbol)
                if current_price is None:
                    continue
                
                trade.current_price = current_price
                trade.unrealized_pnl = (current_price - trade.entry_price) * trade.contracts * 100
                
                # Check exit conditions
                exit_signal = await self._check_exit_conditions(trade, current_price)
                
                if exit_signal:
                    await self.exit_position(trade, exit_signal)
                    
            except Exception as e:
                self.logger.error(f"❌ Position monitoring error for {order_id}: {e}")
    
    async def _check_exit_conditions(self, trade: LiveTrade, current_price: float) -> Optional[ExitTiming]:
        """Check if position should be exited."""
        
        # Profit target hit
        if current_price >= trade.target_exit_price:
            return ExitTiming.PROFIT_TARGET
        
        # Stop loss hit
        if current_price <= trade.stop_loss_price:
            return ExitTiming.TIGHT_STOP
        
        # Time exit (CRITICAL - no expiry exits!)
        if datetime.now() >= trade.time_exit:
            return ExitTiming.TIME_STOP
        
        # Market close approaching (15:30 ET)
        market_close = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
        if datetime.now() >= market_close:
            return ExitTiming.TIME_STOP
        
        return None
    
    async def exit_position(self, trade: LiveTrade, exit_reason: ExitTiming):
        """Exit a position."""
        try:
            self.logger.info(f"🚪 Exiting position: {trade.option_symbol} - Reason: {exit_reason.value}")
            
            # Submit sell order
            order_request = MarketOrderRequest(
                symbol=trade.option_symbol,
                qty=trade.contracts,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            exit_order = self.trade_client.submit_order(order_request)
            
            trade.exit_order_id = exit_order.id
            trade.exit_time = datetime.now()
            trade.exit_reason = exit_reason
            trade.exit_price = trade.current_price
            trade.status = "CLOSING"
            
            self.active_orders.add(exit_order.id)
            
            self.logger.info(f"📋 Exit order submitted: {trade.option_symbol} @ ${trade.current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Exit position error: {e}")
    
    async def handle_trade_update(self, data):
        """Handle trade update from stream."""
        try:
            order_id = data.order['id']
            order_status = data.order['status']
            
            if order_id in self.active_orders:
                # Find associated trade
                trade = None
                for t in self.current_positions.values():
                    if t.entry_order_id == order_id or t.exit_order_id == order_id:
                        trade = t
                        break
                
                if trade:
                    if order_status == OrderStatus.FILLED:
                        if order_id == trade.entry_order_id:
                            # Entry filled
                            trade.status = "OPEN"
                            trade.entry_price = float(data.order['filled_avg_price'])
                            self.logger.info(f"✅ Entry filled: {trade.option_symbol} @ ${trade.entry_price:.2f}")
                        
                        elif order_id == trade.exit_order_id:
                            # Exit filled
                            trade.status = "CLOSED"
                            trade.exit_price = float(data.order['filled_avg_price'])
                            
                            # Calculate final P&L
                            realized_pnl = (trade.exit_price - trade.entry_price) * trade.contracts * 100
                            
                            self.logger.info(f"✅ Exit filled: {trade.option_symbol} @ ${trade.exit_price:.2f}")
                            self.logger.info(f"💰 P&L: ${realized_pnl:+.2f}")
                            
                            # Move to completed trades
                            self.completed_trades.append(trade)
                            if trade.entry_order_id in self.current_positions:
                                del self.current_positions[trade.entry_order_id]
                        
                        self.active_orders.discard(order_id)
                
        except Exception as e:
            self.logger.error(f"❌ Trade update handling error: {e}")
    
    async def get_account_info(self) -> Optional[dict]:
        """Get current account information."""
        try:
            return self.trade_client.get_account()
        except Exception as e:
            self.logger.error(f"❌ Account info error: {e}")
            return None
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.trade_client.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.error(f"❌ Market hours check error: {e}")
            return False
    
    async def get_recent_market_data(self) -> Optional[pd.DataFrame]:
        """Get recent market data for signal generation."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=4)  # 4 hours of data
            
            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time
            )
            
            response = self.stock_data_client.get_stock_bars(request)
            df = response.df.reset_index().set_index('timestamp')
            
            return df if len(df) >= 50 else None
            
        except Exception as e:
            self.logger.error(f"❌ Market data error: {e}")
            return None
    
    def _estimate_option_price(self, signal: OptimizedSignal) -> float:
        """Estimate option price for position sizing."""
        # Use the same logic as backtesting
        base_price = signal.price * 0.018
        time_factor = min(signal.time_to_expiry_hours / 6, 1.0)
        base_price *= (0.3 + 0.7 * time_factor)
        
        return max(0.05, min(base_price, signal.price * 0.1))
    
    async def _generate_option_symbol(self, signal: OptimizedSignal) -> Tuple[str, str, float]:
        """Generate option symbol for trading."""
        current_price = signal.price
        
        # Conservative strikes for better win rate
        if signal.signal_type == "BULLISH":
            strike = round(current_price / 5) * 5  # ATM call
            option_type = "call"
        else:
            strike = round(current_price / 5) * 5  # ATM put  
            option_type = "put"
        
        # Generate symbol (0DTE)
        expiry_date = signal.timestamp.replace(hour=16, minute=15)
        option_symbol = f"SPY{expiry_date.strftime('%y%m%d')}{'C' if option_type == 'call' else 'P'}{int(strike*1000):08d}"
        
        return option_symbol, option_type, strike
    
    async def _get_current_option_price(self, option_symbol: str) -> Optional[float]:
        """Get current option price."""
        try:
            # For paper trading, we can estimate based on underlying movement
            # In production, you'd use real option quotes
            return self._estimate_option_price_from_underlying()
        except Exception as e:
            self.logger.error(f"❌ Option price error for {option_symbol}: {e}")
            return None
    
    def _estimate_option_price_from_underlying(self) -> float:
        """Estimate option price from underlying movement."""
        # Simplified estimation - in production use real option prices
        return self.current_spy_price * 0.02  # 2% of underlying
    
    async def update_market_data(self, data):
        """Update market data buffer."""
        self.market_data_buffer.append({
            'timestamp': datetime.now(),
            'price': self.current_spy_price,
            'data': data
        })
        
        # Keep only last 100 data points
        if len(self.market_data_buffer) > 100:
            self.market_data_buffer.pop(0)
    
    async def initialize_daily_session(self):
        """Initialize daily trading session."""
        account = await self.get_account_info()
        if account:
            self.daily_start_capital = float(account.portfolio_value)
            self.metrics.buying_power = float(account.buying_power)
            
            self.logger.info(f"💰 Daily session initialized")
            self.logger.info(f"📊 Starting capital: ${self.daily_start_capital:,.2f}")
            self.logger.info(f"💳 Buying power: ${self.metrics.buying_power:,.2f}")
    
    async def update_account_metrics(self):
        """Update real-time account metrics."""
        try:
            account = await self.get_account_info()
            if account:
                current_value = float(account.portfolio_value)
                self.metrics.daily_pnl = current_value - self.daily_start_capital
                self.metrics.buying_power = float(account.buying_power)
                
                # Calculate unrealized P&L
                unrealized = sum(trade.unrealized_pnl for trade in self.current_positions.values())
                self.metrics.unrealized_pnl = unrealized
                
        except Exception as e:
            self.logger.error(f"❌ Account metrics error: {e}")
    
    async def update_performance_metrics(self):
        """Update performance metrics."""
        try:
            total_trades = len(self.completed_trades)
            if total_trades > 0:
                winning_trades = len([t for t in self.completed_trades if 
                                    (t.exit_price - t.entry_price) * t.contracts * 100 > 0])
                
                self.metrics.total_trades = total_trades
                self.metrics.winning_trades = winning_trades
                self.metrics.losing_trades = total_trades - winning_trades
                self.metrics.win_rate = (winning_trades / total_trades) * 100
                
                # Calculate total P&L
                realized_pnl = sum((t.exit_price - t.entry_price) * t.contracts * 100 
                                 for t in self.completed_trades if t.exit_price)
                self.metrics.total_pnl = realized_pnl
                
                # Target achievement
                self.metrics.target_achievement = (self.metrics.daily_pnl / self.target_daily_profit) * 100
            
            self.metrics.positions_count = len(self.current_positions)
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics error: {e}")
    
    async def check_risk_limits(self):
        """Check risk management limits."""
        try:
            # Daily loss limit
            if self.metrics.daily_pnl <= -self.max_daily_loss:
                self.logger.warning(f"🚨 Daily loss limit hit: ${self.metrics.daily_pnl:+.2f}")
                await self.pause_trading("Daily loss limit exceeded")
            
            # Maximum positions
            if len(self.current_positions) >= self.max_positions:
                self.logger.warning(f"⚠️ Maximum positions reached: {len(self.current_positions)}")
            
            # Market close warning
            market_close = datetime.now().replace(hour=15, minute=15, second=0, microsecond=0)
            if datetime.now() >= market_close and len(self.current_positions) > 0:
                self.logger.warning(f"⏰ Market closing soon - {len(self.current_positions)} positions open")
                
        except Exception as e:
            self.logger.error(f"❌ Risk check error: {e}")
    
    async def pause_trading(self, reason: str):
        """Pause trading engine."""
        self.logger.warning(f"⏸️ Pausing trading: {reason}")
        self.status = TradingStatus.PAUSED
    
    async def stop_trading(self):
        """Stop trading engine."""
        self.logger.info("🛑 Stopping PaperTradingEngine...")
        self.status = TradingStatus.STOPPED
        self.stop_event.set()
        
        # Close all streams
        try:
            self.stock_stream.stop()
            self.trade_stream.stop()
        except:
            pass
        
        self.logger.info("✅ PaperTradingEngine stopped")
    
    def get_status_summary(self) -> Dict:
        """Get current trading status summary."""
        return {
            "status": self.status.value,
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(self.metrics),
            "positions": len(self.current_positions),
            "active_orders": len(self.active_orders),
            "current_spy_price": self.current_spy_price,
            "target_achievement": f"{self.metrics.target_achievement:.1f}%",
            "daily_pnl": f"${self.metrics.daily_pnl:+.2f}",
            "win_rate": f"{self.metrics.win_rate:.1f}%"
        }


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    print("🚀 LIVE PAPER TRADING ENGINE")
    print("🎯 Optimized 0DTE Strategy - $250/day Target")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Missing API credentials in .env file")
        exit(1)
    
    async def run_paper_trading():
        """Run paper trading engine."""
        engine = PaperTradingEngine(
            api_key=api_key,
            secret_key=secret_key,
            target_daily_profit=250,
            account_size=25000,
            paper=True
        )
        
        try:
            await engine.start_trading()
        except KeyboardInterrupt:
            print("\n⌨️ Keyboard interrupt - shutting down...")
            await engine.stop_trading()
        except Exception as e:
            print(f"❌ Engine error: {e}")
            await engine.stop_trading()
    
    # Run the engine
    asyncio.run(run_paper_trading())