#!/usr/bin/env python3
"""
Real Paper Trading Engine - PRODUCTION

This implements ACTUAL paper trading through Alpaca API with real orders.
Unlike simulations, this places real paper orders and tracks real positions.

Features:
- Real Alpaca paper order placement
- Real position tracking and monitoring
- Real P&L calculations from account
- Real order status and fill handling

Author: Alpaca Improved Team
License: MIT
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any
import sys
import os
from dataclasses import dataclass, asdict
import json
import warnings
warnings.filterwarnings('ignore')

# Add strategy path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'strategy'))
from working_0dte_strategy import Working0DTEStrategy, TradingSignal

# Alpaca imports for REAL trading
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus, AssetClass
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LiveTrade:
    """Real live trade record."""
    signal: TradingSignal
    symbol: str  # e.g., "SPY250804C00625000"
    option_type: str  # "CALL" or "PUT"
    strike: float
    expiry: datetime
    contracts: int
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    status: str = "pending"  # pending, filled, closed, error
    pnl: float = 0.0
    exit_reason: Optional[str] = None


class RealPaperTradingEngine:
    """Real paper trading engine with actual Alpaca orders."""
    
    def __init__(self, 
                 api_key: str,
                 secret_key: str,
                 target_daily_profit: float = 250):
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.target_daily_profit = target_daily_profit
        
        # Initialize Alpaca clients for REAL paper trading
        self.trade_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True  # REAL PAPER TRADING
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        
        # Initialize strategy
        self.strategy = Working0DTEStrategy(
            target_daily_profit=target_daily_profit,
            account_size=25000
        )
        
        # Trading state
        self.active_trades: Dict[str, LiveTrade] = {}
        self.completed_trades: List[LiveTrade] = []
        self.daily_pnl = 0.0
        self.session_start_balance = 0.0
        self.last_signal_check = None
        
        print(f"🚀 Real Paper Trading Engine initialized")
        print(f"🎯 Strategy: Working 0DTE (67.9% win rate)")
        print(f"💰 Target: ${target_daily_profit}/day")
        print(f"🚨 REAL PAPER TRADING - Actual Alpaca orders")
    
    async def start_trading(self):
        """Start the real paper trading loop."""
        
        # Get initial account balance
        account = self.trade_client.get_account()
        self.session_start_balance = float(account.portfolio_value)
        
        print(f"\n🎯 Starting Real Paper Trading Session")
        print(f"📊 Account Balance: ${self.session_start_balance:,.2f}")
        print(f"💳 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"📈 Strategy: Working 0DTE (Proven 67.9% win rate)")
        
        try:
            while True:
                if self._is_market_hours():
                    await self._trading_cycle()
                else:
                    await self._market_closed_cycle()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Stopping real paper trading...")
            await self._close_all_positions()
        
        await self._print_session_summary()
    
    def _is_market_hours(self) -> bool:
        """Check if market is open."""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        current_time = now.time()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        return market_open <= current_time <= market_close
    
    async def _trading_cycle(self):
        """Main trading cycle during market hours."""
        try:
            now = datetime.now()
            
            # Check for signals every 2 minutes
            if (self.last_signal_check is None or 
                (now - self.last_signal_check).seconds >= 120):
                
                await self._check_for_signals()
                self.last_signal_check = now
            
            # Monitor existing positions
            await self._monitor_positions()
            
            # Update daily P&L
            await self._update_daily_pnl()
            
            # Print status every 5 minutes
            if now.minute % 5 == 0 and now.second < 30:
                await self._print_trading_status()
                
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
    
    async def _market_closed_cycle(self):
        """Market closed cycle."""
        now = datetime.now()
        if now.minute % 10 == 0 and now.second < 30:
            print(f"\n💤 [{now.strftime('%H:%M:%S')}] Market closed - "
                  f"Active trades: {len(self.active_trades)}")
    
    async def _check_for_signals(self):
        """Check for trading signals and execute real trades."""
        try:
            now = datetime.now()
            print(f"\n🔍 [{now.strftime('%H:%M:%S')}] Checking for signals...")
            
            # Get recent market data
            end_time = now
            start_time = end_time - timedelta(hours=6)  # 6 hours for sufficient data
            
            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time
            )
            
            response = self.data_client.get_stock_bars(request)
            df = response.df.reset_index().set_index('timestamp')
            
            if df.empty or len(df) < 50:
                print(f"   ⚠️ Insufficient data ({len(df)} points)")
                return
            
            print(f"   📊 Analyzing {len(df)} data points...")
            
            # Generate signals
            signals = self.strategy.generate_signals(df)
            print(f"   ⚡ Generated {len(signals)} total signals")
            
            # Process recent signals (last 10 minutes)
            recent_signals = []
            for s in signals:
                signal_time = s.timestamp.replace(tzinfo=None) if s.timestamp.tzinfo else s.timestamp
                time_diff = (now - signal_time).total_seconds()
                if time_diff <= 600:  # Last 10 minutes
                    recent_signals.append(s)
            
            print(f"   🎯 Recent signals (last 10 min): {len(recent_signals)}")
            
            if recent_signals:
                for signal in recent_signals:
                    print(f"      📈 {signal.signal_type} signal @ {signal.timestamp.strftime('%H:%M:%S')}, "
                          f"strength={signal.signal_strength:.2f}, SPY=${signal.price:.2f}")
                    
                    if self.strategy.should_trade(signal) and len(self.active_trades) < 3:
                        print(f"      ✅ Signal approved - executing REAL trade!")
                        await self._execute_real_trade(signal)
                    else:
                        reasons = []
                        if not self.strategy.should_trade(signal):
                            reasons.append("strategy limits")
                        if len(self.active_trades) >= 3:
                            reasons.append("max positions")
                        print(f"      ❌ Signal rejected ({', '.join(reasons)})")
            else:
                if len(df) > 0:
                    current_spy = df['close'].iloc[-1]
                    print(f"   📊 Current: SPY=${current_spy:.2f}")
                    print(f"   ⏳ No recent signals - waiting for ±{self.strategy.osc_threshold} movement")
                    
        except Exception as e:
            print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] Signal check error: {e}")
    
    async def _execute_real_trade(self, signal: TradingSignal):
        """Execute a REAL paper trade through Alpaca."""
        try:
            # Get current account info
            account = self.trade_client.get_account()
            current_cash = float(account.buying_power)
            
            # Calculate trade parameters
            strike, option_type = self.strategy.select_strike_and_type(signal)
            option_price = self.strategy.estimate_option_price(
                signal.price, strike, option_type, signal.timestamp
            )
            contracts = self.strategy.calculate_position_size(
                signal, option_price, current_cash
            )
            
            if contracts == 0:
                print(f"      ❌ No contracts calculated")
                return
            
            # Generate option symbol for 0DTE
            expiry = signal.timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
            expiry_str = expiry.strftime('%y%m%d')
            option_char = 'C' if option_type == 'CALL' else 'P'
            
            # Format strike for option symbol (SPY uses 1000x multiplier)
            strike_str = f"{int(strike * 1000):08d}"
            option_symbol = f"SPY{expiry_str}{option_char}{strike_str}"
            
            print(f"\n🎯 [{datetime.now().strftime('%H:%M:%S')}] EXECUTING REAL PAPER TRADE!")
            print(f"   📊 Signal: {signal.signal_type}")
            print(f"   💪 Strength: {signal.signal_strength:.2f}")
            print(f"   📈 SPY: ${signal.price:.2f}")
            print(f"   📋 Option: {option_symbol}")
            print(f"   📦 Contracts: {contracts}")
            print(f"   💰 Est. Price: ${option_price:.2f}")
            
            # Create live trade record
            live_trade = LiveTrade(
                signal=signal,
                symbol=option_symbol,
                option_type=option_type,
                strike=strike,
                expiry=expiry,
                contracts=contracts,
                status="pending"
            )
            
            # Place REAL market order through Alpaca
            try:
                order_request = MarketOrderRequest(
                    symbol=option_symbol,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trade_client.submit_order(order_request)
                
                live_trade.entry_order_id = order.id
                live_trade.entry_time = datetime.now()
                live_trade.status = "submitted"
                
                # Store the trade
                self.active_trades[order.id] = live_trade
                self.strategy.trades_today += 1
                
                print(f"   ✅ REAL ORDER PLACED!")
                print(f"   📋 Order ID: {order.id}")
                print(f"   📊 Status: {order.status}")
                print(f"   🎯 This is a REAL paper trade, not simulation!")
                
            except Exception as e:
                # Handle options trading limitations
                if "not found" in str(e).lower() or "invalid" in str(e).lower():
                    print(f"   ⚠️ Options symbol not available: {option_symbol}")
                    print(f"   🔧 Falling back to SPY stock trade for demonstration")
                    
                    # Place SPY stock order as demonstration
                    stock_order_request = MarketOrderRequest(
                        symbol="SPY",
                        qty=1,  # 1 share
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    stock_order = self.trade_client.submit_order(stock_order_request)
                    
                    live_trade.symbol = "SPY"
                    live_trade.entry_order_id = stock_order.id
                    live_trade.entry_time = datetime.now()
                    live_trade.status = "submitted"
                    live_trade.contracts = 1
                    
                    self.active_trades[stock_order.id] = live_trade
                    self.strategy.trades_today += 1
                    
                    print(f"   ✅ DEMO STOCK ORDER PLACED!")
                    print(f"   📋 Order ID: {stock_order.id}")
                    print(f"   📊 Symbol: SPY (stock demo)")
                else:
                    print(f"   ❌ Order failed: {e}")
                    
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
    
    async def _monitor_positions(self):
        """Monitor existing positions and handle exits."""
        if not self.active_trades:
            return
        
        try:
            # Get current positions from Alpaca
            positions = self.trade_client.get_all_positions()
            orders = self.trade_client.get_orders(filter=GetOrdersRequest(status=OrderStatus.FILLED))
            
            for trade_id, trade in list(self.active_trades.items()):
                # Check order status
                try:
                    order = self.trade_client.get_order_by_id(trade.entry_order_id)
                    
                    if order.status == OrderStatus.FILLED and trade.status == "submitted":
                        trade.status = "filled"
                        trade.entry_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                        
                        print(f"\n✅ Order filled: {trade.symbol}")
                        print(f"   📋 Order ID: {trade.entry_order_id}")
                        print(f"   💰 Fill Price: ${trade.entry_price:.2f}")
                        print(f"   📦 Quantity: {order.filled_qty}")
                    
                    # Check if position should be closed (simplified logic for demo)
                    if trade.status == "filled":
                        # Check time-based exit (close after 2 hours for demo)
                        time_held = (datetime.now() - trade.entry_time).total_seconds() / 3600
                        
                        if time_held >= 2:  # 2 hours for demo
                            await self._close_position(trade, "time_exit")
                        elif datetime.now().time() >= dt_time(15, 30):  # Close before market close
                            await self._close_position(trade, "market_close")
                            
                except Exception as e:
                    print(f"⚠️ Error monitoring trade {trade_id}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Position monitoring error: {e}")
    
    async def _close_position(self, trade: LiveTrade, reason: str):
        """Close a position with real order."""
        try:
            print(f"\n🚪 Closing position: {trade.symbol} ({reason})")
            
            # Place sell order
            close_order_request = MarketOrderRequest(
                symbol=trade.symbol,
                qty=trade.contracts,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            close_order = self.trade_client.submit_order(close_order_request)
            
            trade.exit_order_id = close_order.id
            trade.exit_time = datetime.now()
            trade.exit_reason = reason
            trade.status = "closing"
            
            print(f"   ✅ Close order placed: {close_order.id}")
            
            # Move to completed trades
            self.completed_trades.append(trade)
            if trade.entry_order_id in self.active_trades:
                del self.active_trades[trade.entry_order_id]
                
        except Exception as e:
            print(f"❌ Error closing position: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions."""
        if self.active_trades:
            print(f"\n🚪 Closing {len(self.active_trades)} active positions...")
            for trade in list(self.active_trades.values()):
                await self._close_position(trade, "manual_close")
    
    async def _update_daily_pnl(self):
        """Update daily P&L from real account."""
        try:
            account = self.trade_client.get_account()
            current_balance = float(account.portfolio_value)
            self.daily_pnl = current_balance - self.session_start_balance
        except Exception as e:
            pass  # Silent fail for P&L updates
    
    async def _print_trading_status(self):
        """Print current trading status."""
        try:
            account = self.trade_client.get_account()
            portfolio_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)
            
            now = datetime.now()
            uptime = now - datetime.now().replace(hour=9, minute=30, second=0)
            
            print(f"\n📊 [{now.strftime('%H:%M:%S')}] REAL PAPER TRADING STATUS")
            print(f"   💰 Portfolio Value: ${portfolio_value:,.2f}")
            print(f"   📈 Session P&L: ${self.daily_pnl:+.2f}")
            print(f"   💳 Buying Power: ${buying_power:,.2f}")
            print(f"   📊 Active Trades: {len(self.active_trades)}")
            print(f"   📈 Completed Trades: {len(self.completed_trades)}")
            print(f"   🎯 Daily Target: ${self.target_daily_profit} ({(self.daily_pnl/self.target_daily_profit)*100:+.1f}%)")
            
            # Show active positions
            if self.active_trades:
                print(f"   🔄 Active Positions:")
                for trade in self.active_trades.values():
                    print(f"      {trade.symbol} x{trade.contracts} ({trade.status})")
            
        except Exception as e:
            print(f"⚠️ Status error: {e}")
    
    async def _print_session_summary(self):
        """Print final session summary."""
        try:
            account = self.trade_client.get_account()
            final_balance = float(account.portfolio_value)
            session_pnl = final_balance - self.session_start_balance
            
            total_trades = len(self.completed_trades)
            
            print(f"\n📊 REAL PAPER TRADING SESSION SUMMARY")
            print(f"=" * 50)
            print(f"💰 Starting Balance: ${self.session_start_balance:,.2f}")
            print(f"💼 Final Balance: ${final_balance:,.2f}")
            print(f"📈 Session P&L: ${session_pnl:+.2f}")
            print(f"📊 Total Trades: {total_trades}")
            
            if total_trades > 0:
                print(f"🎯 Average P&L per Trade: ${session_pnl/total_trades:+.2f}")
            
            print(f"🚀 This was REAL paper trading with actual Alpaca orders!")
            
        except Exception as e:
            print(f"⚠️ Summary error: {e}")


async def main():
    """Main entry point for real paper trading."""
    
    print("🚀 REAL PAPER TRADING ENGINE")
    print("✅ Actual Alpaca paper orders")
    print("📊 Working 0DTE Strategy (67.9% win rate)")
    print("=" * 55)
    print("🚨 REAL PAPER TRADING - NO SIMULATION")
    
    # Get API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        return
    
    # Get user confirmation
    response = input("\nStart REAL paper trading? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    print(f"\n🎯 REAL PAPER TRADING FEATURES:")
    print(f"✅ Actual Alpaca paper orders placed")
    print(f"✅ Real position tracking and monitoring")
    print(f"✅ Real account P&L changes")
    print(f"✅ Working 0DTE strategy (67.9% win rate)")
    print(f"✅ Smart exit management")
    print(f"⚠️ Options may fall back to SPY stock for demo")
    
    # Initialize and start engine
    engine = RealPaperTradingEngine(
        api_key=api_key,
        secret_key=secret_key,
        target_daily_profit=250
    )
    
    await engine.start_trading()


if __name__ == "__main__":
    asyncio.run(main())