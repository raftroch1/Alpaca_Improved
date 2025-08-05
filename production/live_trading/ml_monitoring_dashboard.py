#!/usr/bin/env python3
"""
ML Live Trading Monitoring Dashboard
====================================

Real-time monitoring dashboard for the ML Live Paper Trader.
Displays performance metrics, position tracking, and alerts.

Run this separately from the main trading engine for monitoring.

Author: Alpaca Improved Team
Version: v1.0
"""

import asyncio
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from dataclasses import asdict

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.join(project_root, 'src'))

from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from dotenv import load_dotenv

# Load environment variables from project root
env_file_path = os.path.join(project_root, '.env')
load_dotenv(env_file_path)

class MLTradingMonitor:
    """Real-time monitoring dashboard for ML trading."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize Alpaca clients
        self.trade_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        
        # Tracking variables
        self.session_start_time = datetime.now()
        self.session_start_balance = 0.0
        self.target_daily_profit = 250.0
        self.last_update = None
        
        # Performance history
        self.balance_history = []
        self.trade_history = []
        
        # Initialize session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize monitoring session."""
        try:
            account = self.trade_client.get_account()
            self.session_start_balance = float(account.portfolio_value)
            
            print("🚀 ML TRADING MONITOR INITIALIZED")
            print(f"📊 Starting Balance: ${self.session_start_balance:,.2f}")
            print(f"🎯 Daily Target: ${self.target_daily_profit}")
            print(f"⏰ Session Start: {self.session_start_time.strftime('%H:%M:%S')}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to initialize monitor: {e}")
    
    async def start_monitoring(self):
        """Start the monitoring dashboard."""
        print("📊 STARTING ML TRADING MONITOR")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 60)
        
        try:
            while True:
                await self._update_dashboard()
                await asyncio.sleep(30)  # Update every 30 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")
            self._print_session_summary()
    
    async def _update_dashboard(self):
        """Update the monitoring dashboard."""
        try:
            # Clear screen for fresh display
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get current account info
            account = self.trade_client.get_account()
            current_balance = float(account.portfolio_value)
            buying_power = float(account.buying_power)
            
            # Calculate session metrics
            session_pnl = current_balance - self.session_start_balance
            session_return = (session_pnl / self.session_start_balance) * 100
            target_progress = (session_pnl / self.target_daily_profit) * 100
            
            # Track balance history
            now = datetime.now()
            self.balance_history.append({
                'timestamp': now,
                'balance': current_balance,
                'pnl': session_pnl
            })
            
            # Keep only last 100 data points
            if len(self.balance_history) > 100:
                self.balance_history = self.balance_history[-100:]
            
            # Get current positions
            positions = self.trade_client.get_all_positions()
            
            # Get recent orders
            orders = self.trade_client.get_orders()
            today_orders = [o for o in orders if o.created_at.date() == now.date()]
            
            # Display dashboard
            self._display_header(now)
            self._display_account_summary(current_balance, session_pnl, session_return, 
                                        target_progress, buying_power)
            self._display_positions(positions)
            self._display_recent_orders(today_orders)
            self._display_performance_chart()
            self._display_alerts(session_pnl, target_progress)
            
            self.last_update = now
            
        except Exception as e:
            print(f"❌ Dashboard update error: {e}")
    
    def _display_header(self, now: datetime):
        """Display dashboard header."""
        session_duration = now - self.session_start_time
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        
        print("🚀 ML LIVE TRADING MONITOR")
        print("=" * 60)
        print(f"⏰ Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Session Duration: {int(hours):02d}:{int(minutes):02d}")
        print(f"🎯 Strategy: ML Daily Target Optimizer ($445/day proven)")
        print()
    
    def _display_account_summary(self, balance: float, pnl: float, return_pct: float,
                                target_progress: float, buying_power: float):
        """Display account summary section."""
        print("💰 ACCOUNT SUMMARY")
        print("-" * 30)
        
        # Account values
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        target_emoji = "🎯" if target_progress >= 100 else "⏳"
        
        print(f"💼 Portfolio Value: ${balance:,.2f}")
        print(f"{pnl_emoji} Session P&L: ${pnl:+.2f} ({return_pct:+.2f}%)")
        print(f"{target_emoji} Target Progress: {target_progress:.1f}% (${self.target_daily_profit})")
        print(f"💳 Buying Power: ${buying_power:,.2f}")
        print()
    
    def _display_positions(self, positions):
        """Display current positions."""
        print("📊 CURRENT POSITIONS")
        print("-" * 30)
        
        if not positions:
            print("   No open positions")
        else:
            total_market_value = 0
            for pos in positions:
                market_value = float(pos.market_value) if pos.market_value else 0
                unrealized_pnl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
                total_market_value += market_value
                
                pnl_emoji = "💚" if unrealized_pnl >= 0 else "❌"
                print(f"   {pnl_emoji} {pos.symbol}: {pos.qty} shares")
                print(f"      Market Value: ${market_value:.2f}")
                print(f"      Unrealized P&L: ${unrealized_pnl:+.2f}")
                print()
            
            print(f"📈 Total Position Value: ${total_market_value:.2f}")
        print()
    
    def _display_recent_orders(self, orders):
        """Display recent orders."""
        print("📋 TODAY'S ORDERS")
        print("-" * 30)
        
        if not orders:
            print("   No orders today")
        else:
            # Group by status
            filled_orders = [o for o in orders if o.status == 'filled']
            pending_orders = [o for o in orders if o.status in ['new', 'accepted', 'pending_new']]
            
            print(f"✅ Filled: {len(filled_orders)}")
            print(f"⏳ Pending: {len(pending_orders)}")
            
            # Show recent filled orders (last 5)
            if filled_orders:
                print("\n   Recent Fills:")
                for order in filled_orders[-5:]:
                    fill_time = order.filled_at.strftime('%H:%M:%S') if order.filled_at else 'N/A'
                    fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0
                    side_emoji = "🟢" if order.side == 'buy' else "🔴"
                    print(f"   {side_emoji} {fill_time}: {order.side.upper()} {order.qty} {order.symbol} @ ${fill_price:.2f}")
            
            # Show pending orders
            if pending_orders:
                print("\n   Pending Orders:")
                for order in pending_orders:
                    submit_time = order.created_at.strftime('%H:%M:%S')
                    side_emoji = "🟡"
                    print(f"   {side_emoji} {submit_time}: {order.side.upper()} {order.qty} {order.symbol} ({order.status})")
        print()
    
    def _display_performance_chart(self):
        """Display simple ASCII performance chart."""
        print("📈 PERFORMANCE TREND (Last 20 Updates)")
        print("-" * 30)
        
        if len(self.balance_history) < 2:
            print("   Insufficient data for chart")
            print()
            return
        
        # Get last 20 data points
        recent_data = self.balance_history[-20:]
        pnl_values = [d['pnl'] for d in recent_data]
        
        # Create simple ASCII chart
        max_pnl = max(pnl_values)
        min_pnl = min(pnl_values)
        
        if max_pnl == min_pnl:
            print("   Flat performance")
            print()
            return
        
        # Normalize to 0-10 scale
        chart_lines = []
        for pnl in pnl_values:
            if max_pnl != min_pnl:
                normalized = ((pnl - min_pnl) / (max_pnl - min_pnl)) * 10
            else:
                normalized = 5
            chart_lines.append(int(normalized))
        
        # Display chart
        for level in range(10, -1, -1):
            line = "   "
            for value in chart_lines:
                if value >= level:
                    line += "█"
                else:
                    line += " "
            
            # Add scale
            if level == 10:
                line += f"  ${max_pnl:+.0f}"
            elif level == 0:
                line += f"  ${min_pnl:+.0f}"
            
            print(line)
        
        print("   " + "─" * len(chart_lines))
        print(f"   Latest: ${pnl_values[-1]:+.2f}")
        print()
    
    def _display_alerts(self, session_pnl: float, target_progress: float):
        """Display alerts and notifications."""
        print("🚨 ALERTS & STATUS")
        print("-" * 30)
        
        alerts = []
        
        # Target achievement alerts
        if target_progress >= 100:
            alerts.append("🎯 DAILY TARGET ACHIEVED!")
        elif target_progress >= 75:
            alerts.append("🔥 Close to daily target!")
        elif target_progress <= -50:
            alerts.append("⚠️ Significant loss - review strategy")
        
        # Market hours alert
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            alerts.append(f"🕘 Market opens in {(market_open - now).total_seconds() / 60:.0f} minutes")
        elif now > market_close:
            alerts.append("🕛 Market closed")
        
        # Performance alerts
        if session_pnl > 100:
            alerts.append("💚 Strong positive performance")
        elif session_pnl < -100:
            alerts.append("❌ Significant negative performance")
        
        if not alerts:
            alerts.append("✅ All systems normal")
        
        for alert in alerts:
            print(f"   {alert}")
        
        print()
        print(f"🔄 Last Update: {self.last_update.strftime('%H:%M:%S') if self.last_update else 'N/A'}")
        print("   Press Ctrl+C to stop monitoring")
    
    def _print_session_summary(self):
        """Print final session summary."""
        try:
            account = self.trade_client.get_account()
            final_balance = float(account.portfolio_value)
            session_pnl = final_balance - self.session_start_balance
            session_return = (session_pnl / self.session_start_balance) * 100
            
            # Get today's orders
            orders = self.trade_client.get_orders()
            today_orders = [o for o in orders if o.created_at.date() == datetime.now().date()]
            filled_orders = [o for o in today_orders if o.status == 'filled']
            
            print("\n📊 ML TRADING MONITOR SESSION SUMMARY")
            print("=" * 50)
            print(f"⏰ Session Duration: {datetime.now() - self.session_start_time}")
            print(f"💰 Starting Balance: ${self.session_start_balance:,.2f}")
            print(f"💼 Final Balance: ${final_balance:,.2f}")
            print(f"📈 Session P&L: ${session_pnl:+.2f} ({session_return:+.2f}%)")
            print(f"🎯 Target Achievement: {(session_pnl/self.target_daily_profit)*100:.1f}%")
            print(f"📋 Orders Filled: {len(filled_orders)}")
            
            if session_pnl >= self.target_daily_profit:
                print("🎉 DAILY TARGET ACHIEVED!")
            elif session_pnl > 0:
                print("💚 Profitable session")
            else:
                print("📉 Loss session - review performance")
            
        except Exception as e:
            print(f"❌ Summary error: {e}")

async def main():
    """Main monitoring function."""
    print("🚀 ML LIVE TRADING MONITOR")
    print("Real-time monitoring for ML Daily Target Optimizer")
    print("=" * 50)
    
    # Get API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        return
    
    # Initialize and start monitor
    monitor = MLTradingMonitor(api_key=api_key, secret_key=secret_key)
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())