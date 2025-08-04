#!/usr/bin/env python3
"""
Live 0DTE Paper Trading Launcher

Simple launcher for the optimized 0DTE paper trading engine.
Includes real-time monitoring dashboard and performance tracking.

Author: Alpaca Improved Team
License: MIT
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List
import json
import threading
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Add the trading engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'trading'))
from paper_trading_engine import PaperTradingEngine, TradingStatus

# Load environment variables
load_dotenv()


class PaperTradingDashboard:
    """Real-time monitoring dashboard for paper trading."""
    
    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
        self.running = False
        self.update_interval = 10  # Update every 10 seconds
        
    def start_monitoring(self):
        """Start real-time monitoring dashboard."""
        self.running = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring dashboard."""
        self.running = False
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._display_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
                time.sleep(5)
    
    def _display_dashboard(self):
        """Display real-time dashboard."""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        print("🚀 LIVE 0DTE PAPER TRADING DASHBOARD")
        print("🎯 Optimized Strategy - $250/day Target")
        print("=" * 60)
        
        # Get current status
        status = self.engine.get_status_summary()
        
        # Header info
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Status: {self._get_status_emoji(status['status'])} {status['status'].upper()}")
        print(f"💹 SPY Price: ${status['current_spy_price']:.2f}")
        
        print("\n" + "=" * 60)
        print("📈 DAILY PERFORMANCE")
        print("=" * 60)
        
        # Performance metrics
        metrics = status['metrics']
        print(f"🎯 Target Achievement: {status['target_achievement']}")
        print(f"💰 Daily P&L: {status['daily_pnl']}")
        print(f"💎 Unrealized P&L: ${metrics['unrealized_pnl']:+.2f}")
        print(f"💳 Buying Power: ${metrics['buying_power']:,.2f}")
        
        print(f"\n📊 TRADING STATISTICS")
        print(f"📈 Total Trades: {metrics['total_trades']}")
        print(f"🏆 Win Rate: {status['win_rate']}")
        print(f"💼 Open Positions: {status['positions']}")
        print(f"📋 Active Orders: {status['active_orders']}")
        
        if metrics['total_trades'] > 0:
            print(f"🥇 Largest Win: ${metrics['largest_win']:+.2f}")
            print(f"🔻 Largest Loss: ${metrics['largest_loss']:+.2f}")
        
        print("\n" + "=" * 60)
        print("📍 CURRENT POSITIONS")
        print("=" * 60)
        
        if status['positions'] > 0:
            self._display_positions()
        else:
            print("📭 No open positions")
        
        print("\n" + "=" * 60)
        print("⚡ CONTROLS: Ctrl+C to stop | Updates every 10s")
        print("=" * 60)
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for trading status."""
        status_emojis = {
            'running': '🟢',
            'stopped': '🔴',
            'paused': '🟡', 
            'starting': '🟠',
            'error': '🔴',
            'market_closed': '🌙'
        }
        return status_emojis.get(status, '⚪')
    
    def _display_positions(self):
        """Display current positions."""
        try:
            for order_id, trade in list(self.engine.current_positions.items()):
                duration = datetime.now() - trade.entry_time
                duration_str = f"{int(duration.total_seconds() / 60)}m"
                
                print(f"📋 {trade.option_symbol}")
                print(f"   📊 {trade.contracts} contracts @ ${trade.entry_price:.2f}")
                print(f"   💰 P&L: ${trade.unrealized_pnl:+.2f} | ⏱️ {duration_str}")
                print(f"   🎯 Target: ${trade.target_exit_price:.2f} | 🛑 Stop: ${trade.stop_loss_price:.2f}")
                print()
        except Exception as e:
            print(f"❌ Position display error: {e}")


async def run_live_paper_trading():
    """Main function to run live paper trading."""
    
    print("🚀 LIVE 0DTE PAPER TRADING LAUNCHER")
    print("🎯 Optimized Strategy - $250/day Target")
    print("=" * 50)
    
    # Get API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API credentials not found!")
        print("📝 Please check your .env file contains:")
        print("   ALPACA_API_KEY=your_paper_key_here")
        print("   ALPACA_SECRET_KEY=your_paper_secret_here")
        return
    
    # Configuration
    config = {
        'target_daily_profit': 250,
        'account_size': 25000,
        'paper': True  # ALWAYS use paper trading for safety
    }
    
    print(f"🔧 Configuration:")
    print(f"   🎯 Daily Target: ${config['target_daily_profit']}")
    print(f"   💰 Account Size: ${config['account_size']:,}")
    print(f"   📋 Paper Trading: {config['paper']}")
    print()
    
    # Initialize engine
    print("⚙️ Initializing paper trading engine...")
    engine = PaperTradingEngine(
        api_key=api_key,
        secret_key=secret_key,
        target_daily_profit=config['target_daily_profit'],
        account_size=config['account_size'],
        paper=config['paper']
    )
    
    # Initialize dashboard
    dashboard = PaperTradingDashboard(engine)
    
    try:
        print("🚀 Starting live paper trading...")
        print("📊 Dashboard will appear in 5 seconds...")
        await asyncio.sleep(5)
        
        # Start monitoring dashboard
        dashboard.start_monitoring()
        
        # Start trading engine
        await engine.start_trading()
        
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt detected...")
        print("🛑 Shutting down paper trading engine...")
        
        # Stop dashboard
        dashboard.stop_monitoring()
        
        # Stop engine
        await engine.stop_trading()
        
        # Final summary
        print("\n📊 FINAL SESSION SUMMARY")
        print("=" * 40)
        status = engine.get_status_summary()
        metrics = status['metrics']
        
        print(f"💰 Daily P&L: {status['daily_pnl']}")
        print(f"📈 Total Trades: {metrics['total_trades']}")
        print(f"🏆 Win Rate: {status['win_rate']}")
        print(f"🎯 Target Achievement: {status['target_achievement']}")
        
        if metrics['total_trades'] > 0:
            print(f"🥇 Best Trade: ${metrics['largest_win']:+.2f}")
            print(f"🔻 Worst Trade: ${metrics['largest_loss']:+.2f}")
        
        print("\n✅ Paper trading session ended safely")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        dashboard.stop_monitoring()
        await engine.stop_trading()
        print("🛑 Emergency shutdown completed")


def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...")
    
    # Check API keys
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Missing API credentials!")
        print("📝 Create a .env file with your Alpaca paper trading keys:")
        print("   ALPACA_API_KEY=your_paper_key_here")
        print("   ALPACA_SECRET_KEY=your_paper_secret_here")
        print("🔗 Get paper trading keys at: https://app.alpaca.markets/paper/dashboard/overview")
        return False
    
    print("✅ API credentials found")
    
    # Check market hours
    current_time = datetime.now().time()
    market_open = dt_time(9, 30)  # 9:30 AM ET
    market_close = dt_time(16, 0)  # 4:00 PM ET
    
    if market_open <= current_time <= market_close:
        print("✅ Market is currently open")
    else:
        print("⚠️ Market is currently closed")
        print("📅 Market hours: 9:30 AM - 4:00 PM ET")
        print("🎯 You can still run the engine - it will wait for market open")
    
    return True


def show_strategy_info():
    """Display strategy information."""
    print("\n📋 STRATEGY INFORMATION")
    print("=" * 40)
    print("🎯 Strategy: Optimized Baseline 0DTE")
    print("💰 Target: $250/day with $25k account")
    print("📈 Win Rate: ~58.7% (backtested)")
    print("⏰ Holding Time: 0.5 - 4 hours")
    print("🔧 Exit Management: Smart exits, NO expiry")
    
    print("\n🛡️ RISK CONTROLS")
    print("=" * 40)
    print("🛑 Daily Stop Loss: 2.5% of capital")
    print("📊 Max Positions: 8 concurrent")
    print("💼 Max Position Size: 10% of capital")
    print("⏰ Time Exits: 30 min before market close")
    
    print("\n⚡ KEY FEATURES")
    print("=" * 40)
    print("✅ Real-time signal generation")
    print("✅ Automated order execution")
    print("✅ Smart exit management")
    print("✅ Risk monitoring")
    print("✅ Live performance tracking")
    print("✅ Paper trading safety")


if __name__ == "__main__":
    print("🚀 ALPACA IMPROVED - LIVE 0DTE PAPER TRADING")
    print("🎯 Bringing our $250/day strategy to LIVE markets!")
    print("=" * 55)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met - please fix and try again")
        exit(1)
    
    # Show strategy info
    show_strategy_info()
    
    # Confirmation
    print("\n🚨 IMPORTANT SAFETY NOTICE")
    print("=" * 40)
    print("📋 This is PAPER TRADING ONLY")
    print("💰 No real money will be traded")
    print("🎯 Perfect for strategy validation")
    print("⚠️ Never use live keys in this script")
    
    print("\n🎯 Ready to start live paper trading?")
    try:
        response = input("Type 'yes' to continue: ").lower().strip()
        if response != 'yes':
            print("👋 Cancelled by user")
            exit(0)
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        exit(0)
    
    # Run the paper trading engine
    try:
        asyncio.run(run_live_paper_trading())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("🛑 Paper trading engine failed to start")
        exit(1)