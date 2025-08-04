#!/usr/bin/env python3
"""
Background 0DTE Paper Trader

Run the proven $250/day strategy in the background as a daemon process.
Includes logging, status monitoring, and safe shutdown capabilities.

Author: Alpaca Improved Team
License: MIT
"""

import os
import sys
import time
import signal
import logging
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Import our paper trader
sys.path.append(os.path.join(os.path.dirname(__file__)))
from simple_0dte_paper_trader import Simple0DTEPaperTrader

# Load environment variables
load_dotenv()


class BackgroundTrader:
    """Background daemon for running the 0DTE paper trading strategy."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the background trader."""
        
        # Create logs directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Trading components
        self.trader: Optional[Simple0DTEPaperTrader] = None
        self.running = False
        self.trade_loop_task: Optional[asyncio.Task] = None
        
        # Signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🚀 Background Trader initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        # Setup main logger
        self.logger = logging.getLogger('BackgroundTrader')
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logs
        log_file = self.log_dir / f"trader_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for status updates
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Performance log
        perf_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        self.perf_logger = logging.getLogger('Performance')
        self.perf_logger.setLevel(logging.INFO)
        perf_handler = logging.FileHandler(perf_file)
        perf_handler.setFormatter(simple_formatter)
        self.perf_logger.addHandler(perf_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"📶 Received signal {signum} - shutting down gracefully...")
        self.stop()
    
    def create_pid_file(self):
        """Create PID file for daemon management."""
        pid_file = Path("trader.pid")
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        return pid_file
    
    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        
        # Check API credentials
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            self.logger.error("❌ Missing API credentials in .env file")
            return False
        
        self.logger.info("✅ API credentials found")
        return True
    
    def initialize_trader(self) -> bool:
        """Initialize the paper trader."""
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            self.trader = Simple0DTEPaperTrader(api_key, secret_key)
            self.logger.info("✅ Paper trader initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trader: {e}")
            return False
    
    def log_performance(self):
        """Log current performance metrics."""
        if not self.trader:
            return
        
        try:
            account = self.trader.get_account_info()
            if account:
                self.perf_logger.info(
                    f"P&L: ${self.trader.daily_pnl:+.2f} | "
                    f"Trades: {self.trader.trades_today} | "
                    f"Positions: {len(self.trader.positions)} | "
                    f"Buying Power: ${account['buying_power']:,.2f}"
                )
        except Exception as e:
            self.logger.warning(f"⚠️ Performance logging error: {e}")
    
    async def trading_loop(self):
        """Main trading loop with error handling."""
        self.logger.info("🎯 Starting trading loop")
        
        last_performance_log = time.time()
        performance_log_interval = 300  # Log every 5 minutes
        
        try:
            while self.running:
                try:
                    # Check market hours
                    if not self.trader.is_market_open():
                        await asyncio.sleep(300)  # Check every 5 minutes when closed
                        continue
                    
                    # Look for signals
                    signals = self.trader.check_for_signals()
                    if signals:
                        self.logger.info(f"⚡ Found {len(signals)} trading signals")
                        
                        for signal in signals:
                            if self.trader.execute_signal(signal):
                                self.logger.info(f"✅ Executed {signal.signal_type} signal")
                            else:
                                self.logger.info(f"⚠️ Skipped {signal.signal_type} signal")
                    
                    # Monitor existing positions
                    if self.trader.positions:
                        self.trader.monitor_positions()
                    
                    # Log performance periodically
                    current_time = time.time()
                    if current_time - last_performance_log >= performance_log_interval:
                        self.log_performance()
                        last_performance_log = current_time
                    
                    # Wait before next check
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"❌ Trading loop error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except asyncio.CancelledError:
            self.logger.info("🛑 Trading loop cancelled")
        except Exception as e:
            self.logger.error(f"❌ Fatal trading loop error: {e}")
        
        self.logger.info("📊 Trading loop ended")
    
    async def start_async(self):
        """Start the async trading components."""
        if not self.check_requirements():
            return False
        
        if not self.initialize_trader():
            return False
        
        self.running = True
        
        # Start trading loop
        self.trade_loop_task = asyncio.create_task(self.trading_loop())
        
        try:
            await self.trade_loop_task
        except asyncio.CancelledError:
            self.logger.info("🛑 Async trading cancelled")
        
        return True
    
    def start(self, daemon: bool = False):
        """Start the background trader."""
        
        if daemon:
            self.logger.info("🔄 Starting in daemon mode...")
            # Create PID file
            pid_file = self.create_pid_file()
            self.logger.info(f"📝 PID file created: {pid_file}")
        
        self.logger.info("🚀 Background 0DTE Paper Trader starting...")
        self.logger.info("🎯 Target: $250/day with proven strategy")
        
        try:
            # Run the async trading loop
            asyncio.run(self.start_async())
            
        except KeyboardInterrupt:
            self.logger.info("⌨️ Keyboard interrupt received")
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error: {e}")
            
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop the trading gracefully."""
        self.logger.info("🛑 Stopping background trader...")
        self.running = False
        
        if self.trade_loop_task and not self.trade_loop_task.done():
            self.trade_loop_task.cancel()
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("🧹 Cleaning up...")
        
        # Log final performance
        if self.trader:
            self.logger.info(f"📊 Final Daily P&L: ${self.trader.daily_pnl:+.2f}")
            self.logger.info(f"📈 Total Trades: {self.trader.trades_today}")
        
        # Remove PID file
        pid_file = Path("trader.pid")
        if pid_file.exists():
            pid_file.unlink()
            self.logger.info("🗑️ PID file removed")
        
        self.logger.info("✅ Cleanup complete")
    
    def status(self) -> dict:
        """Get current status."""
        if not self.trader:
            return {"status": "not_initialized"}
        
        account = self.trader.get_account_info()
        
        return {
            "status": "running" if self.running else "stopped",
            "daily_pnl": self.trader.daily_pnl,
            "trades_today": self.trader.trades_today,
            "positions": len(self.trader.positions),
            "buying_power": account.get('buying_power', 0) if account else 0,
            "market_open": self.trader.is_market_open()
        }


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Background 0DTE Paper Trader")
    parser.add_argument("--daemon", "-d", action="store_true", 
                       help="Run as background daemon")
    parser.add_argument("--log-dir", default="logs", 
                       help="Directory for log files")
    parser.add_argument("--status", "-s", action="store_true",
                       help="Show status and exit")
    
    args = parser.parse_args()
    
    # Check for existing instance
    pid_file = Path("trader.pid")
    if pid_file.exists() and not args.status:
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            
            # Check if process is running
            try:
                os.kill(pid, 0)  # Doesn't actually kill, just checks
                print(f"❌ Trader already running with PID {pid}")
                print(f"🛑 Stop it first with: kill {pid}")
                return
            except OSError:
                # Process not running, remove stale PID file
                pid_file.unlink()
                
        except (ValueError, FileNotFoundError):
            # Invalid or missing PID file
            if pid_file.exists():
                pid_file.unlink()
    
    trader = BackgroundTrader(log_dir=args.log_dir)
    
    if args.status:
        # Show status and exit
        status = trader.status()
        print(f"📊 BACKGROUND TRADER STATUS")
        print(f"Status: {status['status']}")
        if status['status'] != 'not_initialized':
            print(f"Daily P&L: ${status['daily_pnl']:+.2f}")
            print(f"Trades Today: {status['trades_today']}")
            print(f"Positions: {status['positions']}")
            print(f"Market Open: {status['market_open']}")
        return
    
    print("🚀 BACKGROUND 0DTE PAPER TRADER")
    print("🎯 Proven Strategy: $250/day Target")
    
    if args.daemon:
        print("🔄 Running in background daemon mode...")
        print("📝 Logs will be written to:", args.log_dir)
        print("🛑 Stop with: kill $(cat trader.pid)")
    else:
        print("💻 Running in foreground mode...")
        print("🛑 Stop with: Ctrl+C")
    
    print("🚨 PAPER TRADING ONLY - NO REAL MONEY")
    
    if not args.daemon:
        try:
            response = input("Continue? (yes/no): ").lower().strip()
            if response not in ['yes', 'y']:
                print("👋 Cancelled")
                return
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return
    
    # Start the trader
    trader.start(daemon=args.daemon)


if __name__ == "__main__":
    main()