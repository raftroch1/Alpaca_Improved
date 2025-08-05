#!/usr/bin/env python3
"""
Profitable Fixed Strategy Backtest

Tests the improved strategy based on the ONLY profitable backtest (+432% return).
Key improvements:
1. Simple MA Shift signals (±0.3 threshold) 
2. Smart exit management instead of random outcomes
3. No expiry exits - avoid the 50%+ expiry problem
4. Proven position sizing and cost structure

Expected results: 6+ trades/day with 60%+ win rate targeting $250/day.

Author: Alpaca Improved Team
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the strategy to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from profitable_fixed_0dte import ProfitableStrategy, ProfitableSignal, ExitReason

# Load environment variables
load_dotenv()

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class ProfitableBacktester:
    """Backtester for the profitable fixed strategy."""
    
    def __init__(self, initial_capital: float = 25000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.trades = []
        
        # Track daily performance
        self.daily_results = {}
        
    def simulate_profitable_trade(self, 
                                 signal: ProfitableSignal, 
                                 strategy: ProfitableStrategy) -> Optional[Dict]:
        """Simulate a complete trade using the profitable logic."""
        
        if not strategy.should_trade(signal):
            return None
        
        # Select strike and option type
        strike, option_type = strategy.select_strike_and_type(signal)
        
        # Estimate option price
        entry_price = strategy.estimate_option_price(
            signal.price, strike, option_type, signal.timestamp
        )
        
        # Calculate position size
        contracts = strategy.calculate_position_size(signal, entry_price, self.cash)
        
        # Calculate entry costs
        entry_costs = strategy.calculate_realistic_costs(contracts, entry_price)
        total_entry_cost = (contracts * entry_price * 100) + entry_costs
        
        # Check if we have enough capital
        if total_entry_cost > self.cash:
            return None
        
        # Execute entry
        self.cash -= total_entry_cost
        strategy.trades_today += 1
        
        # Smart exit simulation
        exit_price, exit_reason, exit_time = strategy.simulate_smart_exit(
            signal, entry_price, signal.timestamp
        )
        
        # Calculate exit proceeds
        exit_costs = strategy.calculate_realistic_costs(contracts, exit_price)
        gross_exit_value = contracts * exit_price * 100
        net_exit_value = gross_exit_value - exit_costs
        
        # Add proceeds back to cash
        self.cash += net_exit_value
        
        # Calculate P&L
        total_pnl = net_exit_value - total_entry_cost
        
        # Update strategy daily P&L
        strategy.daily_pnl += total_pnl
        
        # Track daily results
        trade_date = signal.timestamp.date()
        if trade_date not in self.daily_results:
            self.daily_results[trade_date] = {'trades': 0, 'pnl': 0}
        
        self.daily_results[trade_date]['trades'] += 1
        self.daily_results[trade_date]['pnl'] += total_pnl
        
        # Time held
        time_held_hours = (exit_time - signal.timestamp).total_seconds() / 3600
        
        # Generate option symbol
        expiry_str = signal.timestamp.strftime('%y%m%d')
        option_char = 'C' if option_type == 'CALL' else 'P'
        option_symbol = f"SPY{expiry_str}{option_char}{int(strike*1000):08d}"
        
        trade_result = {
            'entry_date': signal.timestamp,
            'exit_date': exit_time,
            'signal_type': signal.signal_type,
            'signal_strength': signal.signal_strength,
            'option_symbol': option_symbol,
            'option_type': option_type,
            'strike': strike,
            'contracts': contracts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': total_pnl,
            'total_costs': entry_costs + exit_costs,
            'time_held_hours': time_held_hours,
            'exit_reason': exit_reason.value,
            'underlying_entry': signal.price,
            'win': total_pnl > 0
        }
        
        self.trades.append(trade_result)
        
        return trade_result
    
    def get_performance_analysis(self) -> Dict:
        """Get comprehensive performance analysis."""
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['win']])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum([t['pnl'] for t in self.trades])
        total_costs = sum([t['total_costs'] for t in self.trades])
        
        final_value = self.cash
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Daily analysis
        trading_days = len(self.daily_results)
        avg_daily_pnl = total_pnl / max(trading_days, 1)
        avg_trades_per_day = total_trades / max(trading_days, 1)
        
        # Days hitting target
        target_days = len([day for day, results in self.daily_results.items() 
                          if results['pnl'] >= 250])
        target_hit_rate = (target_days / max(trading_days, 1)) * 100
        
        # Risk metrics
        trade_pnls = [t['pnl'] for t in self.trades]
        best_trade = max(trade_pnls)
        worst_trade = min(trade_pnls)
        
        # Exit analysis
        exit_reasons = {}
        for trade in self.trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Signal strength analysis
        avg_signal_strength = np.mean([t['signal_strength'] for t in self.trades])
        
        # Time analysis
        avg_hold_time = np.mean([t['time_held_hours'] for t in self.trades])
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate_pct": win_rate,
            "total_pnl": total_pnl,
            "total_costs": total_costs,
            "final_value": final_value,
            "total_return_pct": total_return,
            "trading_days": trading_days,
            "avg_daily_pnl": avg_daily_pnl,
            "avg_trades_per_day": avg_trades_per_day,
            "target_days": target_days,
            "target_hit_rate_pct": target_hit_rate,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "exit_reasons": exit_reasons,
            "avg_signal_strength": avg_signal_strength,
            "avg_hold_time_hours": avg_hold_time,
            "cost_per_trade": total_costs / total_trades
        }


def run_profitable_fixed_backtest():
    """Run the profitable fixed strategy backtest."""
    print("🎯 PROFITABLE FIXED 0DTE STRATEGY BACKTEST")
    print("✅ Based on +432% Return Model")
    print("🔧 Smart Exit Management + Simple Signals")
    print("=" * 65)
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please check your .env file")
        return None
    
    # Initialize strategy and backtester
    strategy = ProfitableStrategy(target_daily_profit=250, account_size=25000)
    backtester = ProfitableBacktester(initial_capital=25000)
    
    # Display strategy configuration
    summary = strategy.get_strategy_summary()
    print(f"🔧 Strategy: {summary['strategy_name']}")
    print(f"🎯 Target: ${summary['target_daily_profit']}/day")
    print(f"📊 Based on: {summary['based_on']}")
    
    print(f"\n🔧 Key Improvements:")
    for i, improvement in enumerate(summary['key_improvements'][:4], 1):
        print(f"   {i}. {improvement}")
    
    # Get market data
    print(f"\n📊 Fetching SPY data...")
    
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Use same data range as profitable model
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 8, 15)
    
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start_date,
        end=end_date
    )
    
    response = client.get_stock_bars(request)
    spy_data = response.df.reset_index().set_index('timestamp')
    
    print(f"✅ Retrieved {len(spy_data)} data points")
    print(f"📅 Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
    
    # Generate signals using profitable logic
    print(f"\n⚡ Generating signals with PROFITABLE logic...")
    print(f"🎯 Simple MA Shift (±{strategy.osc_threshold} threshold)")
    
    signals = strategy.generate_profitable_signals(spy_data)
    
    print(f"\n📊 PROFITABLE SIGNAL RESULTS:")
    print(f"🎯 Total Signals: {len(signals)}")
    
    if not signals:
        print("❌ No signals generated - check thresholds")
        return None
    
    # Analyze signal distribution
    signal_types = {'BULLISH': 0, 'BEARISH': 0}
    strengths = []
    
    for signal in signals:
        signal_types[signal.signal_type] += 1
        strengths.append(signal.signal_strength)
    
    print(f"📈 Signal Distribution:")
    for sig_type, count in signal_types.items():
        print(f"   {sig_type}: {count} signals")
    
    print(f"📊 Avg Signal Strength: {np.mean(strengths):.2f}")
    
    # Execute trades
    print(f"\n🔄 Executing trades with SMART exit management...")
    print(f"⚡ Target: {strategy.max_trades_per_day} trades/day max")
    
    executed_trades = 0
    
    for i, signal in enumerate(signals[:100]):  # Limit for testing
        if i % 20 == 0:
            print(f"\r🔄 Processing signal {i+1}/{min(100, len(signals))}: "
                  f"{signal.signal_type} strength={signal.signal_strength:.2f}", end="")
        
        trade = backtester.simulate_profitable_trade(signal, strategy)
        
        if trade:
            executed_trades += 1
            if executed_trades <= 3:  # Show first few trades
                print(f"\n  ✅ Trade {executed_trades}: {trade['option_type']} "
                      f"${trade['strike']} -> P&L: ${trade['pnl']:+.2f} "
                      f"({trade['exit_reason']})")
    
    print(f"\n\n✅ Profitable backtest complete!")
    
    # Display results
    print("\n" + "=" * 65)
    print("📊 PROFITABLE FIXED STRATEGY RESULTS")
    print("=" * 65)
    
    performance = backtester.get_performance_analysis()
    
    if "error" in performance:
        print(f"❌ {performance['error']}")
        return None
    
    print(f"💰 Starting Capital: ${backtester.initial_capital:,.2f}")
    print(f"💼 Final Value: ${performance['final_value']:,.2f}")
    print(f"📈 Total Return: {performance['total_return_pct']:+.2f}%")
    print(f"💵 Total P&L: ${performance['total_pnl']:+,.2f}")
    print(f"💸 Total Costs: ${performance['total_costs']:,.2f}")
    
    print(f"\n🎯 DAILY TARGET ANALYSIS:")
    print(f"🎯 Target: $250/day")
    print(f"📊 Trading Days: {performance['trading_days']}")
    print(f"💵 Avg Daily P&L: ${performance['avg_daily_pnl']:+.2f}")
    print(f"📈 Avg Trades/Day: {performance['avg_trades_per_day']:.1f}")
    print(f"🎯 Days Hit $250+: {performance['target_days']} ({performance['target_hit_rate_pct']:.1f}%)")
    
    print(f"\n📊 TRADING PERFORMANCE:")
    print(f"📈 Total Trades: {performance['total_trades']}")
    print(f"🥇 Winning Trades: {performance['winning_trades']}")
    print(f"📈 Win Rate: {performance['win_rate_pct']:.1f}%")
    print(f"⏱️ Avg Hold Time: {performance['avg_hold_time_hours']:.1f} hours")
    print(f"📊 Avg Signal Strength: {performance['avg_signal_strength']:.2f}")
    print(f"💸 Cost per Trade: ${performance['cost_per_trade']:.2f}")
    
    print(f"\n📋 EXIT ANALYSIS:")
    for reason, count in performance['exit_reasons'].items():
        pct = (count / performance['total_trades']) * 100
        print(f"   {reason.replace('_', ' ').title()}: {count} trades ({pct:.1f}%)")
    
    print(f"\n🎯 RISK METRICS:")
    print(f"💚 Best Trade: ${performance['best_trade']:+.2f}")
    print(f"🔴 Worst Trade: ${performance['worst_trade']:+.2f}")
    
    # Strategy assessment
    print(f"\n🎯 PROFITABLE STRATEGY ASSESSMENT:")
    
    if performance['win_rate_pct'] >= 60 and performance['avg_daily_pnl'] >= 200:
        print(f"🎉 EXCELLENT! Win rate: {performance['win_rate_pct']:.1f}%, "
              f"Daily P&L: ${performance['avg_daily_pnl']:+.2f}")
        print(f"🚀 Ready for live paper trading!")
    elif performance['win_rate_pct'] >= 55 and performance['avg_daily_pnl'] >= 150:
        print(f"💪 GOOD PERFORMANCE! Win rate: {performance['win_rate_pct']:.1f}%, "
              f"Daily P&L: ${performance['avg_daily_pnl']:+.2f}")
        print(f"✅ Ready for careful live testing")
    elif performance['win_rate_pct'] >= 50:
        print(f"🔧 BREAKEVEN RANGE: Win rate: {performance['win_rate_pct']:.1f}%, "
              f"Daily P&L: ${performance['avg_daily_pnl']:+.2f}")
        print(f"⚠️ Consider further optimization")
    else:
        print(f"❌ Below expectations: Win rate: {performance['win_rate_pct']:.1f}%, "
              f"Daily P&L: ${performance['avg_daily_pnl']:+.2f}")
        print(f"🔧 Strategy needs improvement")
    
    print(f"\n📈 NEXT STEPS:")
    if performance['avg_daily_pnl'] >= 200:
        print(f"🎯 Update live trading system with profitable strategy")
        print(f"⚡ Test with current market conditions")
        print(f"📊 Monitor performance vs backtest")
    else:
        print(f"🔧 Fine-tune exit management")
        print(f"📊 Adjust signal thresholds")
        print(f"🎯 Test with different market periods")
    
    return {
        'performance': performance,
        'signals_generated': len(signals),
        'executed_trades': executed_trades,
        'strategy_summary': summary
    }


if __name__ == "__main__":
    print("🎯 PROFITABLE FIXED 0DTE STRATEGY")
    print("✅ Based on +432% Return Backtest")
    print("🔧 Smart Exit + Simple Signals")
    print("=" * 50)
    
    results = run_profitable_fixed_backtest()
    
    if results:
        performance = results['performance']
        if performance['avg_daily_pnl'] >= 200 and performance['win_rate_pct'] >= 55:
            print(f"\n🎉 PROFITABLE STRATEGY VALIDATED!")
            print(f"🚀 READY FOR LIVE IMPLEMENTATION!")
        else:
            print(f"\n🔧 STRATEGY NEEDS REFINEMENT")
            print(f"📊 Continue optimization")
    else:
        print(f"\n⚠️ Strategy validation failed")