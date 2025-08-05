#!/usr/bin/env python3
"""
ML Live Trading Setup Validator
===============================

Validates that all components are properly configured for the ML live trading system.
Run this before starting live trading to ensure everything is ready.

Author: Alpaca Improved Team
Version: v1.0
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import traceback

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'examples', 'backtesting'))

def test_imports() -> Dict[str, bool]:
    """Test all required imports."""
    results = {}
    
    try:
        import pandas as pd
        results['pandas'] = True
    except ImportError:
        results['pandas'] = False
    
    try:
        import numpy as np
        results['numpy'] = True
    except ImportError:
        results['numpy'] = False
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        results['alpaca-py'] = True
    except ImportError:
        results['alpaca-py'] = False
    
    try:
        from ml_daily_target_optimizer import DailyTargetMLOptimizer
        results['ml_optimizer'] = True
    except ImportError:
        results['ml_optimizer'] = False
    
    try:
        from dotenv import load_dotenv
        results['python-dotenv'] = True
    except ImportError:
        results['python-dotenv'] = False
    
    return results

def test_credentials() -> Dict[str, bool]:
    """Test API credentials."""
    results = {}
    
    # Load environment variables from project root
    from dotenv import load_dotenv
    
    # Check for .env file in project root
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    env_file_path = os.path.join(project_root, '.env')
    
    load_dotenv(env_file_path)
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    results['env_file_exists'] = os.path.exists(env_file_path)
    results['api_key_set'] = bool(api_key)
    results['secret_key_set'] = bool(secret_key)
    
    # Test API connection
    if api_key and secret_key:
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
            account = client.get_account()
            results['api_connection'] = True
            results['paper_trading'] = account.trading_blocked == False
            results['buying_power'] = float(account.buying_power) > 0
        except Exception as e:
            results['api_connection'] = False
            results['api_error'] = str(e)
    else:
        results['api_connection'] = False
    
    return results

def test_ml_components() -> Dict[str, bool]:
    """Test ML components."""
    results = {}
    
    try:
        # Test ML optimizer initialization
        from ml_daily_target_optimizer import DailyTargetMLOptimizer
        optimizer = DailyTargetMLOptimizer(target_daily_return=0.01)
        results['ml_optimizer_init'] = True
        
        # Test signal optimization
        test_signal = {
            'timestamp': datetime.now(),
            'signal_type': 'BULLISH',
            'momentum': 2.0,
            'volatility': 0.2,
            'price': 500.0,
            'price_change': 0.01,
            'volume': 1000,
            'recent_win_rate': 0.6,
            'account_value': 25000
        }
        
        ml_signal = optimizer.optimize_signal(test_signal)
        results['signal_processing'] = True
        results['confidence_calculation'] = 0 <= ml_signal.confidence <= 1
        results['position_sizing'] = 0 <= ml_signal.recommended_position_size <= 0.15
        
    except Exception as e:
        results['ml_optimizer_init'] = False
        results['ml_error'] = str(e)
    
    return results

def test_market_data() -> Dict[str, bool]:
    """Test market data access."""
    results = {}
    
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from datetime import timedelta
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            results['data_client_init'] = False
            return results
        
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        
        # Test SPY data retrieval
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        response = client.get_stock_bars(request)
        df = response.df.reset_index()
        
        results['data_client_init'] = True
        results['spy_data_available'] = not df.empty
        results['sufficient_data'] = len(df) >= 10
        
    except Exception as e:
        results['data_client_init'] = False
        results['data_error'] = str(e)
    
    return results

def test_trading_system() -> Dict[str, bool]:
    """Test basic trading system components."""
    results = {}
    
    try:
        # Test live trader import
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ml_live_paper_trader import MLLivePaperTradingEngine, MLLiveTrade
        results['live_trader_import'] = True
        
        # Test initialization (without starting trading)
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if api_key and secret_key:
            engine = MLLivePaperTradingEngine(
                api_key=api_key,
                secret_key=secret_key,
                target_daily_profit=250
            )
            results['engine_init'] = True
            results['ml_optimizer_ready'] = hasattr(engine, 'ml_optimizer')
            results['trade_client_ready'] = hasattr(engine, 'trade_client')
        else:
            results['engine_init'] = False
    
    except Exception as e:
        results['live_trader_import'] = False
        results['system_error'] = str(e)
    
    return results

def print_validation_results():
    """Print comprehensive validation results."""
    print("🚀 ML LIVE TRADING SETUP VALIDATOR")
    print("=" * 50)
    print()
    
    # Test imports
    print("📦 DEPENDENCY CHECK")
    print("-" * 20)
    import_results = test_imports()
    for package, status in import_results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {package}")
    print()
    
    # Test credentials
    print("🔑 CREDENTIALS CHECK") 
    print("-" * 20)
    cred_results = test_credentials()
    for key, status in cred_results.items():
        if key.endswith('_error'):
            continue
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {key.replace('_', ' ').title()}")
    
    if 'api_error' in cred_results:
        print(f"   Error: {cred_results['api_error']}")
    print()
    
    # Test ML components
    print("🤖 ML COMPONENTS CHECK")
    print("-" * 20)
    ml_results = test_ml_components()
    for key, status in ml_results.items():
        if key.endswith('_error'):
            continue
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {key.replace('_', ' ').title()}")
    
    if 'ml_error' in ml_results:
        print(f"   Error: {ml_results['ml_error']}")
    print()
    
    # Test market data
    print("📊 MARKET DATA CHECK")
    print("-" * 20)
    data_results = test_market_data()
    for key, status in data_results.items():
        if key.endswith('_error'):
            continue
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {key.replace('_', ' ').title()}")
    
    if 'data_error' in data_results:
        print(f"   Error: {data_results['data_error']}")
    print()
    
    # Test trading system
    print("🎯 TRADING SYSTEM CHECK")
    print("-" * 20)
    system_results = test_trading_system()
    for key, status in system_results.items():
        if key.endswith('_error'):
            continue
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {key.replace('_', ' ').title()}")
    
    if 'system_error' in system_results:
        print(f"   Error: {system_results['system_error']}")
    print()
    
    # Overall assessment
    all_results = {**import_results, **cred_results, **ml_results, **data_results, **system_results}
    error_keys = [k for k in all_results.keys() if k.endswith('_error')]
    status_keys = [k for k in all_results.keys() if not k.endswith('_error')]
    
    total_checks = len(status_keys)
    passed_checks = sum(1 for k in status_keys if all_results[k])
    
    print("📋 OVERALL ASSESSMENT")
    print("-" * 20)
    print(f"✅ Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 ALL SYSTEMS READY FOR ML LIVE TRADING!")
        print("🚀 You can now run: python ml_live_paper_trader.py")
    else:
        print("⚠️  Some issues need to be resolved before live trading")
        print("🔧 Please fix the ❌ items above and run validation again")
    
    print()
    
    # Quick start reminder
    if passed_checks == total_checks:
        print("🎯 QUICK START:")
        print("1. cd production/live_trading")
        print("2. python ml_live_paper_trader.py")
        print("3. Type 'yes' when prompted")
        print("4. Monitor the $250/day target achievement!")

def main():
    """Main validation function."""
    try:
        print_validation_results()
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print(f"📋 Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()