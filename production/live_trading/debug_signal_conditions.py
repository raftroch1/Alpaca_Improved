#!/usr/bin/env python3
"""
🔍 SIGNAL CONDITION DEBUGGER
================================
Diagnose why no signals are being generated despite $5 SPY drop

This script checks the exact values of:
- Momentum calculation (10-period)
- SMA crossover conditions
- Signal thresholds

Author: Trading Diagnostics
Date: 2025-08-05
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

def get_recent_spy_data():
    """Get recent SPY data for signal debugging"""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API keys not found!")
        return None
    
    data_client = StockHistoricalDataClient(api_key, secret_key)
    
    # Get last 3 days of 5-minute data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3)
    
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_time,
        end=end_time
    )
    
    response = data_client.get_stock_bars(request)
    df = response.df.reset_index().set_index('timestamp')
    
    return df

def analyze_signal_conditions(df):
    """Analyze why signals aren't being generated"""
    
    print("🔍 SIGNAL CONDITION ANALYSIS")
    print("=" * 50)
    
    # Calculate the same indicators as the live trader
    df['sma_short'] = df['close'].rolling(window=5).mean()
    df['sma_long'] = df['close'].rolling(window=20).mean()
    df['momentum'] = (df['close'] / df['close'].shift(10) - 1) * 100
    
    # Get the last 5 data points for analysis
    recent_data = df.tail(10)
    
    print("\n📊 RECENT DATA (Last 10 periods):")
    print(recent_data[['close', 'sma_short', 'sma_long', 'momentum']].round(2))
    
    # Check current conditions
    latest = recent_data.iloc[-1]
    prev = recent_data.iloc[-2]
    
    print(f"\n🔍 CURRENT CONDITIONS:")
    print(f"   📈 SPY Price: ${latest['close']:.2f}")
    print(f"   📊 SMA Short (5): ${latest['sma_short']:.2f}")
    print(f"   📊 SMA Long (20): ${latest['sma_long']:.2f}")
    print(f"   🚀 Momentum (10-period): {latest['momentum']:.2f}%")
    
    print(f"\n📊 SIGNAL REQUIREMENTS:")
    print(f"   🐻 BEARISH: momentum < -1.0% AND sma_short < sma_long")
    print(f"   🐂 BULLISH: momentum > +1.0% AND sma_short > sma_long")
    
    print(f"\n✅ CONDITION CHECK:")
    
    # Check momentum conditions
    if latest['momentum'] < -1.0:
        print(f"   ✅ BEARISH momentum: {latest['momentum']:.2f}% < -1.0%")
    elif latest['momentum'] > 1.0:
        print(f"   ✅ BULLISH momentum: {latest['momentum']:.2f}% > +1.0%")
    else:
        print(f"   ❌ Momentum too weak: {latest['momentum']:.2f}% (need >1% or <-1%)")
    
    # Check SMA crossover
    if latest['sma_short'] > latest['sma_long']:
        print(f"   📈 SMA: SHORT above LONG ({latest['sma_short']:.2f} > {latest['sma_long']:.2f})")
    else:
        print(f"   📉 SMA: SHORT below LONG ({latest['sma_short']:.2f} < {latest['sma_long']:.2f})")
    
    # Final signal determination
    print(f"\n🎯 SIGNAL ANALYSIS:")
    
    if latest['momentum'] > 1.0 and latest['sma_short'] > latest['sma_long']:
        print(f"   🐂 BULLISH SIGNAL WOULD BE GENERATED!")
    elif latest['momentum'] < -1.0 and latest['sma_short'] < latest['sma_long']:
        print(f"   🐻 BEARISH SIGNAL WOULD BE GENERATED!")
    else:
        print(f"   ❌ NO SIGNAL - Conditions not met")
        
        if abs(latest['momentum']) < 1.0:
            print(f"      • Momentum too weak: {latest['momentum']:.2f}%")
        if latest['momentum'] > 1.0 and latest['sma_short'] <= latest['sma_long']:
            print(f"      • Bullish momentum but SMAs not aligned")
        if latest['momentum'] < -1.0 and latest['sma_short'] >= latest['sma_long']:
            print(f"      • Bearish momentum but SMAs not aligned")
    
    # Price change analysis
    price_change = (latest['close'] / prev['close'] - 1) * 100
    print(f"\n📊 RECENT PRICE ACTION:")
    print(f"   💱 Last Period Change: {price_change:.2f}%")
    print(f"   📈 Price: ${prev['close']:.2f} → ${latest['close']:.2f}")
    
    # Suggest threshold adjustments
    print(f"\n💡 THRESHOLD SUGGESTIONS:")
    if abs(latest['momentum']) < 1.0:
        print(f"   📉 Lower momentum threshold from ±1.0% to ±{abs(latest['momentum'])*0.8:.1f}%")
    
    return latest

def suggest_improvements():
    """Suggest improvements to signal generation"""
    print(f"\n🔧 SIGNAL IMPROVEMENT SUGGESTIONS:")
    print(f"=" * 50)
    print(f"1. 📉 Lower momentum threshold from ±1.0% to ±0.5%")
    print(f"2. 🔄 Add volatility-based adaptive thresholds")
    print(f"3. ⚡ Use shorter momentum periods (5 instead of 10)")
    print(f"4. 🎯 Add volume confirmation")
    print(f"5. 📊 Consider RSI or other momentum indicators")

if __name__ == "__main__":
    print("🔍 DEBUGGING SIGNAL CONDITIONS")
    print("🎯 Why no signals despite $5 SPY drop?")
    print("=" * 50)
    
    # Get data
    df = get_recent_spy_data()
    if df is None:
        print("❌ Failed to get market data")
        exit(1)
    
    # Analyze conditions
    latest_conditions = analyze_signal_conditions(df)
    
    # Suggest improvements
    suggest_improvements()
    
    print(f"\n🏁 Analysis complete!")