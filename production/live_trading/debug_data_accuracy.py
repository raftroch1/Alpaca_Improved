#!/usr/bin/env python3
"""
🔍 DATA ACCURACY DIAGNOSTIC TOOL
===============================

Debug and validate data accuracy between different sources.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
env_file_path = os.path.join(project_root, '.env')
load_dotenv(env_file_path)

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def debug_data_sources():
    """Debug data accuracy from multiple sources."""
    
    print("🔍 SPY DATA ACCURACY DIAGNOSTIC")
    print("=" * 50)
    
    # Initialize Alpaca client
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API keys not found")
        return
    
    data_client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key
    )
    
    print(f"📅 Current Time: {datetime.now()}")
    print(f"🌍 Timezone: {datetime.now().astimezone().tzinfo}")
    print()
    
    # Test 1: Latest Quote
    print("🔍 TEST 1: Latest Quote Data")
    print("-" * 30)
    try:
        quote_request = StockLatestQuoteRequest(symbol_or_symbols="SPY")
        latest_quotes = data_client.get_stock_latest_quote(quote_request)
        
        if "SPY" in latest_quotes:
            quote = latest_quotes["SPY"]
            print(f"✅ Latest Quote Retrieved:")
            print(f"   💰 Bid: ${quote.bid_price:.2f}")
            print(f"   💰 Ask: ${quote.ask_price:.2f}")
            print(f"   💰 Mid: ${(quote.bid_price + quote.ask_price)/2:.2f}")
            print(f"   📅 Quote Time: {quote.timestamp}")
            print(f"   📊 Bid Size: {quote.bid_size}")
            print(f"   📊 Ask Size: {quote.ask_size}")
        else:
            print("❌ No quote data available")
    except Exception as e:
        print(f"❌ Quote request failed: {e}")
    
    print()
    
    # Test 2: Recent 1-minute bars
    print("🔍 TEST 2: Recent 1-Minute Bars")
    print("-" * 30)
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=30)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        response = data_client.get_stock_bars(request)
        df = response.df.reset_index()
        
        if not df.empty:
            latest_bar = df.iloc[-1]
            print(f"✅ Latest 1-Min Bar:")
            print(f"   📅 Timestamp: {latest_bar['timestamp']}")
            print(f"   💰 Open: ${latest_bar['open']:.2f}")
            print(f"   💰 High: ${latest_bar['high']:.2f}")
            print(f"   💰 Low: ${latest_bar['low']:.2f}")
            print(f"   💰 Close: ${latest_bar['close']:.2f}")
            print(f"   📊 Volume: {latest_bar['volume']:,}")
            print(f"   🔄 VWAP: ${latest_bar['vwap']:.2f}")
            print()
            
            # Show last 5 bars
            print("📊 Last 5 Bars:")
            for i in range(max(0, len(df)-5), len(df)):
                bar = df.iloc[i]
                print(f"   {bar['timestamp'].strftime('%H:%M')}: ${bar['close']:.2f} (Vol: {bar['volume']:,})")
                
        else:
            print("❌ No bar data available")
    except Exception as e:
        print(f"❌ Bar request failed: {e}")
    
    print()
    
    # Test 3: Recent 5-minute bars (what our system uses)
    print("🔍 TEST 3: Recent 5-Minute Bars (System Data)")
    print("-" * 40)
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        response = data_client.get_stock_bars(request)
        df = response.df.reset_index()
        
        if not df.empty:
            latest_bar = df.iloc[-1]
            print(f"✅ Latest 5-Min Bar (SYSTEM USES THIS):")
            print(f"   📅 Timestamp: {latest_bar['timestamp']}")
            print(f"   💰 Close: ${latest_bar['close']:.2f}")
            print(f"   📊 Volume: {latest_bar['volume']:,}")
            print()
            
            # Show last 3 bars
            print("📊 Last 3 Five-Minute Bars:")
            for i in range(max(0, len(df)-3), len(df)):
                bar = df.iloc[i]
                age_minutes = (datetime.now() - bar['timestamp'].replace(tzinfo=None)).total_seconds() / 60
                print(f"   {bar['timestamp'].strftime('%H:%M')}: ${bar['close']:.2f} ({age_minutes:.1f} min ago)")
                
        else:
            print("❌ No 5-minute bar data available")
    except Exception as e:
        print(f"❌ 5-minute bar request failed: {e}")
    
    print()
    
    # Test 4: Market status
    print("🔍 TEST 4: Market Status Analysis")
    print("-" * 30)
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    print(f"📅 Current Time: {now.strftime('%H:%M:%S')}")
    print(f"🏪 Market Open: {market_open.strftime('%H:%M:%S')}")
    print(f"🏪 Market Close: {market_close.strftime('%H:%M:%S')}")
    
    if market_open <= now <= market_close and now.weekday() < 5:
        print(f"✅ Market is OPEN")
        minutes_since_open = (now - market_open).total_seconds() / 60
        print(f"⏰ Minutes since open: {minutes_since_open:.1f}")
    else:
        print(f"❌ Market is CLOSED")
    
    print()
    
    # Test 5: Data freshness analysis
    print("🔍 TEST 5: Data Freshness Analysis")
    print("-" * 30)
    
    try:
        # Get the exact same data our trading system uses
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        response = data_client.get_stock_bars(request)
        df = response.df.reset_index()
        
        if not df.empty:
            latest_timestamp = df['timestamp'].iloc[-1]
            current_time = datetime.now()
            
            # Convert to naive datetime for comparison
            if hasattr(latest_timestamp, 'tz_localize'):
                latest_timestamp_naive = latest_timestamp.tz_localize(None)
            else:
                latest_timestamp_naive = latest_timestamp.replace(tzinfo=None)
            
            data_age_minutes = (current_time - latest_timestamp_naive).total_seconds() / 60
            
            print(f"📊 Latest Data Timestamp: {latest_timestamp}")
            print(f"🕐 Current Time: {current_time}")
            print(f"⏰ Data Age: {data_age_minutes:.1f} minutes")
            print(f"💰 Latest Price: ${df['close'].iloc[-1]:.2f}")
            
            if data_age_minutes <= 10:
                print(f"✅ Data is FRESH (≤10 min)")
            else:
                print(f"⚠️ Data is STALE (>{data_age_minutes:.1f} min old)")
                
        else:
            print("❌ No recent data available")
            
    except Exception as e:
        print(f"❌ Data freshness check failed: {e}")
    
    print()
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 50)
    print("💡 Compare these prices with your broker/chart to identify discrepancies")
    print("🔧 If prices don't match, we need to adjust data source or timing")


if __name__ == "__main__":
    debug_data_sources()