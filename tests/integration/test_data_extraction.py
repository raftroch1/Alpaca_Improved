"""
Integration tests for data extraction with real Alpaca API connections.

These tests verify that the data extractors work correctly with the actual
Alpaca API. They require valid API credentials to be set in environment variables.

IMPORTANT: These tests use real API calls and count against your rate limits.
Run them sparingly and ensure you have valid credentials configured.
"""

import pytest
import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame

from src.data.extractors.alpaca_extractor import AlpacaDataExtractor
from src.data.extractors.options_chain_extractor import OptionsChainExtractor, OptionType


# Skip tests if no API credentials are provided
skip_integration = pytest.mark.skipif(
    not (os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_SECRET_KEY')),
    reason="API credentials not provided. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
)


@skip_integration
class TestAlpacaDataExtractorIntegration:
    """Integration tests for AlpacaDataExtractor with real API."""
    
    @pytest.fixture(scope="class")
    def extractor(self):
        """Create extractor with real API credentials."""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        return AlpacaDataExtractor(
            api_key=api_key,
            secret_key=secret_key,
            rate_limit=200,  # Use default rate limit
            retries=3
        )
    
    def test_connection(self, extractor):
        """Test API connection with real credentials."""
        result = extractor.test_connection()
        assert result is True, "Failed to connect to Alpaca API. Check your credentials."
    
    def test_get_spy_daily_data(self, extractor):
        """Test getting daily SPY data."""
        result = extractor.get_bars('SPY', TimeFrame.Day, limit=10)
        
        assert not result.empty, "No data returned for SPY"
        assert len(result) <= 10, "More data returned than requested"
        
        # Verify data structure
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Verify data quality
        assert all(result['open'] > 0), "Invalid open prices found"
        assert all(result['high'] >= result['low']), "High < Low found"
        assert all(result['volume'] >= 0), "Negative volume found"
        
        print(f"✅ Successfully extracted {len(result)} bars for SPY")
        print(f"📊 Latest close price: ${result['close'].iloc[-1]:.2f}")
    
    def test_get_multiple_symbols(self, extractor):
        """Test getting data for multiple symbols."""
        symbols = ['SPY', 'QQQ', 'IWM']
        result = extractor.get_bars(symbols, TimeFrame.Day, limit=5)
        
        assert not result.empty, "No data returned for multiple symbols"
        
        # Check that we have data for all symbols
        if hasattr(result.index, 'get_level_values'):
            unique_symbols = result.index.get_level_values('symbol').unique()
            for symbol in symbols:
                assert symbol in unique_symbols, f"No data for {symbol}"
        
        print(f"✅ Successfully extracted data for {symbols}")
    
    def test_get_hourly_data(self, extractor):
        """Test getting hourly data."""
        # Get last 24 hours of hourly data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        result = extractor.get_bars(
            'SPY', 
            TimeFrame.Hour, 
            start=start_time, 
            end=end_time
        )
        
        # During market hours, we should get some data
        print(f"📈 Retrieved {len(result)} hourly bars for SPY")
        
        if not result.empty:
            assert 'open' in result.columns
            assert all(result['open'] > 0)
            print(f"🕐 Latest hourly close: ${result['close'].iloc[-1]:.2f}")
    
    def test_get_quotes(self, extractor):
        """Test getting quote data."""
        result = extractor.get_quotes('SPY', limit=5)
        
        # Quotes might not always be available outside market hours
        if not result.empty:
            assert 'bid' in result.columns or 'ask' in result.columns
            print(f"💰 Retrieved {len(result)} quotes for SPY")
        else:
            print("ℹ️ No quote data available (possibly outside market hours)")
    
    def test_data_validation(self, extractor):
        """Test that data validation works with real data."""
        result = extractor.get_bars('SPY', limit=100, validate_data=True)
        
        assert not result.empty, "No data returned"
        
        # All validation checks should pass
        assert all(result['open'] > 0), "Validation failed: found zero/negative prices"
        assert all(result['high'] >= result['low']), "Validation failed: high < low"
        assert all(result['volume'] >= 0), "Validation failed: negative volume"
        
        print(f"✅ Data validation passed for {len(result)} bars")


@skip_integration
class TestOptionsChainExtractorIntegration:
    """Integration tests for OptionsChainExtractor with real API."""
    
    @pytest.fixture(scope="class")
    def extractor(self):
        """Create options extractor with real API credentials."""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        return OptionsChainExtractor(
            api_key=api_key,
            secret_key=secret_key,
            rate_limit=200,
            retries=3
        )
    
    def test_connection(self, extractor):
        """Test options API connection."""
        result = extractor.test_connection()
        assert result is True, "Failed to connect to Alpaca Options API. Check your credentials and options access."
    
    def test_get_spy_options_chain(self, extractor):
        """Test getting SPY options chain."""
        # Get options chain for next month
        result = extractor.get_options_chain('SPY')
        
        if result:
            assert len(result) > 0, "No options contracts returned"
            
            # Check that we have both calls and puts
            calls = [opt for opt in result if opt.option_type == OptionType.CALL]
            puts = [opt for opt in result if opt.option_type == OptionType.PUT]
            
            assert len(calls) > 0, "No call options found"
            assert len(puts) > 0, "No put options found"
            
            # Verify contract structure
            sample_contract = result[0]
            assert sample_contract.symbol == 'SPY'
            assert sample_contract.strike_price > 0
            assert sample_contract.option_type in [OptionType.CALL, OptionType.PUT]
            
            print(f"✅ Retrieved {len(result)} option contracts for SPY")
            print(f"📞 Calls: {len(calls)}, Puts: {len(puts)}")
            
            # Show sample strikes
            strikes = sorted(set(opt.strike_price for opt in result))
            print(f"💲 Strike range: ${strikes[0]:.0f} - ${strikes[-1]:.0f}")
        else:
            print("ℹ️ No options data available (check if options trading is enabled)")
    
    def test_get_options_with_filters(self, extractor):
        """Test getting options with filters."""
        # Get current SPY price to set reasonable strike range
        from src.data.extractors.alpaca_extractor import AlpacaDataExtractor
        
        data_extractor = AlpacaDataExtractor(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        
        spy_data = data_extractor.get_bars('SPY', limit=1)
        if not spy_data.empty:
            current_price = spy_data['close'].iloc[-1]
            
            # Get options within 10% of current price
            min_strike = current_price * 0.9
            max_strike = current_price * 1.1
            
            result = extractor.get_options_by_strike_range(
                'SPY',
                min_strike=min_strike,
                max_strike=max_strike
            )
            
            if result['calls'] or result['puts']:
                print(f"✅ Filtered options around ${current_price:.2f}")
                print(f"📞 Calls in range: {len(result['calls'])}")
                print(f"📉 Puts in range: {len(result['puts'])}")
            else:
                print("ℹ️ No options found in strike range")
    
    def test_get_option_bars(self, extractor):
        """Test getting historical option bars."""
        # First get an options chain to find a valid option symbol
        chain = extractor.get_options_chain('SPY')
        
        if chain:
            # Use the first option contract
            option_symbol = chain[0].option_symbol
            
            result = extractor.get_option_bars(option_symbol, limit=5)
            
            if not result.empty:
                assert 'open' in result.columns
                assert 'close' in result.columns
                print(f"📊 Retrieved {len(result)} option bars for {option_symbol}")
            else:
                print(f"ℹ️ No historical data available for {option_symbol}")
        else:
            print("ℹ️ Skipping option bars test - no options chain available")


@skip_integration
class TestDataExtractionWorkflow:
    """Test complete data extraction workflows."""
    
    @pytest.fixture(scope="class")
    def extractors(self):
        """Create both extractors."""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        return {
            'data': AlpacaDataExtractor(api_key, secret_key),
            'options': OptionsChainExtractor(api_key, secret_key)
        }
    
    def test_full_spy_analysis_workflow(self, extractors):
        """Test a complete workflow for SPY analysis."""
        data_extractor = extractors['data']
        options_extractor = extractors['options']
        
        print("\n🔍 Starting SPY Analysis Workflow...")
        
        # 1. Get current SPY price
        print("1️⃣ Getting current SPY price...")
        spy_data = data_extractor.get_bars('SPY', limit=1)
        
        if spy_data.empty:
            pytest.skip("Cannot get SPY data")
        
        current_price = spy_data['close'].iloc[-1]
        print(f"   Current SPY price: ${current_price:.2f}")
        
        # 2. Get historical data for trend analysis
        print("2️⃣ Getting historical data...")
        historical_data = data_extractor.get_bars('SPY', limit=20)
        
        if not historical_data.empty:
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            print(f"   20-day annualized volatility: {volatility:.1%}")
        
        # 3. Get options chain
        print("3️⃣ Getting options chain...")
        options = options_extractor.get_options_chain('SPY')
        
        if options:
            # Find ATM options
            atm_calls = [opt for opt in options 
                        if opt.option_type == OptionType.CALL 
                        and abs(opt.strike_price - current_price) < 5]
            
            atm_puts = [opt for opt in options 
                       if opt.option_type == OptionType.PUT 
                       and abs(opt.strike_price - current_price) < 5]
            
            print(f"   ATM Call options: {len(atm_calls)}")
            print(f"   ATM Put options: {len(atm_puts)}")
            
            # Show implied volatility if available
            if atm_calls and atm_calls[0].implied_volatility:
                iv = atm_calls[0].implied_volatility
                print(f"   Implied Volatility: {iv:.1%}")
        
        print("✅ SPY Analysis Workflow Complete!")
    
    def test_multi_symbol_comparison(self, extractors):
        """Test comparing multiple symbols."""
        symbols = ['SPY', 'QQQ', 'IWM']
        data_extractor = extractors['data']
        
        print(f"\n📊 Comparing {symbols}...")
        
        data = data_extractor.get_bars(symbols, limit=5)
        
        if not data.empty:
            # If we have multi-index data, process by symbol
            if hasattr(data.index, 'get_level_values'):
                for symbol in symbols:
                    try:
                        symbol_data = data.xs(symbol, level='symbol')
                        latest_price = symbol_data['close'].iloc[-1]
                        print(f"   {symbol}: ${latest_price:.2f}")
                    except KeyError:
                        print(f"   {symbol}: No data available")
            else:
                # Single symbol data
                latest_price = data['close'].iloc[-1]
                print(f"   Latest close: ${latest_price:.2f}")
        
        print("✅ Multi-symbol comparison complete!")


# Utility functions for integration testing
def check_environment():
    """Check if environment is properly configured for integration tests."""
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Set these variables to run integration tests:")
        for var in missing_vars:
            print(f"  export {var}='your_value_here'")
        return False
    
    print("✅ Environment variables configured for integration tests")
    return True


if __name__ == "__main__":
    """Run integration tests directly."""
    if check_environment():
        print("Running integration tests...")
        pytest.main([__file__, "-v", "-s"])
    else:
        print("Configure environment variables first.") 