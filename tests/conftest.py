"""
Pytest configuration and shared fixtures for Alpaca Improved tests.

This module provides common test fixtures and configuration that can be used
across all test modules in the project.
"""

import pytest
import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock
import tempfile
import shutil

# Test environment variables
TEST_API_KEY = "test_api_key_12345"
TEST_SECRET_KEY = "test_secret_key_67890"


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        'api_key': TEST_API_KEY,
        'secret_key': TEST_SECRET_KEY,
        'rate_limit': 5,  # Low rate limit for testing
        'retries': 2
    }


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=10),
        end=datetime.now(),
        freq='D'
    )
    
    # Generate realistic price data
    base_price = 100.0
    data = []
    
    for i, date in enumerate(dates):
        # Random walk with slight upward bias
        change = (i * 0.1) + (i % 3 - 1) * 0.5
        open_price = base_price + change
        close_price = open_price + (i % 2 - 0.5) * 0.3
        high_price = max(open_price, close_price) + abs(i % 3) * 0.2
        low_price = min(open_price, close_price) - abs(i % 2) * 0.1
        volume = 1000000 + (i * 50000)
        
        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(volume)
        })
    
    return pd.DataFrame(data).set_index('timestamp')


@pytest.fixture
def sample_options_data():
    """Generate sample options data for testing."""
    current_price = 450.0
    expiration = datetime.now() + timedelta(days=30)
    
    options = []
    
    # Generate strikes around current price
    for strike_offset in range(-10, 11, 2):
        strike = current_price + (strike_offset * 5)
        
        # Call option
        call_option = {
            'symbol': 'SPY',
            'option_symbol': f'SPY{expiration.strftime("%y%m%d")}C{int(strike*1000):08d}',
            'strike_price': strike,
            'expiration_date': expiration,
            'option_type': 'C',
            'bid': max(0.1, current_price - strike + 5) if strike < current_price else 0.5,
            'ask': max(0.2, current_price - strike + 5.2) if strike < current_price else 0.7,
            'last_price': max(0.15, current_price - strike + 5.1) if strike < current_price else 0.6,
            'volume': 100 + abs(strike_offset) * 10,
            'open_interest': 1000 + abs(strike_offset) * 50,
            'implied_volatility': 0.20 + abs(strike_offset) * 0.01,
            'delta': max(0.01, min(0.99, 0.5 + (current_price - strike) * 0.01)),
            'gamma': 0.02,
            'theta': -0.05,
            'vega': 0.15,
            'rho': 0.08
        }
        options.append(call_option)
        
        # Put option
        put_option = {
            'symbol': 'SPY',
            'option_symbol': f'SPY{expiration.strftime("%y%m%d")}P{int(strike*1000):08d}',
            'strike_price': strike,
            'expiration_date': expiration,
            'option_type': 'P',
            'bid': max(0.1, strike - current_price + 5) if strike > current_price else 0.5,
            'ask': max(0.2, strike - current_price + 5.2) if strike > current_price else 0.7,
            'last_price': max(0.15, strike - current_price + 5.1) if strike > current_price else 0.6,
            'volume': 80 + abs(strike_offset) * 8,
            'open_interest': 800 + abs(strike_offset) * 40,
            'implied_volatility': 0.22 + abs(strike_offset) * 0.01,
            'delta': max(-0.99, min(-0.01, -0.5 + (current_price - strike) * 0.01)),
            'gamma': 0.02,
            'theta': -0.04,
            'vega': 0.15,
            'rho': -0.06
        }
        options.append(put_option)
    
    return options


@pytest.fixture
def mock_alpaca_response():
    """Create mock Alpaca API response."""
    mock_response = Mock()
    mock_response.df = pd.DataFrame({
        'open': [100.0, 101.0, 102.0],
        'high': [101.0, 102.0, 103.0],
        'low': [99.0, 100.0, 101.0],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200]
    })
    return mock_response


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def integration_credentials():
    """Get real API credentials for integration tests."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        pytest.skip("Integration test credentials not available")
    
    return {
        'api_key': api_key,
        'secret_key': secret_key
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring API credentials"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "real_api: mark test as requiring real API access"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that typically take longer
        if any(keyword in item.name.lower() for keyword in ['workflow', 'analysis', 'complete']):
            item.add_marker(pytest.mark.slow)


# Fixtures for specific test scenarios
@pytest.fixture
def market_hours_data():
    """Generate data that simulates market hours."""
    # Create data for a trading day
    trading_day = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    
    hours = []
    for hour_offset in range(7):  # 9:30 AM to 4:30 PM
        timestamp = trading_day + timedelta(hours=hour_offset)
        hours.append(timestamp)
    
    data = pd.DataFrame({
        'timestamp': hours,
        'open': [100 + i * 0.1 for i in range(len(hours))],
        'high': [100.5 + i * 0.1 for i in range(len(hours))],
        'low': [99.5 + i * 0.1 for i in range(len(hours))],
        'close': [100.3 + i * 0.1 for i in range(len(hours))],
        'volume': [100000 + i * 10000 for i in range(len(hours))]
    }).set_index('timestamp')
    
    return data


@pytest.fixture
def invalid_market_data():
    """Generate invalid market data for testing validation."""
    return pd.DataFrame({
        'open': [100.0, -1.0, 102.0],  # Negative price
        'high': [101.0, 0.0, 103.0],   # Zero price
        'low': [102.0, -1.0, 101.0],   # Low > High
        'close': [100.5, -1.0, 102.5],
        'volume': [1000, -100, 1200]   # Negative volume
    })


# Environment setup helpers
def setup_test_environment():
    """Set up test environment variables."""
    os.environ['ALPACA_API_KEY'] = TEST_API_KEY
    os.environ['ALPACA_SECRET_KEY'] = TEST_SECRET_KEY
    os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
    os.environ['TEST_MODE'] = 'true'


def cleanup_test_environment():
    """Clean up test environment."""
    test_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY', 
        'ALPACA_BASE_URL',
        'TEST_MODE'
    ]
    
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]


# Custom assertions for testing
def assert_valid_market_data(df):
    """Assert that DataFrame contains valid market data."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check columns exist
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check data validity
    assert all(df['open'] > 0), "Found zero or negative open prices"
    assert all(df['high'] > 0), "Found zero or negative high prices"
    assert all(df['low'] > 0), "Found zero or negative low prices"
    assert all(df['close'] > 0), "Found zero or negative close prices"
    assert all(df['volume'] >= 0), "Found negative volume"
    
    # Check price relationships
    assert all(df['high'] >= df['low']), "Found high prices below low prices"
    assert all(df['high'] >= df['open']), "Found high prices below open prices"
    assert all(df['high'] >= df['close']), "Found high prices below close prices"
    assert all(df['low'] <= df['open']), "Found low prices above open prices"
    assert all(df['low'] <= df['close']), "Found low prices above close prices"


def assert_valid_options_data(options):
    """Assert that options list contains valid option contracts."""
    assert isinstance(options, list), "Options should be a list"
    assert len(options) > 0, "Options list should not be empty"
    
    for option in options:
        assert hasattr(option, 'symbol'), "Option missing symbol"
        assert hasattr(option, 'strike_price'), "Option missing strike_price"
        assert hasattr(option, 'option_type'), "Option missing option_type"
        assert hasattr(option, 'expiration_date'), "Option missing expiration_date"
        
        assert option.strike_price > 0, "Invalid strike price"
        assert option.option_type in ['C', 'P'], "Invalid option type" 