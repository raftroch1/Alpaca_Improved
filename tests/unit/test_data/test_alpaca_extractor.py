"""
Unit tests for AlpacaDataExtractor.

These tests verify the functionality of the Alpaca data extractor including
data extraction, validation, error handling, and rate limiting.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from alpaca.data.timeframe import TimeFrame

from src.data.extractors.alpaca_extractor import AlpacaDataExtractor, DataRequest


class TestAlpacaDataExtractor:
    """Test suite for AlpacaDataExtractor."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Alpaca data client."""
        with patch('src.data.extractors.alpaca_extractor.StockHistoricalDataClient') as mock:
            yield mock.return_value
    
    @pytest.fixture
    def extractor(self, mock_client):
        """AlpacaDataExtractor instance for testing."""
        return AlpacaDataExtractor(
            api_key="test_api_key",
            secret_key="test_secret_key",
            rate_limit=5,  # Low rate limit for testing
            retries=2
        )
    
    @pytest.fixture
    def sample_bars_response(self):
        """Sample bars response data."""
        mock_response = Mock()
        mock_response.df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200],
            'timestamp': [
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=1),
                datetime.now()
            ]
        })
        return mock_response
    
    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.rate_limit == 5
        assert extractor.retries == 2
        assert extractor._request_count == 0
        assert isinstance(extractor._last_request_time, datetime)
    
    def test_get_bars_single_symbol(self, extractor, mock_client, sample_bars_response):
        """Test getting bars for a single symbol."""
        mock_client.get_stock_bars.return_value = sample_bars_response
        
        result = extractor.get_bars('SPY', limit=100)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        
        # Verify API call
        mock_client.get_stock_bars.assert_called_once()
        call_args = mock_client.get_stock_bars.call_args[0][0]
        assert call_args.symbol_or_symbols == ['SPY']
        assert call_args.limit == 100
    
    def test_get_bars_multiple_symbols(self, extractor, mock_client, sample_bars_response):
        """Test getting bars for multiple symbols."""
        mock_client.get_stock_bars.return_value = sample_bars_response
        
        symbols = ['SPY', 'QQQ', 'IWM']
        result = extractor.get_bars(symbols, limit=50)
        
        assert isinstance(result, pd.DataFrame)
        
        # Verify API call with multiple symbols
        call_args = mock_client.get_stock_bars.call_args[0][0]
        assert call_args.symbol_or_symbols == symbols
        assert call_args.limit == 50
    
    def test_get_bars_with_date_range(self, extractor, mock_client, sample_bars_response):
        """Test getting bars with specific date range."""
        mock_client.get_stock_bars.return_value = sample_bars_response
        
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        
        result = extractor.get_bars('SPY', start=start, end=end)
        
        call_args = mock_client.get_stock_bars.call_args[0][0]
        assert call_args.start == start
        assert call_args.end == end
    
    def test_get_bars_api_error(self, extractor, mock_client):
        """Test handling of API errors."""
        mock_client.get_stock_bars.side_effect = Exception("API Error")
        
        result = extractor.get_bars('SPY')
        
        assert result.empty
        # Should retry based on retry configuration
        assert mock_client.get_stock_bars.call_count == extractor.retries
    
    def test_get_bars_retry_success(self, extractor, mock_client, sample_bars_response):
        """Test successful retry after initial failure."""
        # First call fails, second succeeds
        mock_client.get_stock_bars.side_effect = [
            Exception("Temporary error"),
            sample_bars_response
        ]
        
        result = extractor.get_bars('SPY')
        
        assert not result.empty
        assert mock_client.get_stock_bars.call_count == 2
    
    def test_data_validation_invalid_prices(self, extractor):
        """Test data validation removes invalid prices."""
        invalid_data = pd.DataFrame({
            'open': [100.0, -1.0, 102.0],  # Negative price
            'high': [101.0, 0.0, 103.0],   # Zero price
            'low': [99.0, -1.0, 101.0],
            'close': [100.5, -1.0, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        cleaned = extractor._validate_bars_data(invalid_data, ['TEST'])
        
        # Should remove the row with invalid prices
        assert len(cleaned) == 2
        assert all(cleaned['open'] > 0)
        assert all(cleaned['high'] > 0)
        assert all(cleaned['low'] > 0)
        assert all(cleaned['close'] > 0)
    
    def test_data_validation_price_consistency(self, extractor):
        """Test data validation checks price consistency."""
        inconsistent_data = pd.DataFrame({
            'open': [100.0, 102.0],
            'high': [99.0, 103.0],  # High < Open (invalid)
            'low': [101.0, 101.0],  # Low > Open (invalid)
            'close': [100.5, 102.5],
            'volume': [1000, 1100]
        })
        
        cleaned = extractor._validate_bars_data(inconsistent_data, ['TEST'])
        
        # Should remove inconsistent rows
        assert len(cleaned) == 1
    
    def test_rate_limiting(self, extractor):
        """Test rate limiting functionality."""
        # Set up extractor to hit rate limit
        extractor._request_count = extractor.rate_limit
        extractor._last_request_time = datetime.now()
        
        with patch('time.sleep') as mock_sleep:
            extractor._check_rate_limit()
            mock_sleep.assert_called_once()
    
    def test_rate_limit_reset(self, extractor):
        """Test rate limit counter resets after time window."""
        # Set old timestamp to trigger reset
        extractor._request_count = extractor.rate_limit
        extractor._last_request_time = datetime.now() - timedelta(seconds=70)
        
        extractor._check_rate_limit()
        
        # Counter should be reset
        assert extractor._request_count == 1
    
    def test_get_quotes(self, extractor, mock_client):
        """Test quote data extraction."""
        mock_response = Mock()
        mock_response.df = pd.DataFrame({
            'bid': [100.0, 101.0],
            'ask': [100.1, 101.1],
            'bid_size': [100, 200],
            'ask_size': [150, 250]
        })
        mock_client.get_stock_quotes.return_value = mock_response
        
        result = extractor.get_quotes('SPY', limit=100)
        
        assert isinstance(result, pd.DataFrame)
        assert 'bid' in result.columns
        assert 'ask' in result.columns
        mock_client.get_stock_quotes.assert_called_once()
    
    def test_test_connection_success(self, extractor, mock_client, sample_bars_response):
        """Test successful connection test."""
        mock_client.get_stock_bars.return_value = sample_bars_response
        
        result = extractor.test_connection()
        
        assert result is True
        mock_client.get_stock_bars.assert_called_once()
    
    def test_test_connection_failure(self, extractor, mock_client):
        """Test failed connection test."""
        mock_client.get_stock_bars.side_effect = Exception("Connection failed")
        
        result = extractor.test_connection()
        
        assert result is False


class TestDataRequest:
    """Test suite for DataRequest dataclass."""
    
    def test_data_request_creation(self):
        """Test DataRequest creation with defaults."""
        request = DataRequest(symbols='SPY')
        
        assert request.symbols == 'SPY'
        assert request.timeframe == TimeFrame.Day
        assert request.start is None
        assert request.end is None
        assert request.limit is None
    
    def test_data_request_with_parameters(self):
        """Test DataRequest creation with all parameters."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        
        request = DataRequest(
            symbols=['SPY', 'QQQ'],
            timeframe=TimeFrame.Hour,
            start=start,
            end=end,
            limit=1000
        )
        
        assert request.symbols == ['SPY', 'QQQ']
        assert request.timeframe == TimeFrame.Hour
        assert request.start == start
        assert request.end == end
        assert request.limit == 1000 