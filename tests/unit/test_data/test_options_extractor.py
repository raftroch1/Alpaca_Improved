"""
Unit tests for OptionsChainExtractor.

These tests verify the functionality of the options chain extractor including
options chain retrieval, Greeks handling, and options-specific data processing.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.extractors.options_chain_extractor import (
    OptionsChainExtractor, 
    OptionContract, 
    OptionType
)


class TestOptionsChainExtractor:
    """Test suite for OptionsChainExtractor."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Alpaca options data client."""
        with patch('src.data.extractors.options_chain_extractor.OptionsHistoricalDataClient') as mock:
            yield mock.return_value
    
    @pytest.fixture
    def extractor(self, mock_client):
        """OptionsChainExtractor instance for testing."""
        return OptionsChainExtractor(
            api_key="test_api_key",
            secret_key="test_secret_key",
            rate_limit=5,
            retries=2
        )
    
    @pytest.fixture
    def sample_option_contracts(self):
        """Sample option contract data."""
        contracts = []
        
        # Call option
        call_contract = Mock()
        call_contract.symbol = "SPY250117C00450000"
        call_contract.strike_price = 450.0
        call_contract.bid = 5.0
        call_contract.ask = 5.2
        call_contract.last_price = 5.1
        call_contract.volume = 100
        call_contract.open_interest = 1000
        call_contract.implied_volatility = 0.25
        call_contract.delta = 0.6
        call_contract.gamma = 0.02
        call_contract.theta = -0.05
        call_contract.vega = 0.15
        call_contract.rho = 0.08
        contracts.append(call_contract)
        
        # Put option
        put_contract = Mock()
        put_contract.symbol = "SPY250117P00450000"
        put_contract.strike_price = 450.0
        put_contract.bid = 8.0
        put_contract.ask = 8.3
        put_contract.last_price = 8.1
        put_contract.volume = 80
        put_contract.open_interest = 800
        put_contract.implied_volatility = 0.28
        put_contract.delta = -0.4
        put_contract.gamma = 0.02
        put_contract.theta = -0.04
        put_contract.vega = 0.15
        put_contract.rho = -0.06
        contracts.append(put_contract)
        
        return contracts
    
    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.rate_limit == 5
        assert extractor.retries == 2
        assert extractor._request_count == 0
        assert isinstance(extractor._last_request_time, datetime)
    
    def test_get_options_chain_success(self, extractor, mock_client, sample_option_contracts):
        """Test successful options chain retrieval."""
        mock_client.get_option_chain.return_value = sample_option_contracts
        
        expiration = datetime.now() + timedelta(days=30)
        result = extractor.get_options_chain('SPY', expiration_date=expiration)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        # Check call option
        call_option = next(opt for opt in result if opt.option_type == OptionType.CALL)
        assert call_option.symbol == 'SPY'
        assert call_option.strike_price == 450.0
        assert call_option.bid == 5.0
        assert call_option.delta == 0.6
        
        # Check put option
        put_option = next(opt for opt in result if opt.option_type == OptionType.PUT)
        assert put_option.symbol == 'SPY'
        assert put_option.strike_price == 450.0
        assert put_option.bid == 8.0
        assert put_option.delta == -0.4
    
    def test_get_options_chain_with_filters(self, extractor, mock_client, sample_option_contracts):
        """Test options chain retrieval with filters."""
        mock_client.get_option_chain.return_value = sample_option_contracts
        
        # Filter for calls only
        result = extractor.get_options_chain(
            'SPY',
            option_type=OptionType.CALL,
            strike_range=(440.0, 460.0)
        )
        
        assert len(result) == 1
        assert result[0].option_type == OptionType.CALL
        assert 440.0 <= result[0].strike_price <= 460.0
    
    def test_get_options_chain_strike_filter(self, extractor, mock_client, sample_option_contracts):
        """Test strike range filtering."""
        # Add contract outside range
        out_of_range = Mock()
        out_of_range.symbol = "SPY250117C00500000"
        out_of_range.strike_price = 500.0
        sample_option_contracts.append(out_of_range)
        
        mock_client.get_option_chain.return_value = sample_option_contracts
        
        result = extractor.get_options_chain(
            'SPY',
            strike_range=(440.0, 460.0)
        )
        
        # Should exclude the 500 strike
        assert len(result) == 2
        assert all(440.0 <= opt.strike_price <= 460.0 for opt in result)
    
    def test_get_options_chain_api_error(self, extractor, mock_client):
        """Test handling of API errors."""
        mock_client.get_option_chain.side_effect = Exception("API Error")
        
        result = extractor.get_options_chain('SPY')
        
        assert result == []
        assert mock_client.get_option_chain.call_count == extractor.retries
    
    def test_get_options_chain_retry_success(self, extractor, mock_client, sample_option_contracts):
        """Test successful retry after initial failure."""
        mock_client.get_option_chain.side_effect = [
            Exception("Temporary error"),
            sample_option_contracts
        ]
        
        result = extractor.get_options_chain('SPY')
        
        assert len(result) == 2
        assert mock_client.get_option_chain.call_count == 2
    
    def test_get_option_bars(self, extractor, mock_client):
        """Test option bars retrieval."""
        mock_response = Mock()
        mock_response.df = pd.DataFrame({
            'open': [5.0, 5.1],
            'high': [5.2, 5.3],
            'low': [4.9, 5.0],
            'close': [5.1, 5.2],
            'volume': [100, 150]
        })
        mock_client.get_option_bars.return_value = mock_response
        
        result = extractor.get_option_bars('SPY250117C00450000')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'open' in result.columns
        mock_client.get_option_bars.assert_called_once()
    
    def test_get_options_by_strike_range(self, extractor, mock_client, sample_option_contracts):
        """Test getting options by strike range."""
        mock_client.get_option_chain.return_value = sample_option_contracts
        
        result = extractor.get_options_by_strike_range('SPY', 440.0, 460.0)
        
        assert 'calls' in result
        assert 'puts' in result
        assert len(result['calls']) == 1
        assert len(result['puts']) == 1
    
    def test_next_monthly_expiration(self, extractor):
        """Test next monthly expiration calculation."""
        expiration = extractor._get_next_monthly_expiration()
        
        assert isinstance(expiration, datetime)
        assert expiration.date() > datetime.now().date()
        assert expiration.weekday() == 4  # Friday
    
    def test_test_connection_success(self, extractor, mock_client, sample_option_contracts):
        """Test successful connection test."""
        mock_client.get_option_chain.return_value = sample_option_contracts
        
        result = extractor.test_connection()
        
        assert result is True
        mock_client.get_option_chain.assert_called_once()
    
    def test_test_connection_failure(self, extractor, mock_client):
        """Test failed connection test."""
        mock_client.get_option_chain.side_effect = Exception("Connection failed")
        
        result = extractor.test_connection()
        
        assert result is False


class TestOptionContract:
    """Test suite for OptionContract dataclass."""
    
    def test_option_contract_creation(self):
        """Test OptionContract creation."""
        expiration = datetime.now() + timedelta(days=30)
        
        contract = OptionContract(
            symbol='SPY',
            option_symbol='SPY250117C00450000',
            strike_price=450.0,
            expiration_date=expiration,
            option_type=OptionType.CALL,
            bid=5.0,
            ask=5.2,
            delta=0.6
        )
        
        assert contract.symbol == 'SPY'
        assert contract.strike_price == 450.0
        assert contract.option_type == OptionType.CALL
        assert contract.bid == 5.0
        assert contract.delta == 0.6
    
    def test_option_contract_optional_fields(self):
        """Test OptionContract with minimal required fields."""
        expiration = datetime.now() + timedelta(days=30)
        
        contract = OptionContract(
            symbol='SPY',
            option_symbol='SPY250117P00450000',
            strike_price=450.0,
            expiration_date=expiration,
            option_type=OptionType.PUT
        )
        
        assert contract.bid is None
        assert contract.ask is None
        assert contract.delta is None


class TestOptionType:
    """Test suite for OptionType enum."""
    
    def test_option_type_values(self):
        """Test OptionType enum values."""
        assert OptionType.CALL.value == "C"
        assert OptionType.PUT.value == "P"
    
    def test_option_type_comparison(self):
        """Test OptionType comparison."""
        assert OptionType.CALL != OptionType.PUT
        assert OptionType.CALL == OptionType.CALL 