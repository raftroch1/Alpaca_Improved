# Testing Data Extractors - Alpaca Improved

This directory contains comprehensive tests for the data extraction layer of Alpaca Improved. The tests are organized into unit tests, integration tests, and end-to-end workflows.

## 🧪 Test Structure

```
tests/
├── unit/                    # Unit tests (mocked, fast)
│   └── test_data/
│       ├── test_alpaca_extractor.py     # Stock data extractor tests
│       └── test_options_extractor.py    # Options data extractor tests
├── integration/             # Integration tests (real API)
│   └── test_data_extraction.py          # Real API integration tests
├── fixtures/                # Test data and utilities
│   └── sample_data.py                   # Sample data generators
└── conftest.py             # Pytest configuration and fixtures
```

## 🚀 Quick Start

### 1. Set Up API Credentials

For integration tests that connect to real APIs, set your Alpaca credentials:

```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
```

### 2. Run Quick Connection Test

```bash
# Test basic connectivity
python scripts/test_data_extraction.py --connection-only

# Test stock data only
python scripts/test_data_extraction.py --stocks-only

# Test options data only (requires options permissions)
python scripts/test_data_extraction.py --options-only
```

### 3. Run Full Test Suite

```bash
# Run all unit tests (fast, no API calls)
pytest tests/unit/ -v

# Run integration tests (requires API credentials)
pytest tests/integration/ -v -s

# Run all tests
pytest tests/ -v
```

## 📊 Test Types

### Unit Tests
- **Fast execution** (< 1 second each)
- **No external dependencies** (mocked APIs)
- **High coverage** of core functionality
- **Run on every commit**

Example:
```bash
pytest tests/unit/test_data/test_alpaca_extractor.py::TestAlpacaDataExtractor::test_get_bars_single_symbol -v
```

### Integration Tests
- **Real API connections** (requires credentials)
- **End-to-end workflows** with actual data
- **Rate limit aware** (run sparingly)
- **Validate real data quality**

Example:
```bash
pytest tests/integration/test_data_extraction.py::TestAlpacaDataExtractorIntegration::test_get_spy_daily_data -v -s
```

## 🎯 Testing Your Data Extractors

### Phase 2 Testing Workflow

1. **Start with Unit Tests** (Development):
   ```bash
   # Test individual components
   pytest tests/unit/test_data/ -v
   ```

2. **Quick Integration Check** (Validation):
   ```bash
   # Verify API connectivity
   python scripts/test_data_extraction.py --connection-only
   ```

3. **Full Integration Tests** (Comprehensive):
   ```bash
   # Test with real data
   pytest tests/integration/ -v -s
   ```

4. **Interactive Demo** (Exploration):
   ```bash
   # See your extractors in action
   python examples/data_extraction_demo.py
   ```

### Test What Matters

#### For Stock Data Extractor:
- ✅ **Connection**: Can connect to Alpaca API
- ✅ **Data Quality**: Prices are valid, volumes are positive
- ✅ **Multiple Symbols**: Handle batch requests correctly
- ✅ **Date Ranges**: Historical data retrieval works
- ✅ **Rate Limiting**: Respects API limits
- ✅ **Error Handling**: Graceful failure and retries

#### For Options Data Extractor:
- ✅ **Options Chains**: Retrieve complete option chains
- ✅ **Greeks**: Greeks data is present and reasonable
- ✅ **Filtering**: Strike range and option type filters work
- ✅ **Expiration Logic**: Monthly expiration calculation
- ✅ **Data Validation**: Option prices and relationships are valid

## 🛠️ Test Configuration

### Environment Variables
```bash
# Required for integration tests
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key

# Optional test configuration
TEST_MODE=true
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Pytest Markers
```bash
# Run only unit tests
pytest -m "not integration"

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only real API tests
pytest -m real_api
```

### Custom Test Execution
```bash
# Verbose output with print statements
pytest -v -s

# Stop on first failure
pytest -x

# Run specific test class
pytest tests/unit/test_data/test_alpaca_extractor.py::TestAlpacaDataExtractor

# Run with coverage
pytest --cov=src/data tests/unit/test_data/
```

## 📈 Expected Test Results

### Successful Output Example:
```
🧪 DATA EXTRACTION TESTING
==================================================
Time: 2024-01-15 10:30:00

🔍 Testing API Connectivity...
----------------------------------------
📈 Testing stock data API...
✅ Stock data API: Connected
🎯 Testing options data API...
✅ Options data API: Connected

📈 Testing Stock Data Extraction...
----------------------------------------
1. Single symbol (SPY) daily data...
   ✅ Got 5 bars, latest close: $450.25
2. Multiple symbols (SPY, QQQ)...
   ✅ Got data: 10 total bars
3. Date range query (last 7 days)...
   ✅ Got 5 bars for date range
4. Data validation...
   ✅ Data validation passed for 10 bars

Stock Data Tests: 4/4 passed

🎯 Testing Options Data Extraction...
----------------------------------------
1. SPY options chain...
   ✅ Got 156 contracts (78 calls, 78 puts)
2. Strike range filtering...
   ✅ Filtered: 12 calls, 12 puts in range
3. Next expiration calculation...
   ✅ Next expiration: 2024-02-16 (32 days)

Options Data Tests: 3/3 passed

==================================================
🎉 ALL TESTS PASSED!
Your data extractors are working correctly.
```

## 🚨 Troubleshooting

### Common Issues:

1. **API Credentials Not Found**
   ```
   ❌ Missing API credentials!
   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
   ```
   **Solution**: Set your API credentials as environment variables

2. **Options API Access Denied**
   ```
   ❌ Options data API: Failed
   (Options access may require special permissions)
   ```
   **Solution**: Enable options trading in your Alpaca account

3. **Rate Limit Exceeded**
   ```
   Warning: Rate limit reached, sleeping for 45.2s
   ```
   **Solution**: Run tests less frequently or use paper trading account

4. **No Data Outside Market Hours**
   ```
   ℹ️ No quote data available (possibly outside market hours)
   ```
   **Solution**: Normal behavior - some data types only available during market hours

## 🎯 What to Test in Phase 2

Based on your TASKS.md, here's what you should focus on testing:

### ✅ **Already Implemented & Ready to Test:**
- Basic Alpaca data extraction
- Options chain retrieval
- Data validation and quality checks
- Rate limiting and error handling

### 🚧 **Phase 2 Priorities to Test:**
1. **Real-time Data Streaming** - Test live market data feeds
2. **Polygon.io Integration** - Test alternative data source validation  
3. **Technical Indicators** - Test TA-Lib integration for signal generation
4. **Performance Optimization** - Test with large datasets

### 📋 **Next Phase Testing:**
- Strategy backtesting integration
- Live trading data feeds
- Multi-timeframe data processing

## 🎉 Success Criteria

Your data extractors are working correctly when:
- ✅ All unit tests pass
- ✅ Integration tests connect successfully  
- ✅ Real SPY data is retrieved and validated
- ✅ Options chains are complete with Greeks
- ✅ Demo script runs without errors
- ✅ Data quality checks pass consistently

Now you're ready to move forward with Phase 2 development! 🚀 