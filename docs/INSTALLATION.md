# Installation Guide - Alpaca Improved

This guide will walk you through setting up the Alpaca Improved options trading platform on your system.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB+ recommended for backtesting)
- **Storage**: At least 10GB free space for data storage

### Account Requirements
- **Alpaca Trading Account**: [Sign up here](https://alpaca.markets/)
- **API Keys**: Generate paper trading keys (live trading keys optional)

## 🚀 Quick Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/alpaca_improved.git
cd alpaca_improved
```

### 2. Set Up Python Environment
```bash
# Using venv (recommended)
python -m venv alpaca_env
source alpaca_env/bin/activate  # On Windows: alpaca_env\Scripts\activate

# Using conda (alternative)
conda create -n alpaca_improved python=3.9
conda activate alpaca_improved
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Configure Environment
```bash
# Copy environment template
cp config/.env.example .env

# Edit .env with your Alpaca credentials
nano .env  # or your preferred editor
```

### 5. Verify Installation
```bash
python -m pytest tests/unit/test_installation.py
python examples/quick_start.py
```

## 🔧 Detailed Installation

### Python Environment Setup

#### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv alpaca_env

# Activate environment
# On Windows:
alpaca_env\Scripts\activate

# On macOS/Linux:
source alpaca_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Option 2: Conda Environment
```bash
# Create conda environment
conda create -n alpaca_improved python=3.9

# Activate environment
conda activate alpaca_improved

# Install pip in conda environment
conda install pip
```

#### Option 3: Poetry (Advanced)
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate shell
poetry shell
```

### Core Dependencies Installation

#### Method 1: Requirements File
```bash
pip install -r requirements.txt
```

#### Method 2: Manual Installation
```bash
# Core trading libraries
pip install alpaca-py>=0.20.0
pip install backtrader>=1.9.78
pip install vectorbt-pro  # or vectorbt for free version

# Data manipulation
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install pytz>=2022.1

# Technical analysis
pip install ta-lib>=0.4.25
pip install pandas-ta>=0.3.14

# API and networking
pip install requests>=2.28.0
pip install websocket-client>=1.4.0
pip install aiohttp>=3.8.0

# Configuration and utilities
pip install python-dotenv>=0.19.0
pip install pyyaml>=6.0
pip install click>=8.0.0

# Database (optional)
pip install sqlalchemy>=1.4.0
pip install psycopg2-binary>=2.9.0  # PostgreSQL
```

### Development Dependencies (Optional)

```bash
# Testing
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0
pip install pytest-asyncio>=0.20.0

# Code quality
pip install black>=22.0.0
pip install flake8>=5.0.0
pip install mypy>=0.990
pip install isort>=5.10.0

# Development tools
pip install pre-commit>=2.20.0
pip install jupyter>=1.0.0
pip install ipython>=8.0.0
```

## 🔑 Configuration Setup

### 1. Alpaca API Configuration

Create your `.env` file:
```bash
cp config/.env.example .env
```

Edit `.env` with your credentials:
```env
# Alpaca API Configuration
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# ALPACA_BASE_URL=https://api.alpaca.markets     # Live trading (when ready)

# Data Configuration
ALPACA_DATA_URL=https://data.alpaca.markets
POLYGON_API_KEY=your_polygon_key_here  # Optional, for additional data

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_STORAGE_PATH=./data
BACKTEST_DATA_PATH=./data/backtest

# Database Configuration (optional)
DATABASE_URL=sqlite:///./alpaca_improved.db
# DATABASE_URL=postgresql://user:password@localhost/alpaca_improved

# Bot Configuration (optional)
DISCORD_BOT_TOKEN=your_discord_bot_token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

### 2. Application Configuration

The main configuration file is `config/config.yaml`:
```yaml
# Application Configuration
app:
  name: "Alpaca Improved"
  version: "1.0.0"
  environment: "development"

# Trading Configuration
trading:
  default_symbol: "SPY"
  paper_trading: true
  max_position_size: 1000
  risk_per_trade: 0.02

# Backtesting Configuration
backtesting:
  initial_cash: 100000
  commission: 0.0
  slippage: 0.0
  data_frequency: "1D"

# Data Configuration
data:
  storage_format: "parquet"
  compression: "snappy"
  cache_ttl: 3600
```

## 🧪 Verification & Testing

### 1. Run Installation Tests
```bash
# Test basic installation
python -c "import alpaca; print('Alpaca SDK: OK')"
python -c "import backtrader; print('Backtrader: OK')"
python -c "import vectorbt; print('VectorBT: OK')"

# Run unit tests
python -m pytest tests/unit/ -v

# Test API connectivity
python scripts/test_api_connection.py
```

### 2. Run Example Strategy
```bash
# Run a simple example
python examples/simple_strategy.py

# Run options data extraction test
python examples/options_data_example.py
```

## 🐳 Docker Installation (Optional)

### 1. Using Docker Compose
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Manual Docker Build
```bash
# Build image
docker build -t alpaca-improved .

# Run container
docker run -p 8080:8080 \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  alpaca-improved
```

## 🔧 Platform-Specific Instructions

### Windows Installation

#### Prerequisites
```powershell
# Install Python from python.org or Microsoft Store
# Install Git for Windows
# Install Visual Studio Build Tools (for some packages)
```

#### TA-Lib Installation on Windows
```powershell
# Option 1: Pre-compiled wheel
pip install TA-Lib

# Option 2: Manual installation
# Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
```

### macOS Installation

#### Prerequisites
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install TA-Lib dependencies
brew install ta-lib
```

### Linux Installation (Ubuntu/Debian)

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3.9 python3.9-venv python3.9-dev
sudo apt install build-essential

# Install TA-Lib dependencies
sudo apt install libta-lib-dev
```

## 🚨 Troubleshooting

### Common Issues

#### TA-Lib Installation Errors
```bash
# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib

# Windows - use pre-compiled wheel or conda
conda install -c conda-forge ta-lib
```

#### Alpaca API Connection Issues
```python
# Test API connectivity
python -c "
from alpaca.trading.client import TradingClient
client = TradingClient('your_key', 'your_secret', paper=True)
print(client.get_account())
"
```

#### VectorBT Installation Issues
```bash
# Free version
pip install vectorbt

# Pro version (requires license)
pip install vectorbt-pro
```

#### Permission Issues on Linux/macOS
```bash
# If you get permission errors
sudo chown -R $USER:$USER ~/.local/lib/python3.9/site-packages/
```

### Performance Optimization

#### For Large Dataset Processing
```bash
# Install optional performance libraries
pip install numba>=0.56.0
pip install bottleneck>=1.3.0
pip install numexpr>=2.8.0
```

#### For Database Performance
```bash
# PostgreSQL optimizations
pip install psycopg2-binary
pip install sqlalchemy[postgresql]
```

## 🎯 Next Steps

After successful installation:

1. **Read the Documentation**: Check out [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. **Run Examples**: Explore the `examples/` directory
3. **Configure Trading**: Set up your paper trading environment
4. **Develop Strategies**: Create your first custom strategy
5. **Run Backtests**: Test strategies with historical data

## 🆘 Getting Help

If you encounter issues:

1. **Check Documentation**: Review the troubleshooting section above
2. **GitHub Issues**: [Open an issue](https://github.com/yourusername/alpaca_improved/issues)
3. **Community**: Join our [Discord server](https://discord.gg/alpaca-improved)
4. **Professional Support**: Contact support@alpaca-improved.com

---

**⚠️ Important**: Always start with paper trading and thoroughly test your strategies before considering live trading with real money. 