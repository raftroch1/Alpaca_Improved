"""
Configuration Management for Alpaca Improved

This module provides comprehensive configuration management with support for:
- YAML configuration files
- Environment variable overrides
- Type validation and defaults
- Multiple environment support (dev, test, staging, prod)
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    url: str = Field(default="postgresql://user:password@localhost:5432/alpaca_improved")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="alpaca_improved")
    user: str = Field(default="alpaca_user")
    password: str = Field(default="")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False)


class RedisConfig(BaseModel):
    """Redis configuration settings."""
    
    url: str = Field(default="redis://localhost:6379/0")
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: str = Field(default="")
    decode_responses: bool = Field(default=True)
    socket_connect_timeout: int = Field(default=5)
    socket_timeout: int = Field(default=5)


class AlpacaConfig(BaseModel):
    """Alpaca API configuration settings."""
    
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    base_url: str = Field(default="https://paper-api.alpaca.markets")
    data_url: str = Field(default="https://data.alpaca.markets")
    environment: str = Field(default="paper")
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['paper', 'live']:
            raise ValueError('environment must be either "paper" or "live"')
        return v


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    max_position_size_percent: float = Field(default=5.0, ge=0.1, le=100.0)
    max_daily_loss_percent: float = Field(default=2.0, ge=0.1, le=50.0)
    max_open_positions: int = Field(default=10, ge=1, le=100)
    force_paper_trading: bool = Field(default=True)


class MarketDataConfig(BaseModel):
    """Market data provider configuration."""
    
    primary: str = Field(default="alpaca")
    backup: list = Field(default_factory=lambda: ["alpha_vantage", "yahoo_finance"])
    alpha_vantage_api_key: str = Field(default="")
    yahoo_finance_enabled: bool = Field(default=True)
    iex_cloud_api_key: str = Field(default="")
    real_time_enabled: bool = Field(default=True)
    historical_data_buffer_days: int = Field(default=365)


class BacktestingConfig(BaseModel):
    """Backtesting configuration settings."""
    
    initial_capital: float = Field(default=100000.0, ge=1000.0)
    commission: float = Field(default=0.0, ge=0.0)
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="today")
    benchmark: str = Field(default="SPY")
    risk_free_rate: float = Field(default=0.02, ge=0.0, le=1.0)


class StrategyConfig(BaseModel):
    """Strategy configuration settings."""
    
    max_positions: int = Field(default=5, ge=1, le=50)
    position_size: float = Field(default=0.1, ge=0.01, le=1.0)
    stop_loss: float = Field(default=0.05, ge=0.001, le=1.0)
    take_profit: float = Field(default=0.15, ge=0.001, le=10.0)


class DiscordBotConfig(BaseModel):
    """Discord bot configuration."""
    
    token: str = Field(default="")
    guild_id: str = Field(default="")
    channel_id: str = Field(default="")
    command_prefix: str = Field(default="!")
    enabled: bool = Field(default=False)


class TelegramBotConfig(BaseModel):
    """Telegram bot configuration."""
    
    token: str = Field(default="")
    chat_id: str = Field(default="")
    enabled: bool = Field(default=False)


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""
    
    sentry_dsn: str = Field(default="")
    performance_enabled: bool = Field(default=True)
    email_alerts_enabled: bool = Field(default=False)
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_username: str = Field(default="")


class AppConfig(BaseModel):
    """Application-level configuration."""
    
    name: str = Field(default="Alpaca Improved")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    timezone: str = Field(default="America/New_York")
    secret_key: str = Field(default="")
    
    @validator('environment')
    def validate_app_environment(cls, v):
        if v not in ['development', 'testing', 'staging', 'production']:
            raise ValueError('environment must be one of: development, testing, staging, production')
        return v


class Config(BaseModel):
    """Main configuration class that combines all settings."""
    
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    discord_bot: DiscordBotConfig = Field(default_factory=DiscordBotConfig)
    telegram_bot: TelegramBotConfig = Field(default_factory=TelegramBotConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.alpaca.environment == "paper" or self.risk.force_paper_trading
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.environment == "production"
    
    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.app.debug and not self.is_production


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports syntax: ${VAR_NAME:default_value}
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    else:
        return value


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            raw_config = yaml.safe_load(file)
        
        # Substitute environment variables
        config = substitute_env_vars(raw_config)
        
        return config or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def get_config_path() -> Path:
    """
    Determine the appropriate configuration file path based on environment.
    
    Priority:
    1. CONFIG_PATH environment variable
    2. config/config.{environment}.yaml
    3. config/config.yaml
    4. Default configuration (no file)
    """
    # Check for explicit config path
    config_path_env = os.getenv('CONFIG_PATH')
    if config_path_env:
        return Path(config_path_env)
    
    # Determine base directory (project root)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up from src/alpaca_improved/
    config_dir = project_root / "config"
    
    # Check environment-specific config
    environment = os.getenv('APP_ENV', 'development')
    env_config_path = config_dir / f"config.{environment}.yaml"
    if env_config_path.exists():
        return env_config_path
    
    # Check default config
    default_config_path = config_dir / "config.yaml"
    if default_config_path.exists():
        return default_config_path
    
    # Return None if no config file found (will use defaults)
    return None


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Create a Config object from a dictionary, handling nested structures.
    
    Args:
        config_dict: Dictionary containing configuration values
        
    Returns:
        Config object with validated settings
    """
    # Extract nested configurations
    app_config = AppConfig(**config_dict.get('app', {}))
    
    # Handle database config
    db_config_dict = config_dict.get('database', {})
    if 'postgresql' in db_config_dict:
        db_config_dict = db_config_dict['postgresql']
    database_config = DatabaseConfig(**db_config_dict)
    
    # Handle redis config
    redis_config_dict = config_dict.get('database', {}).get('redis', {})
    if not redis_config_dict:
        redis_config_dict = config_dict.get('redis', {})
    redis_config = RedisConfig(**redis_config_dict)
    
    # Handle trading config
    trading_config = config_dict.get('trading', {})
    alpaca_config = AlpacaConfig(**trading_config.get('alpaca', {}))
    risk_config = RiskConfig(**trading_config.get('risk', {}))
    
    # Handle market data config
    market_data_dict = config_dict.get('market_data', {})
    market_data_config = MarketDataConfig(
        primary=market_data_dict.get('providers', {}).get('primary', 'alpaca'),
        backup=market_data_dict.get('providers', {}).get('backup', ['alpha_vantage', 'yahoo_finance']),
        alpha_vantage_api_key=market_data_dict.get('alpha_vantage', {}).get('api_key', ''),
        yahoo_finance_enabled=market_data_dict.get('yahoo_finance', {}).get('enabled', True),
        iex_cloud_api_key=market_data_dict.get('iex_cloud', {}).get('api_key', ''),
        real_time_enabled=market_data_dict.get('updates', {}).get('real_time_enabled', True),
        historical_data_buffer_days=market_data_dict.get('updates', {}).get('historical_data_buffer_days', 365),
    )
    
    # Handle backtesting config
    backtesting_dict = config_dict.get('backtesting', {}).get('defaults', {})
    backtesting_config = BacktestingConfig(**backtesting_dict)
    
    # Handle strategy config
    strategy_dict = config_dict.get('strategies', {}).get('defaults', {})
    strategy_config = StrategyConfig(**strategy_dict)
    
    # Handle bot configs
    bots_config = config_dict.get('bots', {})
    discord_config = DiscordBotConfig(**bots_config.get('discord', {}))
    telegram_config = TelegramBotConfig(**bots_config.get('telegram', {}))
    
    # Handle monitoring config
    monitoring_dict = config_dict.get('monitoring', {})
    monitoring_config = MonitoringConfig(
        sentry_dsn=monitoring_dict.get('sentry', {}).get('dsn', ''),
        performance_enabled=monitoring_dict.get('performance', {}).get('enabled', True),
        email_alerts_enabled=bool(monitoring_dict.get('email', {}).get('smtp_host')),
        smtp_host=monitoring_dict.get('email', {}).get('smtp_host', ''),
        smtp_port=monitoring_dict.get('email', {}).get('smtp_port', 587),
        smtp_username=monitoring_dict.get('email', {}).get('username', ''),
    )
    
    return Config(
        app=app_config,
        database=database_config,
        redis=redis_config,
        alpaca=alpaca_config,
        risk=risk_config,
        market_data=market_data_config,
        backtesting=backtesting_config,
        strategy=strategy_config,
        discord_bot=discord_config,
        telegram_bot=telegram_config,
        monitoring=monitoring_config,
    )


def get_config() -> Config:
    """
    Load and return the application configuration.
    
    This function loads configuration from:
    1. YAML configuration file (if available)
    2. Environment variables
    3. Default values
    
    Returns:
        Config object with all application settings
    """
    try:
        config_path = get_config_path()
        
        if config_path and config_path.exists():
            config_dict = load_yaml_config(config_path)
            return create_config_from_dict(config_dict)
        else:
            # Use defaults with environment variable overrides
            return create_config_from_dict({})
            
    except Exception as e:
        # If configuration loading fails, log the error and use defaults
        print(f"Warning: Error loading configuration: {e}")
        print("Using default configuration values")
        return create_config_from_dict({})


# Global configuration instance
_config: Optional[Config] = None


def get_global_config() -> Config:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def reload_config() -> Config:
    """Reload the global configuration from file."""
    global _config
    _config = None
    return get_global_config()


# Convenience function for backward compatibility
config = get_global_config 