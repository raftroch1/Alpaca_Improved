"""
Logging utilities for Alpaca Improved

This module provides comprehensive logging functionality using loguru with:
- Configurable log levels and formats
- Multiple output handlers (console, file, rotating files)
- Structured logging for trading activities
- Performance monitoring integration
- Error tracking and alerting
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# Remove default handler to avoid duplicate logs
logger.remove()

# Global configuration for logging setup
_logging_configured = False


def setup_logging(config: Any) -> None:
    """
    Set up comprehensive logging configuration based on application config.
    
    Args:
        config: Configuration object containing logging settings
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Determine log level from config
    log_level = getattr(config.app, 'debug', True) and "DEBUG" or "INFO"
    if hasattr(config, 'logging'):
        log_level = getattr(config.logging, 'level', log_level)
    
    # Console handler with colored output
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    
    # Main application log file with rotation
    logger.add(
        logs_dir / "alpaca_improved.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level="DEBUG",
        rotation="daily",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    
    # Error-only log file
    logger.add(
        logs_dir / "errors.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{exception}"
        ),
        level="ERROR",
        rotation="weekly",
        retention="12 weeks",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    
    # Trading-specific log file
    logger.add(
        logs_dir / "trading.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{extra[trade_type]} | "
            "{extra[symbol]} | "
            "{message}"
        ),
        level="INFO",
        filter=lambda record: "trade_type" in record["extra"],
        rotation="daily",
        retention="90 days",
        compression="gz",
        enqueue=True,
    )
    
    # Strategy performance log file
    logger.add(
        logs_dir / "performance.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{extra[strategy_name]} | "
            "{extra[metric_type]} | "
            "{message}"
        ),
        level="INFO",
        filter=lambda record: "strategy_name" in record["extra"],
        rotation="daily",
        retention="365 days",
        compression="gz",
        enqueue=True,
    )
    
    # Mark logging as configured
    _logging_configured = True
    
    logger.info("Logging system initialized successfully")


def get_logger(name: Optional[str] = None) -> Any:
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Optional name to bind to the logger
        
    Returns:
        Logger instance configured for the application
    """
    if name:
        return logger.bind(logger_name=name)
    return logger


def log_trade_event(
    trade_type: str,
    symbol: str,
    action: str,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    strategy: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log a trading event with structured data.
    
    Args:
        trade_type: Type of trade (buy, sell, options, etc.)
        symbol: Trading symbol
        action: Action taken (open, close, modify, etc.)
        quantity: Number of shares/contracts
        price: Execution price
        strategy: Strategy name
        **kwargs: Additional context data
    """
    log_data = {
        "trade_type": trade_type,
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "strategy": strategy,
        **kwargs
    }
    
    # Filter out None values
    log_data = {k: v for k, v in log_data.items() if v is not None}
    
    message_parts = [f"{action.upper()} {trade_type}"]
    if quantity:
        message_parts.append(f"{quantity}")
    message_parts.append(f"{symbol}")
    if price:
        message_parts.append(f"@ ${price:.2f}")
    if strategy:
        message_parts.append(f"[{strategy}]")
    
    message = " ".join(message_parts)
    
    logger.bind(**log_data).info(message)


def log_strategy_performance(
    strategy_name: str,
    metric_type: str,
    metric_value: float,
    period: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log strategy performance metrics.
    
    Args:
        strategy_name: Name of the strategy
        metric_type: Type of metric (return, sharpe, drawdown, etc.)
        metric_value: Value of the metric
        period: Time period for the metric
        **kwargs: Additional context data
    """
    log_data = {
        "strategy_name": strategy_name,
        "metric_type": metric_type,
        "metric_value": metric_value,
        "period": period,
        **kwargs
    }
    
    # Filter out None values
    log_data = {k: v for k, v in log_data.items() if v is not None}
    
    message = f"{metric_type.upper()}: {metric_value:.4f}"
    if period:
        message += f" ({period})"
    
    logger.bind(**log_data).info(message)


def log_market_data_event(
    symbol: str,
    data_type: str,
    event: str,
    **kwargs: Any
) -> None:
    """
    Log market data events.
    
    Args:
        symbol: Trading symbol
        data_type: Type of data (price, volume, options_chain, etc.)
        event: Event description
        **kwargs: Additional context data
    """
    log_data = {
        "data_type": data_type,
        "symbol": symbol,
        **kwargs
    }
    
    message = f"[{data_type.upper()}] {symbol}: {event}"
    
    logger.bind(**log_data).info(message)


def log_api_call(
    api_name: str,
    endpoint: str,
    method: str,
    status_code: Optional[int] = None,
    response_time: Optional[float] = None,
    error: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Log API calls for monitoring and debugging.
    
    Args:
        api_name: Name of the API (alpaca, alpha_vantage, etc.)
        endpoint: API endpoint called
        method: HTTP method
        status_code: Response status code
        response_time: Response time in seconds
        error: Error message if any
        **kwargs: Additional context data
    """
    log_data = {
        "api_name": api_name,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time": response_time,
        **kwargs
    }
    
    # Filter out None values
    log_data = {k: v for k, v in log_data.items() if v is not None}
    
    message_parts = [f"[{api_name.upper()}]", f"{method}", endpoint]
    
    if status_code:
        message_parts.append(f"({status_code})")
    
    if response_time:
        message_parts.append(f"{response_time:.3f}s")
    
    message = " ".join(message_parts)
    
    if error:
        logger.bind(**log_data).error(f"{message} - {error}")
    elif status_code and status_code >= 400:
        logger.bind(**log_data).warning(message)
    else:
        logger.bind(**log_data).debug(message)


def log_backtest_event(
    strategy_name: str,
    event_type: str,
    message: str,
    **kwargs: Any
) -> None:
    """
    Log backtesting events.
    
    Args:
        strategy_name: Name of the strategy being backtested
        event_type: Type of event (start, end, trade, error, etc.)
        message: Event message
        **kwargs: Additional context data
    """
    log_data = {
        "strategy_name": strategy_name,
        "event_type": event_type,
        **kwargs
    }
    
    formatted_message = f"[BACKTEST] [{strategy_name}] {event_type.upper()}: {message}"
    
    if event_type.lower() == "error":
        logger.bind(**log_data).error(formatted_message)
    elif event_type.lower() == "warning":
        logger.bind(**log_data).warning(formatted_message)
    else:
        logger.bind(**log_data).info(formatted_message)


class ContextLogger:
    """
    Context manager for adding structured context to log messages.
    
    Example:
        with ContextLogger(strategy="momentum", symbol="SPY"):
            logger.info("Processing trade signal")
    """
    
    def __init__(self, **context: Any):
        """
        Initialize context logger.
        
        Args:
            **context: Context data to bind to all log messages
        """
        self.context = context
        self.logger = logger.bind(**context)
    
    def __enter__(self):
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Convenience functions for common logging patterns
def info(message: str, **kwargs: Any) -> None:
    """Log info message with optional context."""
    if kwargs:
        logger.bind(**kwargs).info(message)
    else:
        logger.info(message)


def debug(message: str, **kwargs: Any) -> None:
    """Log debug message with optional context."""
    if kwargs:
        logger.bind(**kwargs).debug(message)
    else:
        logger.debug(message)


def warning(message: str, **kwargs: Any) -> None:
    """Log warning message with optional context."""
    if kwargs:
        logger.bind(**kwargs).warning(message)
    else:
        logger.warning(message)


def error(message: str, **kwargs: Any) -> None:
    """Log error message with optional context."""
    if kwargs:
        logger.bind(**kwargs).error(message)
    else:
        logger.error(message)


def critical(message: str, **kwargs: Any) -> None:
    """Log critical message with optional context."""
    if kwargs:
        logger.bind(**kwargs).critical(message)
    else:
        logger.critical(message)


# Export the main logger instance
__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
    "log_trade_event",
    "log_strategy_performance",
    "log_market_data_event",
    "log_api_call",
    "log_backtest_event",
    "ContextLogger",
    "info",
    "debug",
    "warning",
    "error",
    "critical",
] 