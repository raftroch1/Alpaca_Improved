"""
Trading Strategies for Alpaca Improved

This package contains:
- Base strategy template classes
- Common strategy utilities and indicators
- Pre-built strategy implementations
- Strategy validation and testing tools
"""

from .base import BaseStrategy, StrategyConfig
from .options_base import BaseOptionsStrategy

__all__ = [
    "BaseStrategy",
    "StrategyConfig", 
    "BaseOptionsStrategy",
] 