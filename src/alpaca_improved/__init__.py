"""
Alpaca Improved - Advanced Options Trading Platform

A comprehensive, production-ready options trading platform built on top of the 
Alpaca Trading API ecosystem with sophisticated backtesting and strategy development capabilities.

Author: Alpaca Improved Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Alpaca Improved Team"
__email__ = "team@alpacaimproved.com"
__description__ = "Advanced Options Trading Platform built on Alpaca's ecosystem"

# Core modules
from .config import Config, get_config
from .utils.logger import setup_logging

# Initialize default configuration and logging
config = get_config()
setup_logging(config)

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "Config",
    "get_config",
    "config",
] 