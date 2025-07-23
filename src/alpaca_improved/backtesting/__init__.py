"""
Backtesting Framework for Alpaca Improved

This package provides comprehensive backtesting capabilities with:
- Backtrader integration for event-driven backtesting
- VectorBT integration for vectorized backtesting
- Strategy performance analysis and comparison
- Risk metrics calculation
- Automated backtesting workflows
"""

from .backtrader_engine import BacktraderEngine
from .vectorbt_engine import VectorBTEngine
from .performance_analyzer import PerformanceAnalyzer
from .backtest_runner import BacktestRunner

__all__ = [
    "BacktraderEngine",
    "VectorBTEngine", 
    "PerformanceAnalyzer",
    "BacktestRunner",
] 