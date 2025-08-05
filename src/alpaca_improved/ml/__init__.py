"""
Machine Learning Integration Module
==================================

Advanced ML capabilities for trading strategy optimization and risk management.
Inspired by Liu Algo Trader's ML integration approach.

Components:
- TradeProbabilityEstimator: ML-based trade success prediction
- AdaptiveParameterOptimizer: Real-time strategy parameter optimization
- PerformanceAnalyzer: ML-enhanced performance analytics

Author: Alpaca Improved Team
"""

from .trade_probability_estimator import (
    TradeProbabilityEstimator,
    TradeFeatures,
    TradePrediction,
    create_ml_enhanced_backtest_data
)

__all__ = [
    'TradeProbabilityEstimator',
    'TradeFeatures', 
    'TradePrediction',
    'create_ml_enhanced_backtest_data'
]