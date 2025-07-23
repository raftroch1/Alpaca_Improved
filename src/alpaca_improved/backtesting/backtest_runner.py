"""
Comprehensive Backtesting Runner for Alpaca Improved

This module provides a unified interface for running backtests using either
Backtrader or VectorBT engines, with comprehensive performance analysis and comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import Config, get_global_config
from ..strategies.base import BaseStrategy, StrategyConfig
from ..data.market_data import MarketDataManager
from ..utils.logger import get_logger, log_backtest_event


class BacktestEngine(Enum):
    """Available backtesting engines."""
    BACKTRADER = "backtrader"
    VECTORBT = "vectorbt"
    BOTH = "both"


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Capital and costs
    initial_capital: float = 100000.0
    commission: float = 0.0  # Alpaca is commission-free
    slippage: float = 0.001  # 0.1% slippage
    
    # Engine selection
    engine: BacktestEngine = BacktestEngine.BACKTRADER
    
    # Data configuration
    data_frequency: str = "1D"  # 1D, 1H, 15T, etc.
    benchmark_symbol: str = "SPY"
    
    # Analysis configuration
    calculate_detailed_metrics: bool = True
    generate_plots: bool = True
    save_results: bool = True
    output_dir: Optional[Path] = None
    
    # Performance thresholds
    min_sharpe_ratio: float = 1.0
    max_drawdown_percent: float = 20.0
    min_win_rate: float = 0.4
    
    # Validation
    enable_walk_forward: bool = False
    walk_forward_periods: int = 6
    validation_split: float = 0.2  # 20% for out-of-sample testing
    
    # Risk management
    position_size_method: str = "fixed"  # fixed, percentage, kelly
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Container for backtest results."""
    strategy_name: str
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    engine_used: str
    
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    var_95: float
    cvar_95: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio evolution
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    
    # Additional metrics
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float
    
    # Execution details
    execution_time: float
    memory_usage: float
    
    # Raw engine results
    raw_results: Any = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key performance metrics."""
        return {
            "strategy_name": self.strategy_name,
            "engine": self.engine_used,
            "period": f"{self.start_date.date()} to {self.end_date.date()}",
            "total_return": f"{self.total_return:.2%}",
            "annual_return": f"{self.annual_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "total_trades": self.total_trades,
            "profit_factor": f"{self.profit_factor:.2f}",
            "alpha": f"{self.alpha:.2%}",
            "beta": f"{self.beta:.2f}",
        }


class BacktestRunner:
    """
    Comprehensive backtesting runner that supports multiple engines and strategies.
    
    This class provides a unified interface for:
    - Running backtests with different engines (Backtrader, VectorBT)
    - Comprehensive performance analysis
    - Walk-forward analysis and validation
    - Strategy comparison and optimization
    - Automated reporting and visualization
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        data_manager: Optional[MarketDataManager] = None
    ):
        """
        Initialize the backtest runner.
        
        Args:
            config: Application configuration
            data_manager: Market data manager instance
        """
        self.config = config or get_global_config()
        self.logger = get_logger(self.__class__.__name__)
        self.data_manager = data_manager or MarketDataManager(self.config)
        
        # Results storage
        self.results_history: List[BacktestResults] = []
        
        # Setup output directory
        self.output_dir = Path(self.config.storage.data_dir) / "backtest_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Backtest runner initialized")
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        backtest_config: BacktestConfig,
        symbols: List[str],
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResults:
        """
        Run a comprehensive backtest for a strategy.
        
        Args:
            strategy: Strategy instance to backtest
            backtest_config: Backtest configuration
            symbols: List of symbols to trade
            strategy_params: Additional strategy parameters
            
        Returns:
            BacktestResults object with comprehensive metrics
        """
        log_backtest_event(
            strategy_name=strategy.config.name,
            event_type="start",
            message=f"Starting backtest with {backtest_config.engine.value} engine"
        )
        
        start_time = datetime.now()
        
        try:
            # Prepare data
            market_data = self._prepare_market_data(
                symbols, backtest_config.start_date, backtest_config.end_date
            )
            
            # Run backtest based on selected engine
            if backtest_config.engine == BacktestEngine.BACKTRADER:
                results = self._run_backtrader_backtest(
                    strategy, backtest_config, market_data, strategy_params
                )
            elif backtest_config.engine == BacktestEngine.VECTORBT:
                results = self._run_vectorbt_backtest(
                    strategy, backtest_config, market_data, strategy_params
                )
            elif backtest_config.engine == BacktestEngine.BOTH:
                # Run both engines and compare
                bt_results = self._run_backtrader_backtest(
                    strategy, backtest_config, market_data, strategy_params
                )
                vbt_results = self._run_vectorbt_backtest(
                    strategy, backtest_config, market_data, strategy_params
                )
                results = self._compare_engine_results(bt_results, vbt_results)
            else:
                raise ValueError(f"Unknown backtest engine: {backtest_config.engine}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            results.execution_time = execution_time
            
            # Store results
            self.results_history.append(results)
            
            # Generate reports if requested
            if backtest_config.save_results:
                self._save_results(results, backtest_config)
            
            if backtest_config.generate_plots:
                self._generate_plots(results, backtest_config)
            
            log_backtest_event(
                strategy_name=strategy.config.name,
                event_type="complete",
                message=f"Backtest completed in {execution_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            log_backtest_event(
                strategy_name=strategy.config.name,
                event_type="error",
                message=f"Backtest failed: {str(e)}"
            )
            raise
    
    def run_walk_forward_analysis(
        self,
        strategy: BaseStrategy,
        backtest_config: BacktestConfig,
        symbols: List[str],
        train_periods: int = 252,  # 1 year
        test_periods: int = 63,   # 3 months
        step_size: int = 21       # 1 month
    ) -> List[BacktestResults]:
        """
        Run walk-forward analysis to validate strategy robustness.
        
        Args:
            strategy: Strategy to test
            backtest_config: Base backtest configuration
            symbols: Symbols to trade
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step_size: Step size for rolling forward
            
        Returns:
            List of backtest results for each walk-forward period
        """
        self.logger.info("Starting walk-forward analysis")
        
        results = []
        current_start = backtest_config.start_date
        
        while current_start + timedelta(days=train_periods + test_periods) <= backtest_config.end_date:
            # Define train and test periods
            train_end = current_start + timedelta(days=train_periods)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_periods)
            
            # Create configuration for this period
            period_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=backtest_config.initial_capital,
                commission=backtest_config.commission,
                engine=backtest_config.engine,
                save_results=False,  # Don't save individual results
                generate_plots=False
            )
            
            try:
                # Run backtest for this period
                period_result = self.run_backtest(strategy, period_config, symbols)
                period_result.strategy_name += f"_WF_{len(results)+1}"
                results.append(period_result)
                
                self.logger.info(
                    f"Completed walk-forward period {len(results)}: "
                    f"{test_start.date()} to {test_end.date()}"
                )
                
            except Exception as e:
                self.logger.error(f"Walk-forward period failed: {e}")
                continue
            
            # Move forward
            current_start += timedelta(days=step_size)
        
        # Generate walk-forward analysis report
        self._generate_walk_forward_report(results, backtest_config)
        
        return results
    
    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        backtest_config: BacktestConfig,
        symbols: List[str]
    ) -> Dict[str, BacktestResults]:
        """
        Compare multiple strategies using the same backtest configuration.
        
        Args:
            strategies: List of strategies to compare
            backtest_config: Backtest configuration
            symbols: Symbols to trade
            
        Returns:
            Dictionary mapping strategy names to their results
        """
        self.logger.info(f"Comparing {len(strategies)} strategies")
        
        results = {}
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, backtest_config, symbols)
                results[strategy.config.name] = result
                
            except Exception as e:
                self.logger.error(f"Strategy {strategy.config.name} failed: {e}")
                continue
        
        # Generate comparison report
        self._generate_comparison_report(results, backtest_config)
        
        return results
    
    def _prepare_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Prepare market data for backtesting."""
        market_data = {}
        
        for symbol in symbols:
            try:
                data = self.data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                market_data[symbol] = data
                
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
        
        return market_data
    
    def _run_backtrader_backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        market_data: Dict[str, pd.DataFrame],
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResults:
        """Run backtest using Backtrader engine."""
        # Implementation would use the BacktraderEngine
        # This is a simplified version
        
        # Simulate basic backtest results for now
        # In real implementation, this would use the actual Backtrader integration
        return self._create_mock_results(strategy, config, "backtrader")
    
    def _run_vectorbt_backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        market_data: Dict[str, pd.DataFrame],
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResults:
        """Run backtest using VectorBT engine."""
        # Implementation would use the VectorBTEngine
        # This is a simplified version
        
        # Simulate basic backtest results for now
        # In real implementation, this would use the actual VectorBT integration
        return self._create_mock_results(strategy, config, "vectorbt")
    
    def _create_mock_results(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        engine: str
    ) -> BacktestResults:
        """Create mock results for demonstration purposes."""
        
        # Generate mock portfolio values
        days = (config.end_date - config.start_date).days
        dates = pd.date_range(config.start_date, config.end_date, freq='D')
        
        # Simulate portfolio performance with some randomness
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
        portfolio_values = pd.Series(
            config.initial_capital * (1 + returns).cumprod(),
            index=dates
        )
        
        returns_series = portfolio_values.pct_change().dropna()
        
        # Calculate basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / days) - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return BacktestResults(
            strategy_name=strategy.config.name,
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            engine_used=engine,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio * 1.1,  # Mock value
            max_drawdown=max_drawdown,
            max_drawdown_duration=30,  # Mock value
            volatility=volatility,
            var_95=returns_series.quantile(0.05),
            cvar_95=returns_series[returns_series <= returns_series.quantile(0.05)].mean(),
            total_trades=100,  # Mock value
            winning_trades=60,  # Mock value
            losing_trades=40,   # Mock value
            win_rate=0.6,
            avg_win=0.025,      # Mock value
            avg_loss=-0.015,    # Mock value
            profit_factor=1.67,  # Mock value
            portfolio_values=portfolio_values,
            returns=returns_series,
            positions=pd.DataFrame(),  # Mock empty
            trades=pd.DataFrame(),     # Mock empty
            benchmark_return=0.08,     # Mock 8% benchmark
            alpha=annual_return - 0.08,
            beta=1.2,                  # Mock beta
            information_ratio=0.5,     # Mock value
            calmar_ratio=annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            omega_ratio=2.1,           # Mock value
            tail_ratio=0.8,            # Mock value
            execution_time=0.0,
            memory_usage=0.0
        )
    
    def _compare_engine_results(
        self,
        bt_results: BacktestResults,
        vbt_results: BacktestResults
    ) -> BacktestResults:
        """Compare results from different engines."""
        # For now, return the Backtrader results
        # In a real implementation, this would provide detailed comparison
        bt_results.engine_used = "backtrader_vs_vectorbt"
        return bt_results
    
    def _save_results(self, results: BacktestResults, config: BacktestConfig) -> None:
        """Save backtest results to file."""
        output_file = self.output_dir / f"{results.strategy_name}_backtest_results.json"
        
        # Convert results to JSON-serializable format
        results_dict = {
            "strategy_name": results.strategy_name,
            "summary": results.get_summary(),
            "config": {
                "start_date": config.start_date.isoformat(),
                "end_date": config.end_date.isoformat(),
                "initial_capital": config.initial_capital,
                "engine": config.engine.value,
            },
            "performance_metrics": {
                "total_return": results.total_return,
                "annual_return": results.annual_return,
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
            }
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def _generate_plots(self, results: BacktestResults, config: BacktestConfig) -> None:
        """Generate performance plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"{results.strategy_name} - Backtest Results", fontsize=16)
            
            # Portfolio value over time
            axes[0, 0].plot(results.portfolio_values.index, results.portfolio_values.values)
            axes[0, 0].set_title("Portfolio Value")
            axes[0, 0].set_ylabel("Value ($)")
            
            # Returns distribution
            axes[0, 1].hist(results.returns.dropna(), bins=50, alpha=0.7)
            axes[0, 1].set_title("Returns Distribution")
            axes[0, 1].set_xlabel("Daily Returns")
            
            # Drawdown
            rolling_max = results.portfolio_values.expanding().max()
            drawdown = (results.portfolio_values - rolling_max) / rolling_max
            axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
            axes[1, 0].set_title("Drawdown")
            axes[1, 0].set_ylabel("Drawdown (%)")
            
            # Rolling Sharpe ratio
            rolling_sharpe = results.returns.rolling(252).mean() / results.returns.rolling(252).std() * np.sqrt(252)
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
            axes[1, 1].set_title("Rolling Sharpe Ratio (1Y)")
            axes[1, 1].set_ylabel("Sharpe Ratio")
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{results.strategy_name}_performance_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance plots saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def _generate_walk_forward_report(
        self,
        results: List[BacktestResults],
        config: BacktestConfig
    ) -> None:
        """Generate walk-forward analysis report."""
        if not results:
            return
        
        # Calculate stability metrics
        returns = [r.annual_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        
        stability_metrics = {
            "periods_tested": len(results),
            "avg_annual_return": np.mean(returns),
            "std_annual_return": np.std(returns),
            "avg_sharpe_ratio": np.mean(sharpe_ratios),
            "std_sharpe_ratio": np.std(sharpe_ratios),
            "positive_periods": sum(1 for r in returns if r > 0),
            "win_rate_periods": sum(1 for r in returns if r > 0) / len(returns)
        }
        
        self.logger.info(f"Walk-forward analysis completed: {stability_metrics}")
    
    def _generate_comparison_report(
        self,
        results: Dict[str, BacktestResults],
        config: BacktestConfig
    ) -> None:
        """Generate strategy comparison report."""
        if not results:
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                "Strategy": name,
                "Total Return": f"{result.total_return:.2%}",
                "Annual Return": f"{result.annual_return:.2%}",
                "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                "Max Drawdown": f"{result.max_drawdown:.2%}",
                "Win Rate": f"{result.win_rate:.2%}",
                "Profit Factor": f"{result.profit_factor:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        report_file = self.output_dir / "strategy_comparison.csv"
        comparison_df.to_csv(report_file, index=False)
        
        self.logger.info(f"Strategy comparison saved to {report_file}")
    
    def get_best_strategy(self, metric: str = "sharpe_ratio") -> Optional[BacktestResults]:
        """Get the best performing strategy based on a specific metric."""
        if not self.results_history:
            return None
        
        return max(self.results_history, key=lambda x: getattr(x, metric, 0)) 