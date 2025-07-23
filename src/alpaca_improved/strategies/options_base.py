"""
Options-Specific Base Strategy for Alpaca Improved

This module extends the base strategy with comprehensive options trading capabilities,
including Greeks calculations, implied volatility analysis, and options-specific risk management.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data import OptionsHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, OptionChainRequest

from .base import BaseStrategy, StrategyConfig, TradingSignal, SignalType, Position
from ..utils.logger import get_logger, log_trade_event


class OptionsStrategyType(Enum):
    """Types of options strategies."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"


class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionsContract:
    """Represents an options contract."""
    symbol: str  # Underlying symbol
    option_symbol: str  # Full options symbol
    strike_price: float
    expiration_date: datetime
    option_type: OptionType
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last_price
    
    @property
    def days_to_expiration(self) -> int:
        """Calculate days to expiration."""
        return (self.expiration_date.date() - datetime.now().date()).days
    
    @property
    def time_to_expiration(self) -> float:
        """Calculate time to expiration in years."""
        return self.days_to_expiration / 365.0
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money."""
        if self.option_type == OptionType.CALL:
            return self.strike_price < (self.last_price or 0)
        else:
            return self.strike_price > (self.last_price or 0)


@dataclass
class OptionsPosition(Position):
    """Represents an options position with Greeks."""
    option_type: OptionType = OptionType.CALL
    strike_price: float = 0.0
    expiration_date: Optional[datetime] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    @property
    def delta_exposure(self) -> float:
        """Calculate delta exposure."""
        return (self.delta or 0) * self.quantity * 100  # Each contract represents 100 shares
    
    @property
    def theta_decay(self) -> float:
        """Calculate daily theta decay."""
        return (self.theta or 0) * self.quantity
    
    @property
    def vega_exposure(self) -> float:
        """Calculate vega exposure."""
        return (self.vega or 0) * self.quantity


@dataclass
class OptionsSignal(TradingSignal):
    """Options-specific trading signal."""
    option_symbol: str = ""
    strike_price: float = 0.0
    expiration_date: Optional[datetime] = None
    option_type: OptionType = OptionType.CALL
    strategy_type: OptionsStrategyType = OptionsStrategyType.LONG_CALL
    target_delta: Optional[float] = None
    max_dte: int = 45  # Maximum days to expiration
    min_dte: int = 7   # Minimum days to expiration


@dataclass
class OptionsStrategyConfig(StrategyConfig):
    """Configuration specific to options strategies."""
    # Options-specific parameters
    max_dte: int = 45  # Maximum days to expiration
    min_dte: int = 7   # Minimum days to expiration
    target_delta: float = 0.3  # Target delta for options selection
    delta_range: Tuple[float, float] = (0.2, 0.8)  # Acceptable delta range
    theta_threshold: float = -0.1  # Minimum theta (for selling strategies)
    iv_percentile_min: float = 30  # Minimum IV percentile
    iv_percentile_max: float = 70  # Maximum IV percentile
    max_gamma_exposure: float = 1000  # Maximum gamma exposure
    max_vega_exposure: float = 5000  # Maximum vega exposure
    profit_target: float = 0.5  # 50% profit target
    loss_limit: float = 2.0  # 200% loss limit (for credit strategies)
    
    # Greeks management
    portfolio_delta_limit: float = 100  # Maximum portfolio delta
    portfolio_theta_target: float = -50  # Target portfolio theta (for income strategies)
    rebalance_delta_threshold: float = 50  # Delta threshold for rebalancing


class BaseOptionsStrategy(BaseStrategy):
    """
    Base class for options trading strategies.
    
    This class extends BaseStrategy with options-specific functionality including:
    - Options chain data retrieval
    - Greeks calculations and monitoring
    - Options-specific risk management
    - Multi-leg strategy support
    """
    
    def __init__(
        self,
        config: OptionsStrategyConfig,
        **kwargs
    ):
        """
        Initialize the options strategy.
        
        Args:
            config: Options strategy configuration
            **kwargs: Additional arguments passed to BaseStrategy
        """
        super().__init__(config, **kwargs)
        self.options_config = config
        self.options_data_client = self._create_options_data_client()
        
        # Options-specific tracking
        self.options_positions: Dict[str, OptionsPosition] = {}
        self.portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        self.logger.info(f"Options strategy '{self.config.name}' initialized")
    
    def _create_options_data_client(self) -> OptionsHistoricalDataClient:
        """Create Alpaca options data client."""
        return OptionsHistoricalDataClient(
            api_key=self.app_config.alpaca.api_key,
            secret_key=self.app_config.alpaca.secret_key
        )
    
    @abstractmethod
    def analyze_options_chain(
        self,
        symbol: str,
        expiration_date: datetime,
        options_chain: List[OptionsContract]
    ) -> List[OptionsSignal]:
        """
        Analyze options chain and generate trading signals.
        
        Args:
            symbol: Underlying symbol
            expiration_date: Options expiration date
            options_chain: List of options contracts
            
        Returns:
            List of options trading signals
        """
        pass
    
    @abstractmethod
    def calculate_strategy_greeks(
        self,
        contracts: List[OptionsContract],
        quantities: List[int]
    ) -> Dict[str, float]:
        """
        Calculate combined Greeks for a multi-leg strategy.
        
        Args:
            contracts: List of options contracts
            quantities: List of quantities (positive for long, negative for short)
            
        Returns:
            Dictionary of combined Greeks
        """
        pass
    
    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[datetime] = None
    ) -> List[OptionsContract]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration_date: Specific expiration date (optional)
            
        Returns:
            List of options contracts
        """
        try:
            # If no expiration specified, get next monthly expiration
            if expiration_date is None:
                expiration_date = self._get_next_monthly_expiration()
            
            request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date=expiration_date
            )
            
            chain_data = self.options_data_client.get_option_chain(request)
            
            # Convert to OptionsContract objects
            contracts = []
            for contract_data in chain_data:
                contract = OptionsContract(
                    symbol=symbol,
                    option_symbol=contract_data.symbol,
                    strike_price=contract_data.strike_price,
                    expiration_date=expiration_date,
                    option_type=OptionType.CALL if 'C' in contract_data.symbol else OptionType.PUT,
                    bid=getattr(contract_data, 'bid', None),
                    ask=getattr(contract_data, 'ask', None),
                    last_price=getattr(contract_data, 'last_price', None),
                    volume=getattr(contract_data, 'volume', None),
                    open_interest=getattr(contract_data, 'open_interest', None),
                    implied_volatility=getattr(contract_data, 'implied_volatility', None),
                    delta=getattr(contract_data, 'delta', None),
                    gamma=getattr(contract_data, 'gamma', None),
                    theta=getattr(contract_data, 'theta', None),
                    vega=getattr(contract_data, 'vega', None),
                    rho=getattr(contract_data, 'rho', None),
                )
                contracts.append(contract)
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error fetching options chain for {symbol}: {e}")
            return []
    
    def filter_options_by_criteria(
        self,
        contracts: List[OptionsContract],
        option_type: Optional[OptionType] = None,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        min_volume: Optional[int] = None,
        min_open_interest: Optional[int] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None
    ) -> List[OptionsContract]:
        """
        Filter options contracts by various criteria.
        
        Args:
            contracts: List of options contracts
            option_type: Filter by option type (call/put)
            min_delta: Minimum delta
            max_delta: Maximum delta
            min_volume: Minimum volume
            min_open_interest: Minimum open interest
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            Filtered list of options contracts
        """
        filtered = contracts
        
        if option_type:
            filtered = [c for c in filtered if c.option_type == option_type]
        
        if min_delta is not None:
            filtered = [c for c in filtered if c.delta and c.delta >= min_delta]
        
        if max_delta is not None:
            filtered = [c for c in filtered if c.delta and c.delta <= max_delta]
        
        if min_volume is not None:
            filtered = [c for c in filtered if c.volume and c.volume >= min_volume]
        
        if min_open_interest is not None:
            filtered = [c for c in filtered if c.open_interest and c.open_interest >= min_open_interest]
        
        if min_dte is not None:
            filtered = [c for c in filtered if c.days_to_expiration >= min_dte]
        
        if max_dte is not None:
            filtered = [c for c in filtered if c.days_to_expiration <= max_dte]
        
        return filtered
    
    def select_optimal_contract(
        self,
        contracts: List[OptionsContract],
        target_delta: Optional[float] = None,
        prefer_high_volume: bool = True
    ) -> Optional[OptionsContract]:
        """
        Select the optimal contract from a list based on criteria.
        
        Args:
            contracts: List of options contracts
            target_delta: Target delta value
            prefer_high_volume: Whether to prefer higher volume contracts
            
        Returns:
            Optimal contract or None if no suitable contract found
        """
        if not contracts:
            return None
        
        # Filter by basic liquidity requirements
        liquid_contracts = [
            c for c in contracts
            if (c.volume or 0) > 0 and (c.open_interest or 0) > 0
        ]
        
        if not liquid_contracts:
            return None
        
        # If target delta specified, find closest match
        if target_delta is not None:
            liquid_contracts.sort(
                key=lambda c: abs((c.delta or 0) - target_delta)
            )
        
        # If prefer high volume, sort by volume descending
        if prefer_high_volume:
            liquid_contracts.sort(
                key=lambda c: (c.volume or 0) * (c.open_interest or 0),
                reverse=True
            )
        
        return liquid_contracts[0]
    
    def execute_options_trade(
        self,
        option_symbol: str,
        quantity: int,
        side: OrderSide,
        strategy_type: OptionsStrategyType = OptionsStrategyType.LONG_CALL
    ) -> bool:
        """
        Execute an options trade.
        
        Args:
            option_symbol: Options contract symbol
            quantity: Number of contracts
            side: Order side (buy/sell)
            strategy_type: Type of options strategy
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        try:
            order_request = MarketOrderRequest(
                symbol=option_symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                asset_class=AssetClass.OPTION
            )
            
            order = self.trading_client.submit_order(order_request)
            
            # Log the trade
            log_trade_event(
                trade_type="option",
                symbol=option_symbol,
                action=side.value,
                quantity=quantity,
                strategy=self.config.name,
                strategy_type=strategy_type.value
            )
            
            self.logger.info(f"Options trade executed: {side.value} {quantity} {option_symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing options trade: {e}")
            return False
    
    def update_portfolio_greeks(self) -> None:
        """Update portfolio-level Greeks from all positions."""
        self.portfolio_greeks = {
            'delta': sum(pos.delta_exposure for pos in self.options_positions.values()),
            'gamma': sum((pos.gamma or 0) * pos.quantity for pos in self.options_positions.values()),
            'theta': sum(pos.theta_decay for pos in self.options_positions.values()),
            'vega': sum(pos.vega_exposure for pos in self.options_positions.values()),
            'rho': sum((pos.rho or 0) * pos.quantity for pos in self.options_positions.values()),
        }
    
    def check_portfolio_risk_limits(self) -> bool:
        """
        Check if portfolio is within risk limits.
        
        Returns:
            True if within limits, False otherwise
        """
        self.update_portfolio_greeks()
        
        # Check delta limit
        if abs(self.portfolio_greeks['delta']) > self.options_config.portfolio_delta_limit:
            self.logger.warning(f"Portfolio delta {self.portfolio_greeks['delta']} exceeds limit")
            return False
        
        # Check gamma exposure
        if abs(self.portfolio_greeks['gamma']) > self.options_config.max_gamma_exposure:
            self.logger.warning(f"Portfolio gamma {self.portfolio_greeks['gamma']} exceeds limit")
            return False
        
        # Check vega exposure
        if abs(self.portfolio_greeks['vega']) > self.options_config.max_vega_exposure:
            self.logger.warning(f"Portfolio vega {self.portfolio_greeks['vega']} exceeds limit")
            return False
        
        return True
    
    def _get_next_monthly_expiration(self) -> datetime:
        """Get the next monthly options expiration date."""
        now = datetime.now()
        
        # Third Friday of current month
        third_friday = self._get_third_friday(now.year, now.month)
        
        # If we've passed this month's expiration, get next month's
        if now.date() > third_friday.date():
            if now.month == 12:
                third_friday = self._get_third_friday(now.year + 1, 1)
            else:
                third_friday = self._get_third_friday(now.year, now.month + 1)
        
        return third_friday
    
    def _get_third_friday(self, year: int, month: int) -> datetime:
        """Get the third Friday of a given month."""
        # First day of the month
        first_day = datetime(year, month, 1)
        
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday
    
    def get_options_performance_metrics(self) -> Dict[str, Any]:
        """Get options-specific performance metrics."""
        base_metrics = self.get_performance_metrics()
        
        # Calculate options-specific metrics
        total_theta_income = sum(
            pos.theta_decay for pos in self.options_positions.values()
            if pos.theta_decay < 0  # Positive theta income
        )
        
        options_metrics = {
            "portfolio_delta": self.portfolio_greeks['delta'],
            "portfolio_gamma": self.portfolio_greeks['gamma'],
            "portfolio_theta": self.portfolio_greeks['theta'],
            "portfolio_vega": self.portfolio_greeks['vega'],
            "portfolio_rho": self.portfolio_greeks['rho'],
            "options_positions": len(self.options_positions),
            "daily_theta_income": total_theta_income,
        }
        
        return {**base_metrics, **options_metrics} 