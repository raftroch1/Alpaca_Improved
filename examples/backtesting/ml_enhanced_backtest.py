#!/usr/bin/env python3
"""
ML-ENHANCED BACKTEST - Turn -4.76% into Profitable Returns
========================================================

Integrates Liu Algo Trader style ML probability estimation with our proven
risk management framework to optimize strategy performance.

🎯 GOAL: Transform our -4.76% return into profitable results using:
- ML-based trade success probability prediction
- Adaptive parameter optimization
- Real-time strategy learning

Building on our proven 93.59 percentage point improvement from risk management!

Author: Alpaca Improved Team
Version: ML Enhanced Backtest v1.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Add the parent directory to path to import from enhanced_risk_management_backtest
sys.path.append(os.path.dirname(__file__))

# Import our previous proven risk management framework
from enhanced_risk_management_backtest import (
    ProfessionalOptionsBacktester,
    ProfessionalTrade,
    ExitReason,
    MAShiftSignal,
    generate_ma_shift_signals,
    get_spy_data,
    get_alpaca_option_price
)

# Import our new ML components
from alpaca_improved.ml import (
    TradeProbabilityEstimator,
    TradePrediction,
    create_ml_enhanced_backtest_data
)

class MLEnhancedBacktester(ProfessionalOptionsBacktester):
    """
    Enhanced backtester with ML-based probability estimation and adaptive optimization
    """
    
    def __init__(self, initial_capital: float = 25000):
        super().__init__(initial_capital)
        
        # ML Components
        self.ml_estimator = TradeProbabilityEstimator()
        self.ml_predictions = []
        self.ml_performance_history = []
        
        # Adaptive parameters (start with proven defaults)
        self.adaptive_profit_target_15 = 0.15
        self.adaptive_profit_target_25 = 0.25  
        self.adaptive_stop_loss = 0.20
        self.adaptive_position_size = 0.04
        
        # ML learning parameters
        self.min_trades_for_ml = 10  # Minimum trades before ML kicks in
        self.learning_window = 20    # Window for recent performance analysis
        self.confidence_threshold = 0.6  # Minimum ML confidence to adjust parameters
        
        # Performance tracking
        self.ml_accuracy = 0.0
        self.parameter_evolution = []
        
    def train_ml_model(self, historical_trades: List[Dict]) -> Dict:
        """Train the ML model on historical trade data"""
        
        if len(historical_trades) < self.min_trades_for_ml:
            self.logger.info(f"Need {self.min_trades_for_ml - len(historical_trades)} more trades for ML training")
            return {"status": "insufficient_data"}
        
        # Convert our trade format to ML format
        ml_trade_data = create_ml_enhanced_backtest_data(historical_trades)
        
        # Train the model
        training_result = self.ml_estimator.train_model(ml_trade_data)
        
        if training_result["status"] == "success":
            self.logger.info(f"🤖 ML Model trained! Accuracy: {training_result['accuracy']:.3f}")
            self.logger.info(f"🎯 Top features: {training_result.get('feature_importance', {})}")
            
        return training_result
    
    def get_ml_prediction(self, signal: MAShiftSignal, recent_performance: Dict) -> TradePrediction:
        """Get ML prediction for trade success"""
        
        # Prepare trade data for ML prediction
        trade_data = {
            'signal_confidence': 0.7,  # Base confidence
            'volatility': 0.2,  # Market volatility estimate
            'timestamp': signal.timestamp,
            'expiry_date': signal.timestamp + timedelta(days=7),
            'underlying_price': signal.price,
            'strike_price': signal.price * 1.02 if signal.signal_type == 'BULLISH' else signal.price * 0.98,
            'volume': 5000,
            'recent_price_changes': [0.01, -0.005, 0.008],  # Sample market data
            'recent_win_rate': recent_performance.get('win_rate', 0.5)
        }
        
        return self.ml_estimator.predict_trade_success(trade_data)
    
    def adaptive_parameter_optimization(self, recent_trades: List[ProfessionalTrade]):
        """Optimize parameters based on recent ML performance"""
        
        if len(recent_trades) < 5:
            return  # Need minimum trades for optimization
        
        # Analyze recent performance
        recent_wins = sum(1 for t in recent_trades if t.net_pnl > 0)
        recent_win_rate = recent_wins / len(recent_trades)
        recent_avg_pnl = np.mean([t.net_pnl for t in recent_trades])
        
        # Adaptive adjustment based on performance
        if recent_win_rate > 0.6:  # Performing well
            # Increase position size and profit targets slightly
            self.adaptive_position_size = min(self.adaptive_position_size * 1.1, 0.06)
            self.adaptive_profit_target_15 = min(self.adaptive_profit_target_15 * 1.05, 0.20)
        elif recent_win_rate < 0.4:  # Performing poorly  
            # Decrease position size and tighten stops
            self.adaptive_position_size = max(self.adaptive_position_size * 0.9, 0.02)
            self.adaptive_stop_loss = max(self.adaptive_stop_loss * 0.95, 0.15)
        
        # Track parameter evolution
        self.parameter_evolution.append({
            'timestamp': datetime.now(),
            'win_rate': recent_win_rate,
            'avg_pnl': recent_avg_pnl,
            'position_size': self.adaptive_position_size,
            'profit_target_15': self.adaptive_profit_target_15,
            'stop_loss': self.adaptive_stop_loss
        })
        
        self.logger.info(f"🔧 Adaptive optimization: WR={recent_win_rate:.2f}, "
                        f"PS={self.adaptive_position_size:.3f}, PT={self.adaptive_profit_target_15:.3f}")
    
    def execute_ml_enhanced_trade(self, signal: MAShiftSignal) -> Optional[ProfessionalTrade]:
        """Execute trade with ML enhancement and adaptive parameters"""
        
        # Calculate recent performance for ML input
        recent_trades = self.trades[-self.learning_window:] if len(self.trades) >= self.learning_window else self.trades
        recent_performance = {
            'win_rate': sum(1 for t in recent_trades if t.net_pnl > 0) / len(recent_trades) if recent_trades else 0.5,
            'avg_pnl': np.mean([t.net_pnl for t in recent_trades]) if recent_trades else 0.0
        }
        
        # Get ML prediction if model is trained
        ml_prediction = None
        if self.ml_estimator.is_trained:
            ml_prediction = self.get_ml_prediction(signal, recent_performance)
            self.ml_predictions.append(ml_prediction)
            
            # Skip trade if ML prediction is very pessimistic
            if ml_prediction.success_probability < 0.3:
                self.logger.info(f"🤖 ML suggests SKIP trade (probability: {ml_prediction.success_probability:.3f})")
                return None
                
            self.logger.info(f"🤖 ML Prediction: {ml_prediction.success_probability:.3f} probability, "
                           f"confidence: {ml_prediction.confidence:.3f}")
        
        # Build options chain
        self.logger.info(f"🔍 Building options chain for SPY @ ${signal.price:.2f}")
        
        # Determine strike price (slightly OTM for better premium)
        if signal.signal_type == 'BULLISH':
            strike_price = round(signal.price * 1.02)  # 2% OTM call
        else:
            strike_price = round(signal.price * 0.98)  # 2% OTM put
        
        # Get option contract details
        expiry_date = signal.timestamp + timedelta(days=7)
        option_type = 'C' if signal.signal_type == 'BULLISH' else 'P'
        option_symbol = f"SPY{expiry_date.strftime('%y%m%d')}{option_type}{strike_price:08d}"
        
        # Get entry price
        entry_price = get_alpaca_option_price(option_symbol, signal.timestamp)
        if entry_price is None:
            return None
        
        # Calculate position size using ML recommendation if available
        if ml_prediction and ml_prediction.confidence > self.confidence_threshold:
            position_size_pct = ml_prediction.recommended_position_size
            profit_target_15 = ml_prediction.suggested_profit_target
            stop_loss_pct = ml_prediction.suggested_stop_loss
        else:
            # Use adaptive parameters
            position_size_pct = self.adaptive_position_size
            profit_target_15 = self.adaptive_profit_target_15
            stop_loss_pct = self.adaptive_stop_loss
        
        # Calculate contracts to buy
        position_value = self.cash * position_size_pct
        entry_cost_per_contract = entry_price * 100
        contracts_to_buy = max(1, int(position_value / entry_cost_per_contract))
        
        # Calculate costs
        entry_cost = contracts_to_buy * entry_price * 100
        commission = contracts_to_buy * 0.65
        bid_ask_cost = entry_cost * self.bid_ask_spread_pct
        slippage_cost = entry_cost * self.slippage_pct
        total_entry_cost = entry_cost + commission + bid_ask_cost + slippage_cost
        
        # Check if we have enough cash
        if total_entry_cost > self.cash:
            return None
        
        # Execute entry
        self.cash -= total_entry_cost
        
        # Simulate trade execution with ML-enhanced exit strategy
        exit_timestamp, exit_price, exit_reason = self.simulate_ml_enhanced_exit(
            signal, entry_price, profit_target_15, stop_loss_pct, ml_prediction
        )
        
        # Calculate exit costs
        exit_proceeds = contracts_to_buy * exit_price * 100
        exit_commission = contracts_to_buy * 0.65
        exit_bid_ask_cost = exit_proceeds * self.bid_ask_spread_pct
        exit_slippage_cost = exit_proceeds * self.slippage_pct
        total_exit_costs = exit_commission + exit_bid_ask_cost + exit_slippage_cost
        net_exit_proceeds = exit_proceeds - total_exit_costs
        
        # Calculate P&L
        gross_pnl = exit_proceeds - entry_cost
        net_pnl = net_exit_proceeds - total_entry_cost
        total_costs = commission + exit_commission + bid_ask_cost + exit_bid_ask_cost + slippage_cost + exit_slippage_cost
        
        # Add proceeds to cash
        self.cash += net_exit_proceeds
        
        # Calculate hold time
        hold_time_hours = (exit_timestamp - signal.timestamp).total_seconds() / 3600
        
        # Create trade record
        trade = ProfessionalTrade(
            entry_date=signal.timestamp,
            exit_date=exit_timestamp,
            signal_type=signal.signal_type,
            option_symbol=option_symbol,
            option_type=option_type,
            strike=strike_price,
            contracts=contracts_to_buy,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            commission=commission + exit_commission,
            bid_ask_cost=bid_ask_cost + exit_bid_ask_cost,
            slippage_cost=slippage_cost + exit_slippage_cost,
            total_costs=total_costs,
            exit_reason=exit_reason,
            hold_time_hours=hold_time_hours,
            underlying_price=signal.price,
            risk_reward_ratio=abs(gross_pnl / (entry_cost * stop_loss_pct)) if stop_loss_pct > 0 else 0.0,
            ml_probability=ml_prediction.success_probability if ml_prediction else None,
            ml_confidence=ml_prediction.confidence if ml_prediction else None
        )
        
        self.trades.append(trade)
        
        # Update equity curve
        portfolio_value = self.cash
        self.equity_curve.append({
            'date': exit_timestamp,
            'value': portfolio_value
        })
        
        # Update ML model performance if we had a prediction
        if ml_prediction:
            actual_result = trade.net_pnl > 0
            self.ml_estimator.update_model_performance(ml_prediction, actual_result)
        
        self.logger.info(f"✅ ML-ENHANCED TRADE EXECUTED!")
        self.logger.info(f"   Option: {option_symbol}")
        self.logger.info(f"   Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
        self.logger.info(f"   Exit Reason: {exit_reason.value}")
        self.logger.info(f"   Hold Time: {hold_time_hours:.1f} hours")
        self.logger.info(f"   Net P&L: ${net_pnl:.2f}")
        if ml_prediction:
            self.logger.info(f"   ML Probability: {ml_prediction.success_probability:.3f}")
        
        return trade
    
    def simulate_ml_enhanced_exit(self, signal: MAShiftSignal, entry_price: float, 
                                 profit_target_15: float, stop_loss_pct: float,
                                 ml_prediction: Optional[TradePrediction]) -> Tuple[datetime, float, ExitReason]:
        """Simulate exit with ML-enhanced timing"""
        
        # Use ML-suggested hold time if available
        if ml_prediction and ml_prediction.confidence > self.confidence_threshold:
            max_hold_hours = ml_prediction.hold_time_estimate
        else:
            max_hold_hours = 72  # Default 3 days
        
        # Calculate target and stop prices
        profit_target_15_price = entry_price * (1 + profit_target_15)
        profit_target_25_price = entry_price * (1 + self.adaptive_profit_target_25)
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        # Simulate price movement and exit logic
        current_time = signal.timestamp
        hours_elapsed = 0
        trailing_stop_price = stop_loss_price
        
        while hours_elapsed < max_hold_hours:
            # Simulate hourly price checks
            current_time += timedelta(hours=1)
            hours_elapsed += 1
            
            # Get simulated current option price
            current_price = get_alpaca_option_price(
                f"SPY{(signal.timestamp + timedelta(days=7)).strftime('%y%m%d')}{'C' if signal.signal_type == 'BULLISH' else 'P'}{round(signal.price * (1.02 if signal.signal_type == 'BULLISH' else 0.98)):08d}",
                current_time
            )
            
            if current_price is None:
                current_price = entry_price * 0.8  # Assume time decay
            
            # Check exit conditions with ML enhancement
            
            # 1. Profit target exits (ML may adjust these dynamically)
            if current_price >= profit_target_25_price:
                return current_time, current_price, ExitReason.PROFIT_TARGET_25
            elif current_price >= profit_target_15_price:
                # ML confidence affects when we take profits
                if ml_prediction and ml_prediction.confidence > 0.7:
                    # High confidence - let it run a bit more
                    if hours_elapsed > 6:  # But take profit after 6 hours minimum
                        return current_time, current_price, ExitReason.PROFIT_TARGET_15
                else:
                    return current_time, current_price, ExitReason.PROFIT_TARGET_15
            
            # 2. Stop loss exit
            if current_price <= stop_loss_price:
                return current_time, current_price, ExitReason.STOP_LOSS_20
            
            # 3. Trailing stop (implement if price moved favorably)
            if current_price > entry_price * 1.05:  # If up 5%
                new_trailing_stop = current_price * 0.9  # Trail at 10% below peak
                trailing_stop_price = max(trailing_stop_price, new_trailing_stop)
                
                if current_price <= trailing_stop_price:
                    return current_time, current_price, ExitReason.TRAILING_STOP
        
        # Time exit - hold period expired
        final_price = entry_price * 0.85  # Assume time decay
        return current_time, final_price, ExitReason.TIME_EXIT

def run_ml_enhanced_backtest() -> Dict:
    """Run the ML-enhanced backtest"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    print("🤖 ML-ENHANCED BACKTEST")
    print("🎯 TURNING -4.76% INTO PROFITABLE RETURNS!")
    print("=" * 80)
    print("🚀 ML-ENHANCED PROFESSIONAL BACKTEST")
    print("🎯 Building on our proven 93.59% improvement with ML optimization!")
    print("=" * 80)
    
    # Get market data
    print("📊 Fetching SPY market data...")
    spy_data = get_spy_data()
    print(f"✅ Retrieved {len(spy_data)} days of SPY data")
    
    # Generate signals
    print("🎯 Generating trading signals...")
    signals = generate_ma_shift_signals(spy_data)
    tradeable_signals = [s for s in signals if s.timestamp.weekday() < 5]
    print(f"📈 Generated {len(tradeable_signals)} tradeable signals")
    
    # Initialize ML-enhanced backtester
    backtester = MLEnhancedBacktester(initial_capital=25000)
    
    print("🔄 Executing trades with ML ENHANCEMENT...")
    print()
    
    # Train ML model after first batch of trades
    training_trades = []
    
    for i, signal in enumerate(tradeable_signals, 1):
        print(f"📊 Processing signal {i}: {signal.signal_type} @ ${signal.price:.2f}")
        
        # Execute trade
        trade = backtester.execute_ml_enhanced_trade(signal)
        
        if trade:
            # Convert trade to training format and add to training data
            training_trade = {
                'net_pnl': trade.net_pnl,
                'entry_date': trade.entry_date,
                'underlying_price': trade.underlying_price,
                'strike': trade.strike
            }
            training_trades.append(training_trade)
            
            # Train ML model after accumulating enough trades
            if len(training_trades) == backtester.min_trades_for_ml:
                print(f"\n🤖 Training ML model with {len(training_trades)} trades...")
                training_result = backtester.train_ml_model(training_trades)
                print()
            elif len(training_trades) > backtester.min_trades_for_ml and len(training_trades) % 10 == 0:
                # Retrain every 10 trades to adapt
                print(f"\n🔄 Retraining ML model with {len(training_trades)} trades...")
                backtester.train_ml_model(training_trades)
                print()
            
            # Adaptive parameter optimization every 5 trades
            if len(backtester.trades) % 5 == 0:
                backtester.adaptive_parameter_optimization(backtester.trades[-5:])
    
    # Calculate results
    print("=" * 80)
    print("🎉 ML-ENHANCED RESULTS")
    print("=" * 80)
    
    total_trades = len(backtester.trades)
    winning_trades = len([t for t in backtester.trades if t.net_pnl > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_net_pnl = sum([t.net_pnl for t in backtester.trades])
    final_value = backtester.cash
    total_return = ((final_value - 25000) / 25000) * 100
    
    print(f"💰 Starting Capital: ${25000:,}")
    print(f"📊 Signals Generated: {len(tradeable_signals)}")
    print(f"🎯 Trades Executed: {total_trades}")
    print(f"🥇 Winning Trades: {winning_trades}")
    print(f"📈 Win Rate: {win_rate:.1f}%")
    print(f"💰 Total P&L: ${total_net_pnl:.2f}")
    print(f"📈 Total Return: {total_return:.2f}%")
    print(f"💼 Final Value: ${final_value:.2f}")
    
    # ML Performance metrics
    if backtester.ml_estimator.is_trained:
        ml_insights = backtester.ml_estimator.get_model_insights()
        print(f"\n🤖 ML PERFORMANCE:")
        print(f"✅ Model Accuracy: {ml_insights['model_accuracy']:.3f}")
        print(f"📊 Recent Accuracy: {ml_insights['recent_accuracy']:.3f}")
        print(f"🎯 Predictions Made: {ml_insights['prediction_count']}")
        
        # Show parameter evolution
        if backtester.parameter_evolution:
            latest_params = backtester.parameter_evolution[-1]
            print(f"🔧 Final Adaptive Parameters:")
            print(f"   Position Size: {latest_params['position_size']:.3f}")
            print(f"   Profit Target: {latest_params['profit_target_15']:.3f}")
            print(f"   Stop Loss: {latest_params['stop_loss']:.3f}")
    
    # Compare to previous results
    print(f"\n🚀 COMPARISON TO PREVIOUS RESULTS:")
    print(f"   Disaster (No Risk Mgmt): -98.35% (CATASTROPHIC)")
    print(f"   Professional Risk Mgmt: -4.76% (MANAGED)")
    print(f"   ML Enhanced: {total_return:.2f}% (OPTIMIZED)")
    
    if total_return > -4.76:
        improvement = total_return - (-4.76)
        print(f"   ML Improvement: +{improvement:.2f} percentage points! 🎉")
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'final_value': final_value,
        'ml_accuracy': backtester.ml_estimator.model_accuracy if backtester.ml_estimator.is_trained else 0,
        'trades': backtester.trades,
        'parameter_evolution': backtester.parameter_evolution
    }

if __name__ == "__main__":
    results = run_ml_enhanced_backtest()