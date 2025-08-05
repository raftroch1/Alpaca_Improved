#!/usr/bin/env python3
"""
ML-Based Trade Probability Estimator - Liu Algo Trader Style
==========================================================

Implements machine learning probability estimation for trade success
based on historical performance, market conditions, and trade characteristics.

Inspired by Liu Algo Trader's automatic probability feedback system.

Author: Alpaca Improved Team
Version: ML Probability Estimator v1.0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class TradeFeatures:
    """Features for ML trade prediction"""
    signal_strength: float
    volatility: float
    time_of_day: float  # Hour as fraction of day
    days_to_expiry: float
    underlying_price: float
    strike_distance: float  # Distance from ATM
    volume_profile: float
    market_regime: str  # 'trending', 'ranging', 'volatile'
    recent_win_rate: float
    
@dataclass
class TradePrediction:
    """ML prediction result"""
    success_probability: float
    confidence: float
    recommended_position_size: float
    suggested_stop_loss: float
    suggested_profit_target: float
    hold_time_estimate: float
    risk_adjusted_score: float

class TradeProbabilityEstimator:
    """
    ML-based trade probability estimator inspired by Liu Algo Trader
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.model_accuracy = 0.0
        
        # Model performance tracking
        self.prediction_history = []
        self.recent_accuracy = 0.0
        
        # Adaptive parameters
        self.base_stop_loss = 0.20
        self.base_profit_target = 0.15
        self.base_position_size = 0.04
        
    def extract_features(self, trade_data: Dict) -> TradeFeatures:
        """Extract features from trade data for ML prediction"""
        
        # Calculate signal strength (simplified)
        signal_strength = abs(trade_data.get('signal_confidence', 0.5))
        
        # Market volatility (VIX proxy)
        volatility = trade_data.get('volatility', 0.2)
        
        # Time of day factor
        trade_time = trade_data.get('timestamp', datetime.now())
        time_of_day = trade_time.hour / 24.0
        
        # Days to expiry
        expiry_date = trade_data.get('expiry_date', trade_time + timedelta(days=7))
        days_to_expiry = (expiry_date - trade_time).days
        
        # Strike distance from ATM
        underlying_price = trade_data.get('underlying_price', 500)
        strike_price = trade_data.get('strike_price', underlying_price)
        strike_distance = abs(strike_price - underlying_price) / underlying_price
        
        # Volume profile (normalized)
        volume_profile = min(trade_data.get('volume', 1000) / 10000, 1.0)
        
        # Market regime detection (simplified)
        price_changes = trade_data.get('recent_price_changes', [0])
        volatility_score = np.std(price_changes) if len(price_changes) > 1 else 0.1
        
        if volatility_score > 0.02:
            market_regime = 'volatile'
        elif abs(np.mean(price_changes)) > 0.01:
            market_regime = 'trending'
        else:
            market_regime = 'ranging'
            
        # Recent win rate
        recent_win_rate = trade_data.get('recent_win_rate', 0.5)
        
        return TradeFeatures(
            signal_strength=signal_strength,
            volatility=volatility,
            time_of_day=time_of_day,
            days_to_expiry=days_to_expiry,
            underlying_price=underlying_price,
            strike_distance=strike_distance,
            volume_profile=volume_profile,
            market_regime=market_regime,
            recent_win_rate=recent_win_rate
        )
    
    def prepare_training_data(self, historical_trades: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical trades"""
        
        features_list = []
        targets = []
        
        for trade in historical_trades:
            # Extract features
            features = self.extract_features(trade)
            
            # Convert to numerical array
            feature_array = [
                features.signal_strength,
                features.volatility,
                features.time_of_day,
                features.days_to_expiry,
                features.underlying_price / 1000,  # Normalize
                features.strike_distance,
                features.volume_profile,
                1.0 if features.market_regime == 'trending' else 0.0,
                1.0 if features.market_regime == 'volatile' else 0.0,
                features.recent_win_rate
            ]
            
            features_list.append(feature_array)
            
            # Target: 1 for profitable trade, 0 for loss
            is_profitable = trade.get('net_pnl', 0) > 0
            targets.append(1 if is_profitable else 0)
        
        return np.array(features_list), np.array(targets)
    
    def train_model(self, historical_trades: List[Dict]) -> Dict:
        """Train the ML model on historical trade data"""
        
        if len(historical_trades) < 10:
            self.logger.warning("Insufficient training data. Need at least 10 trades.")
            return {"status": "insufficient_data", "trades_needed": 10 - len(historical_trades)}
        
        # Prepare data
        X, y = self.prepare_training_data(historical_trades)
        
        if len(np.unique(y)) < 2:
            self.logger.warning("Need both winning and losing trades for training")
            return {"status": "insufficient_variety"}
        
        # Split data
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            # Use all data for training if dataset is small
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        
        # Store feature importance
        feature_names = [
            'signal_strength', 'volatility', 'time_of_day', 'days_to_expiry',
            'underlying_price', 'strike_distance', 'volume_profile',
            'trending_regime', 'volatile_regime', 'recent_win_rate'
        ]
        
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        self.is_trained = True
        
        self.logger.info(f"Model trained successfully. Accuracy: {self.model_accuracy:.3f}")
        self.logger.info(f"Top features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return {
            "status": "success",
            "accuracy": self.model_accuracy,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": self.feature_importance
        }
    
    def predict_trade_success(self, trade_data: Dict) -> TradePrediction:
        """Predict trade success probability and suggest parameters"""
        
        if not self.is_trained:
            # Return default prediction
            return TradePrediction(
                success_probability=0.5,
                confidence=0.0,
                recommended_position_size=self.base_position_size,
                suggested_stop_loss=self.base_stop_loss,
                suggested_profit_target=self.base_profit_target,
                hold_time_estimate=24.0,
                risk_adjusted_score=0.5
            )
        
        # Extract and scale features
        features = self.extract_features(trade_data)
        feature_array = np.array([[
            features.signal_strength,
            features.volatility,
            features.time_of_day,
            features.days_to_expiry,
            features.underlying_price / 1000,
            features.strike_distance,
            features.volume_profile,
            1.0 if features.market_regime == 'trending' else 0.0,
            1.0 if features.market_regime == 'volatile' else 0.0,
            features.recent_win_rate
        ]])
        
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Get prediction and probability
        success_probability = self.model.predict_proba(feature_array_scaled)[0][1]
        confidence = max(self.model.predict_proba(feature_array_scaled)[0]) - 0.5
        
        # Adaptive parameter suggestion based on probability
        position_size_multiplier = min(success_probability * 2, 1.5)
        recommended_position_size = self.base_position_size * position_size_multiplier
        
        # Adjust stop loss based on confidence
        suggested_stop_loss = self.base_stop_loss * (1 - confidence * 0.5)
        
        # Adjust profit target based on probability
        suggested_profit_target = self.base_profit_target * (1 + success_probability * 0.5)
        
        # Estimate hold time based on market regime
        hold_time_base = 18.0  # Base 18 hours
        if features.market_regime == 'volatile':
            hold_time_estimate = hold_time_base * 0.7  # Shorter holds in volatile markets
        elif features.market_regime == 'trending':
            hold_time_estimate = hold_time_base * 1.3  # Longer holds in trending markets
        else:
            hold_time_estimate = hold_time_base
        
        # Risk-adjusted score
        risk_adjusted_score = success_probability * confidence * (1 - features.volatility)
        
        return TradePrediction(
            success_probability=success_probability,
            confidence=confidence,
            recommended_position_size=recommended_position_size,
            suggested_stop_loss=suggested_stop_loss,
            suggested_profit_target=suggested_profit_target,
            hold_time_estimate=hold_time_estimate,
            risk_adjusted_score=risk_adjusted_score
        )
    
    def update_model_performance(self, prediction: TradePrediction, actual_result: bool):
        """Update model performance tracking"""
        
        self.prediction_history.append({
            'predicted_probability': prediction.success_probability,
            'actual_result': actual_result,
            'timestamp': datetime.now()
        })
        
        # Keep only recent predictions (last 50)
        if len(self.prediction_history) > 50:
            self.prediction_history = self.prediction_history[-50:]
        
        # Calculate recent accuracy
        if len(self.prediction_history) >= 10:
            recent_predictions = self.prediction_history[-10:]
            predicted_results = [p['predicted_probability'] > 0.5 for p in recent_predictions]
            actual_results = [p['actual_result'] for p in recent_predictions]
            self.recent_accuracy = accuracy_score(actual_results, predicted_results)
        
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'model_accuracy': self.model_accuracy,
            'base_params': {
                'stop_loss': self.base_stop_loss,
                'profit_target': self.base_profit_target,
                'position_size': self.base_position_size
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data['feature_importance']
            self.model_accuracy = model_data['model_accuracy']
            
            base_params = model_data.get('base_params', {})
            self.base_stop_loss = base_params.get('stop_loss', self.base_stop_loss)
            self.base_profit_target = base_params.get('profit_target', self.base_profit_target)
            self.base_position_size = base_params.get('position_size', self.base_position_size)
            
            self.logger.info(f"Model loaded from {filepath}. Accuracy: {self.model_accuracy:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_insights(self) -> Dict:
        """Get insights about model performance and feature importance"""
        
        insights = {
            'is_trained': self.is_trained,
            'model_accuracy': self.model_accuracy,
            'recent_accuracy': self.recent_accuracy,
            'prediction_count': len(self.prediction_history),
            'feature_importance': self.feature_importance,
            'top_features': sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5] if self.feature_importance else [],
            'current_parameters': {
                'base_stop_loss': self.base_stop_loss,
                'base_profit_target': self.base_profit_target,
                'base_position_size': self.base_position_size
            }
        }
        
        return insights

def create_ml_enhanced_backtest_data(historical_trades: List[Dict]) -> List[Dict]:
    """Convert backtest trade data to ML training format"""
    
    ml_trades = []
    for i, trade in enumerate(historical_trades):
        # Calculate recent win rate (last 10 trades)
        start_idx = max(0, i - 10)
        recent_trades = historical_trades[start_idx:i] if i > 0 else []
        recent_wins = sum(1 for t in recent_trades if t.get('net_pnl', 0) > 0)
        recent_win_rate = recent_wins / len(recent_trades) if recent_trades else 0.5
        
        ml_trade = {
            'net_pnl': trade.get('net_pnl', 0),
            'signal_confidence': 0.7,  # Default confidence
            'volatility': 0.2,  # Default volatility
            'timestamp': trade.get('entry_date', datetime.now()),
            'expiry_date': trade.get('entry_date', datetime.now()) + timedelta(days=7),
            'underlying_price': trade.get('underlying_price', 500),
            'strike_price': trade.get('strike', 500),
            'volume': 5000,  # Default volume
            'recent_price_changes': [0.01, -0.005, 0.008],  # Sample price changes
            'recent_win_rate': recent_win_rate
        }
        ml_trades.append(ml_trade)
    
    return ml_trades