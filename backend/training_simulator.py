"""
Real-time ML Training Simulator
Shows live training progress with mock trades, win/loss rates, and pip tracking
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import random
from collections import deque
from motor.motor_asyncio import AsyncIOMotorClient
import os

class TrainingSimulator:
    def __init__(self, db_client):
        self.db = db_client
        self.is_training = False
        self.training_progress = {}
        self.mock_trades = deque(maxlen=100)
        self.training_metrics = {
            'xgboost': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0,
                'accuracy': 0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0,
                'avg_pips_per_trade': 0,
                'win_rate': 0,
                'last_10_trades': []
            },
            'catboost': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0,
                'accuracy': 0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0,
                'avg_pips_per_trade': 0,
                'win_rate': 0,
                'last_10_trades': []
            },
            'tpot': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0,
                'accuracy': 0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0,
                'avg_pips_per_trade': 0,
                'win_rate': 0,
                'last_10_trades': []
            },
            'prophet': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0,
                'accuracy': 0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0,
                'avg_pips_per_trade': 0,
                'win_rate': 0,
                'last_10_trades': []
            }
        }
        self.training_stage = "idle"
        self.training_epoch = 0
        self.total_epochs = 200
        
    async def start_training_simulation(self, symbol: str = "XAUUSD"):
        """Start real-time training simulation"""
        if self.is_training:
            return {"error": "Training already in progress"}
        
        self.is_training = True
        self.training_stage = "initializing"
        self.training_epoch = 0
        self.symbol = symbol
        
        # Reset metrics
        for model in self.training_metrics:
            self.training_metrics[model] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0,
                'accuracy': 0,
                'current_streak': 0,
                'best_streak': 0,
                'worst_streak': 0,
                'avg_pips_per_trade': 0,
                'win_rate': 0,
                'last_10_trades': []
            }
        
        # Start training in background
        asyncio.create_task(self._run_training_simulation())
        
        return {"status": "Training started", "symbol": symbol}
    
    async def _run_training_simulation(self):
        """Run the actual training simulation"""
        try:
            # Phase 1: Data preparation
            self.training_stage = "data_preparation"
            await asyncio.sleep(2)
            
            # Phase 2: Model training with live trades
            self.training_stage = "training"
            
            for epoch in range(self.total_epochs):
                self.training_epoch = epoch + 1
                
                # Simulate training for each model
                for model_name in ['xgboost', 'catboost', 'tpot', 'prophet']:
                    await self._simulate_model_training(model_name)
                
                # Small delay between epochs
                await asyncio.sleep(0.1)
                
                if not self.is_training:  # Allow stopping
                    break
            
            # Phase 3: Model evaluation
            self.training_stage = "evaluating"
            await asyncio.sleep(2)
            
            # Phase 4: Complete
            self.training_stage = "completed"
            self.is_training = False
            
        except Exception as e:
            self.training_stage = "error"
            self.is_training = False
            print(f"Training simulation error: {e}")
    
    async def _simulate_model_training(self, model_name: str):
        """Simulate training for a specific model with mock trades"""
        # Simulate a trade decision
        action = random.choice(['BUY', 'SELL'])
        
        # Get current "price" for simulation
        base_prices = {
            'XAUUSD': 2650.0,
            'EURUSD': 1.0500,
            'EURJPY': 164.0,
            'USDJPY': 156.0,
            'NASDAQ': 20000.0
        }
        
        current_price = base_prices.get(self.symbol, 2650.0) + random.uniform(-50, 50)
        
        # Simulate trade outcome based on model learning
        model_accuracy = min(0.5 + (self.training_epoch / self.total_epochs) * 0.3, 0.8)
        is_winning_trade = random.random() < model_accuracy
        
        # Calculate pips (simplified)
        if 'USD' in self.symbol:
            pip_value = 10000 if self.symbol != 'XAUUSD' else 100
        else:
            pip_value = 100
        
        # Generate realistic pip movement
        if is_winning_trade:
            pips = random.uniform(5, 50)  # Winning trade
        else:
            pips = -random.uniform(5, 30)  # Losing trade
        
        # Update metrics
        metrics = self.training_metrics[model_name]
        metrics['trades'] += 1
        
        if is_winning_trade:
            metrics['wins'] += 1
            metrics['current_streak'] = max(0, metrics['current_streak']) + 1
        else:
            metrics['losses'] += 1
            metrics['current_streak'] = min(0, metrics['current_streak']) - 1
        
        metrics['total_pips'] += pips
        metrics['win_rate'] = (metrics['wins'] / metrics['trades']) * 100
        metrics['accuracy'] = metrics['win_rate'] / 100
        metrics['avg_pips_per_trade'] = metrics['total_pips'] / metrics['trades']
        
        # Update streaks
        if abs(metrics['current_streak']) > abs(metrics['best_streak']):
            if metrics['current_streak'] > 0:
                metrics['best_streak'] = metrics['current_streak']
        
        if abs(metrics['current_streak']) > abs(metrics['worst_streak']):
            if metrics['current_streak'] < 0:
                metrics['worst_streak'] = metrics['current_streak']
        
        # Add to last 10 trades
        trade_result = {
            'action': action,
            'pips': pips,
            'price': current_price,
            'timestamp': datetime.now().isoformat(),
            'is_win': is_winning_trade,
            'epoch': self.training_epoch
        }
        
        metrics['last_10_trades'].append(trade_result)
        if len(metrics['last_10_trades']) > 10:
            metrics['last_10_trades'].pop(0)
        
        # Add to global mock trades
        mock_trade = {
            'id': f"{model_name}_{self.training_epoch}_{metrics['trades']}",
            'model': model_name,
            'symbol': self.symbol,
            'action': action,
            'entry_price': current_price,
            'pips': pips,
            'profit': pips * 10,  # Simplified profit calculation
            'is_win': is_winning_trade,
            'timestamp': datetime.now(),
            'epoch': self.training_epoch,
            'model_accuracy': model_accuracy
        }
        
        self.mock_trades.append(mock_trade)
        
        # Store in database
        await self.db.mock_trades.insert_one(mock_trade)
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'stage': self.training_stage,
            'epoch': self.training_epoch,
            'total_epochs': self.total_epochs,
            'progress_percentage': (self.training_epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0,
            'symbol': getattr(self, 'symbol', 'XAUUSD')
        }
    
    def get_training_metrics(self) -> Dict:
        """Get current training metrics for all models"""
        return {
            'training_status': self.get_training_status(),
            'model_metrics': self.training_metrics,
            'recent_trades': list(self.mock_trades)[-20:],  # Last 20 trades
            'overall_stats': self._calculate_overall_stats()
        }
    
    def _calculate_overall_stats(self) -> Dict:
        """Calculate overall training statistics"""
        total_trades = sum(metrics['trades'] for metrics in self.training_metrics.values())
        total_wins = sum(metrics['wins'] for metrics in self.training_metrics.values())
        total_losses = sum(metrics['losses'] for metrics in self.training_metrics.values())
        total_pips = sum(metrics['total_pips'] for metrics in self.training_metrics.values())
        
        return {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'overall_win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'total_pips': total_pips,
            'avg_pips_per_trade': total_pips / total_trades if total_trades > 0 else 0,
            'best_performing_model': self._get_best_model(),
            'worst_performing_model': self._get_worst_model()
        }
    
    def _get_best_model(self) -> str:
        """Get the best performing model"""
        best_model = 'xgboost'
        best_win_rate = 0
        
        for model_name, metrics in self.training_metrics.items():
            if metrics['win_rate'] > best_win_rate:
                best_win_rate = metrics['win_rate']
                best_model = model_name
        
        return best_model
    
    def _get_worst_model(self) -> str:
        """Get the worst performing model"""
        worst_model = 'xgboost'
        worst_win_rate = 100
        
        for model_name, metrics in self.training_metrics.items():
            if metrics['win_rate'] < worst_win_rate:
                worst_win_rate = metrics['win_rate']
                worst_model = model_name
        
        return worst_model
    
    async def stop_training(self):
        """Stop the training simulation"""
        self.is_training = False
        self.training_stage = "stopped"
        return {"status": "Training stopped"}
    
    async def get_model_comparison(self) -> Dict:
        """Get detailed model comparison"""
        comparison = {}
        
        for model_name, metrics in self.training_metrics.items():
            comparison[model_name] = {
                'win_rate': metrics['win_rate'],
                'total_pips': metrics['total_pips'],
                'avg_pips_per_trade': metrics['avg_pips_per_trade'],
                'best_streak': metrics['best_streak'],
                'worst_streak': metrics['worst_streak'],
                'total_trades': metrics['trades'],
                'profitability': 'Profitable' if metrics['total_pips'] > 0 else 'Loss',
                'grade': self._calculate_grade(metrics)
            }
        
        return comparison
    
    def _calculate_grade(self, metrics: Dict) -> str:
        """Calculate performance grade for a model"""
        win_rate = metrics['win_rate']
        avg_pips = metrics['avg_pips_per_trade']
        
        if win_rate >= 70 and avg_pips >= 10:
            return 'A+'
        elif win_rate >= 60 and avg_pips >= 5:
            return 'A'
        elif win_rate >= 50 and avg_pips >= 0:
            return 'B'
        elif win_rate >= 40:
            return 'C'
        else:
            return 'D'
    
    async def get_live_trade_feed(self) -> List[Dict]:
        """Get live trade feed for real-time updates"""
        recent_trades = []
        
        # Get last 10 trades from database
        cursor = self.db.mock_trades.find().sort("timestamp", -1).limit(10)
        async for trade in cursor:
            recent_trades.append({
                'id': trade['id'],
                'model': trade['model'],
                'symbol': trade['symbol'],
                'action': trade['action'],
                'pips': trade['pips'],
                'profit': trade['profit'],
                'is_win': trade['is_win'],
                'timestamp': trade['timestamp'].isoformat(),
                'model_accuracy': trade['model_accuracy']
            })
        
        return recent_trades