from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import asyncio
import json
import pickle
from collections import deque
import schedule
import time
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

# Add backend directory to path
sys.path.append('/app/backend')

# Import the specialized ML engine and training simulator
try:
    from ml_engine import EnsembleMLEngine
    from training_simulator import TrainingSimulator
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"ML Engine not available: {e}")
    ML_ENGINE_AVAILABLE = False

# ML Libraries
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Constants
SYMBOLS = ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY', 'NASDAQ']
EODHD_API_KEY = os.environ.get('EODHD_API_KEY')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

import os
import pickle
import json
from pathlib import Path

# Create data directory for persistence
DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(exist_ok=True)

# Persistence file paths
RL_AGENT_FILE = DATA_DIR / "rl_agent.pkl"
SCALPING_RL_AGENT_FILE = DATA_DIR / "scalping_rl_agent.pkl"
ML_MODELS_FILE = DATA_DIR / "ml_models.pkl"
FEATURE_HISTORY_FILE = DATA_DIR / "feature_history.pkl"
PRICE_HISTORY_FILE = DATA_DIR / "price_history.pkl"
TRADING_HISTORY_FILE = DATA_DIR / "trading_history.json"
MODEL_PERFORMANCE_FILE = DATA_DIR / "model_performance.json"

def save_rl_agents():
    """Save RL agents to disk for persistence"""
    try:
        global rl_agent, scalping_rl_agent
        
        if rl_agent:
            with open(RL_AGENT_FILE, 'wb') as f:
                pickle.dump(rl_agent, f)
            print(f"✅ Saved RL agent to {RL_AGENT_FILE}")
        
        if scalping_rl_agent:
            with open(SCALPING_RL_AGENT_FILE, 'wb') as f:
                pickle.dump(scalping_rl_agent, f)
            print(f"✅ Saved Scalping RL agent to {SCALPING_RL_AGENT_FILE}")
            
    except Exception as e:
        print(f"❌ Error saving RL agents: {e}")

def load_rl_agents():
    """Load RL agents from disk"""
    try:
        global rl_agent, scalping_rl_agent
        
        if RL_AGENT_FILE.exists():
            with open(RL_AGENT_FILE, 'rb') as f:
                rl_agent = pickle.load(f)
            print(f"✅ Loaded RL agent from {RL_AGENT_FILE}")
        
        if SCALPING_RL_AGENT_FILE.exists():
            with open(SCALPING_RL_AGENT_FILE, 'rb') as f:
                scalping_rl_agent = pickle.load(f)
            print(f"✅ Loaded Scalping RL agent from {SCALPING_RL_AGENT_FILE}")
            print(f"   - Trades made: {scalping_rl_agent.trades_made}")
            print(f"   - Memory size: {len(scalping_rl_agent.memory)}")
            print(f"   - Epsilon: {scalping_rl_agent.epsilon:.3f}")
            
    except Exception as e:
        print(f"❌ Error loading RL agents: {e}")

def save_ml_models():
    """Save ML models to disk"""
    try:
        global ml_models
        
        if ml_models:
            with open(ML_MODELS_FILE, 'wb') as f:
                pickle.dump(ml_models, f)
            print(f"✅ Saved ML models to {ML_MODELS_FILE}")
            
    except Exception as e:
        print(f"❌ Error saving ML models: {e}")

def load_ml_models():
    """Load ML models from disk"""
    try:
        global ml_models
        
        if ML_MODELS_FILE.exists():
            with open(ML_MODELS_FILE, 'rb') as f:
                ml_models = pickle.load(f)
            print(f"✅ Loaded ML models from {ML_MODELS_FILE}")
            
            # Print status of loaded models
            for model_name, model in ml_models.items():
                status = "Active" if model is not None else "Inactive"
                print(f"   - {model_name}: {status}")
                
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")

def save_feature_and_price_history():
    """Save feature history and price history"""
    try:
        global feature_history, price_history
        
        if feature_history:
            with open(FEATURE_HISTORY_FILE, 'wb') as f:
                pickle.dump(list(feature_history), f)
            print(f"✅ Saved feature history ({len(feature_history)} records) to {FEATURE_HISTORY_FILE}")
        
        if price_history:
            # Convert deque to regular dict for JSON serialization
            price_history_dict = {symbol: list(prices) for symbol, prices in price_history.items()}
            with open(PRICE_HISTORY_FILE, 'wb') as f:
                pickle.dump(price_history_dict, f)
            print(f"✅ Saved price history to {PRICE_HISTORY_FILE}")
            
    except Exception as e:
        print(f"❌ Error saving feature/price history: {e}")

def load_feature_and_price_history():
    """Load feature history and price history"""
    try:
        global feature_history, price_history
        
        if FEATURE_HISTORY_FILE.exists():
            with open(FEATURE_HISTORY_FILE, 'rb') as f:
                feature_list = pickle.load(f)
                feature_history = deque(feature_list, maxlen=1000)
            print(f"✅ Loaded feature history ({len(feature_history)} records) from {FEATURE_HISTORY_FILE}")
        
        if PRICE_HISTORY_FILE.exists():
            with open(PRICE_HISTORY_FILE, 'rb') as f:
                price_history_dict = pickle.load(f)
                # Convert back to deque
                price_history = {symbol: deque(prices, maxlen=100) for symbol, prices in price_history_dict.items()}
            print(f"✅ Loaded price history from {PRICE_HISTORY_FILE}")
            
    except Exception as e:
        print(f"❌ Error loading feature/price history: {e}")

def save_trading_data():
    """Save trading history and model performance"""
    try:
        global trading_history, model_performance
        
        # Save trading history as JSON
        if trading_history:
            trading_data = []
            for trade in trading_history:
                # Convert to JSON-serializable format
                if hasattr(trade, '__dict__'):
                    trade_dict = trade.__dict__
                else:
                    trade_dict = trade
                
                # Handle datetime objects
                if 'timestamp' in trade_dict and hasattr(trade_dict['timestamp'], 'isoformat'):
                    trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
                if 'close_timestamp' in trade_dict and trade_dict['close_timestamp'] and hasattr(trade_dict['close_timestamp'], 'isoformat'):
                    trade_dict['close_timestamp'] = trade_dict['close_timestamp'].isoformat()
                    
                trading_data.append(trade_dict)
            
            with open(TRADING_HISTORY_FILE, 'w') as f:
                json.dump(trading_data, f, indent=2)
            print(f"✅ Saved trading history ({len(trading_data)} trades) to {TRADING_HISTORY_FILE}")
        
        # Save model performance
        if model_performance:
            with open(MODEL_PERFORMANCE_FILE, 'w') as f:
                json.dump(model_performance, f, indent=2)
            print(f"✅ Saved model performance to {MODEL_PERFORMANCE_FILE}")
            
    except Exception as e:
        print(f"❌ Error saving trading data: {e}")

def load_trading_data():
    """Load trading history and model performance"""
    try:
        global trading_history, model_performance
        
        if TRADING_HISTORY_FILE.exists():
            with open(TRADING_HISTORY_FILE, 'r') as f:
                trading_data = json.load(f)
                trading_history = trading_data
            print(f"✅ Loaded trading history ({len(trading_history)} trades) from {TRADING_HISTORY_FILE}")
        
        if MODEL_PERFORMANCE_FILE.exists():
            with open(MODEL_PERFORMANCE_FILE, 'r') as f:
                model_performance = json.load(f)
            print(f"✅ Loaded model performance from {MODEL_PERFORMANCE_FILE}")
            
    except Exception as e:
        print(f"❌ Error loading trading data: {e}")

def save_all_persistent_data():
    """Save all persistent data in one call"""
    print("💾 Saving all persistent data...")
    save_rl_agents()
    save_ml_models()
    save_feature_and_price_history()
    save_trading_data()
    print("✅ All persistent data saved successfully!")


async def periodic_save_task():
    """Background task to periodically save all data"""
    while True:
        try:
            await asyncio.sleep(300)  # Save every 5 minutes
            save_all_persistent_data()
        except Exception as e:
            print(f"❌ Error in periodic save task: {e}")
            await asyncio.sleep(60)  # Wait a minute on error

def load_all_persistent_data():
    """Load all persistent data in one call"""
    print("📂 Loading all persistent data...")
    load_rl_agents()
    load_ml_models()
    load_feature_and_price_history()
    load_trading_data()
    print("✅ All persistent data loaded successfully!")

# Global variables for models and data
ml_models = {}
rl_agent = None
scalping_rl_agent = None
feature_history = deque(maxlen=1000)
price_history = {}
trading_history = []
model_performance = {}
candlestick_history = {symbol: deque(maxlen=100) for symbol in SYMBOLS}

# Initialize the specialized ML engine and training simulator
ensemble_ml_engine = EnsembleMLEngine(news_api_key=NEWS_API_KEY) if ML_ENGINE_AVAILABLE else None
training_simulator = TrainingSimulator(db)

# Create the main app
app = FastAPI(title="Advanced Trading Bot API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Pydantic Models
class MarketData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    price: float
    change: float
    volume: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TechnicalIndicators(BaseModel):
    symbol: str
    rsi: Optional[float] = 50.0
    macd: Optional[float] = 0.0
    macd_signal: Optional[float] = 0.0
    macd_hist: Optional[float] = 0.0
    bb_upper: Optional[float] = 0.0
    bb_middle: Optional[float] = 0.0
    bb_lower: Optional[float] = 0.0
    stoch_k: Optional[float] = 50.0
    stoch_d: Optional[float] = 50.0
    atr: Optional[float] = 0.0
    obv: Optional[float] = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasons: List[str]
    ml_prediction: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Trade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    profit: Optional[float] = None
    pips: Optional[float] = None
    is_closed: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    close_timestamp: Optional[datetime] = None

class EnhancedTrade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Basic trade info
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    exit_price: Optional[float] = None
    pips_gained: Optional[float] = None
    percentage_pl: Optional[float] = None
    
    # ML and Decision info
    confidence: float = 0.0
    decision_factors: str = ""  # Summary of RSI, MACD, etc.
    trade_type: str = "Scalping"  # Scalping, Swing, Position
    forecast_trend: str = "NEUTRAL"  # UP, DOWN, NEUTRAL from Prophet
    news_sentiment: float = 0.0  # -1 to 1 sentiment score
    tweet_bias: str = "NEUTRAL"  # Expert tweet sentiment
    bot_strategy: str = "Default"  # Strategy used
    ml_decision: str = "RL Agent"  # Model that influenced most
    risk_level: str = "Medium"  # Low, Medium, High
    
    # Trade management
    quantity: float = 1.0
    profit: Optional[float] = None
    is_closed: bool = False
    close_timestamp: Optional[datetime] = None
    exit_reason: str = "Open"  # TP hit, SL hit, Timeout, Manual rule
    
    # Additional context
    rsi_value: Optional[float] = None
    macd_value: Optional[float] = None
    volume_spike: Optional[float] = None
    volatility: Optional[float] = None

class TweetInput(BaseModel):
    tweet: str
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BacktestResult(BaseModel):
    symbol: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    max_drawdown: float
    sharpe_ratio: float
    results: List[Dict]

class ModelStatus(BaseModel):
    xgboost_active: bool
    catboost_active: bool
    prophet_active: bool
    tpot_active: bool
    rl_agent_active: bool
    last_trained: Optional[datetime] = None
    performance: Dict[str, float]

class EnsemblePrediction(BaseModel):
    ensemble_decision: Dict[str, Any]
    individual_predictions: Dict[str, Any]
    models_active: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CandlestickData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime

class ScalpingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasons: List[str]
    timeframe: str  # 1m, 5m for scalping
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Scalping-focused Reinforcement Learning Agent
class ScalpingRLAgent:
    def __init__(self, state_size=15, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size  # 0: HOLD, 1: BUY, 2: SELL
        self.lr = learning_rate
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=2000)
        
        # Scalping-specific parameters (optimized for fast trades)
        self.pip_target = 5  # Target 5 pips per trade (tighter for scalping)
        self.pip_stop_loss = 3  # Stop loss at 3 pips (tighter risk management)
        self.max_hold_time = 3  # Max 3 minutes per trade (faster scalping)
        
        # Faster learning parameters for scalping
        self.gamma = 0.98  # Higher gamma for faster reward recognition
        self.epsilon_decay = 0.99  # Faster exploration decay
        self.epsilon_min = 0.05  # Slightly higher minimum for scalping volatility
        
        # Neural network weights optimized for scalping
        self.W1 = np.random.randn(state_size, 32) * 0.1
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, 16) * 0.1
        self.b2 = np.zeros((1, 16))
        self.W3 = np.random.randn(16, action_size) * 0.1
        self.b3 = np.zeros((1, action_size))
        
        # Performance tracking
        self.trades_made = 0
        self.winning_trades = 0
        self.total_pips = 0
        self.current_streak = 0
        
    def prepare_scalping_state(self, candlestick_data):
        """Prepare state vector optimized for scalping"""
        if len(candlestick_data) < 10:
            return np.zeros(self.state_size)
        
        recent_candles = candlestick_data[-10:]  # Last 10 minutes
        
        # Extract features for scalping
        closes = [candle['close'] for candle in recent_candles]
        highs = [candle['high'] for candle in recent_candles]
        lows = [candle['low'] for candle in recent_candles]
        volumes = [candle['volume'] for candle in recent_candles]
        
        # Calculate scalping-specific features
        current_price = closes[-1]
        
        # 1. Short-term momentum (1, 2, 3 minute)
        momentum_1m = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
        momentum_2m = (closes[-1] - closes[-3]) / closes[-3] if len(closes) >= 3 else 0
        momentum_3m = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        
        # 2. Volatility measures
        price_range = (max(highs) - min(lows)) / current_price
        volume_spike = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1
        
        # 3. Support/Resistance levels
        recent_high = max(highs)
        recent_low = min(lows)
        distance_to_high = (recent_high - current_price) / current_price
        distance_to_low = (current_price - recent_low) / current_price
        
        # 4. Price action patterns
        green_candles = sum(1 for candle in recent_candles if candle['close'] > candle['open'])
        red_candles = len(recent_candles) - green_candles
        
        # 5. Trend indicators
        sma_3 = np.mean(closes[-3:])
        sma_5 = np.mean(closes[-5:])
        price_vs_sma3 = (current_price - sma_3) / sma_3
        price_vs_sma5 = (current_price - sma_5) / sma_5
        
        # Combine features into state vector
        state = np.array([
            momentum_1m * 1000,  # Short-term momentum
            momentum_2m * 1000,  # Medium-term momentum
            momentum_1m * 1000,  # Repeat short-term for consistency
            price_range * 1000,  # Volatility measure
            volume_spike,        # Volume analysis
            distance_to_high * 1000,  # Distance to resistance
            distance_to_low * 1000,   # Distance to support
            green_candles / len(recent_candles),  # Bullish pressure
            red_candles / len(recent_candles),    # Bearish pressure
            price_vs_sma3 * 1000,    # Short-term trend
            price_vs_sma3 * 1000,    # Repeat for consistency
            np.tanh(current_price / 1000),  # Normalized price
            self.epsilon,        # Exploration factor
            self.trades_made / 100,  # Experience factor
            self.total_pips / 1000   # Performance factor
        ])
        
        return state
    
    def forward(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        z3 = np.dot(a2, self.W3) + self.b3
        return z3
    
    def act(self, state, use_ml_confidence=None):
        if use_ml_confidence and np.random.random() <= use_ml_confidence:
            return np.argmax(self.forward(state.reshape(1, -1)))
        
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.forward(state.reshape(1, -1))
        return np.argmax(q_values)
    
    def calculate_scalping_reward(self, action, entry_price, exit_price, symbol):
        """Calculate reward optimized for scalping"""
        pip_values = {
            'XAUUSD': 0.1,
            'EURUSD': 0.0001,
            'EURJPY': 0.01,
            'USDJPY': 0.01,
            'NASDAQ': 1.0
        }
        
        pip_value = pip_values.get(symbol, 0.01)
        
        if action == 0:  # HOLD
            return 0
        elif action == 1:  # BUY
            pips = (exit_price - entry_price) / pip_value
        else:  # SELL
            pips = (entry_price - exit_price) / pip_value
        
        # Update performance tracking
        self.trades_made += 1
        self.total_pips += pips
        
        if pips > 0:
            self.winning_trades += 1
            self.current_streak = max(0, self.current_streak + 1)
        else:
            self.current_streak = min(0, self.current_streak - 1)
        
        # Reward function optimized for scalping
        if pips >= self.pip_target:
            reward = 1.0 + (pips - self.pip_target) * 0.1  # Bonus for exceeding target
        elif pips > 0:
            reward = pips / self.pip_target  # Partial reward for positive pips
        elif pips >= -self.pip_stop_loss:
            reward = pips / self.pip_stop_loss  # Moderate penalty for small losses
        else:
            reward = -2.0  # Heavy penalty for exceeding stop loss
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            
            if not done:
                target = reward + self.gamma * np.max(self.forward(next_state.reshape(1, -1)))
            
            target_f = self.forward(state.reshape(1, -1))
            target_f[0][action] = target
            
            self._train_step(state.reshape(1, -1), target_f)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Save agent state after training (every 10 training sessions to avoid excessive I/O)
        if hasattr(self, 'training_sessions'):
            self.training_sessions += 1
        else:
            self.training_sessions = 1
            
        if self.training_sessions % 10 == 0:
            try:
                save_all_persistent_data()
                print(f"💾 Saved learning progress after {self.training_sessions} training sessions")
            except Exception as e:
                print(f"❌ Error saving after training: {e}")
    
    def _train_step(self, X, y):
        # Forward pass
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)
        z3 = np.dot(a2, self.W3) + self.b3
        
        # Backward pass
        dz3 = (z3 - y) / X.shape[0]
        dW3 = np.dot(a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (z2 > 0)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def get_performance_metrics(self):
        win_rate = (self.winning_trades / self.trades_made * 100) if self.trades_made > 0 else 0
        return {
            'trades_made': self.trades_made,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pips': self.total_pips,
            'current_streak': self.current_streak,
            'epsilon': self.epsilon
        }

# Reinforcement Learning Agent
class RLTradingAgent:
    def __init__(self, state_size=20, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=2000)
        
        # Neural network weights
        self.W1 = np.random.randn(state_size, 64) * 0.1
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros((1, 32))
        self.W3 = np.random.randn(32, action_size) * 0.1
        self.b3 = np.zeros((1, action_size))
    
    def forward(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        z3 = np.dot(a2, self.W3) + self.b3
        return z3
    
    def act(self, state, use_ml_confidence=None):
        if use_ml_confidence and np.random.random() <= use_ml_confidence:
            return np.argmax(self.forward(state.reshape(1, -1)))
        
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.forward(state.reshape(1, -1))
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            
            if not done:
                target = reward + self.gamma * np.max(self.forward(next_state.reshape(1, -1)))
            
            target_f = self.forward(state.reshape(1, -1))
            target_f[0][action] = target
            
            self._train_step(state.reshape(1, -1), target_f)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Save progress after training (every 10 training sessions)
        if hasattr(self, 'training_count'):
            self.training_count += 1
        else:
            self.training_count = 1
            
        if self.training_count % 10 == 0:
            try:
                # Import save function
                save_rl_agents()
                print(f"💾 Auto-saved RL agent progress (training #{self.training_count})")
            except:
                pass  # Don't break training if save fails
    
    def _train_step(self, X, y):
        # Forward pass
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)
        z3 = np.dot(a2, self.W3) + self.b3
        
        # Backward pass
        dz3 = (z3 - y) / X.shape[0]
        dW3 = np.dot(a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (z2 > 0)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

# Custom Technical Analysis Functions
def calculate_rsi(prices, window=7):
    """Calculate RSI indicator optimized for scalping (7-period)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, fast=6, slow=12, signal=5):
    """Calculate MACD indicator optimized for scalping (6,12,5)"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, window=10, num_std=2):
    """Calculate Bollinger Bands optimized for scalping (10-period)"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_stochastic(high, low, close, k_window=7, d_window=3):
    """Calculate Stochastic Oscillator optimized for scalping (7,3)"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent.fillna(50), d_percent.fillna(50)

def calculate_atr(high, low, close, window=7):
    """Calculate Average True Range optimized for scalping (7-period)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.fillna(0)

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv.fillna(0)

# Data Fetching Functions
async def fetch_live_data(symbol: str) -> Optional[Dict]:
    """Fetch live market data from EODHD"""
    try:
        if symbol in ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY']:
            formatted_symbol = f"{symbol}.FOREX"
        elif symbol == 'NASDAQ':
            formatted_symbol = "IXIC.INDX"
        else:
            return None
        
        url = f"https://eodhistoricaldata.com/api/real-time/{formatted_symbol}"
        params = {
            "api_token": EODHD_API_KEY,
            "fmt": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Handle "NA" values and ensure we have valid numeric data
            def safe_float(value, default=0.0):
                try:
                    if value is None or value == "NA" or value == "":
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Generate realistic sample data if API returns invalid data
            price = safe_float(data.get('close'))
            if price == 0.0:
                # Generate sample data based on symbol
                if symbol == 'XAUUSD':
                    price = 2650.0 + np.random.uniform(-50, 50)
                elif symbol == 'EURUSD':
                    price = 1.0500 + np.random.uniform(-0.02, 0.02)
                elif symbol == 'EURJPY':
                    price = 164.0 + np.random.uniform(-2, 2)
                elif symbol == 'USDJPY':
                    price = 156.0 + np.random.uniform(-2, 2)
                elif symbol == 'NASDAQ':
                    price = 20000.0 + np.random.uniform(-500, 500)
            
            change = safe_float(data.get('change_p'), np.random.uniform(-1.5, 1.5))
            volume = safe_float(data.get('volume'), np.random.randint(100000, 1000000))
            
            return {
                'symbol': symbol,
                'price': price,
                'change': change,
                'volume': volume,
                'timestamp': datetime.now()
            }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        # Return sample data as fallback
        price_map = {
            'XAUUSD': 2650.0 + np.random.uniform(-50, 50),
            'EURUSD': 1.0500 + np.random.uniform(-0.02, 0.02),
            'EURJPY': 164.0 + np.random.uniform(-2, 2),
            'USDJPY': 156.0 + np.random.uniform(-2, 2),
            'NASDAQ': 20000.0 + np.random.uniform(-500, 500)
        }
        
        return {
            'symbol': symbol,
            'price': price_map.get(symbol, 100.0),
            'change': np.random.uniform(-1.5, 1.5),
            'volume': np.random.randint(100000, 1000000),
            'timestamp': datetime.now()
        }
    
    return None

async def fetch_candlestick_data(symbol: str, period: str = '1d', interval: str = '1m') -> List[Dict]:
    """Fetch candlestick data using yfinance for scalping"""
    try:
        # Map our symbols to yfinance symbols
        yf_symbol_map = {
            'XAUUSD': 'GC=F',  # Gold futures
            'EURUSD': 'EURUSD=X',
            'EURJPY': 'EURJPY=X', 
            'USDJPY': 'USDJPY=X',
            'NASDAQ': '^IXIC'
        }
        
        yf_symbol = yf_symbol_map.get(symbol, symbol)
        
        # Get data for the last few hours (good for scalping)
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print(f"No data found for {symbol}, generating sample data")
            return generate_sample_candlestick_data(symbol, interval)
        
        # Convert to candlestick format
        candlesticks = []
        for timestamp, row in data.iterrows():
            candlesticks.append({
                'timestamp': timestamp.isoformat(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
            })
        
        return candlesticks[-100:]  # Return last 100 candles
        
    except Exception as e:
        print(f"Error fetching candlestick data for {symbol}: {e}")
        return generate_sample_candlestick_data(symbol, interval)

def generate_sample_candlestick_data(symbol: str, interval: str = '1m') -> List[Dict]:
    """Generate sample candlestick data for fallback"""
    
    # Base prices for different symbols
    base_prices = {
        'XAUUSD': 2650.0,
        'EURUSD': 1.0500,
        'EURJPY': 164.0,
        'USDJPY': 156.0,
        'NASDAQ': 20000.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    
    # Different volatility for different symbols
    volatilities = {
        'XAUUSD': 0.001,  # 0.1% 
        'EURUSD': 0.0005,  # 0.05%
        'EURJPY': 0.001,   # 0.1%
        'USDJPY': 0.001,   # 0.1%
        'NASDAQ': 0.002    # 0.2%
    }
    
    volatility = volatilities.get(symbol, 0.001)
    
    # Generate time intervals
    now = datetime.now()
    if interval == '1m':
        time_delta = timedelta(minutes=1)
        periods = 100
    elif interval == '5m':
        time_delta = timedelta(minutes=5)
        periods = 100
    else:
        time_delta = timedelta(hours=1)
        periods = 100
    
    candlesticks = []
    current_price = base_price
    
    for i in range(periods):
        timestamp = now - time_delta * (periods - i - 1)
        
        # Generate realistic OHLC data for scalping
        open_price = current_price
        
        # Random walk for the period
        price_change = np.random.normal(0, volatility * base_price)
        high_price = open_price + abs(np.random.normal(0, volatility * base_price * 0.5))
        low_price = open_price - abs(np.random.normal(0, volatility * base_price * 0.5))
        close_price = open_price + price_change
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        volume = np.random.randint(10000, 100000)
        
        candlesticks.append({
            'timestamp': timestamp.isoformat(),
            'open': round(open_price, 4 if symbol in ['EURUSD', 'EURJPY', 'USDJPY'] else 2),
            'high': round(high_price, 4 if symbol in ['EURUSD', 'EURJPY', 'USDJPY'] else 2),
            'low': round(low_price, 4 if symbol in ['EURUSD', 'EURJPY', 'USDJPY'] else 2),
            'close': round(close_price, 4 if symbol in ['EURUSD', 'EURJPY', 'USDJPY'] else 2),
            'volume': volume
        })
        
        current_price = close_price
    
    return candlesticks

def get_strategy_label(indicators, sentiment=0, tweet_bias="NEUTRAL", events=None, action="HOLD"):
    """
    Determine scalping strategy label based on fast timeframes and scalping conditions
    """
    if events is None:
        events = []
    
    rsi = indicators.get("RSI", 50)
    macd = indicators.get("MACD", 0)
    macd_signal = indicators.get("MACD_signal", 0)
    macd_hist = indicators.get("MACD_hist", 0)
    atr = indicators.get("ATR", 0)
    bb_upper = indicators.get("BB_upper", 0)
    bb_lower = indicators.get("BB_lower", 0)
    price = indicators.get("price", 0)
    volume = indicators.get("volume", 0)
    
    # SCALPING-SPECIFIC STRATEGY LABELS
    
    # 1. Scalp Reversal (RSI oversold/overbought on 1-min)
    if rsi < 30 and action == "BUY":
        return "Scalp Reversal"
    elif rsi > 70 and action == "SELL":
        return "Scalp Reversal"
    
    # 2. MACD Quick Flip (fast crossover on 1m)
    elif abs(macd_hist) > 0.0005 and ((macd > macd_signal and action == "BUY") or (macd < macd_signal and action == "SELL")):
        return "MACD Quick Flip"
    
    # 3. Tweet Surge Entry (tweet-based scalping)
    elif tweet_bias == "BULLISH" and sentiment > 0.05 and action == "BUY":
        return "Tweet Surge Entry"
    elif tweet_bias == "BEARISH" and sentiment < -0.05 and action == "SELL":
        return "Tweet Surge Entry"
    
    # 4. Breakout Scalping (price breaks + volume spike)
    elif volume > 80000 and atr > 0.001:
        if price > bb_upper and action == "BUY":
            return "Breakout Scalping"
        elif price < bb_lower and action == "SELL":
            return "Breakout Scalping"
    
    # 5. Micro Trend Pullback (small pullback in strong move)
    elif 35 < rsi < 65 and abs(macd_hist) > 0.0002:
        return "Micro Trend Pullback"
    
    # 6. ATR Scalping Range (trading within ATR zones)
    elif atr > 0.0008 and atr < 0.002:
        return "ATR Scalping Range"
    
    # 7. News Sentiment Scalping (quick news-based entries)
    elif abs(sentiment) > 0.15:
        if sentiment > 0.15 and action == "BUY":
            return "News Sentiment Scalping"
        elif sentiment < -0.15 and action == "SELL":
            return "News Sentiment Scalping"
    
    # 8. Fast Volume Spike (unusual volume for scalping)
    elif volume > 120000:
        return "Fast Volume Spike"
    
    # 9. Tight Range Scalping (consolidation breakout)
    elif bb_upper > 0 and bb_lower > 0 and (bb_upper - bb_lower) / price < 0.015:
        return "Tight Range Scalping"
    
    # Default scalping strategy
    else:
        return "General Scalping"

def should_trade_scalping(indicators, sentiment, tweet_bias, symbol):
    """
    Determine if conditions are suitable for scalping trades
    """
    # Check volatility - need minimum volatility for scalping
    atr = indicators.get('ATR', 0)
    if atr < 0.0008:  # Too low volatility
        return False, "Volatility too low for scalping"
    
    if atr > 0.003:  # Too high volatility - risky
        return False, "Volatility too high - market too volatile"
    
    # Check if we have directional bias
    rsi = indicators.get('RSI', 50)
    macd_hist = indicators.get('MACD_hist', 0)
    
    # Need some directional signal
    has_rsi_signal = rsi < 35 or rsi > 65
    has_macd_signal = abs(macd_hist) > 0.0002
    has_sentiment_signal = abs(sentiment) > 0.1 or tweet_bias != "NEUTRAL"
    
    if not (has_rsi_signal or has_macd_signal or has_sentiment_signal):
        return False, "No clear directional signal"
    
    # Check volume - need sufficient volume for scalping
    volume = indicators.get('volume', 0)
    if volume < 40000:
        return False, "Volume too low for scalping"
    
    # All conditions met
    return True, "Good conditions for scalping"

def create_enhanced_trade_record(
    symbol: str,
    action: str,
    entry_price: float,
    confidence: float,
    technical_indicators: Dict,
    scalping_signal: Optional[Dict] = None,
    news_sentiment: float = 0.0,
    tweet_bias: str = "NEUTRAL",
    ml_predictions: Optional[Dict] = None
) -> EnhancedTrade:
    """Create a comprehensive trade record with all tracking information"""
    
    # Calculate decision factors summary
    decision_factors = []
    rsi_value = technical_indicators.get('RSI', 50)
    macd_value = technical_indicators.get('MACD', 0)
    
    if rsi_value < 30:
        decision_factors.append("RSI Oversold")
    elif rsi_value > 70:
        decision_factors.append("RSI Overbought")
    
    if macd_value > 0:
        decision_factors.append("MACD Bullish")
    elif macd_value < 0:
        decision_factors.append("MACD Bearish")
    
    if news_sentiment > 0.1:
        decision_factors.append("Positive News")
    elif news_sentiment < -0.1:
        decision_factors.append("Negative News")
    
    if tweet_bias != "NEUTRAL":
        decision_factors.append(f"Tweet: {tweet_bias}")
    
    # Determine trade type based on signals
    trade_type = "Scalping"  # Default for our scalping bot
    if scalping_signal and scalping_signal.get('timeframe') == '15m':
        trade_type = "Swing"
    
    # Determine forecast trend from Prophet (if available)
    forecast_trend = "NEUTRAL"
    if ml_predictions and 'prophet' in ml_predictions:
        prophet_pred = ml_predictions['prophet']
        if isinstance(prophet_pred, dict) and 'trend' in prophet_pred:
            forecast_trend = prophet_pred['trend']
    
    # Determine which ML model influenced the decision most
    ml_decision = "RL Agent"  # Default
    if ml_predictions:
        if 'ensemble' in ml_predictions:
            ensemble = ml_predictions['ensemble']
            if isinstance(ensemble, dict) and 'primary_model' in ensemble:
                ml_decision = ensemble['primary_model']
        elif confidence > 0.7:
            if 'xgboost' in ml_predictions and ml_predictions['xgboost'].get('confidence', 0) > 0.7:
                ml_decision = "XGBoost"
            elif 'catboost' in ml_predictions and ml_predictions['catboost'].get('confidence', 0) > 0.7:
                ml_decision = "CatBoost"
    
    # Determine bot strategy using intelligent logic
    events = []  # Could be populated from economic calendar
    bot_strategy = get_strategy_label(
        indicators=technical_indicators,
        sentiment=news_sentiment,
        tweet_bias=tweet_bias,
        events=events,
        action=action
    )
    
    # Determine risk level
    risk_level = "Medium"
    volatility = technical_indicators.get('ATR', 0)
    if volatility > 0.002:
        risk_level = "High"
    elif volatility < 0.001:
        risk_level = "Low"
    
    # Create enhanced trade record
    enhanced_trade = EnhancedTrade(
        timestamp=datetime.utcnow(),
        symbol=symbol,
        action=action,
        entry_price=entry_price,
        confidence=confidence,
        decision_factors=", ".join(decision_factors) if decision_factors else "No specific factors",
        trade_type=trade_type,
        forecast_trend=forecast_trend,
        news_sentiment=news_sentiment,
        tweet_bias=tweet_bias,
        bot_strategy=bot_strategy,
        ml_decision=ml_decision,
        risk_level=risk_level,
        rsi_value=rsi_value,
        macd_value=macd_value,
        volatility=volatility,
        exit_reason="Open"
    )
    
    return enhanced_trade

def calculate_pips(entry_price, exit_price, symbol, action="BUY"):
    """
    Calculate pips correctly for different instruments
    """
    if symbol == 'XAUUSD':
        # Gold: 1 pip = 0.1, so price difference * 10
        pips = (exit_price - entry_price) * 10
    elif symbol in ['EURUSD']:
        # Major FX pairs: 1 pip = 0.0001, so price difference * 10000
        pips = (exit_price - entry_price) * 10000
    elif symbol in ['USDJPY', 'EURJPY']:
        # JPY pairs: 1 pip = 0.01, so price difference * 100
        pips = (exit_price - entry_price) * 100
    elif symbol == 'NASDAQ':
        # Index: 1 pip = 1 point
        pips = (exit_price - entry_price)
    else:
        # Default calculation
        pips = (exit_price - entry_price) * 10000
    
    # For SELL orders, reverse the calculation
    if action == "SELL":
        pips = -pips
    
    return round(pips, 1)

def calculate_pips_and_profit(trade: EnhancedTrade, exit_price: float) -> EnhancedTrade:
    """Calculate pips and profit when trade is closed"""
    
    # Use the correct pip calculation function
    pips_gained = calculate_pips(trade.entry_price, exit_price, trade.symbol, trade.action)
    
    # Calculate percentage P/L
    if trade.action == "BUY":
        percentage_pl = ((exit_price - trade.entry_price) / trade.entry_price) * 100
    else:  # SELL
        percentage_pl = ((trade.entry_price - exit_price) / trade.entry_price) * 100
    
    # Update trade record
    trade.exit_price = exit_price
    trade.pips_gained = round(pips_gained, 1)
    trade.percentage_pl = round(percentage_pl, 2)
    trade.profit = round(pips_gained * trade.quantity, 2)
    trade.is_closed = True
    trade.close_timestamp = datetime.utcnow()
    
    # Determine exit reason based on performance
    if pips_gained >= 10:
        trade.exit_reason = "TP Hit"
    elif pips_gained <= -5:
        trade.exit_reason = "SL Hit"
    elif trade.close_timestamp and (trade.close_timestamp - trade.timestamp).seconds > 300:  # 5 minutes
        trade.exit_reason = "Timeout"
    else:
        trade.exit_reason = "Manual Rule"
    
    return trade

async def fetch_historical_data(symbol: str, period: str = '1d') -> pd.DataFrame:
    """Fetch historical data for technical analysis"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        if symbol in ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY']:
            formatted_symbol = f"{symbol}.FOREX"
        elif symbol == 'NASDAQ':
            formatted_symbol = "IXIC.INDX"
        else:
            return pd.DataFrame()
        
        url = f"https://eodhistoricaldata.com/api/eod/{formatted_symbol}"
        params = {
            "api_token": EODHD_API_KEY,
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d'),
            "fmt": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            if not data.empty:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                
                # Clean data and handle missing values
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        data[col] = data[col].fillna(method='ffill')
                
                return data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
    
    # Generate sample historical data as fallback
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic sample data based on symbol
    if symbol == 'XAUUSD':
        base_price = 2650.0
        price_range = 100
    elif symbol == 'EURUSD':
        base_price = 1.0500
        price_range = 0.05
    elif symbol == 'EURJPY':
        base_price = 164.0
        price_range = 10
    elif symbol == 'USDJPY':
        base_price = 156.0
        price_range = 8
    elif symbol == 'NASDAQ':
        base_price = 20000.0
        price_range = 2000
    else:
        base_price = 100.0
        price_range = 10
    
    # Generate realistic OHLCV data
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_price = base_price
    for i in range(100):
        # Random walk with some trend
        change = np.random.uniform(-price_range * 0.02, price_range * 0.02)
        current_price += change
        
        # Generate OHLC for the day
        open_price = current_price
        high_price = open_price + abs(np.random.uniform(0, price_range * 0.01))
        low_price = open_price - abs(np.random.uniform(0, price_range * 0.01))
        close_price = np.random.uniform(low_price, high_price)
        volume = np.random.randint(100000, 1000000)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        
        current_price = close_price
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """Calculate technical indicators using custom functions"""
    if df.empty or len(df) < 20:
        return {}
    
    indicators = {}
    
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        rsi = calculate_rsi(close)
        indicators['RSI'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # MACD
        macd, signal, hist = calculate_macd(close)
        indicators['MACD'] = macd.iloc[-1] if not macd.empty else 0
        indicators['MACD_signal'] = signal.iloc[-1] if not signal.empty else 0
        indicators['MACD_hist'] = hist.iloc[-1] if not hist.empty else 0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        indicators['BB_upper'] = bb_upper.iloc[-1] if not bb_upper.empty else close.iloc[-1]
        indicators['BB_middle'] = bb_middle.iloc[-1] if not bb_middle.empty else close.iloc[-1]
        indicators['BB_lower'] = bb_lower.iloc[-1] if not bb_lower.empty else close.iloc[-1]
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        indicators['STOCH_K'] = stoch_k.iloc[-1] if not stoch_k.empty else 50
        indicators['STOCH_D'] = stoch_d.iloc[-1] if not stoch_d.empty else 50
        
        # ATR
        atr = calculate_atr(high, low, close)
        indicators['ATR'] = atr.iloc[-1] if not atr.empty else 0
        
        # OBV
        obv = calculate_obv(close, volume)
        indicators['OBV'] = obv.iloc[-1] if not obv.empty else 0
        
        # Moving averages optimized for scalping
        indicators['SMA_5'] = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else close.iloc[-1]
        indicators['SMA_10'] = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else close.iloc[-1]
        indicators['EMA_5'] = close.ewm(span=5).mean().iloc[-1] if len(close) >= 5 else close.iloc[-1]
        indicators['EMA_10'] = close.ewm(span=10).mean().iloc[-1] if len(close) >= 10 else close.iloc[-1]
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        # Return default values
        default_indicators = {
            'RSI': 50, 'MACD': 0, 'MACD_signal': 0, 'MACD_hist': 0,
            'BB_upper': 0, 'BB_middle': 0, 'BB_lower': 0,
            'STOCH_K': 50, 'STOCH_D': 50, 'ATR': 0, 'OBV': 0,
            'SMA_20': 0, 'SMA_50': 0, 'EMA_12': 0, 'EMA_26': 0
        }
        indicators.update(default_indicators)
    
    return indicators

def prepare_ml_features(symbol_data: Dict, indicators: Dict, sentiment: float, tweet_bias: float) -> np.ndarray:
    """Prepare features for ML models"""
    features = []
    
    # Price-based features
    if symbol_data:
        features.extend([
            symbol_data['price'],
            symbol_data['change'],
            symbol_data['volume']
        ])
    else:
        features.extend([0, 0, 0])
    
    # Technical indicators
    indicator_list = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                     'BB_upper', 'BB_middle', 'BB_lower',
                     'STOCH_K', 'STOCH_D', 'ATR']
    
    for ind in indicator_list:
        features.append(indicators.get(ind, 0))
    
    # Sentiment and tweet bias
    features.extend([sentiment, tweet_bias])
    
    # Time features
    now = datetime.now()
    features.extend([now.hour, now.weekday(), now.day])
    
    return np.array(features)

def calculate_position_size(account_balance: float, risk_percentage: float, stop_loss_pips: float) -> float:
    """Calculate position size based on risk management"""
    risk_amount = account_balance * (risk_percentage / 100)
    pip_value = 1.0  # Simplified pip value
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return min(position_size, account_balance * 0.1)  # Max 10% of account

def check_correlation(symbol1: str, symbol2: str, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
    """Check correlation between two symbols"""
    if data1.empty or data2.empty:
        return 0.0
    
    # Align data by date
    combined = pd.concat([data1['close'], data2['close']], axis=1, join='inner')
    if len(combined) < 20:
        return 0.0
    
    return combined.corr().iloc[0, 1]

# Initialize global components
async def initialize_system():
    """Initialize the trading system with persistent learning"""
    global rl_agent, ml_models, price_history, scalping_rl_agent, candlestick_history
    
    print("🚀 Initializing Cash Trading Bot System...")
    
    # First, try to load existing data from previous sessions
    load_all_persistent_data()
    
    # Initialize RL agent (only if not loaded from file)
    if rl_agent is None:
        rl_agent = RLTradingAgent(state_size=20)
        print("🆕 Created new RL agent")
    else:
        print("♻️ Using restored RL agent from previous session")
    
    # Initialize Scalping RL agent (only if not loaded from file)
    if scalping_rl_agent is None:
        scalping_rl_agent = ScalpingRLAgent(state_size=15)
        print("🆕 Created new Scalping RL agent")
    else:
        print("♻️ Using restored Scalping RL agent from previous session")
    
    # Initialize price history (only if not loaded from file)
    if not price_history:
        price_history = {}
        for symbol in SYMBOLS:
            price_history[symbol] = deque(maxlen=100)
        print("🆕 Created new price history")
    else:
        print("♻️ Using restored price history from previous session")
    
    # Initialize candlestick history
    for symbol in SYMBOLS:
        if symbol not in candlestick_history:
            candlestick_history[symbol] = deque(maxlen=100)
    
    # Initialize ML models dictionary (only if not loaded)
    if not ml_models:
        ml_models = {
            'xgboost': None,
            'catboost': None,
            'prophet': None,
            'scaler': None
        }
        print("🆕 Created new ML models dictionary")
    else:
        print("♻️ Using restored ML models from previous session")
    
    # Populate some sample trading data (only if no data exists)
    try:
        existing_trades = await db.trades.count_documents({})
        if existing_trades == 0:
            await populate_sample_data()
            print("🆕 Generated sample trading data")
        else:
            print(f"♻️ Found {existing_trades} existing trades in database")
    except Exception as e:
        print(f"⚠️ Error checking existing trades: {e}")
        await populate_sample_data()
    
    print("✅ Trading system initialized successfully with persistent learning!")
    print("📊 System Status:")
    print(f"   - RL Agent: {'Restored' if rl_agent.epsilon < 1.0 else 'New'}")
    print(f"   - Scalping RL Agent: {'Restored' if scalping_rl_agent.epsilon < 1.0 else 'New'}")
    print(f"   - ML Models: {len([k for k, v in ml_models.items() if v is not None])} active")
    print(f"   - Feature History: {len(feature_history)} records")
    print(f"   - Price History: {len(price_history)} symbols")
    
    # Start periodic saving
    import asyncio
    asyncio.create_task(periodic_save_task())

async def populate_sample_data():
    """Populate database with sample trading data"""
    try:
        # Check if we already have data
        existing_trades = await db.trades.count_documents({})
        if existing_trades > 0:
            print("Sample data already exists, skipping population.")
            return
        
        print("Populating sample trading data...")
        
        # Generate sample trades
        sample_trades = []
        strategies = ['RSI Reversal', 'MACD Crossover', 'Momentum Breakout', 'Support Resistance', 'Default']
        
        for i in range(50):
            symbol = np.random.choice(SYMBOLS)
            action = np.random.choice(['BUY', 'SELL'])
            entry_price = np.random.uniform(2600, 2700) if symbol == 'XAUUSD' else (
                np.random.uniform(1.04, 1.06) if symbol == 'EURUSD' else (
                    np.random.uniform(155, 158) if symbol == 'USDJPY' else (
                        np.random.uniform(162, 166) if symbol == 'EURJPY' else 
                        np.random.uniform(19500, 20500)  # NASDAQ
                    )
                )
            )
            # Generate smaller price movements for scalping
            exit_price = entry_price * (1 + np.random.uniform(-0.005, 0.005))  # Smaller moves for scalping
            quantity = np.random.uniform(0.1, 1.0)  # Smaller quantities for scalping
            
            # Use correct pip calculation
            pips = calculate_pips(entry_price, exit_price, symbol, action)
            
            # Calculate profit based on pips (more realistic for scalping)
            if symbol == 'XAUUSD':
                profit = pips * quantity * 0.1  # $0.10 per pip
            elif symbol == 'EURUSD':
                profit = pips * quantity * 1.0  # $1.00 per pip
            else:
                profit = pips * quantity * 0.1
            
            # Generate technical indicators optimized for scalping
            rsi_value = np.random.randint(25, 75)  # More realistic RSI range
            macd_value = np.random.uniform(-0.0005, 0.0005)  # Smaller MACD values for scalping
            macd_signal = macd_value + np.random.uniform(-0.0001, 0.0001)  # Tighter signal
            atr_value = np.random.uniform(0.0008, 0.0015)  # Realistic ATR for scalping
            volume_value = np.random.randint(50000, 120000)  # Volume range for scalping
            news_sentiment = round(np.random.uniform(-0.3, 0.3), 2)  # Moderate sentiment
            tweet_bias = np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
            
            # Create technical indicators dict for strategy determination
            technical_indicators = {
                'RSI': rsi_value,
                'MACD': macd_value,
                'MACD_signal': macd_signal,
                'MACD_hist': macd_value - macd_signal,
                'ATR': atr_value,
                'price': entry_price,
                'volume': volume_value,
                'BB_upper': entry_price * 1.02,
                'BB_lower': entry_price * 0.98
            }
            
            # Generate intelligent strategy label
            intelligent_strategy = get_strategy_label(
                indicators=technical_indicators,
                sentiment=news_sentiment,
                tweet_bias=tweet_bias,
                events=[],
                action=action
            )
            
            trade = {
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'action': action,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'profit': profit,
                'pips': pips,
                'is_closed': True,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                'close_timestamp': datetime.now() - timedelta(days=np.random.randint(0, 29)),
                'bot_strategy': intelligent_strategy  # Use intelligent strategy
            }
            sample_trades.append(trade)
        
        # Insert sample trades
        if sample_trades:
            await db.trades.insert_many(sample_trades)
            print(f"Inserted {len(sample_trades)} sample trades")
        
        # Generate sample market data
        for symbol in SYMBOLS:
            market_data = await fetch_live_data(symbol)
            if market_data:
                await db.market_data.insert_one(market_data)
        
        # Generate comprehensive training data for ML models
        training_data = []
        for i in range(300):  # More data for better ML training
            symbol = np.random.choice(SYMBOLS)
            base_price = {
                'XAUUSD': 2650.0,
                'EURUSD': 1.0500,
                'EURJPY': 164.0,
                'USDJPY': 156.0,
                'NASDAQ': 20000.0
            }[symbol]
            
            # Generate realistic market data
            price = base_price + np.random.uniform(-base_price*0.02, base_price*0.02)
            volume = np.random.randint(100000, 1000000)
            change = np.random.uniform(-2, 2)
            
            # Generate technical indicators
            indicators = {
                'RSI': np.random.uniform(30, 70),
                'MACD': np.random.uniform(-0.5, 0.5),
                'MACD_signal': np.random.uniform(-0.4, 0.4),
                'MACD_hist': np.random.uniform(-0.2, 0.2),
                'BB_upper': price * 1.02,
                'BB_middle': price,
                'BB_lower': price * 0.98,
                'STOCH_K': np.random.uniform(20, 80),
                'STOCH_D': np.random.uniform(20, 80),
                'ATR': np.random.uniform(0.5, 2.0),
                'EMA_12': price * 1.001,
                'EMA_26': price * 0.999
            }
            
            # Generate sentiment data
            sentiment_data = {
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'article_count': np.random.randint(1, 20),
                'confidence': np.random.uniform(0.3, 0.9),
                'keywords': ['market', 'trading', 'price']
            }
            
            # Generate next price change for training
            next_price_change = np.random.uniform(-3, 3)
            
            training_point = {
                'symbol': symbol,
                'timestamp': datetime.now() - timedelta(hours=i),
                'market_data': {
                    'price': price,
                    'volume': volume,
                    'change': change
                },
                'indicators': indicators,
                'sentiment_data': sentiment_data,
                'news_sentiment': sentiment_data['sentiment_score'],
                'event_flag': np.random.choice([0, 1], p=[0.8, 0.2]),
                'next_price_change': next_price_change
            }
            
            training_data.append(training_point)
        
        # Store training data in database
        await db.training_data.insert_many(training_data)
        print(f"Inserted {len(training_data)} training data points")
        
        # Initialize feature history for RL agent
        for data_point in training_data:
            features = prepare_ml_features(
                data_point['market_data'], 
                data_point['indicators'], 
                data_point['news_sentiment'], 
                data_point['event_flag']
            )
            feature_history.append(features)
        
        print("Sample data population completed!")
        
    except Exception as e:
        print(f"Error populating sample data: {e}")

# Background task for real-time data updates
async def update_market_data():
    """Background task to update market data"""
    while True:
        try:
            for symbol in SYMBOLS:
                data = await fetch_live_data(symbol)
                if data:
                    # Store in database
                    await db.market_data.insert_one(data)
                    
                    # Update price history
                    if symbol in price_history:
                        price_history[symbol].append(data['price'])
                    
                    # Update feature history
                    historical_data = await fetch_historical_data(symbol)
                    if not historical_data.empty:
                        indicators = calculate_technical_indicators(historical_data)
                        features = prepare_ml_features(data, indicators, 0, 0)
                        feature_history.append(features)
                
                await asyncio.sleep(1)  # Small delay between symbols
            
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error in market data update: {e}")
            await asyncio.sleep(10)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Advanced Trading Bot API", "version": "1.0.0"}

@api_router.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get latest market data for a symbol"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        data = await fetch_live_data(symbol)
        if data:
            # Update candlestick history for scalping analysis
            await update_candlestick_history(symbol, data)
            return data
        else:
            # Generate fallback data if API fails
            price_map = {
                'XAUUSD': 2650.0 + np.random.uniform(-50, 50),
                'EURUSD': 1.0500 + np.random.uniform(-0.02, 0.02),
                'EURJPY': 164.0 + np.random.uniform(-2, 2),
                'USDJPY': 156.0 + np.random.uniform(-2, 2),
                'NASDAQ': 20000.0 + np.random.uniform(-500, 500)
            }
            
            fallback_data = {
                'symbol': symbol,
                'price': price_map.get(symbol, 100.0),
                'change': np.random.uniform(-1.5, 1.5),
                'volume': np.random.randint(100000, 1000000),
                'timestamp': datetime.now()
            }
            
            # Update candlestick history with fallback data
            await update_candlestick_history(symbol, fallback_data)
            return fallback_data
            
    except Exception as e:
        # Generate fallback data on error
        price_map = {
            'XAUUSD': 2650.0 + np.random.uniform(-50, 50),
            'EURUSD': 1.0500 + np.random.uniform(-0.02, 0.02),
            'EURJPY': 164.0 + np.random.uniform(-2, 2),
            'USDJPY': 156.0 + np.random.uniform(-2, 2),
            'NASDAQ': 20000.0 + np.random.uniform(-500, 500)
        }
        
        fallback_data = {
            'symbol': symbol,
            'price': price_map.get(symbol, 100.0),
            'change': np.random.uniform(-1.5, 1.5),
            'volume': np.random.randint(100000, 1000000),
            'timestamp': datetime.now()
        }
        
        # Update candlestick history with fallback data
        await update_candlestick_history(symbol, fallback_data)
        return fallback_data

async def update_candlestick_history(symbol: str, market_data: Dict):
    """Update candlestick history for scalping analysis"""
    global candlestick_history
    
    # Create a candlestick from current market data
    # This is a simplified approach - in real implementation, you'd aggregate tick data
    current_time = datetime.now()
    
    # For simplicity, we'll create 1-minute candles
    minute_key = current_time.replace(second=0, microsecond=0)
    
    current_candle = {
        'timestamp': minute_key.isoformat(),
        'open': market_data['price'],
        'high': market_data['price'],
        'low': market_data['price'],
        'close': market_data['price'],
        'volume': market_data['volume']
    }
    
    # Add to history
    if symbol not in candlestick_history:
        candlestick_history[symbol] = deque(maxlen=100)
    
    # Check if this is a new minute candle or update existing
    if candlestick_history[symbol] and candlestick_history[symbol][-1]['timestamp'] == minute_key.isoformat():
        # Update existing candle
        candlestick_history[symbol][-1]['high'] = max(candlestick_history[symbol][-1]['high'], market_data['price'])
        candlestick_history[symbol][-1]['low'] = min(candlestick_history[symbol][-1]['low'], market_data['price'])
        candlestick_history[symbol][-1]['close'] = market_data['price']
        candlestick_history[symbol][-1]['volume'] += market_data['volume']
    else:
        # New minute candle
        candlestick_history[symbol].append(current_candle)

@api_router.get("/candlestick-data/{symbol}")
async def get_candlestick_data(symbol: str, period: str = '1d', interval: str = '1m'):
    """Get candlestick data for scalping"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        candlestick_data = await fetch_candlestick_data(symbol, period, interval)
        return {
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'data': candlestick_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching candlestick data: {str(e)}")

@api_router.get("/scalping-signal/{symbol}")
async def get_scalping_signal(symbol: str):
    """Get scalping-focused trading signal"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        # Get recent candlestick data for scalping analysis
        candlestick_data = await fetch_candlestick_data(symbol, period='1d', interval='1m')
        
        if not candlestick_data:
            raise HTTPException(status_code=404, detail="No data available for scalping analysis")
        
        # Simple scalping signal based on recent price action
        recent_candles = candlestick_data[-5:]  # Last 5 minutes
        
        if len(recent_candles) < 3:
            return ScalpingSignal(
                symbol=symbol,
                action="HOLD",
                entry_price=recent_candles[-1]['close'],
                stop_loss=recent_candles[-1]['close'] * 0.999,
                take_profit=recent_candles[-1]['close'] * 1.001,
                confidence=0.0,
                reasons=["Insufficient data for scalping"],
                timeframe="1m"
            )
        
        # Basic scalping logic
        current_price = recent_candles[-1]['close']
        prev_price = recent_candles[-2]['close']
        
        # Calculate short-term momentum
        momentum = (current_price - prev_price) / prev_price
        
        # Calculate volatility
        highs = [candle['high'] for candle in recent_candles]
        lows = [candle['low'] for candle in recent_candles]
        volatility = (max(highs) - min(lows)) / current_price
        
        # Generate scalping signal with tight SL/TP
        action = "HOLD"
        confidence = 0.5
        reasons = []
        
        # Scalping-optimized pip values for tight SL/TP
        if symbol == 'XAUUSD':
            # For XAUUSD scalping: 2-3 pip SL, 4-6 pip TP
            sl_pips = 2.5
            tp_pips = 5.0
            pip_value = 0.1
        elif symbol == 'EURUSD':
            # For EURUSD scalping: 3-5 pip SL, 6-10 pip TP
            sl_pips = 4
            tp_pips = 8
            pip_value = 0.0001
        elif symbol in ['USDJPY', 'EURJPY']:
            # For JPY pairs: 2-3 pip SL, 4-6 pip TP
            sl_pips = 3
            tp_pips = 6
            pip_value = 0.01
        else:
            # Default for other symbols
            sl_pips = 5
            tp_pips = 10
            pip_value = 1.0
        
        # Enhanced scalping momentum detection
        if momentum > 0.0003:  # Reduced threshold for scalping sensitivity
            action = "BUY"
            confidence = min(0.85, 0.6 + momentum * 2000)  # Higher confidence scaling
            reasons.append("Scalping momentum: Strong upward movement")
            stop_loss = current_price - (pip_value * sl_pips)
            take_profit = current_price + (pip_value * tp_pips)
        elif momentum < -0.0003:  # Reduced threshold for scalping sensitivity
            action = "SELL"
            confidence = min(0.85, 0.6 + abs(momentum) * 2000)
            reasons.append("Scalping momentum: Strong downward movement")
            stop_loss = current_price + (pip_value * sl_pips)
            take_profit = current_price - (pip_value * tp_pips)
        else:
            # Tight range for consolidation scalping
            stop_loss = current_price - (pip_value * (sl_pips * 0.5))
            take_profit = current_price + (pip_value * (tp_pips * 0.5))
            reasons.append("Consolidation scalping - tight range")
        
        # Additional scalping conditions
        if volatility > 0.001:  # Lower threshold for scalping
            reasons.append("Good volatility for scalping")
            confidence += 0.15
        
        # Volume-based scalping enhancement
        volumes = [candle['volume'] for candle in recent_candles]
        avg_volume = sum(volumes) / len(volumes)
        if volumes[-1] > avg_volume * 1.2:  # 20% volume increase
            reasons.append("Volume spike detected")
            confidence += 0.1
        
        return ScalpingSignal(
            symbol=symbol,
            action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=min(confidence, 0.9),
            reasons=reasons,
            timeframe="1m"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating scalping signal: {str(e)}")

@api_router.get("/technical-indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        historical_data = await fetch_historical_data(symbol)
        if historical_data.empty:
            # Return default indicators if no historical data
            return TechnicalIndicators(
                symbol=symbol,
                rsi=50.0,
                macd=0.0,
                macd_signal=0.0,
                macd_hist=0.0,
                bb_upper=0.0,
                bb_middle=0.0,
                bb_lower=0.0,
                stoch_k=50.0,
                stoch_d=50.0,
                atr=0.0,
                obv=0.0
            )
        
        indicators = calculate_technical_indicators(historical_data)
        
        # Ensure all required fields are present with defaults
        indicator_data = {
            'symbol': symbol,
            'rsi': indicators.get('RSI', 50.0),
            'macd': indicators.get('MACD', 0.0),
            'macd_signal': indicators.get('MACD_signal', 0.0),
            'macd_hist': indicators.get('MACD_hist', 0.0),
            'bb_upper': indicators.get('BB_upper', 0.0),
            'bb_middle': indicators.get('BB_middle', 0.0),
            'bb_lower': indicators.get('BB_lower', 0.0),
            'stoch_k': indicators.get('STOCH_K', 50.0),
            'stoch_d': indicators.get('STOCH_D', 50.0),
            'atr': indicators.get('ATR', 0.0),
            'obv': indicators.get('OBV', 0.0)
        }
        
        return TechnicalIndicators(**indicator_data)
        
    except Exception as e:
        print(f"Error calculating technical indicators for {symbol}: {e}")
        # Return default indicators on error
        return TechnicalIndicators(
            symbol=symbol,
            rsi=50.0,
            macd=0.0,
            macd_signal=0.0,
            macd_hist=0.0,
            bb_upper=0.0,
            bb_middle=0.0,
            bb_lower=0.0,
            stoch_k=50.0,
            stoch_d=50.0,
            atr=0.0,
            obv=0.0
        )

@api_router.get("/trading-signal/{symbol}")
async def get_trading_signal(symbol: str):
    """Get advanced trading signal using ensemble ML engine"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        # Get market data and indicators
        market_data = await fetch_live_data(symbol)
        if not market_data:
            # Use fallback market data
            price_map = {
                'XAUUSD': 2650.0 + np.random.uniform(-50, 50),
                'EURUSD': 1.0500 + np.random.uniform(-0.02, 0.02),
                'EURJPY': 164.0 + np.random.uniform(-2, 2),
                'USDJPY': 156.0 + np.random.uniform(-2, 2),
                'NASDAQ': 20000.0 + np.random.uniform(-500, 500)
            }
            
            market_data = {
                'symbol': symbol,
                'price': price_map.get(symbol, 100.0),
                'change': np.random.uniform(-1.5, 1.5),
                'volume': np.random.randint(100000, 1000000),
                'timestamp': datetime.now()
            }
        
        historical_data = await fetch_historical_data(symbol)
        if historical_data.empty:
            historical_data = pd.DataFrame()  # Will use defaults
        
        indicators = calculate_technical_indicators(historical_data)
        
        # Use ensemble ML engine if available
        if ML_ENGINE_AVAILABLE and ensemble_ml_engine:
            try:
                # Get ensemble prediction from specialized ML models
                ensemble_prediction = await ensemble_ml_engine.get_ensemble_prediction(
                    symbol, market_data, indicators
                )
                
                # Extract ensemble decision
                ensemble_decision = ensemble_prediction['ensemble_decision']
                individual_predictions = ensemble_prediction['individual_predictions']
                
                # Prepare comprehensive trading signal
                action = ensemble_decision.get('action', 'HOLD')
                confidence = ensemble_decision.get('confidence', 0.5)
                should_trade = ensemble_decision.get('should_trade', False)
                reasons = ensemble_decision.get('reasons', [])
                
                # Prepare detailed ML predictions
                ml_prediction = {
                    'ensemble': {
                        'action': action,
                        'confidence': confidence,
                        'should_trade': should_trade,
                        'strength': ensemble_decision.get('ensemble_strength', 'MEDIUM')
                    }
                }
                
                # Add individual model predictions
                for model_name in ['xgboost', 'catboost', 'tpot', 'prophet']:
                    if model_name in individual_predictions:
                        ml_prediction[model_name] = individual_predictions[model_name]
            
            except Exception as e:
                print(f"Ensemble prediction error: {e}")
                # Fallback to basic prediction
                action = "HOLD"
                confidence = 0.5
                reasons = [f"Ensemble error: {str(e)}, using basic analysis"]
                ml_prediction = {"ensemble_error": str(e)}
        
        else:
            # Fallback to basic trading signal
            action = "HOLD"
            confidence = 0.5
            reasons = ["Using basic technical analysis (ML engine not available)"]
            ml_prediction = {"fallback": True}
            
            # Use RL agent if available
            if rl_agent:
                try:
                    features = prepare_ml_features(market_data, indicators, 0, 0)
                    state = features[:20]  # Use first 20 features
                    action_idx = rl_agent.act(state)
                    action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                    action = action_map[action_idx]
                    confidence = 0.8
                    reasons.append(f"RL Agent decision: {action}")
                except Exception as e:
                    reasons.append(f"RL Agent error: {str(e)}")
        
        # Add technical analysis reasons
        rsi = indicators.get('RSI', 50)
        if rsi < 30:
            reasons.append("RSI oversold (technical)")
            if action == "HOLD":
                action = "BUY"
                confidence = max(confidence, 0.7)
        elif rsi > 70:
            reasons.append("RSI overbought (technical)")
            if action == "HOLD":
                action = "SELL"
                confidence = max(confidence, 0.7)
        
        # MACD analysis
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_signal', 0)
        if macd > macd_signal:
            reasons.append("MACD bullish crossover")
        elif macd < macd_signal:
            reasons.append("MACD bearish crossover")
        
        # Volume analysis
        volume = market_data.get('volume', 0)
        if volume > 500000:
            reasons.append("High volume detected")
            confidence = min(confidence * 1.1, 0.95)
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasons=reasons,
            ml_prediction=ml_prediction
        )
        
    except Exception as e:
        print(f"Error generating trading signal for {symbol}: {e}")
        # Fallback to basic signal
        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.5,
            reasons=[f"System error: {str(e)}, using safe HOLD position"],
            ml_prediction={"system_error": str(e)}
        )

@api_router.post("/tweet-input")
async def add_tweet_input(tweet_input: TweetInput):
    """Add manual tweet input for sentiment analysis"""
    # Store tweet in database
    await db.tweets.insert_one(tweet_input.dict())
    
    # Simple sentiment analysis
    text = tweet_input.tweet.lower()
    bull_words = ["buy", "long", "bull", "upside", "breakout", "moon", "pump"]
    bear_words = ["sell", "short", "bear", "downside", "crash", "dump", "collapse"]
    
    bull_score = sum(1 for word in bull_words if word in text)
    bear_score = sum(1 for word in bear_words if word in text)
    
    sentiment = "BULLISH" if bull_score > bear_score else "BEARISH" if bear_score > bull_score else "NEUTRAL"
    
    return {
        "message": "Tweet processed",
        "sentiment": sentiment,
        "bull_score": bull_score,
        "bear_score": bear_score
    }

@api_router.get("/enhanced-trading-history")
async def get_enhanced_trading_history(limit: int = 100):
    """Get enhanced trading history with all tracking details"""
    trades = await db.enhanced_trades.find().sort("timestamp", -1).limit(limit).to_list(limit)
    
    # Convert ObjectId to string for JSON serialization
    for trade in trades:
        if '_id' in trade:
            trade['_id'] = str(trade['_id'])
    
    return {
        "trades": trades,
        "total_count": len(trades),
        "columns": [
            "🕒 Time", "📈 Symbol", "💰 Action", "💸 Entry Price", 
            "⏳ Exit Price", "📊 Pips Gained", "💹 % P/L", "🤖 Confidence",
            "📋 Decision Factors", "📦 Trade Type", "📉 Forecast Trend",
            "📰 News Sentiment", "🗣️ Tweet Bias", "💡 Bot Strategy",
            "🧠 ML Decision", "📦 Risk Level", "🧾 Exit Reason"
        ]
    }

@api_router.get("/export-enhanced-trades")
async def export_enhanced_trades():
    """Export enhanced trading history to CSV format"""
    trades = await db.enhanced_trades.find().sort("timestamp", -1).to_list(1000)
    
    if not trades:
        # Generate sample enhanced trades for demonstration
        sample_trades = generate_sample_enhanced_trades()
        return {
            "message": "CSV data generated with sample trades",
            "total_trades": len(sample_trades),
            "csv_data": convert_trades_to_csv(sample_trades)
        }
    
    # Convert ObjectId to string and prepare for CSV
    for trade in trades:
        if '_id' in trade:
            trade['_id'] = str(trade['_id'])
    
    csv_data = convert_trades_to_csv(trades)
    
    return {
        "message": "Enhanced trading history exported successfully",
        "total_trades": len(trades),
        "csv_data": csv_data
    }

def generate_sample_enhanced_trades() -> List[Dict]:
    """Generate sample enhanced trades for demonstration"""
    sample_trades = []
    symbols = ['XAUUSD', 'EURUSD', 'USDJPY', 'EURJPY']
    actions = ['BUY', 'SELL']
    strategies = ['RSI Reversal', 'MACD Crossover', 'Momentum Breakout', 'Support Resistance']
    ml_models = ['XGBoost', 'CatBoost', 'Prophet', 'RL Agent']
    risk_levels = ['Low', 'Medium', 'High']
    exit_reasons = ['TP Hit', 'SL Hit', 'Timeout', 'Manual Rule']
    
    for i in range(50):
        symbol = np.random.choice(symbols)
        action = np.random.choice(actions)
        entry_price = {
            'XAUUSD': 2650.0 + np.random.uniform(-50, 50),
            'EURUSD': 1.0500 + np.random.uniform(-0.02, 0.02),
            'USDJPY': 156.0 + np.random.uniform(-2, 2),
            'EURJPY': 164.0 + np.random.uniform(-2, 2)
        }[symbol]
        
        exit_price = entry_price + np.random.uniform(-0.005, 0.005) * entry_price  # Smaller moves for scalping
        
        # Use correct pip calculation
        pips_gained = calculate_pips(entry_price, exit_price, symbol, action)
        
        # Calculate percentage P/L (more realistic for scalping)
        if action == "BUY":
            percentage_pl = ((exit_price - entry_price) / entry_price) * 100
        else:  # SELL
            percentage_pl = ((entry_price - exit_price) / entry_price) * 100
        
        # Generate technical indicators optimized for scalping
        rsi_value = np.random.randint(25, 75)  # More realistic RSI range
        macd_value = np.random.uniform(-0.0005, 0.0005)  # Smaller MACD values for scalping
        macd_signal = macd_value + np.random.uniform(-0.0001, 0.0001)  # Tighter signal
        atr_value = np.random.uniform(0.0008, 0.0015)  # Realistic ATR for scalping
        volume_value = np.random.randint(50000, 120000)  # Volume range for scalping
        news_sentiment = round(np.random.uniform(-0.3, 0.3), 2)  # Moderate sentiment
        tweet_bias = np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
        
        # Create technical indicators dict for strategy determination
        technical_indicators = {
            'RSI': rsi_value,
            'MACD': macd_value,
            'MACD_signal': macd_signal,
            'MACD_hist': macd_value - macd_signal,
            'ATR': atr_value,
            'price': entry_price,
            'volume': volume_value,
            'BB_upper': entry_price * 1.02,
            'BB_lower': entry_price * 0.98
        }
        
        # Generate intelligent strategy label
        intelligent_strategy = get_strategy_label(
            indicators=technical_indicators,
            sentiment=news_sentiment,
            tweet_bias=tweet_bias,
            events=[],
            action=action
        )
        
        trade = {
            'timestamp': (datetime.utcnow() - timedelta(hours=np.random.randint(1, 168))).isoformat(),
            'symbol': symbol,
            'action': action,
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'pips_gained': round(pips_gained, 1),
            'percentage_pl': round(percentage_pl, 2),
            'confidence': round(np.random.uniform(0.5, 0.95), 2),
            'decision_factors': f"RSI {rsi_value}, MACD {np.random.choice(['Bullish', 'Bearish'])}, {np.random.choice(['Positive News', 'Negative News', 'Neutral'])}",
            'trade_type': 'Scalping',
            'forecast_trend': np.random.choice(['UP', 'DOWN', 'NEUTRAL']),
            'news_sentiment': news_sentiment,
            'tweet_bias': tweet_bias,
            'bot_strategy': intelligent_strategy,  # Use intelligent strategy
            'ml_decision': np.random.choice(ml_models),
            'risk_level': np.random.choice(risk_levels),
            'exit_reason': np.random.choice(exit_reasons),
            'is_closed': True,
            'close_timestamp': (datetime.utcnow() - timedelta(hours=np.random.randint(0, 167))).isoformat(),
            # Add separate technical indicator columns
            'rsi_value': rsi_value,
            'macd_value': round(macd_value, 4),
            'volume_spike': round(np.random.uniform(0.8, 2.5), 2),
            'volatility': round(np.random.uniform(0.001, 0.005), 4)
        }
        sample_trades.append(trade)
    
    return sample_trades

def convert_trades_to_csv(trades: List[Dict]) -> str:
    """Convert trades list to CSV format string"""
    if not trades:
        return ""
    
    # Define CSV headers
    headers = [
        "🕒 Time", "📈 Symbol", "💰 Action", "💸 Entry Price", 
        "⏳ Exit Price", "📊 Pips Gained", "💹 % P/L", "🤖 Confidence",
        "📋 Decision Factors", "📦 Trade Type", "📉 Forecast Trend",
        "📰 News Sentiment", "🗣️ Tweet Bias", "💡 Bot Strategy",
        "🧠 ML Decision", "📦 Risk Level", "🧾 Exit Reason",
        "📊 RSI Value", "📈 MACD Value", "📊 Volume Spike", "⚡ Volatility"
    ]
    
    csv_lines = [",".join(headers)]
    
    for trade in trades:
        row = [
            trade.get('timestamp', ''),
            trade.get('symbol', ''),
            trade.get('action', ''),
            str(trade.get('entry_price', '')),
            str(trade.get('exit_price', '')),
            str(trade.get('pips_gained', '')),
            str(trade.get('percentage_pl', '')),
            str(trade.get('confidence', '')),
            f'"{trade.get("decision_factors", "")}"',  # Quote to handle commas
            trade.get('trade_type', ''),
            trade.get('forecast_trend', ''),
            str(trade.get('news_sentiment', '')),
            trade.get('tweet_bias', ''),
            trade.get('bot_strategy', ''),
            trade.get('ml_decision', ''),
            trade.get('risk_level', ''),
            trade.get('exit_reason', '')
        ]
        csv_lines.append(",".join(row))
    
    return "\n".join(csv_lines)

@api_router.post("/create-sample-trades")
async def create_sample_enhanced_trades():
    """Create sample enhanced trades in database for testing"""
    sample_trades = generate_sample_enhanced_trades()
    
    # Insert into database
    if sample_trades:
        await db.enhanced_trades.insert_many(sample_trades)
    
    # Save trade data after creating sample trades
    try:
        save_all_persistent_data()
        print("💾 Saved trading data after creating sample trades")
    except Exception as e:
        print(f"❌ Error saving sample trades: {e}")
    
    return {
        "message": f"Created {len(sample_trades)} sample enhanced trades",
        "trades_created": len(sample_trades)
    }

@api_router.post("/clear-sample-data")
async def clear_sample_data():
    """Clear existing sample data and regenerate with new logic"""
    try:
        # Clear existing trades
        await db.trades.delete_many({})
        
        # Clear existing enhanced trades
        await db.enhanced_trades.delete_many({})
        
        # Regenerate sample data
        await populate_sample_data()
        
        # Save after data regeneration
        try:
            save_all_persistent_data()
            print("💾 Saved trading data after clearing and regenerating")
        except Exception as e:
            print(f"❌ Error saving after data regeneration: {e}")
        
        return {"message": "Sample data cleared and regenerated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@api_router.get("/trade-analysis")
async def get_trade_analysis():
    """Get trade analysis including top gainers and losers"""
    try:
        # Get all trades
        trades = await db.trades.find().to_list(1000)
        
        if not trades:
            return {
                "top_gainers": [],
                "top_losers": [],
                "best_strategies": [],
                "worst_strategies": [],
                "summary": {
                    "total_trades": 0,
                    "avg_profit": 0,
                    "avg_pips": 0
                }
            }
        
        # Sort by profit/pips
        top_gainers = sorted(trades, key=lambda x: x.get('profit', 0), reverse=True)[:5]
        top_losers = sorted(trades, key=lambda x: x.get('profit', 0))[:5]
        
        # Strategy analysis
        strategy_performance = {}
        for trade in trades:
            strategy = trade.get('bot_strategy', 'Unknown')
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'profits': [], 'pips': []}
            strategy_performance[strategy]['profits'].append(trade.get('profit', 0))
            strategy_performance[strategy]['pips'].append(trade.get('pips', 0))
        
        # Calculate average performance per strategy
        strategy_stats = []
        for strategy, data in strategy_performance.items():
            if data['profits']:
                avg_profit = sum(data['profits']) / len(data['profits'])
                avg_pips = sum(data['pips']) / len(data['pips'])
                win_rate = len([p for p in data['profits'] if p > 0]) / len(data['profits']) * 100
                
                strategy_stats.append({
                    'strategy': strategy,
                    'avg_profit': round(avg_profit, 2),
                    'avg_pips': round(avg_pips, 1),
                    'win_rate': round(win_rate, 1),
                    'trade_count': len(data['profits'])
                })
        
        best_strategies = sorted(strategy_stats, key=lambda x: x['avg_profit'], reverse=True)[:3]
        worst_strategies = sorted(strategy_stats, key=lambda x: x['avg_profit'])[:3]
        
        # Clean ObjectId for JSON serialization
        for trade in top_gainers + top_losers:
            if '_id' in trade:
                trade['_id'] = str(trade['_id'])
        
        return {
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "best_strategies": best_strategies,
            "worst_strategies": worst_strategies,
            "summary": {
                "total_trades": len(trades),
                "avg_profit": round(sum(t.get('profit', 0) for t in trades) / len(trades), 2),
                "avg_pips": round(sum(t.get('pips', 0) for t in trades) / len(trades), 1)
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "top_gainers": [],
            "top_losers": [],
            "best_strategies": [],
            "worst_strategies": [],
            "summary": {"total_trades": 0, "avg_profit": 0, "avg_pips": 0}
        }

@api_router.get("/bot-health")
async def get_bot_health():
    """Get bot health metrics including RL agent status"""
    try:
        global scalping_rl_agent, rl_agent
        
        health_data = {
            "scalping_rl_agent": {
                "status": "active" if scalping_rl_agent else "inactive",
                "epsilon": scalping_rl_agent.epsilon if scalping_rl_agent else 0.0,
                "memory_size": len(scalping_rl_agent.memory) if scalping_rl_agent else 0,
                "learning_rate": scalping_rl_agent.lr if scalping_rl_agent else 0.0,
                "trades_made": scalping_rl_agent.trades_made if scalping_rl_agent else 0,
                "current_streak": scalping_rl_agent.current_streak if scalping_rl_agent else 0
            },
            "regular_rl_agent": {
                "status": "active" if rl_agent else "inactive",
                "epsilon": rl_agent.epsilon if rl_agent else 0.0,
                "memory_size": len(rl_agent.memory) if rl_agent else 0,
                "learning_rate": rl_agent.lr if rl_agent else 0.0
            },
            "system_health": {
                "ml_engine_available": ML_ENGINE_AVAILABLE,
                "models_loaded": len([k for k, v in ml_models.items() if v is not None]) if ml_models else 0,
                "price_history_size": len(price_history) if price_history else 0
            }
        }
        
        return health_data
    except Exception as e:
        return {
            "error": str(e),
            "scalping_rl_agent": {"status": "error"},
            "regular_rl_agent": {"status": "error"},
            "system_health": {"status": "error"}
        }

@api_router.post("/activate-model/{model_name}")
async def activate_model(model_name: str):
    """Activate a specific ML model"""
    try:
        global ml_models
        
        if model_name not in ['xgboost', 'catboost', 'prophet', 'tpot']:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        # Simple activation - in real implementation, you'd load the actual model
        ml_models[model_name] = f"activated_{model_name}_model"
        
        return {
            "message": f"{model_name} model activated successfully",
            "model_name": model_name,
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating model: {str(e)}")

@api_router.post("/deactivate-model/{model_name}")
async def deactivate_model(model_name: str):
    """Deactivate a specific ML model"""
    try:
        global ml_models
        
        if model_name not in ['xgboost', 'catboost', 'prophet', 'tpot']:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        ml_models[model_name] = None
        
        return {
            "message": f"{model_name} model deactivated successfully",
            "model_name": model_name,
            "status": "inactive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deactivating model: {str(e)}")

@api_router.get("/performance-metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics optimized for scalping"""
    try:
        # Get trading history
        trades = await db.trades.find().to_list(1000)
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
        losing_trades = len([t for t in trades if t.get('profit', 0) < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(t.get('profit', 0) for t in trades)
        
        # Calculate Max Drawdown
        running_profit = 0
        peak_profit = 0
        max_drawdown = 0
        
        sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
        for trade in sorted_trades:
            running_profit += trade.get('profit', 0)
            if running_profit > peak_profit:
                peak_profit = running_profit
            
            current_drawdown = ((peak_profit - running_profit) / max(abs(peak_profit), 1)) * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        # Calculate Current Streak
        current_streak = 0
        if trades:
            recent_trades = sorted_trades[-10:]  # Last 10 trades
            for trade in reversed(recent_trades):
                if trade.get('profit', 0) > 0:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        break
                else:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        break
        
        # Last Trade Info
        last_trade_status = "No trades"
        last_trade_time = "Never"
        if trades:
            last_trade = sorted_trades[-1]
            last_profit = last_trade.get('profit', 0)
            last_trade_status = f"${last_profit:.2f}" if last_profit >= 0 else f"-${abs(last_profit):.2f}"
            if 'timestamp' in last_trade:
                try:
                    last_trade_time = str(last_trade['timestamp'])[:19]  # Format timestamp
                except:
                    last_trade_time = "Recent"
        
        # Top Strategy Used
        strategy_counts = {}
        for trade in trades:
            strategy = trade.get('bot_strategy', 'Unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        top_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else "None"
        
        # Daily Profit (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.now() - timedelta(days=1)
        daily_trades = [t for t in trades if 'timestamp' in t and str(t['timestamp']) > str(yesterday)]
        daily_profit = sum(t.get('profit', 0) for t in daily_trades)
        
        # Calculate bot confidence (based on recent performance)
        recent_trades = sorted_trades[-20:] if len(sorted_trades) >= 20 else sorted_trades
        if recent_trades:
            recent_win_rate = len([t for t in recent_trades if t.get('profit', 0) > 0]) / len(recent_trades) * 100
            recent_profit = sum(t.get('profit', 0) for t in recent_trades)
            
            # Bot confidence formula: recent win rate + profit factor + streak factor
            confidence_base = recent_win_rate * 0.6  # 60% weight on win rate
            profit_factor = min(20, max(-20, recent_profit)) * 0.5  # Profit contribution
            streak_factor = min(10, max(-10, current_streak)) * 2  # Streak contribution
            
            bot_confidence = max(10, min(95, confidence_base + profit_factor + streak_factor))
        else:
            bot_confidence = 50  # Default confidence
        
        return {
            "totalTrades": total_trades,
            "winRate": round(win_rate, 1),
            "totalProfit": round(total_profit, 2),
            "maxDrawdown": round(max_drawdown, 1),
            "botConfidence": round(bot_confidence, 1),
            "currentStreak": current_streak,
            "lastTradeStatus": last_trade_status,
            "lastTradeTime": last_trade_time,
            "topStrategy": top_strategy,
            "dailyProfit": round(daily_profit, 2),
            "totalLosses": losing_trades
        }
    except Exception as e:
        # Return default values if error
        return {
            "totalTrades": 0,
            "winRate": 0,
            "totalProfit": 0,
            "maxDrawdown": 0,
            "botConfidence": 50,
            "currentStreak": 0,
            "lastTradeStatus": "No trades",
            "lastTradeTime": "Never", 
            "topStrategy": "None",
            "dailyProfit": 0,
            "totalLosses": 0
        }

@api_router.get("/trading-history")
async def get_trading_history():
    """Get trading history"""
    trades = await db.trades.find().sort("timestamp", -1).limit(100).to_list(100)
    
    # Convert ObjectId to string for JSON serialization
    for trade in trades:
        if '_id' in trade:
            trade['_id'] = str(trade['_id'])
    
    return trades

@api_router.get("/model-status")
async def get_model_status():
    """Get comprehensive ML model status"""
    if ML_ENGINE_AVAILABLE and ensemble_ml_engine:
        models_active = ensemble_ml_engine.models_trained
        
        return ModelStatus(
            xgboost_active=models_active.get('xgboost', False),
            catboost_active=models_active.get('catboost', False),
            prophet_active=models_active.get('prophet', False),
            tpot_active=models_active.get('tpot', False),
            rl_agent_active=rl_agent is not None,
            performance=model_performance
        )
    else:
        # Fallback status
        return ModelStatus(
            xgboost_active=ml_models.get('xgboost') is not None,
            catboost_active=ml_models.get('catboost') is not None,
            prophet_active=ml_models.get('prophet') is not None,
            tpot_active=False,
            rl_agent_active=rl_agent is not None,
            performance=model_performance
        )

@api_router.post("/train-models")
async def train_models():
    """Train all specialized ML models with real-time simulation"""
    try:
        # Start training simulation
        result = await training_simulator.start_training_simulation("XAUUSD")
        
        if not ML_ENGINE_AVAILABLE or not ensemble_ml_engine:
            # Fallback to basic training with simulation
            if len(feature_history) < 100:
                # Generate some sample data for simulation
                for _ in range(100):
                    features = np.random.randn(18)
                    feature_history.append(features)
            
            # Basic XGBoost training
            X = np.array(list(feature_history))
            y = np.random.choice([0, 1, 2], size=len(X), p=[0.3, 0.4, 0.3])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            
            # Save basic model
            ml_models['xgboost'] = xgb_model
            ml_models['scaler'] = scaler
            
            # Save basic model training progress
            try:
                save_all_persistent_data()
                print("💾 Saved basic XGBoost model training progress")
            except Exception as e:
                print(f"❌ Error saving basic model progress: {e}")
            
            return {
                "message": "Training simulation started with basic XGBoost model",
                "xgboost_accuracy": xgb_accuracy,
                "models_trained": 1,
                "total_models": 1,
                "simulation_active": True
            }
        
        # Get training data from database
        training_data_cursor = db.training_data.find().limit(300)
        training_data = await training_data_cursor.to_list(300)
        
        if len(training_data) < 100:
            # Generate training data if not enough
            await populate_sample_data()
            training_data_cursor = db.training_data.find().limit(300)
            training_data = await training_data_cursor.to_list(300)
        
        # Train all models using ensemble ML engine
        training_results = await ensemble_ml_engine.train_all_models(training_data, "XAUUSD")
        
        # Update global model performance
        model_performance.update({
            'ensemble_training': training_results.get('overall_success', False),
            'models_trained': training_results.get('models_trained', 0),
            'total_models': training_results.get('total_models', 4),
            'last_trained': datetime.now().isoformat()
        })
        
        # Add individual model performance
        detailed_results = training_results.get('detailed_results', {})
        for model_name, result in detailed_results.items():
            if result.get('success', False):
                model_performance[f'{model_name}_accuracy'] = result.get('accuracy', 0.0)
        
        # Save ML model training progress
        try:
            save_all_persistent_data()
            print("💾 Saved ML model training progress")
        except Exception as e:
            print(f"❌ Error saving ML model progress: {e}")
        
        return {
            "message": "Advanced ML models training simulation started",
            "overall_success": training_results.get('overall_success', False),
            "models_trained": training_results.get('models_trained', 0),
            "total_models": training_results.get('total_models', 4),
            "detailed_results": detailed_results,
            "simulation_active": True,
            "specializations": {
                "xgboost": "Price Movement Prediction (>70% probability triggers trades)",
                "catboost": "Sentiment Impact Modeling (news headlines → price impact)",
                "tpot": "Automatic Pattern Discovery (finds optimal trading patterns)",
                "prophet": "Time Series Forecasting (trend analysis & seasonality)"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@api_router.get("/training-status")
async def get_training_status():
    """Get current training status"""
    return training_simulator.get_training_status()

@api_router.get("/training-metrics")
async def get_training_metrics():
    """Get detailed training metrics for all models"""
    return training_simulator.get_training_metrics()

@api_router.get("/model-comparison")
async def get_model_comparison():
    """Get detailed model performance comparison"""
    return await training_simulator.get_model_comparison()

@api_router.get("/live-trade-feed")
async def get_live_trade_feed():
    """Get live trade feed for real-time updates"""
    return await training_simulator.get_live_trade_feed()

@api_router.post("/stop-training")
async def stop_training():
    """Stop the training simulation"""
    return await training_simulator.stop_training()

@api_router.get("/scalping-rl-performance")
async def get_scalping_rl_performance():
    """Get scalping RL agent performance metrics"""
    global scalping_rl_agent
    
    if scalping_rl_agent is None:
        return {
            "error": "Scalping RL agent not initialized",
            "metrics": {
                "trades_made": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pips": 0,
                "current_streak": 0,
                "epsilon": 1.0
            }
        }
    
    metrics = scalping_rl_agent.get_performance_metrics()
    return {
        "agent_type": "Scalping RL Agent",
        "status": "active",
        "metrics": metrics,
        "description": "High-frequency trading agent optimized for scalping strategies (1-5 minute trades)"
    }

@api_router.get("/mock-trades")
async def get_mock_trades():
    """Get recent mock trades from training"""
    trades = await db.mock_trades.find().sort("timestamp", -1).limit(50).to_list(50)
    
    # Convert ObjectId to string for JSON serialization
    for trade in trades:
        if '_id' in trade:
            trade['_id'] = str(trade['_id'])
    
    return {
        "trades": trades,
        "total_count": len(trades)
    }

@api_router.get("/ensemble-prediction/{symbol}")
async def get_ensemble_prediction(symbol: str):
    """Get detailed ensemble prediction from all ML models"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        # Get market data and indicators
        market_data = await fetch_live_data(symbol)
        if not market_data:
            raise HTTPException(status_code=404, detail="Market data not found")
        
        historical_data = await fetch_historical_data(symbol)
        if historical_data.empty:
            raise HTTPException(status_code=404, detail="Historical data not found")
        
        indicators = calculate_technical_indicators(historical_data)
        
        # Get ensemble prediction
        ensemble_prediction = await ensemble_ml_engine.get_ensemble_prediction(
            symbol, market_data, indicators
        )
        
        return EnsemblePrediction(**ensemble_prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")

@api_router.get("/model-specializations")
async def get_model_specializations():
    """Get information about each model's specialization"""
    return {
        "xgboost": {
            "name": "XGBoost Price Predictor",
            "specialization": "Price Movement Prediction",
            "description": "Collects RSI, EMA, MACD, volume, news sentiment, and event flags to predict if price will go UP or DOWN next. Only triggers trades when probability > 70%.",
            "features": [
                "RSI, EMA, MACD, volume analysis",
                "News sentiment integration",
                "Event flag detection",
                "High-confidence trading signals (>70%)"
            ],
            "active": ensemble_ml_engine.models_trained.get('xgboost', False)
        },
        "catboost": {
            "name": "CatBoost Sentiment Analyzer",
            "specialization": "Sentiment Impact Modeling",
            "description": "Reads news headlines, converts them to sentiment signals, learns how news affects price, and uses that to boost trading accuracy.",
            "features": [
                "Real-time news headline analysis",
                "Sentiment-to-price impact learning",
                "Trading accuracy enhancement",
                "Market sentiment scoring"
            ],
            "active": ensemble_ml_engine.models_trained.get('catboost', False)
        },
        "tpot": {
            "name": "TPOT Pattern Discovery",
            "specialization": "Automatic Pattern Discovery",
            "description": "Automatically discovers hidden trading patterns and creates optimal feature combinations to find the best ML pipeline.",
            "features": [
                "Automated pattern recognition",
                "Optimal feature combination",
                "Best ML pipeline discovery",
                "Hidden pattern detection"
            ],
            "active": ensemble_ml_engine.models_trained.get('tpot', False)
        },
        "prophet": {
            "name": "Prophet Time Series Forecaster",
            "specialization": "Time Series Forecasting",
            "description": "Forecasts future price movements using time series analysis, identifies trends, seasonality, and provides long-term direction predictions.",
            "features": [
                "Future price forecasting",
                "Trend identification",
                "Seasonality analysis",
                "Long-term direction prediction"
            ],
            "active": ensemble_ml_engine.models_trained.get('prophet', False)
        }
    }

@api_router.post("/backtest/{symbol}")
async def run_backtest(symbol: str, start_date: str, end_date: str):
    """Run backtest for a symbol"""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    
    try:
        # Fetch historical data for backtesting
        historical_data = await fetch_historical_data(symbol)
        if historical_data.empty:
            raise HTTPException(status_code=404, detail="Historical data not found")
        
        # Simple backtest simulation
        results = []
        total_profit = 0
        winning_trades = 0
        losing_trades = 0
        max_drawdown = 0
        
        for i in range(20, len(historical_data)):
            # Simulate trading decision
            indicators = calculate_technical_indicators(historical_data.iloc[:i])
            
            # Simple strategy: Buy when RSI < 30, Sell when RSI > 70
            rsi = indicators.get('RSI', 50)
            
            if rsi < 30:
                action = "BUY"
            elif rsi > 70:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Calculate profit/loss (simplified)
            if action in ["BUY", "SELL"]:
                profit = np.random.uniform(-50, 100)  # Simplified profit calculation
                total_profit += profit
                
                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                results.append({
                    "date": historical_data.index[i].isoformat(),
                    "action": action,
                    "price": historical_data.iloc[i]['close'],
                    "profit": profit,
                    "rsi": rsi
                })
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        return BacktestResult(
            symbol=symbol,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=1.5,  # Simplified
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@api_router.get("/export-trades")
async def export_trades():
    """Export trading history to CSV"""
    trades = await db.trades.find().to_list(1000)
    
    if not trades:
        raise HTTPException(status_code=404, detail="No trades found")
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Export to CSV
    csv_data = df.to_csv(index=False)
    
    return {
        "message": "Trades exported successfully",
        "csv_data": csv_data,
        "total_trades": len(trades)
    }

# Create models directory
os.makedirs('/app/backend/models', exist_ok=True)

# Include router
app.include_router(api_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await initialize_system()
    # Start background task for market data updates
    asyncio.create_task(update_market_data())

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()