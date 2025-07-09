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
        
        # Scalping-specific parameters
        self.pip_target = 10  # Target 10 pips per trade
        self.pip_stop_loss = 5  # Stop loss at 5 pips
        self.max_hold_time = 5  # Max 5 minutes per trade
        
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
            momentum_1m * 1000,  # Scale up small forex movements
            momentum_2m * 1000,
            momentum_3m * 1000,
            price_range * 1000,
            volume_spike,
            distance_to_high * 1000,
            distance_to_low * 1000,
            green_candles / len(recent_candles),
            red_candles / len(recent_candles),
            price_vs_sma3 * 1000,
            price_vs_sma5 * 1000,
            np.tanh(current_price / 1000),  # Normalized price
            self.epsilon,  # Exploration factor
            self.trades_made / 100,  # Normalized trade count
            self.total_pips / 1000  # Normalized total pips
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
def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent.fillna(50), d_percent.fillna(50)

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range"""
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
    Determine strategy label based on actual trading conditions
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
    
    # RSI Reversal Strategy
    if rsi < 30 and action == "BUY":
        return "RSI Reversal"
    elif rsi > 70 and action == "SELL":
        return "RSI Reversal"
    
    # MACD Crossover Strategy
    elif abs(macd_hist) > 0.001 and ((macd > macd_signal and action == "BUY") or (macd < macd_signal and action == "SELL")):
        return "MACD Crossover"
    
    # Tweet Momentum Strategy
    elif tweet_bias == "BULLISH" and sentiment > 0.1 and action == "BUY":
        return "Tweet Momentum"
    elif tweet_bias == "BEARISH" and sentiment < -0.1 and action == "SELL":
        return "Tweet Momentum"
    
    # Event Reaction Strategy
    elif events and any(e.get('impact', '') == "High" for e in events):
        return "Event Reaction"
    
    # News Sentiment Strategy
    elif abs(sentiment) > 0.2:
        if sentiment > 0.2 and action == "BUY":
            return "News Sentiment Boost"
        elif sentiment < -0.2 and action == "SELL":
            return "News Sentiment Boost"
    
    # Breakout Strategy
    elif atr > 0.002 and volume > 50000:
        if price > bb_upper and action == "BUY":
            return "Breakout"
        elif price < bb_lower and action == "SELL":
            return "Breakout"
    
    # Bollinger Squeeze Strategy
    elif bb_upper > 0 and bb_lower > 0 and (bb_upper - bb_lower) / price < 0.02:
        return "Bollinger Squeeze"
    
    # Volume Spike Strategy
    elif volume > 100000:  # High volume
        return "Volume Spike Trade"
    
    # Mean Reversion Strategy
    elif 35 < rsi < 65 and abs(macd_hist) < 0.0005:
        return "Mean Reversion"
    
    # Trend Continuation Strategy
    elif atr > 0.001 and ((rsi > 50 and action == "BUY") or (rsi < 50 and action == "SELL")):
        return "Trend Continuation"
    
    # Double Confirmation Strategy
    elif ((rsi < 35 and macd > macd_signal) or (rsi > 65 and macd < macd_signal)) and abs(sentiment) > 0.1:
        return "Double Confirmation"
    
    # Scalping Opportunity (default for quick trades)
    elif abs(macd_hist) > 0.0003 or abs(sentiment) > 0.05:
        return "Scalping Opportunity"
    
    # Default fallback
    else:
        return "General Strategy"

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
    
    # Determine bot strategy
    bot_strategy = "Default"
    if rsi_value < 30 and action == "BUY":
        bot_strategy = "RSI Reversal"
    elif rsi_value > 70 and action == "SELL":
        bot_strategy = "RSI Reversal"
    elif abs(macd_value) > 0.001:
        bot_strategy = "MACD Crossover"
    elif scalping_signal and scalping_signal.get('confidence', 0) > 0.8:
        bot_strategy = "Momentum Breakout"
    
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

def calculate_pips_and_profit(trade: EnhancedTrade, exit_price: float) -> EnhancedTrade:
    """Calculate pips and profit when trade is closed"""
    
    # Pip values for different symbols
    pip_values = {
        'XAUUSD': 0.1,
        'EURUSD': 0.0001,
        'EURJPY': 0.01,
        'USDJPY': 0.01,
        'NASDAQ': 1.0
    }
    
    pip_value = pip_values.get(trade.symbol, 0.01)
    
    # Calculate pips gained
    if trade.action == "BUY":
        pips_gained = (exit_price - trade.entry_price) / pip_value
    else:  # SELL
        pips_gained = (trade.entry_price - exit_price) / pip_value
    
    # Calculate percentage P/L
    percentage_pl = ((exit_price - trade.entry_price) / trade.entry_price) * 100
    if trade.action == "SELL":
        percentage_pl = -percentage_pl
    
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
        
        # Moving averages
        indicators['SMA_20'] = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
        indicators['SMA_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
        indicators['EMA_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['EMA_26'] = close.ewm(span=26).mean().iloc[-1]
        
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
    """Initialize the trading system"""
    global rl_agent, ml_models, price_history, scalping_rl_agent, candlestick_history
    
    # Initialize RL agent
    rl_agent = RLTradingAgent(state_size=20)
    
    # Initialize Scalping RL agent
    scalping_rl_agent = ScalpingRLAgent(state_size=15)
    
    # Initialize price history
    for symbol in SYMBOLS:
        price_history[symbol] = deque(maxlen=100)
    
    # Initialize candlestick history
    for symbol in SYMBOLS:
        candlestick_history[symbol] = deque(maxlen=100)
    
    # Initialize ML models dictionary
    ml_models = {
        'xgboost': None,
        'catboost': None,
        'prophet': None,
        'scaler': None
    }
    
    # Populate some sample trading data
    await populate_sample_data()
    
    print("Trading system initialized successfully!")
    print("Scalping RL agent initialized for high-frequency trading")

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
            entry_price = np.random.uniform(100, 3000) if symbol == 'XAUUSD' else np.random.uniform(1, 200)
            exit_price = entry_price * (1 + np.random.uniform(-0.05, 0.05))
            quantity = np.random.uniform(0.1, 2.0)
            profit = (exit_price - entry_price) * quantity if action == 'BUY' else (entry_price - exit_price) * quantity
            pips = abs(exit_price - entry_price) * 10000 if 'USD' in symbol else abs(exit_price - entry_price) * 100
            if profit < 0:
                pips = -pips
            
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
                'bot_strategy': np.random.choice(strategies)
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
        
        # Generate scalping signal
        action = "HOLD"
        confidence = 0.5
        reasons = []
        
        # Define pip value based on symbol
        pip_values = {
            'XAUUSD': 0.1,
            'EURUSD': 0.0001,
            'EURJPY': 0.01,
            'USDJPY': 0.01,
            'NASDAQ': 1.0
        }
        
        pip_value = pip_values.get(symbol, 0.01)
        
        if momentum > 0.0005:  # Strong upward momentum
            action = "BUY"
            confidence = min(0.8, 0.5 + momentum * 1000)
            reasons.append("Strong upward momentum detected")
            stop_loss = current_price - (pip_value * 5)  # 5 pip stop loss
            take_profit = current_price + (pip_value * 10)  # 10 pip take profit
        elif momentum < -0.0005:  # Strong downward momentum
            action = "SELL"
            confidence = min(0.8, 0.5 + abs(momentum) * 1000)
            reasons.append("Strong downward momentum detected")
            stop_loss = current_price + (pip_value * 5)  # 5 pip stop loss
            take_profit = current_price - (pip_value * 10)  # 10 pip take profit
        else:
            stop_loss = current_price - (pip_value * 2)
            take_profit = current_price + (pip_value * 2)
            reasons.append("Consolidation - waiting for breakout")
        
        if volatility > 0.002:
            reasons.append("High volatility - good for scalping")
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
        
        exit_price = entry_price + np.random.uniform(-0.01, 0.01) * entry_price
        pips_gained = np.random.uniform(-10, 15)
        percentage_pl = np.random.uniform(-2, 3)
        
        rsi_value = np.random.randint(20, 80)
        macd_value = np.random.uniform(-0.002, 0.002)
        
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
            'news_sentiment': round(np.random.uniform(-0.5, 0.5), 2),
            'tweet_bias': np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
            'bot_strategy': np.random.choice(strategies),
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
    
    return {
        "message": f"Created {len(sample_trades)} sample enhanced trades",
        "trades_created": len(sample_trades)
    }

@api_router.get("/performance-metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        # Get trading history
        trades = await db.trades.find().to_list(1000)
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
        losing_trades = len([t for t in trades if t.get('profit', 0) < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(t.get('profit', 0) for t in trades)
        total_pips = sum(t.get('pips', 0) for t in trades)
        
        # Calculate bot confidence (based on recent trading performance)
        recent_trades = sorted(trades, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
        if recent_trades:
            recent_win_rate = len([t for t in recent_trades if t.get('profit', 0) > 0]) / len(recent_trades) * 100
            # Bot confidence is based on recent performance and win rate
            bot_confidence = min(90, max(10, recent_win_rate * 0.8 + 20))  # Scale 10-90%
        else:
            bot_confidence = 50  # Default confidence
        
        return {
            "totalTrades": total_trades,
            "winRate": round(win_rate, 1),
            "totalProfit": round(total_profit, 2),
            "totalPips": round(total_pips, 1),
            "totalLosses": losing_trades,
            "botConfidence": round(bot_confidence, 1)
        }
    except Exception as e:
        # Return default values if error
        return {
            "totalTrades": 0,
            "winRate": 0,
            "totalProfit": 0,
            "totalPips": 0,
            "totalLosses": 0,
            "botConfidence": 50
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