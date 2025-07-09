"""
Advanced ML Engine for Autonomous Trading Bot
Each model has a specialized role for optimal trading performance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import pickle
from typing import Dict, List, Tuple, Optional
import asyncio
from textblob import TextBlob
from newsapi import NewsApiClient
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from prophet import Prophet
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

class XGBoostPricePredictor:
    """
    XGBoost for Price Movement Prediction
    - Collects RSI, EMA, MACD, volume, news sentiment, and event flags
    - Predicts: "Will price go UP or DOWN next?"
    - Triggers trades when probability > 70%
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'rsi', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d', 'atr',
            'volume', 'volume_sma', 'price_change', 'volatility',
            'news_sentiment', 'event_flag', 'hour', 'day_of_week'
        ]
        self.is_trained = False
        self.last_accuracy = 0.0
        
    def prepare_features(self, market_data: Dict, indicators: Dict, 
                        news_sentiment: float, event_flag: int) -> np.ndarray:
        """Prepare features for XGBoost prediction"""
        now = datetime.now()
        
        # Price and volume features
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        price_change = market_data.get('change', 0)
        
        # Calculate volatility (simplified)
        volatility = abs(price_change) / 100
        
        # Volume moving average (simplified)
        volume_sma = volume * 0.95  # Simplified calculation
        
        features = [
            indicators.get('RSI', 50),
            indicators.get('EMA_12', price),
            indicators.get('EMA_26', price),
            indicators.get('MACD', 0),
            indicators.get('MACD_signal', 0),
            indicators.get('MACD_hist', 0),
            indicators.get('BB_upper', price),
            indicators.get('BB_middle', price),
            indicators.get('BB_lower', price),
            indicators.get('STOCH_K', 50),
            indicators.get('STOCH_D', 50),
            indicators.get('ATR', 0),
            volume,
            volume_sma,
            price_change,
            volatility,
            news_sentiment,
            event_flag,
            now.hour,
            now.weekday()
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Dict]) -> Dict:
        """Train XGBoost model for price movement prediction"""
        if len(training_data) < 100:
            return {"success": False, "error": "Insufficient training data"}
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for data_point in training_data:
            features = self.prepare_features(
                data_point['market_data'],
                data_point['indicators'],
                data_point.get('news_sentiment', 0),
                data_point.get('event_flag', 0)
            )
            X_list.append(features.flatten())
            
            # Label: 0=DOWN, 1=SIDEWAYS, 2=UP based on next price movement
            next_price_change = data_point.get('next_price_change', 0)
            if next_price_change > 0.5:  # >0.5% change
                label = 2  # UP
            elif next_price_change < -0.5:  # <-0.5% change
                label = 0  # DOWN
            else:
                label = 1  # SIDEWAYS
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Fit model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.last_accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True
        
        # Save model
        joblib.dump(self.model, '/app/backend/models/xgboost_price_predictor.pkl')
        joblib.dump(self.scaler, '/app/backend/models/xgboost_scaler.pkl')
        
        return {
            "success": True,
            "accuracy": self.last_accuracy,
            "model_type": "XGBoost Price Predictor",
            "features_used": len(self.feature_names)
        }
    
    def predict(self, market_data: Dict, indicators: Dict, 
                news_sentiment: float = 0, event_flag: int = 0) -> Dict:
        """Make price movement prediction"""
        if not self.is_trained or self.model is None:
            return {"prediction": "HOLD", "probability": 0.5, "confidence": "LOW"}
        
        features = self.prepare_features(market_data, indicators, news_sentiment, event_flag)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        prediction_idx = np.argmax(probabilities)
        max_prob = probabilities[prediction_idx]
        
        # Map prediction
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        prediction = action_map[prediction_idx]
        
        # Determine confidence and trading signal
        confidence = "HIGH" if max_prob > 0.7 else "MEDIUM" if max_prob > 0.6 else "LOW"
        should_trade = max_prob > 0.7  # Only trade when probability > 70%
        
        return {
            "prediction": prediction,
            "probability": float(max_prob),
            "confidence": confidence,
            "should_trade": should_trade,
            "probabilities": {
                "DOWN": float(probabilities[0]),
                "HOLD": float(probabilities[1]),
                "UP": float(probabilities[2])
            }
        }

class CatBoostSentimentAnalyzer:
    """
    CatBoost for Sentiment Impact Modeling
    - Reads news headlines and converts them to sentiment signals
    - Learns how news affects price movements
    - Boosts trading accuracy with sentiment analysis
    """
    
    def __init__(self, news_api_key: str):
        self.model = None
        self.news_client = NewsApiClient(api_key=news_api_key) if news_api_key else None
        self.is_trained = False
        self.sentiment_history = []
        
    def extract_news_sentiment(self, symbol: str, hours_back: int = 24) -> Dict:
        """Extract sentiment from recent news headlines"""
        if not self.news_client:
            # Fallback: simulate sentiment based on symbol
            base_sentiment = {
                'XAUUSD': 0.1,   # Slightly positive (safe haven)
                'EURUSD': -0.05, # Slightly negative
                'EURJPY': 0.02,  # Neutral
                'USDJPY': 0.08,  # Positive
                'NASDAQ': 0.15   # Positive (tech growth)
            }
            return {
                "sentiment_score": base_sentiment.get(symbol, 0.0),
                "article_count": 5,
                "keywords": ["market", "trading", symbol],
                "confidence": 0.6
            }
        
        try:
            # Search for recent news
            search_terms = {
                'XAUUSD': 'gold price OR precious metals',
                'EURUSD': 'EUR USD OR euro dollar',
                'EURJPY': 'EUR JPY OR euro yen',
                'USDJPY': 'USD JPY OR dollar yen',
                'NASDAQ': 'NASDAQ OR tech stocks'
            }
            
            query = search_terms.get(symbol, f"{symbol} trading")
            
            articles = self.news_client.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(hours=hours_back)).isoformat(),
                page_size=50
            )
            
            if not articles['articles']:
                return {"sentiment_score": 0.0, "article_count": 0, "confidence": 0.0}
            
            # Analyze sentiment
            total_sentiment = 0
            sentiment_scores = []
            keywords = []
            
            for article in articles['articles']:
                text = f"{article['title']} {article['description'] or ''}"
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                sentiment_scores.append(sentiment)
                total_sentiment += sentiment
                
                # Extract keywords
                words = text.lower().split()
                market_keywords = ['bull', 'bear', 'rise', 'fall', 'surge', 'crash', 'growth', 'decline']
                keywords.extend([word for word in words if word in market_keywords])
            
            avg_sentiment = total_sentiment / len(articles['articles'])
            confidence = min(len(articles['articles']) / 20, 1.0)  # Higher confidence with more articles
            
            return {
                "sentiment_score": avg_sentiment,
                "article_count": len(articles['articles']),
                "keywords": list(set(keywords)),
                "confidence": confidence,
                "individual_scores": sentiment_scores
            }
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return {"sentiment_score": 0.0, "article_count": 0, "confidence": 0.0}
    
    def train(self, training_data: List[Dict]) -> Dict:
        """Train CatBoost model to learn sentiment impact on price"""
        if len(training_data) < 50:
            return {"success": False, "error": "Insufficient sentiment training data"}
        
        # Prepare features: sentiment metrics -> price impact
        X_list = []
        y_list = []
        
        for data_point in training_data:
            sentiment_data = data_point.get('sentiment_data', {})
            features = [
                sentiment_data.get('sentiment_score', 0),
                sentiment_data.get('article_count', 0),
                sentiment_data.get('confidence', 0),
                len(sentiment_data.get('keywords', [])),
                data_point.get('market_data', {}).get('volume', 0) / 1000000,  # Volume in millions
                data_point.get('indicators', {}).get('RSI', 50) / 100,  # Normalized RSI
            ]
            X_list.append(features)
            
            # Target: price impact (next hour price change)
            price_impact = data_point.get('next_price_change', 0)
            if price_impact > 0.3:
                y_list.append(1)  # Positive impact
            elif price_impact < -0.3:
                y_list.append(-1)  # Negative impact
            else:
                y_list.append(0)  # Neutral impact
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Convert to CatBoost format
        y = y + 1  # Convert to 0, 1, 2 for classification
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train CatBoost
        self.model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            verbose=False,
            random_seed=42
        )
        
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True
        
        # Save model
        self.model.save_model('/app/backend/models/catboost_sentiment_model.cbm')
        
        return {
            "success": True,
            "accuracy": accuracy,
            "model_type": "CatBoost Sentiment Analyzer",
            "training_samples": len(training_data)
        }
    
    def analyze_sentiment_impact(self, symbol: str, market_data: Dict, indicators: Dict) -> Dict:
        """Analyze how sentiment affects price movement"""
        sentiment_data = self.extract_news_sentiment(symbol)
        
        if not self.is_trained or self.model is None:
            return {
                "sentiment_impact": "NEUTRAL",
                "confidence": 0.5,
                "raw_sentiment": sentiment_data['sentiment_score'],
                "news_volume": sentiment_data['article_count']
            }
        
        # Prepare features
        features = [
            sentiment_data['sentiment_score'],
            sentiment_data['article_count'],
            sentiment_data['confidence'],
            len(sentiment_data.get('keywords', [])),
            market_data.get('volume', 0) / 1000000,
            indicators.get('RSI', 50) / 100
        ]
        
        # Predict sentiment impact
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        impact_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        sentiment_impact = impact_map[prediction]
        confidence = float(np.max(probabilities))
        
        return {
            "sentiment_impact": sentiment_impact,
            "confidence": confidence,
            "raw_sentiment": sentiment_data['sentiment_score'],
            "news_volume": sentiment_data['article_count'],
            "keywords": sentiment_data.get('keywords', []),
            "boost_factor": confidence * (1 if sentiment_impact == "POSITIVE" else -1 if sentiment_impact == "NEGATIVE" else 0)
        }

class TPOTPatternDiscovery:
    """
    TPOT for Automatic Pattern Discovery
    - Automatically discovers hidden trading patterns
    - Creates optimal feature combinations
    - Finds the best ML pipeline for each symbol
    """
    
    def __init__(self):
        self.tpot = None
        self.best_pipeline = None
        self.is_trained = False
        self.pattern_features = []
        
    def discover_patterns(self, training_data: List[Dict], symbol: str) -> Dict:
        """Use TPOT to automatically discover trading patterns"""
        if len(training_data) < 200:
            return {"success": False, "error": "Insufficient data for pattern discovery"}
        
        # Prepare extended feature set for pattern discovery
        X_list = []
        y_list = []
        
        for i, data_point in enumerate(training_data[:-1]):  # Exclude last item for target
            market_data = data_point.get('market_data', {})
            indicators = data_point.get('indicators', {})
            
            # Extended feature set for pattern discovery
            features = [
                # Basic technical indicators
                indicators.get('RSI', 50),
                indicators.get('MACD', 0),
                indicators.get('MACD_signal', 0),
                indicators.get('STOCH_K', 50),
                indicators.get('STOCH_D', 50),
                indicators.get('ATR', 0),
                
                # Price features
                market_data.get('price', 0),
                market_data.get('change', 0),
                market_data.get('volume', 0),
                
                # Derived features
                indicators.get('RSI', 50) - 50,  # RSI deviation from neutral
                indicators.get('MACD', 0) - indicators.get('MACD_signal', 0),  # MACD histogram
                
                # Time features
                datetime.now().hour,
                datetime.now().weekday(),
                
                # Volatility features
                abs(market_data.get('change', 0)),  # Absolute change
                market_data.get('volume', 0) / 1000000,  # Volume in millions
            ]
            
            X_list.append(features)
            
            # Target: next period's significant price movement
            next_data = training_data[i + 1] if i + 1 < len(training_data) else training_data[i]
            next_change = next_data.get('market_data', {}).get('change', 0)
            
            # Binary classification: significant move or not
            significant_threshold = 1.0  # 1% change threshold
            y_list.append(1 if abs(next_change) > significant_threshold else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        if len(np.unique(y)) < 2:  # Need at least 2 classes
            return {"success": False, "error": "Insufficient class diversity"}
        
        # Use TPOT for automated ML pipeline discovery
        self.tpot = TPOTClassifier(
            generations=5,  # Reduced for faster training
            population_size=20,
            verbosity=0,
            random_state=42,
            config_dict='TPOT light',  # Faster configuration
            scoring='accuracy',
            cv=3,
            max_time_mins=5  # Limit training time
        )
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Fit TPOT
            self.tpot.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.tpot.score(X_test, y_test)
            self.best_pipeline = self.tpot.fitted_pipeline_
            self.is_trained = True
            
            # Save the best pipeline
            joblib.dump(self.best_pipeline, f'/app/backend/models/tpot_pattern_{symbol}.pkl')
            
            # Get feature importances if available
            feature_importance = {}
            if hasattr(self.best_pipeline, 'feature_importances_'):
                feature_names = [f'feature_{i}' for i in range(len(X[0]))]
                feature_importance = dict(zip(feature_names, self.best_pipeline.feature_importances_))
            
            return {
                "success": True,
                "accuracy": accuracy,
                "model_type": "TPOT Auto-Discovered Pipeline",
                "pipeline": str(self.best_pipeline),
                "feature_importance": feature_importance,
                "patterns_discovered": len(feature_importance)
            }
            
        except Exception as e:
            return {"success": False, "error": f"TPOT training failed: {str(e)}"}
    
    def predict_pattern(self, market_data: Dict, indicators: Dict) -> Dict:
        """Use discovered patterns to make predictions"""
        if not self.is_trained or self.best_pipeline is None:
            return {"pattern_detected": False, "confidence": 0.0}
        
        # Prepare features (same as training)
        features = [
            indicators.get('RSI', 50),
            indicators.get('MACD', 0),
            indicators.get('MACD_signal', 0),
            indicators.get('STOCH_K', 50),
            indicators.get('STOCH_D', 50),
            indicators.get('ATR', 0),
            market_data.get('price', 0),
            market_data.get('change', 0),
            market_data.get('volume', 0),
            indicators.get('RSI', 50) - 50,
            indicators.get('MACD', 0) - indicators.get('MACD_signal', 0),
            datetime.now().hour,
            datetime.now().weekday(),
            abs(market_data.get('change', 0)),
            market_data.get('volume', 0) / 1000000,
        ]
        
        X = np.array(features).reshape(1, -1)
        
        try:
            prediction = self.best_pipeline.predict(X)[0]
            
            # Get probability if available
            confidence = 0.5
            if hasattr(self.best_pipeline, 'predict_proba'):
                proba = self.best_pipeline.predict_proba(X)[0]
                confidence = np.max(proba)
            
            return {
                "pattern_detected": bool(prediction),
                "confidence": float(confidence),
                "signal_strength": "STRONG" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "WEAK"
            }
            
        except Exception as e:
            return {"pattern_detected": False, "confidence": 0.0, "error": str(e)}

class ProphetTimeSeriesForecaster:
    """
    Prophet for Time Series Forecasting
    - Forecasts future price movements using time series analysis
    - Identifies trends, seasonality, and holiday effects
    - Provides long-term price direction predictions
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.forecast_horizon = 24  # 24 hours ahead
        
    def prepare_time_series_data(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Prepare data for Prophet time series forecasting"""
        data = []
        
        for point in historical_data:
            data.append({
                'ds': point.get('timestamp', datetime.now()),
                'y': point.get('market_data', {}).get('price', 0)
            })
        
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        return df
    
    def train(self, historical_data: List[Dict]) -> Dict:
        """Train Prophet model for time series forecasting"""
        if len(historical_data) < 50:
            return {"success": False, "error": "Insufficient historical data for time series"}
        
        # Prepare time series data
        df = self.prepare_time_series_data(historical_data)
        
        if df.empty or df['y'].isna().all():
            return {"success": False, "error": "Invalid time series data"}
        
        # Initialize Prophet with market-specific parameters
        self.model = Prophet(
            changepoint_prior_scale=0.05,  # Lower for financial data
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Not relevant for short-term trading
            interval_width=0.8
        )
        
        # Add custom seasonalities for trading hours
        self.model.add_seasonality(
            name='trading_hours',
            period=1,  # Daily
            fourier_order=3
        )
        
        try:
            # Fit the model
            self.model.fit(df)
            self.is_trained = True
            
            # Generate forecast for evaluation
            future = self.model.make_future_dataframe(periods=24, freq='H')
            forecast = self.model.predict(future)
            
            # Calculate simple accuracy metric
            actual_prices = df['y'].tail(10).values
            forecasted_prices = forecast['yhat'].tail(10).values
            mape = np.mean(np.abs((actual_prices - forecasted_prices) / actual_prices)) * 100
            accuracy = max(0, 100 - mape) / 100
            
            # Save model
            with open('/app/backend/models/prophet_forecaster.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            return {
                "success": True,
                "accuracy": accuracy,
                "model_type": "Prophet Time Series Forecaster",
                "forecast_horizon": self.forecast_horizon,
                "training_periods": len(df)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Prophet training failed: {str(e)}"}
    
    def forecast(self, periods: int = None) -> Dict:
        """Generate price forecast"""
        if not self.is_trained or self.model is None:
            return {"forecast": [], "trend": "UNKNOWN", "confidence": 0.0}
        
        periods = periods or self.forecast_horizon
        
        try:
            # Generate future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq='H')
            forecast = self.model.predict(future)
            
            # Extract key information
            latest_forecast = forecast.tail(periods)
            current_price = forecast['yhat'].iloc[-periods-1] if len(forecast) > periods else forecast['yhat'].iloc[0]
            future_price = latest_forecast['yhat'].iloc[-1]
            
            # Determine trend
            price_change = (future_price - current_price) / current_price * 100
            if price_change > 1.0:
                trend = "STRONG_UP"
            elif price_change > 0.3:
                trend = "UP"
            elif price_change < -1.0:
                trend = "STRONG_DOWN"
            elif price_change < -0.3:
                trend = "DOWN"
            else:
                trend = "SIDEWAYS"
            
            # Calculate confidence based on uncertainty intervals
            uncertainty = np.mean(latest_forecast['yhat_upper'] - latest_forecast['yhat_lower'])
            relative_uncertainty = uncertainty / np.mean(latest_forecast['yhat'])
            confidence = max(0, 1 - relative_uncertainty)
            
            # Prepare forecast data
            forecast_data = []
            for _, row in latest_forecast.iterrows():
                forecast_data.append({
                    "timestamp": row['ds'].isoformat(),
                    "predicted_price": row['yhat'],
                    "lower_bound": row['yhat_lower'],
                    "upper_bound": row['yhat_upper']
                })
            
            return {
                "forecast": forecast_data,
                "trend": trend,
                "confidence": float(confidence),
                "price_change_percent": float(price_change),
                "forecast_horizon_hours": periods,
                "current_price": float(current_price),
                "target_price": float(future_price)
            }
            
        except Exception as e:
            return {"forecast": [], "trend": "UNKNOWN", "confidence": 0.0, "error": str(e)}

class EnsembleMLEngine:
    """
    Master ML Engine that coordinates all specialized models
    """
    
    def __init__(self, news_api_key: str = None):
        self.xgboost_predictor = XGBoostPricePredictor()
        self.catboost_sentiment = CatBoostSentimentAnalyzer(news_api_key)
        self.tpot_patterns = TPOTPatternDiscovery()
        self.prophet_forecaster = ProphetTimeSeriesForecaster()
        
        self.models_trained = {
            'xgboost': False,
            'catboost': False,
            'tpot': False,
            'prophet': False
        }
        
    async def train_all_models(self, training_data: List[Dict], symbol: str = "XAUUSD") -> Dict:
        """Train all specialized ML models"""
        results = {}
        
        # Train XGBoost Price Predictor
        print("Training XGBoost Price Predictor...")
        xgb_result = self.xgboost_predictor.train(training_data)
        results['xgboost'] = xgb_result
        self.models_trained['xgboost'] = xgb_result.get('success', False)
        
        # Train CatBoost Sentiment Analyzer
        print("Training CatBoost Sentiment Analyzer...")
        cat_result = self.catboost_sentiment.train(training_data)
        results['catboost'] = cat_result
        self.models_trained['catboost'] = cat_result.get('success', False)
        
        # Train TPOT Pattern Discovery
        print("Running TPOT Pattern Discovery...")
        tpot_result = self.tpot_patterns.discover_patterns(training_data, symbol)
        results['tpot'] = tpot_result
        self.models_trained['tpot'] = tpot_result.get('success', False)
        
        # Train Prophet Forecaster
        print("Training Prophet Time Series Forecaster...")
        prophet_result = self.prophet_forecaster.train(training_data)
        results['prophet'] = prophet_result
        self.models_trained['prophet'] = prophet_result.get('success', False)
        
        return {
            "overall_success": any(self.models_trained.values()),
            "models_trained": sum(self.models_trained.values()),
            "total_models": 4,
            "detailed_results": results
        }
    
    async def get_ensemble_prediction(self, symbol: str, market_data: Dict, 
                                    indicators: Dict) -> Dict:
        """Get combined prediction from all models"""
        predictions = {}
        
        # Get XGBoost price prediction
        if self.models_trained['xgboost']:
            xgb_pred = self.xgboost_predictor.predict(market_data, indicators)
            predictions['xgboost'] = xgb_pred
        
        # Get CatBoost sentiment analysis
        if self.models_trained['catboost']:
            sentiment_analysis = self.catboost_sentiment.analyze_sentiment_impact(
                symbol, market_data, indicators
            )
            predictions['catboost'] = sentiment_analysis
        
        # Get TPOT pattern detection
        if self.models_trained['tpot']:
            pattern_pred = self.tpot_patterns.predict_pattern(market_data, indicators)
            predictions['tpot'] = pattern_pred
        
        # Get Prophet forecast
        if self.models_trained['prophet']:
            forecast = self.prophet_forecaster.forecast(periods=6)  # 6 hours ahead
            predictions['prophet'] = forecast
        
        # Combine predictions into ensemble decision
        ensemble_decision = self._combine_predictions(predictions)
        
        return {
            "ensemble_decision": ensemble_decision,
            "individual_predictions": predictions,
            "models_active": self.models_trained
        }
    
    def _combine_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from all models into final ensemble decision"""
        # Initialize weights and scores
        weights = {'xgboost': 0.4, 'catboost': 0.2, 'tpot': 0.2, 'prophet': 0.2}
        
        # Collect signals
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0
        reasons = []
        
        # XGBoost signals (highest weight)
        if 'xgboost' in predictions:
            xgb = predictions['xgboost']
            if xgb.get('should_trade', False):
                if xgb['prediction'] == 'BUY':
                    buy_signals += weights['xgboost'] * xgb['probability']
                    reasons.append(f"XGBoost: {xgb['prediction']} ({xgb['probability']:.1%})")
                elif xgb['prediction'] == 'SELL':
                    sell_signals += weights['xgboost'] * xgb['probability']
                    reasons.append(f"XGBoost: {xgb['prediction']} ({xgb['probability']:.1%})")
            
            total_confidence += weights['xgboost'] * xgb.get('probability', 0.5)
        
        # CatBoost sentiment boost
        if 'catboost' in predictions:
            cat = predictions['catboost']
            boost = cat.get('boost_factor', 0) * weights['catboost']
            if cat['sentiment_impact'] == 'POSITIVE':
                buy_signals += boost
                reasons.append(f"Sentiment: {cat['sentiment_impact']} (News: {cat['news_volume']})")
            elif cat['sentiment_impact'] == 'NEGATIVE':
                sell_signals += abs(boost)
                reasons.append(f"Sentiment: {cat['sentiment_impact']} (News: {cat['news_volume']})")
            
            total_confidence += weights['catboost'] * cat.get('confidence', 0.5)
        
        # TPOT pattern signals
        if 'tpot' in predictions:
            tpot = predictions['tpot']
            if tpot.get('pattern_detected', False):
                pattern_strength = weights['tpot'] * tpot.get('confidence', 0.5)
                buy_signals += pattern_strength  # Assume positive pattern
                reasons.append(f"Pattern: {tpot.get('signal_strength', 'DETECTED')}")
            
            total_confidence += weights['tpot'] * tpot.get('confidence', 0.5)
        
        # Prophet trend signals
        if 'prophet' in predictions:
            prophet = predictions['prophet']
            trend = prophet.get('trend', 'UNKNOWN')
            prophet_conf = prophet.get('confidence', 0.5) * weights['prophet']
            
            if 'UP' in trend:
                buy_signals += prophet_conf
                reasons.append(f"Forecast: {trend}")
            elif 'DOWN' in trend:
                sell_signals += prophet_conf
                reasons.append(f"Forecast: {trend}")
            
            total_confidence += prophet_conf
        
        # Make final decision
        signal_diff = buy_signals - sell_signals
        
        if signal_diff > 0.3:  # Strong buy signal
            final_action = "BUY"
        elif signal_diff < -0.3:  # Strong sell signal
            final_action = "SELL"
        else:
            final_action = "HOLD"
        
        # Normalize confidence
        final_confidence = min(total_confidence, 1.0)
        
        # Determine if should trade (ensemble confidence > 70%)
        should_trade = final_confidence > 0.7 and final_action != "HOLD"
        
        return {
            "action": final_action,
            "confidence": final_confidence,
            "should_trade": should_trade,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "reasons": reasons,
            "ensemble_strength": "STRONG" if final_confidence > 0.8 else "MEDIUM" if final_confidence > 0.6 else "WEAK"
        }