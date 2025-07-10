import requests
import json
import sys
import os
import time
from datetime import datetime

class TradingBotAPITester:
    def __init__(self, base_url=None):
        # Use the production URL from frontend/.env
        if base_url is None:
            # Read the backend URL from frontend/.env
            try:
                with open('/app/frontend/.env', 'r') as f:
                    for line in f:
                        if line.startswith('REACT_APP_BACKEND_URL='):
                            base_url = line.split('=')[1].strip()
                            break
            except:
                base_url = "http://127.0.0.1:8001"  # fallback
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.symbols = ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY', 'GBPUSD']  # Updated symbols as per Twelve Data integration

    def run_test(self, name, method, endpoint, expected_status=200, data=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, allow_redirects=True)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, allow_redirects=True)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"Response: {response.json()}")
                except:
                    print(f"Response: {response.text}")
                return False, None

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, None

    def test_health_check(self):
        """Test API health check endpoint"""
        success, response = self.run_test(
            "API Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_market_data(self, symbol):
        """Test market data endpoint for a symbol"""
        success, response = self.run_test(
            f"Market Data for {symbol}",
            "GET",
            f"market-data/{symbol}",
            200
        )
        if success and response:
            print(f"  Symbol: {response.get('symbol')}")
            print(f"  Price: {response.get('price')}")
            print(f"  Change: {response.get('change')}")
        return success

    def test_technical_indicators(self, symbol):
        """Test technical indicators endpoint for a symbol"""
        success, response = self.run_test(
            f"Technical Indicators for {symbol}",
            "GET",
            f"technical-indicators/{symbol}",
            200
        )
        if success and response:
            print(f"  RSI: {response.get('rsi')}")
            print(f"  MACD: {response.get('macd')}")
            print(f"  Bollinger Bands: {response.get('bb_upper')}, {response.get('bb_middle')}, {response.get('bb_lower')}")
        return success

    def test_trading_signal(self, symbol):
        """Test trading signal endpoint for a symbol"""
        success, response = self.run_test(
            f"Trading Signal for {symbol}",
            "GET",
            f"trading-signal/{symbol}",
            200
        )
        if success and response:
            print(f"  Action: {response.get('action')}")
            print(f"  Confidence: {response.get('confidence')}")
            print(f"  Reasons: {response.get('reasons')}")
        return success

    def test_tweet_input(self):
        """Test tweet input endpoint"""
        tweet_data = {
            "tweet": "I think XAUUSD is going to break out to the upside soon! #bullish",
            "symbol": "XAUUSD"
        }
        success, response = self.run_test(
            "Tweet Input Analysis",
            "POST",
            "tweet-input",
            200,
            data=tweet_data
        )
        if success and response:
            print(f"  Sentiment: {response.get('sentiment')}")
            print(f"  Bull Score: {response.get('bull_score')}")
            print(f"  Bear Score: {response.get('bear_score')}")
        return success

    def test_trading_history(self):
        """Test trading history endpoint"""
        success, response = self.run_test(
            "Trading History",
            "GET",
            "trading-history",
            200
        )
        if success and response:
            print(f"  Number of trades: {len(response)}")
        return success

    def test_model_status(self):
        """Test model status endpoint"""
        success, response = self.run_test(
            "Model Status",
            "GET",
            "model-status",
            200
        )
        if success and response:
            print(f"  XGBoost Active: {response.get('xgboost_active')}")
            print(f"  CatBoost Active: {response.get('catboost_active')}")
            print(f"  Prophet Active: {response.get('prophet_active')}")
            print(f"  RL Agent Active: {response.get('rl_agent_active')}")
        return success

    def test_train_models(self):
        """Test train models endpoint"""
        success, response = self.run_test(
            "Train Models",
            "POST",
            "train-models",
            200
        )
        if success and response:
            print(f"  Message: {response.get('message')}")
            print(f"  XGBoost Accuracy: {response.get('xgboost_accuracy')}")
            print(f"  CatBoost Accuracy: {response.get('catboost_accuracy')}")
        return success

    def test_candlestick_data(self, symbol, interval='1m'):
        """Test candlestick data endpoint for a symbol with specific interval"""
        success, response = self.run_test(
            f"Candlestick Data for {symbol} (interval: {interval})",
            "GET",
            f"candlestick-data/{symbol}?interval={interval}",
            200
        )
        if success and response:
            print(f"  Symbol: {response.get('symbol')}")
            print(f"  Interval: {response.get('interval')}")
            data = response.get('data', [])
            print(f"  Number of candles: {len(data)}")
            if data:
                print(f"  Sample candle: {data[0]}")
                # Verify candle structure
                candle = data[0]
                required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                all_fields_present = all(field in candle for field in required_fields)
                if all_fields_present:
                    print("  ✅ Candle structure is valid")
                else:
                    print("  ❌ Candle structure is missing fields")
                    success = False
        return success

    def test_scalping_signal(self, symbol):
        """Test scalping signal endpoint for a symbol"""
        success, response = self.run_test(
            f"Scalping Signal for {symbol}",
            "GET",
            f"scalping-signal/{symbol}",
            200
        )
        if success and response:
            print(f"  Symbol: {response.get('symbol')}")
            print(f"  Action: {response.get('action')}")
            print(f"  Entry Price: {response.get('entry_price')}")
            print(f"  Stop Loss: {response.get('stop_loss')}")
            print(f"  Take Profit: {response.get('take_profit')}")
            print(f"  Confidence: {response.get('confidence')}")
            print(f"  Timeframe: {response.get('timeframe')}")
            
            # Verify signal structure
            required_fields = ['symbol', 'action', 'entry_price', 'stop_loss', 'take_profit', 'confidence', 'reasons', 'timeframe']
            all_fields_present = all(field in response for field in required_fields)
            if all_fields_present:
                print("  ✅ Signal structure is valid")
            else:
                print("  ❌ Signal structure is missing fields")
                success = False
        return success

    def test_scalping_rl_performance(self):
        """Test scalping RL performance endpoint"""
        success, response = self.run_test(
            "Scalping RL Performance",
            "GET",
            "scalping-rl-performance",
            200
        )
        if success and response:
            print(f"  Trades Made: {response.get('trades_made')}")
            print(f"  Winning Trades: {response.get('winning_trades')}")
            print(f"  Win Rate: {response.get('win_rate')}")
            print(f"  Total Pips: {response.get('total_pips')}")
            print(f"  Current Streak: {response.get('current_streak')}")
            
            # Verify performance metrics structure
            required_fields = ['trades_made', 'winning_trades', 'win_rate', 'total_pips', 'current_streak']
            all_fields_present = all(field in response for field in required_fields)
            if all_fields_present:
                print("  ✅ Performance metrics structure is valid")
            else:
                print("  ❌ Performance metrics structure is missing fields")
                success = False
        return success
        
    def test_create_sample_trades(self):
        """Test create sample trades endpoint for persistence"""
        success, response = self.run_test(
            "Create Sample Trades",
            "POST",
            "create-sample-trades",
            200
        )
        if success and response:
            print(f"  Message: {response.get('message')}")
            print(f"  Trades Created: {response.get('trades_created')}")
        return success
        
    def test_clear_sample_data(self):
        """Test clear sample data endpoint"""
        success, response = self.run_test(
            "Clear Sample Data",
            "POST",
            "clear-sample-data",
            200
        )
        if success and response:
            print(f"  Message: {response.get('message')}")
        return success
        
    def test_performance_metrics(self):
        """Test performance metrics endpoint"""
        success, response = self.run_test(
            "Performance Metrics",
            "GET",
            "performance-metrics",
            200
        )
        if success and response:
            print(f"  Win Rate: {response.get('win_rate')}%")
            print(f"  Total Trades: {response.get('total_trades')}")
            print(f"  Total Pips: {response.get('total_pips')}")
            
            # Verify metrics structure
            required_fields = ['win_rate', 'total_trades', 'total_pips', 'profit_factor']
            all_fields_present = all(field in response for field in required_fields)
            if all_fields_present:
                print("  ✅ Performance metrics structure is valid")
            else:
                print("  ❌ Performance metrics structure is missing fields")
                success = False
        return success
        
    def test_persistence_files(self):
        """Test if persistence files are created in the data directory"""
        print("\n🔍 Testing Persistence Files...")
        data_dir = "/app/data"
        
        # List of expected persistence files
        expected_files = [
            "rl_agent.pkl",
            "scalping_rl_agent.pkl",
            "ml_models.pkl",
            "feature_history.pkl",
            "price_history.pkl",
            "trading_history.json",
            "model_performance.json"
        ]
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Failed - Data directory {data_dir} does not exist")
            return False
            
        print(f"✅ Data directory {data_dir} exists")
        
        # Check for each expected file
        all_files_exist = True
        for file_name in expected_files:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_name} exists (size: {file_size} bytes)")
                
                # Check if file is not empty
                if file_size == 0:
                    print(f"⚠️ Warning: {file_name} is empty")
            else:
                print(f"❌ {file_name} does not exist")
                all_files_exist = False
                
        self.tests_run += 1
        if all_files_exist:
            self.tests_passed += 1
            
        return all_files_exist
        
    def test_periodic_save(self):
        """Test that periodic save is working"""
        print("\n🔍 Testing Periodic Save Functionality...")
        
        # First, get the current modification time of a persistence file
        rl_agent_file = "/app/data/scalping_rl_agent.pkl"
        
        if not os.path.exists(rl_agent_file):
            print(f"❌ Failed - {rl_agent_file} does not exist")
            return False
            
        initial_mtime = os.path.getmtime(rl_agent_file)
        initial_time = datetime.fromtimestamp(initial_mtime)
        print(f"Initial modification time: {initial_time}")
        
        # Train the model to trigger a save
        self.test_train_models()
        
        # Check if the file was updated
        new_mtime = os.path.getmtime(rl_agent_file)
        new_time = datetime.fromtimestamp(new_mtime)
        print(f"New modification time: {new_time}")
        
        # Check if the file was updated
        was_updated = new_mtime > initial_mtime
        
        self.tests_run += 1
        if was_updated:
            self.tests_passed += 1
            print("✅ Persistence file was updated after training")
        else:
            print("❌ Persistence file was not updated after training")
            
        return was_updated

    def test_bot_trading_status(self):
        """Test bot trading status endpoint - Fix ObjectId serialization errors"""
        success, response = self.run_test(
            "Bot Trading Status",
            "GET",
            "bot-trading-status",
            200
        )
        if success and response:
            print(f"  Bot Active: {response.get('bot_active')}")
            print(f"  Current Balance: {response.get('current_balance')}")
            print(f"  Total Trades: {response.get('total_trades')}")
            print(f"  Win Rate: {response.get('win_rate')}")
            print(f"  Learning Progress: {response.get('learning_progress')}")
            
            # Verify no ObjectId serialization errors
            try:
                json.dumps(response)  # This will fail if there are ObjectId issues
                print("  ✅ No ObjectId serialization errors")
            except TypeError as e:
                if "ObjectId" in str(e):
                    print("  ❌ ObjectId serialization error detected")
                    success = False
                else:
                    print(f"  ❌ Other serialization error: {e}")
                    success = False
        return success

    def test_manual_trade(self, symbol):
        """Test manual trade endpoint - Test autonomous trading logic"""
        trade_data = {
            "action": "BUY",
            "quantity": 0.1,
            "stop_loss": 0.01,
            "take_profit": 0.02
        }
        success, response = self.run_test(
            f"Manual Trade for {symbol}",
            "POST",
            f"manual-trade/{symbol}",
            200,
            data=trade_data
        )
        if success and response:
            print(f"  Trade ID: {response.get('trade_id')}")
            print(f"  Symbol: {response.get('symbol')}")
            print(f"  Action: {response.get('action')}")
            print(f"  Entry Price: {response.get('entry_price')}")
            print(f"  Status: {response.get('status')}")
            
            # Verify trade structure
            required_fields = ['trade_id', 'symbol', 'action', 'entry_price', 'status']
            all_fields_present = all(field in response for field in required_fields)
            if all_fields_present:
                print("  ✅ Trade structure is valid")
            else:
                print("  ❌ Trade structure is missing fields")
                success = False
        return success

    def test_twelve_data_integration(self, symbol):
        """Test Twelve Data API integration with rate limiting"""
        success, response = self.run_test(
            f"Twelve Data Integration for {symbol}",
            "GET",
            f"market-data/{symbol}",
            200
        )
        if success and response:
            print(f"  Symbol: {response.get('symbol')}")
            print(f"  Price: {response.get('price')}")
            print(f"  Change: {response.get('change')}")
            print(f"  Timestamp: {response.get('timestamp')}")
            
            # Verify this is real-time data (not cached/fallback)
            if 'timestamp' in response:
                timestamp = response['timestamp']
                if isinstance(timestamp, str):
                    try:
                        parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_diff = datetime.now() - parsed_time.replace(tzinfo=None)
                        if time_diff.total_seconds() < 300:  # Within 5 minutes
                            print("  ✅ Real-time data confirmed")
                        else:
                            print("  ⚠️ Data may be cached or delayed")
                    except:
                        print("  ⚠️ Could not parse timestamp")
            
            # Verify price is realistic for the symbol
            price = response.get('price', 0)
            if symbol == 'XAUUSD' and 2500 < price < 3000:
                print("  ✅ XAUUSD price is realistic")
            elif symbol == 'EURUSD' and 0.9 < price < 1.2:
                print("  ✅ EURUSD price is realistic")
            elif symbol in ['EURJPY', 'USDJPY'] and 100 < price < 200:
                print("  ✅ JPY pair price is realistic")
            elif symbol == 'GBPUSD' and 1.0 < price < 1.5:
                print("  ✅ GBPUSD price is realistic")
            else:
                print(f"  ⚠️ Price {price} may not be realistic for {symbol}")
        return success

    def test_continuous_learning_loop(self):
        """Test continuous learning loop status"""
        success, response = self.run_test(
            "Continuous Learning Loop Status",
            "GET",
            "learning-loop-status",
            200
        )
        if success and response:
            print(f"  Loop Active: {response.get('loop_active')}")
            print(f"  Last Decision: {response.get('last_decision_time')}")
            print(f"  Next Decision: {response.get('next_decision_time')}")
            print(f"  Decisions Made: {response.get('decisions_made')}")
            print(f"  Learning Rate: {response.get('learning_rate')}")
        return success

    def test_enhanced_reward_function(self):
        """Test enhanced reward function with currency-specific optimization"""
        success, response = self.run_test(
            "Enhanced Reward Function Status",
            "GET",
            "reward-function-status",
            200
        )
        if success and response:
            print(f"  Currency Multipliers: {response.get('currency_multipliers')}")
            print(f"  Reward Components: {response.get('reward_components')}")
            print(f"  Last Reward: {response.get('last_reward')}")
        return success

    def test_strategy_learning(self):
        """Test strategy learning and curriculum learning"""
        success, response = self.run_test(
            "Strategy Learning Status",
            "GET",
            "strategy-learning-status",
            200
        )
        if success and response:
            print(f"  Current Curriculum Stage: {response.get('curriculum_stage')}")
            print(f"  Stage Name: {response.get('stage_name')}")
            print(f"  Strategy Performance: {response.get('strategy_performance')}")
            print(f"  Currency Performance: {response.get('currency_performance')}")
        return success

    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis in enhanced state preparation"""
        success, response = self.run_test(
            "Multi-timeframe Analysis",
            "GET",
            "multi-timeframe-analysis/XAUUSD",
            200
        )
        if success and response:
            print(f"  Timeframes: {response.get('timeframes')}")
            print(f"  Momentum Analysis: {response.get('momentum_analysis')}")
            print(f"  Session Features: {response.get('session_features')}")
        return success

    def test_mock_trades(self):
        """Test mock trades endpoint for ObjectId serialization"""
        success, response = self.run_test(
            "Mock Trades",
            "GET",
            "mock-trades",
            200
        )
        if success:
            if isinstance(response, list):
                print(f"  Number of mock trades: {len(response)}")
                if response:
                    print(f"  Sample trade: {response[0]}")
            else:
                print(f"  Response: {response}")
        return success

    def test_rate_limiting(self):
        """Test Twelve Data API rate limiting (8 calls per minute)"""
        print("\n🔍 Testing Rate Limiting (8 calls per minute)...")
        
        # Make 9 rapid calls to test rate limiting
        calls_made = 0
        rate_limited = False
        
        for i in range(9):
            success, response = self.run_test(
                f"Rate Limit Test Call {i+1}",
                "GET",
                "market-data/XAUUSD",
                200
            )
            calls_made += 1
            
            if not success:
                # Check if this was due to rate limiting
                if "rate limit" in str(response).lower() or calls_made >= 8:
                    rate_limited = True
                    print(f"  ✅ Rate limiting activated after {calls_made} calls")
                    break
            
            time.sleep(0.5)  # Small delay between calls
        
        self.tests_run += 1
        if rate_limited or calls_made <= 8:
            self.tests_passed += 1
            return True
        else:
            print(f"  ❌ Rate limiting not working - made {calls_made} calls without limit")
            return False

    def test_autonomous_trading_decisions(self):
        """Test that bot makes autonomous trading decisions every 2 minutes"""
        print("\n🔍 Testing Autonomous Trading Decisions...")
        
        # Get initial status
        success1, response1 = self.run_test(
            "Initial Bot Status",
            "GET",
            "bot-trading-status",
            200
        )
        
        if not success1:
            return False
        
        initial_decisions = response1.get('decisions_made', 0)
        print(f"  Initial decisions made: {initial_decisions}")
        
        # Wait for 2.5 minutes to see if a new decision is made
        print("  Waiting 2.5 minutes for autonomous decision...")
        time.sleep(150)  # 2.5 minutes
        
        # Get updated status
        success2, response2 = self.run_test(
            "Updated Bot Status",
            "GET",
            "bot-trading-status",
            200
        )
        
        if not success2:
            return False
        
        new_decisions = response2.get('decisions_made', 0)
        print(f"  New decisions made: {new_decisions}")
        
        self.tests_run += 1
        if new_decisions > initial_decisions:
            self.tests_passed += 1
            print("  ✅ Autonomous trading decision detected")
            return True
        else:
            print("  ❌ No autonomous trading decision in 2.5 minutes")
            return False

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Trading Bot API Tests")
        print("=" * 50)
        
        # Health check
        self.test_health_check()
        
        # Test yfinance integration via candlestick data endpoint
        print("\n📊 Testing yfinance Integration and Candlestick Data API")
        print("-" * 50)
        for symbol in self.symbols:
            # Test with different intervals for XAUUSD
            if symbol == 'XAUUSD':
                for interval in ['1m', '5m', '15m']:
                    self.test_candlestick_data(symbol, interval)
            else:
                self.test_candlestick_data(symbol)
        
        # Test scalping signal API
        print("\n📈 Testing Scalping Signal API")
        print("-" * 50)
        for symbol in self.symbols:
            self.test_scalping_signal(symbol)
        
        # Test scalping RL agent performance
        print("\n🤖 Testing Scalping RL Agent Performance")
        print("-" * 50)
        self.test_scalping_rl_performance()
        
        # Test ObjectId serialization fix
        print("\n🔄 Testing ObjectId Serialization Fix")
        print("-" * 50)
        self.test_trading_history()
        self.test_mock_trades()
        
        # Test persistence implementation
        print("\n💾 Testing Persistent Learning Implementation")
        print("-" * 50)
        self.test_persistence_files()
        self.test_train_models()
        self.test_periodic_save()
        self.test_create_sample_trades()
        self.test_performance_metrics()
        
        # Test other endpoints
        print("\n🔍 Testing Other API Endpoints")
        print("-" * 50)
        for symbol in self.symbols:
            self.test_market_data(symbol)
            self.test_technical_indicators(symbol)
            self.test_trading_signal(symbol)
        
        self.test_tweet_input()
        self.test_model_status()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Tests passed: {self.tests_passed}/{self.tests_run} ({(self.tests_passed/self.tests_run*100):.1f}%)")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = TradingBotAPITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)