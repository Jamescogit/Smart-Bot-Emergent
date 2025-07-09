import requests
import json
import sys
from datetime import datetime

class TradingBotAPITester:
    def __init__(self, base_url="https://2a7364d9-f987-4163-81cb-725f1a053226.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.symbols = ['XAUUSD', 'EURUSD', 'USDJPY']  # Symbols to test as per requirements

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
        
        # Test other endpoints
        print("\n🔍 Testing Other API Endpoints")
        print("-" * 50)
        for symbol in self.symbols:
            self.test_market_data(symbol)
            self.test_technical_indicators(symbol)
            self.test_trading_signal(symbol)
        
        self.test_tweet_input()
        self.test_model_status()
        self.test_train_models()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Tests passed: {self.tests_passed}/{self.tests_run} ({(self.tests_passed/self.tests_run*100):.1f}%)")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = TradingBotAPITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)