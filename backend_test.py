import requests
import json
import sys
from datetime import datetime

class TradingBotAPITester:
    def __init__(self, base_url="https://bd0ec55c-d60f-495b-ac89-f54a07678a7d.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.symbols = ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY', 'NASDAQ']

    def run_test(self, name, method, endpoint, expected_status=200, data=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

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

    def test_backtest(self, symbol):
        """Test backtest endpoint for a symbol"""
        today = datetime.now().strftime('%Y-%m-%d')
        # 30 days ago
        start_date = datetime.now().replace(day=datetime.now().day - 30).strftime('%Y-%m-%d')
        
        success, response = self.run_test(
            f"Backtest for {symbol}",
            "POST",
            f"backtest/{symbol}?start_date={start_date}&end_date={today}",
            200
        )
        if success and response:
            print(f"  Total Trades: {response.get('total_trades')}")
            print(f"  Win Rate: {response.get('win_rate')}")
            print(f"  Total Profit: {response.get('total_profit')}")
        return success

    def test_export_trades(self):
        """Test export trades endpoint"""
        success, response = self.run_test(
            "Export Trades",
            "GET",
            "export-trades",
            200
        )
        if success and response:
            print(f"  Message: {response.get('message')}")
            print(f"  Total Trades: {response.get('total_trades')}")
            # Check if CSV data exists
            if isinstance(response, dict) and 'csv_data' in response:
                print(f"  CSV Data Length: {len(response.get('csv_data'))} characters")
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Trading Bot API Tests")
        print("=" * 50)
        
        # Health check
        self.test_health_check()
        
        # Test for each symbol
        for symbol in self.symbols:
            self.test_market_data(symbol)
            self.test_technical_indicators(symbol)
            self.test_trading_signal(symbol)
        
        # Other endpoints
        self.test_tweet_input()
        self.test_trading_history()
        self.test_model_status()
        self.test_train_models()
        self.test_backtest(self.symbols[0])  # Test backtest for first symbol
        self.test_export_trades()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Tests passed: {self.tests_passed}/{self.tests_run} ({(self.tests_passed/self.tests_run*100):.1f}%)")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = TradingBotAPITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)