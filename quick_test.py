#!/usr/bin/env python3

import requests
import json

# Test specific endpoints that are failing
base_url = "https://70f00b2b-8b2d-489d-9db9-864872c61a38.preview.emergentagent.com/api"

def test_endpoint(name, endpoint):
    print(f"\n🔍 Testing {name}...")
    try:
        response = requests.get(f"{base_url}/{endpoint}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Success - Response type: {type(data)}")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())[:5]}...")  # First 5 keys
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                return False
        else:
            print(f"❌ HTTP Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

# Test the problematic endpoints
print("🚀 Quick Test of Problematic Endpoints")
print("=" * 50)

results = {}
results['market_data_xauusd'] = test_endpoint("Market Data XAUUSD", "market-data/XAUUSD")
results['market_data_eurusd'] = test_endpoint("Market Data EURUSD", "market-data/EURUSD")
results['bot_trading_status'] = test_endpoint("Bot Trading Status", "bot-trading-status")
results['scalping_rl_performance'] = test_endpoint("Scalping RL Performance", "scalping-rl-performance")

# Test endpoints that should exist but returned 404
results['learning_loop_status'] = test_endpoint("Learning Loop Status", "learning-loop-status")
results['reward_function_status'] = test_endpoint("Reward Function Status", "reward-function-status")
results['strategy_learning_status'] = test_endpoint("Strategy Learning Status", "strategy-learning-status")
results['multi_timeframe_analysis'] = test_endpoint("Multi-timeframe Analysis", "multi-timeframe-analysis/XAUUSD")

print("\n" + "=" * 50)
print("📊 QUICK TEST SUMMARY")
print("=" * 50)
for endpoint, success in results.items():
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{endpoint}: {status}")

passed = sum(results.values())
total = len(results)
print(f"\nOverall: {passed}/{total} ({passed/total*100:.1f}%)")