#!/usr/bin/env python3
"""Test persistence functionality"""

import requests
import json
import time

def test_persistence():
    BASE_URL = "http://localhost:8001/api"
    
    print("🧪 Testing Persistent Learning Functionality...")
    print("=" * 50)
    
    # Test 1: Check if we can trigger ML model training
    print("\n1. Testing ML model training...")
    try:
        response = requests.post(f"{BASE_URL}/train-models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ML training triggered successfully")
            print(f"   - Models trained: {data.get('models_trained', 0)}")
            print(f"   - Training active: {data.get('simulation_active', False)}")
        else:
            print(f"❌ ML training failed: {response.status_code}")
    except Exception as e:
        print(f"❌ ML training error: {e}")
    
    # Test 2: Check scalping RL agent performance
    print("\n2. Testing Scalping RL Agent...")
    try:
        response = requests.get(f"{BASE_URL}/scalping-rl-performance")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Scalping RL agent accessible")
            print(f"   - Trades made: {data.get('trades_made', 0)}")
            print(f"   - Memory size: {data.get('memory_size', 0)}")
            print(f"   - Epsilon: {data.get('epsilon', 0.0):.3f}")
        else:
            print(f"❌ Scalping RL agent failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Scalping RL agent error: {e}")
    
    # Test 3: Check performance metrics
    print("\n3. Testing Performance Metrics...")
    try:
        response = requests.get(f"{BASE_URL}/performance-metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Performance metrics accessible")
            print(f"   - Total trades: {data.get('total_trades', 0)}")
            print(f"   - Win rate: {data.get('win_rate', 0.0):.2f}%")
            print(f"   - Bot confidence: {data.get('bot_confidence', 0.0):.2f}")
        else:
            print(f"❌ Performance metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Performance metrics error: {e}")
    
    # Test 4: Check data directory for persistence files
    print("\n4. Checking persistence files...")
    import os
    data_dir = "/app/data"
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    
    if files:
        print(f"✅ Found {len(files)} persistence files:")
        for file in files:
            file_path = os.path.join(data_dir, file)
            size = os.path.getsize(file_path)
            print(f"   - {file}: {size} bytes")
    else:
        print("⚠️  No persistence files found yet (this is normal on first run)")
    
    print("\n" + "=" * 50)
    print("🎯 Persistence test completed!")

if __name__ == "__main__":
    test_persistence()