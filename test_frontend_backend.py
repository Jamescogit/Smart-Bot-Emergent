#!/usr/bin/env python3
"""Test if frontend can reach backend"""

import requests
import time

def test_frontend_to_backend():
    print("🧪 Testing Frontend-Backend Connection...")
    print("=" * 50)
    
    # Test the exact URL the frontend should be using
    frontend_backend_url = "http://localhost:8001"
    
    endpoints_to_test = [
        "/api/performance-metrics",
        "/api/model-status", 
        "/api/trading-history",
        "/api/market-data/XAUUSD"
    ]
    
    for endpoint in endpoints_to_test:
        url = f"{frontend_backend_url}{endpoint}"
        try:
            print(f"\n📡 Testing: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS - Status: {response.status_code}")
                
                # Show a sample of the data
                if isinstance(data, dict):
                    if len(data) > 3:
                        sample_keys = list(data.keys())[:3]
                        sample_data = {k: data[k] for k in sample_keys}
                        print(f"   Sample data: {sample_data}")
                    else:
                        print(f"   Data: {data}")
                elif isinstance(data, list):
                    print(f"   Data type: List with {len(data)} items")
                    if len(data) > 0:
                        print(f"   First item: {data[0] if len(str(data[0])) < 100 else str(data[0])[:100] + '...'}")
                else:
                    print(f"   Data: {data}")
                    
            else:
                print(f"❌ FAILED - Status: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ CONNECTION ERROR: {e}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Frontend-Backend connection test completed!")

if __name__ == "__main__":
    test_frontend_to_backend()