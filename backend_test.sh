#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base URL
BASE_URL="https://2a7364d9-f987-4163-81cb-725f1a053226.preview.emergentagent.com"
API_URL="${BASE_URL}/api"

# Test symbols
SYMBOLS=("XAUUSD" "EURUSD" "USDJPY")

# Test counters
TESTS_RUN=0
TESTS_PASSED=0

# Function to run a test
run_test() {
    local name=$1
    local endpoint=$2
    local expected_status=$3
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo -e "\n${BLUE}🔍 Testing ${name}...${NC}"
    
    # Run curl with -s (silent) and -o /dev/null (discard output) and -w to get status code
    # Also use -L to follow redirects
    status_code=$(curl -s -o /tmp/curl_response.txt -w "%{http_code}" -L "${API_URL}/${endpoint}")
    
    if [ "$status_code" -eq "$expected_status" ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo -e "${GREEN}✅ Passed - Status: ${status_code}${NC}"
        echo -e "Response: $(cat /tmp/curl_response.txt | head -c 300)..."
        return 0
    else
        echo -e "${RED}❌ Failed - Expected ${expected_status}, got ${status_code}${NC}"
        echo -e "Response: $(cat /tmp/curl_response.txt)"
        return 1
    fi
}

echo -e "${BLUE}🚀 Starting Trading Bot API Tests${NC}"
echo "=================================================="

# Test API health check
run_test "API Health Check" "" 200

# Test yfinance Integration and Candlestick Data API
echo -e "\n${BLUE}📊 Testing yfinance Integration and Candlestick Data API${NC}"
echo "--------------------------------------------------"

for symbol in "${SYMBOLS[@]}"; do
    if [ "$symbol" == "XAUUSD" ]; then
        for interval in "1m" "5m" "15m"; do
            run_test "Candlestick Data for ${symbol} (interval: ${interval})" "candlestick-data/${symbol}?interval=${interval}" 200
        done
    else
        run_test "Candlestick Data for ${symbol}" "candlestick-data/${symbol}?interval=1m" 200
    fi
done

# Test Scalping Signal API
echo -e "\n${BLUE}📈 Testing Scalping Signal API${NC}"
echo "--------------------------------------------------"

for symbol in "${SYMBOLS[@]}"; do
    run_test "Scalping Signal for ${symbol}" "scalping-signal/${symbol}" 200
done

# Test Scalping RL Agent Performance
echo -e "\n${BLUE}🤖 Testing Scalping RL Agent Performance${NC}"
echo "--------------------------------------------------"

run_test "Scalping RL Performance" "scalping-rl-performance" 200

# Test ObjectId Serialization Fix
echo -e "\n${BLUE}🔄 Testing ObjectId Serialization Fix${NC}"
echo "--------------------------------------------------"

run_test "Trading History" "trading-history" 200
run_test "Mock Trades" "mock-trades" 200

# Test Other API Endpoints
echo -e "\n${BLUE}🔍 Testing Other API Endpoints${NC}"
echo "--------------------------------------------------"

for symbol in "${SYMBOLS[@]}"; do
    run_test "Market Data for ${symbol}" "market-data/${symbol}" 200
    run_test "Technical Indicators for ${symbol}" "technical-indicators/${symbol}" 200
    run_test "Trading Signal for ${symbol}" "trading-signal/${symbol}" 200
done

# Print summary
echo -e "\n=================================================="
PERCENTAGE=$(( (TESTS_PASSED * 100) / TESTS_RUN ))
echo -e "${BLUE}📊 Tests passed: ${TESTS_PASSED}/${TESTS_RUN} (${PERCENTAGE}%)${NC}"

if [ $TESTS_PASSED -eq $TESTS_RUN ]; then
    exit 0
else
    exit 1
fi