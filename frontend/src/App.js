import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Brain, Target, AlertCircle, Download, Upload, Play, Pause, RefreshCw } from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const SYMBOLS = ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY', 'NASDAQ'];

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('XAUUSD');
  const [marketData, setMarketData] = useState({});
  const [technicalIndicators, setTechnicalIndicators] = useState({});
  const [tradingSignals, setTradingSignals] = useState({});
  const [tradingHistory, setTradingHistory] = useState([]);
  const [modelStatus, setModelStatus] = useState({});
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [tweetInput, setTweetInput] = useState('');
  const [tweets, setTweets] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [priceHistory, setPriceHistory] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState({});
  const [riskSettings, setRiskSettings] = useState({
    maxDrawdown: 10,
    positionSize: 1,
    riskPercentage: 2
  });
  
  const intervalRef = useRef(null);

  // Fetch market data for all symbols
  const fetchMarketData = async () => {
    try {
      const promises = SYMBOLS.map(symbol => 
        axios.get(`${API}/market-data/${symbol}`)
          .then(response => ({ symbol, data: response.data }))
          .catch(error => ({ symbol, data: null, error }))
      );
      
      const results = await Promise.all(promises);
      const newMarketData = {};
      
      results.forEach(({ symbol, data }) => {
        if (data) {
          newMarketData[symbol] = data;
        }
      });
      
      setMarketData(newMarketData);
      
      // Update price history for selected symbol
      if (newMarketData[selectedSymbol]) {
        setPriceHistory(prev => {
          const newHistory = [...prev, {
            time: new Date().toLocaleTimeString(),
            price: newMarketData[selectedSymbol].price,
            change: newMarketData[selectedSymbol].change
          }];
          return newHistory.slice(-50); // Keep last 50 points
        });
      }
    } catch (error) {
      console.error('Error fetching market data:', error);
    }
  };

  // Fetch technical indicators
  const fetchTechnicalIndicators = async (symbol) => {
    try {
      const response = await axios.get(`${API}/technical-indicators/${symbol}`);
      setTechnicalIndicators(prev => ({
        ...prev,
        [symbol]: response.data
      }));
    } catch (error) {
      console.error('Error fetching technical indicators:', error);
    }
  };

  // Fetch trading signals
  const fetchTradingSignals = async (symbol) => {
    try {
      const response = await axios.get(`${API}/trading-signal/${symbol}`);
      setTradingSignals(prev => ({
        ...prev,
        [symbol]: response.data
      }));
    } catch (error) {
      console.error('Error fetching trading signals:', error);
    }
  };

  // Fetch trading history
  const fetchTradingHistory = async () => {
    try {
      const response = await axios.get(`${API}/trading-history`);
      setTradingHistory(response.data);
      
      // Calculate performance metrics
      const totalTrades = response.data.length;
      const winningTrades = response.data.filter(trade => trade.profit > 0).length;
      const losingTrades = response.data.filter(trade => trade.profit < 0).length;
      const totalProfit = response.data.reduce((sum, trade) => sum + (trade.profit || 0), 0);
      const totalPips = response.data.reduce((sum, trade) => sum + (trade.pips || 0), 0);
      
      setPerformanceMetrics({
        totalTrades,
        winningTrades,
        losingTrades,
        winRate: totalTrades > 0 ? (winningTrades / totalTrades * 100).toFixed(1) : 0,
        totalProfit: totalProfit.toFixed(2),
        totalPips: totalPips.toFixed(1)
      });
    } catch (error) {
      console.error('Error fetching trading history:', error);
    }
  };

  // Fetch model status
  const fetchModelStatus = async () => {
    try {
      const response = await axios.get(`${API}/model-status`);
      setModelStatus(response.data);
    } catch (error) {
      console.error('Error fetching model status:', error);
    }
  };

  // Train ML models
  const trainModels = async () => {
    setIsTraining(true);
    try {
      const response = await axios.post(`${API}/train-models`);
      alert(`Models trained successfully! XGBoost: ${response.data.xgboost_accuracy.toFixed(3)}, CatBoost: ${response.data.catboost_accuracy.toFixed(3)}`);
      await fetchModelStatus();
    } catch (error) {
      console.error('Error training models:', error);
      alert('Error training models');
    } finally {
      setIsTraining(false);
    }
  };

  // Submit tweet
  const submitTweet = async () => {
    if (!tweetInput.trim()) return;
    
    try {
      const response = await axios.post(`${API}/tweet-input`, {
        tweet: tweetInput,
        symbol: selectedSymbol
      });
      
      setTweets(prev => [{
        id: Date.now(),
        tweet: tweetInput,
        symbol: selectedSymbol,
        sentiment: response.data.sentiment,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 9)]);
      
      setTweetInput('');
    } catch (error) {
      console.error('Error submitting tweet:', error);
    }
  };

  // Run backtest
  const runBacktest = async () => {
    try {
      const endDate = new Date().toISOString().split('T')[0];
      const startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
      
      const response = await axios.post(`${API}/backtest/${selectedSymbol}?start_date=${startDate}&end_date=${endDate}`);
      setBacktestResults(response.data);
    } catch (error) {
      console.error('Error running backtest:', error);
    }
  };

  // Export trades
  const exportTrades = async () => {
    try {
      const response = await axios.get(`${API}/export-trades`);
      const blob = new Blob([response.data.csv_data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trades_${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
    } catch (error) {
      console.error('Error exporting trades:', error);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    if (isAutoRefresh) {
      const interval = setInterval(() => {
        fetchMarketData();
        fetchTechnicalIndicators(selectedSymbol);
        fetchTradingSignals(selectedSymbol);
        fetchTradingHistory();
        fetchModelStatus();
      }, 3000); // Update every 3 seconds
      
      intervalRef.current = interval;
      return () => clearInterval(interval);
    }
  }, [isAutoRefresh, selectedSymbol]);

  // Initial data fetch
  useEffect(() => {
    fetchMarketData();
    fetchTechnicalIndicators(selectedSymbol);
    fetchTradingSignals(selectedSymbol);
    fetchTradingHistory();
    fetchModelStatus();
  }, [selectedSymbol]);

  // Get signal color
  const getSignalColor = (action) => {
    switch(action) {
      case 'BUY': return 'text-green-500';
      case 'SELL': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getSignalIcon = (action) => {
    switch(action) {
      case 'BUY': return <TrendingUp className="w-5 h-5" />;
      case 'SELL': return <TrendingDown className="w-5 h-5" />;
      default: return <Activity className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 p-4 border-b border-gray-700">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-blue-500" />
            Advanced Trading Bot
          </h1>
          
          <div className="flex items-center gap-4">
            <select 
              value={selectedSymbol} 
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="bg-gray-700 text-white p-2 rounded border border-gray-600"
            >
              {SYMBOLS.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
            
            <button
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
              className={`flex items-center gap-2 px-4 py-2 rounded ${isAutoRefresh ? 'bg-green-600' : 'bg-gray-600'}`}
            >
              {isAutoRefresh ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isAutoRefresh ? 'Pause' : 'Resume'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <div className="p-6 space-y-6">
        {/* Market Data Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {SYMBOLS.map(symbol => (
            <div key={symbol} className={`bg-gray-800 p-4 rounded-lg border-2 ${selectedSymbol === symbol ? 'border-blue-500' : 'border-gray-700'}`}>
              <h3 className="font-semibold text-lg mb-2">{symbol}</h3>
              {marketData[symbol] && (
                <>
                  <p className="text-2xl font-bold">${marketData[symbol].price?.toFixed(4)}</p>
                  <p className={`text-sm ${marketData[symbol].change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {marketData[symbol].change >= 0 ? '+' : ''}{marketData[symbol].change?.toFixed(2)}%
                  </p>
                  <p className="text-xs text-gray-400">Vol: {marketData[symbol].volume?.toLocaleString()}</p>
                </>
              )}
            </div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Technical Analysis */}
          <div className="space-y-6">
            {/* Technical Indicators */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Technical Indicators
              </h3>
              {technicalIndicators[selectedSymbol] && (
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span>RSI:</span>
                    <span className={`font-bold ${technicalIndicators[selectedSymbol].rsi < 30 ? 'text-green-500' : technicalIndicators[selectedSymbol].rsi > 70 ? 'text-red-500' : 'text-gray-300'}`}>
                      {technicalIndicators[selectedSymbol].rsi?.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>MACD:</span>
                    <span className="font-bold">{technicalIndicators[selectedSymbol].macd?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ATR:</span>
                    <span className="font-bold">{technicalIndicators[selectedSymbol].atr?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Stoch K:</span>
                    <span className="font-bold">{technicalIndicators[selectedSymbol].stoch_k?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Stoch D:</span>
                    <span className="font-bold">{technicalIndicators[selectedSymbol].stoch_d?.toFixed(2)}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Trading Signal */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Target className="w-5 h-5" />
                Trading Signal
              </h3>
              {tradingSignals[selectedSymbol] && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <span className={`text-2xl font-bold ${getSignalColor(tradingSignals[selectedSymbol].action)}`}>
                      {tradingSignals[selectedSymbol].action}
                    </span>
                    {getSignalIcon(tradingSignals[selectedSymbol].action)}
                  </div>
                  <div className="text-sm text-gray-400">
                    Confidence: {(tradingSignals[selectedSymbol].confidence * 100).toFixed(1)}%
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold">Reasons:</p>
                    {tradingSignals[selectedSymbol].reasons?.map((reason, idx) => (
                      <p key={idx} className="text-xs text-gray-300">• {reason}</p>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ML Model Status */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5" />
                ML Models
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span>XGBoost:</span>
                  <span className={`px-2 py-1 rounded text-xs ${modelStatus.xgboost_active ? 'bg-green-600' : 'bg-red-600'}`}>
                    {modelStatus.xgboost_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span>CatBoost:</span>
                  <span className={`px-2 py-1 rounded text-xs ${modelStatus.catboost_active ? 'bg-green-600' : 'bg-red-600'}`}>
                    {modelStatus.catboost_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Prophet:</span>
                  <span className={`px-2 py-1 rounded text-xs ${modelStatus.prophet_active ? 'bg-green-600' : 'bg-red-600'}`}>
                    {modelStatus.prophet_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RL Agent:</span>
                  <span className={`px-2 py-1 rounded text-xs ${modelStatus.rl_agent_active ? 'bg-green-600' : 'bg-red-600'}`}>
                    {modelStatus.rl_agent_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
              <button
                onClick={trainModels}
                disabled={isTraining}
                className="w-full mt-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 p-2 rounded flex items-center justify-center gap-2"
              >
                {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                {isTraining ? 'Training...' : 'Train Models'}
              </button>
            </div>
          </div>

          {/* Middle Column - Charts */}
          <div className="space-y-6">
            {/* Price Chart */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4">Price Chart - {selectedSymbol}</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={priceHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="time" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#3B82F6" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Tweet Input */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4">Expert Tweet Analysis</h3>
              <div className="space-y-4">
                <textarea
                  value={tweetInput}
                  onChange={(e) => setTweetInput(e.target.value)}
                  placeholder="Enter expert trader tweet for sentiment analysis..."
                  className="w-full p-3 bg-gray-700 border border-gray-600 rounded resize-none"
                  rows="3"
                />
                <button
                  onClick={submitTweet}
                  className="w-full bg-blue-600 hover:bg-blue-700 p-2 rounded"
                >
                  Analyze Tweet
                </button>
              </div>
              
              {/* Recent Tweets */}
              <div className="mt-4 space-y-2 max-h-40 overflow-y-auto">
                {tweets.map(tweet => (
                  <div key={tweet.id} className="bg-gray-700 p-3 rounded text-sm">
                    <div className="flex justify-between items-start">
                      <span className="font-semibold">{tweet.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        tweet.sentiment === 'BULLISH' ? 'bg-green-600' : 
                        tweet.sentiment === 'BEARISH' ? 'bg-red-600' : 'bg-gray-600'
                      }`}>
                        {tweet.sentiment}
                      </span>
                    </div>
                    <p className="text-gray-300 mt-1">{tweet.tweet}</p>
                    <p className="text-xs text-gray-400 mt-1">{tweet.timestamp}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Performance & Controls */}
          <div className="space-y-6">
            {/* Performance Metrics */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4">Performance Metrics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Total Trades:</span>
                  <span className="font-bold">{performanceMetrics.totalTrades || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Win Rate:</span>
                  <span className="font-bold text-green-500">{performanceMetrics.winRate || 0}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Total Profit:</span>
                  <span className={`font-bold ${parseFloat(performanceMetrics.totalProfit) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    ${performanceMetrics.totalProfit || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Total Pips:</span>
                  <span className={`font-bold ${parseFloat(performanceMetrics.totalPips) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {performanceMetrics.totalPips || 0}
                  </span>
                </div>
              </div>
            </div>

            {/* Risk Management */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                Risk Management
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1">Max Drawdown (%)</label>
                  <input
                    type="number"
                    value={riskSettings.maxDrawdown}
                    onChange={(e) => setRiskSettings(prev => ({ ...prev, maxDrawdown: e.target.value }))}
                    className="w-full p-2 bg-gray-700 border border-gray-600 rounded"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Position Size</label>
                  <input
                    type="number"
                    value={riskSettings.positionSize}
                    onChange={(e) => setRiskSettings(prev => ({ ...prev, positionSize: e.target.value }))}
                    className="w-full p-2 bg-gray-700 border border-gray-600 rounded"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Risk per Trade (%)</label>
                  <input
                    type="number"
                    value={riskSettings.riskPercentage}
                    onChange={(e) => setRiskSettings(prev => ({ ...prev, riskPercentage: e.target.value }))}
                    className="w-full p-2 bg-gray-700 border border-gray-600 rounded"
                  />
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="bg-gray-800 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-4">Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={runBacktest}
                  className="w-full bg-purple-600 hover:bg-purple-700 p-3 rounded flex items-center justify-center gap-2"
                >
                  <Activity className="w-4 h-4" />
                  Run Backtest
                </button>
                <button
                  onClick={exportTrades}
                  className="w-full bg-green-600 hover:bg-green-700 p-3 rounded flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Export Trades (CSV)
                </button>
              </div>
            </div>

            {/* Backtest Results */}
            {backtestResults && (
              <div className="bg-gray-800 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4">Backtest Results</h3>
                <div className="space-y-2 text-sm">
                  <p><strong>Symbol:</strong> {backtestResults.symbol}</p>
                  <p><strong>Total Trades:</strong> {backtestResults.total_trades}</p>
                  <p><strong>Win Rate:</strong> {backtestResults.win_rate.toFixed(1)}%</p>
                  <p><strong>Total Profit:</strong> ${backtestResults.total_profit.toFixed(2)}</p>
                  <p><strong>Sharpe Ratio:</strong> {backtestResults.sharpe_ratio.toFixed(2)}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Trading History Table */}
        <div className="bg-gray-800 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4">Recent Trading History</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left p-2">Symbol</th>
                  <th className="text-left p-2">Action</th>
                  <th className="text-left p-2">Entry Price</th>
                  <th className="text-left p-2">Exit Price</th>
                  <th className="text-left p-2">Profit</th>
                  <th className="text-left p-2">Pips</th>
                  <th className="text-left p-2">Status</th>
                  <th className="text-left p-2">Time</th>
                </tr>
              </thead>
              <tbody>
                {tradingHistory.slice(0, 10).map((trade, idx) => (
                  <tr key={idx} className="border-b border-gray-700">
                    <td className="p-2 font-medium">{trade.symbol}</td>
                    <td className={`p-2 ${getSignalColor(trade.action)}`}>{trade.action}</td>
                    <td className="p-2">${trade.entry_price?.toFixed(4)}</td>
                    <td className="p-2">${trade.exit_price?.toFixed(4) || '-'}</td>
                    <td className={`p-2 ${trade.profit >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      ${trade.profit?.toFixed(2) || '-'}
                    </td>
                    <td className={`p-2 ${trade.pips >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {trade.pips?.toFixed(1) || '-'}
                    </td>
                    <td className="p-2">
                      <span className={`px-2 py-1 rounded text-xs ${trade.is_closed ? 'bg-gray-600' : 'bg-blue-600'}`}>
                        {trade.is_closed ? 'Closed' : 'Open'}
                      </span>
                    </td>
                    <td className="p-2 text-gray-400">
                      {new Date(trade.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;