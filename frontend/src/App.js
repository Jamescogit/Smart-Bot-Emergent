import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Brain, Target, AlertCircle, Download, Upload, Play, Pause, RefreshCw, Award, Zap, Users, DollarSign, TrendingDownIcon, BarChart3, PieChart as PieChartIcon } from 'lucide-react';
import CandlestickChart from './CandlestickChart';
import ChartErrorBoundary from './ChartErrorBoundary';
import EnhancedTradeHistory from './EnhancedTradeHistory';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
const API = `${BACKEND_URL}/api`;

const SYMBOLS = ['XAUUSD', 'EURUSD', 'EURJPY', 'USDJPY', 'NASDAQ'];

// Colors for Cash-style theme
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

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
  
  // New state for training simulation
  const [trainingStatus, setTrainingStatus] = useState({});
  const [trainingMetrics, setTrainingMetrics] = useState({});
  const [liveTradeFeeds, setLiveTradeFeeds] = useState([]);
  const [modelComparison, setModelComparison] = useState({});
  const [mockTrades, setMockTrades] = useState([]);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);
  
  // New state for candlestick data and scalping
  const [candlestickData, setCandlestickData] = useState({});
  const [scalpingSignals, setScalpingSignals] = useState({});
  const [scalpingRLPerformance, setScalpingRLPerformance] = useState({});
  const [chartInterval, setChartInterval] = useState('1m');
  const [autoRefreshInterval, setAutoRefreshInterval] = useState(30); // 30 seconds
  const [currentView, setCurrentView] = useState('dashboard'); // dashboard, trade-history
  
  const intervalRef = useRef(null);
  const trainingIntervalRef = useRef(null);

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
    } catch (error) {
      console.error('Error fetching trading history:', error);
    }
  };

  // Fetch performance metrics
  const fetchPerformanceMetrics = async () => {
    try {
      const response = await axios.get(`${API}/performance-metrics`);
      setPerformanceMetrics(response.data);
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
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

  // Fetch training status
  const fetchTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API}/training-status`);
      setTrainingStatus(response.data);
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  // Fetch training metrics
  const fetchTrainingMetrics = async () => {
    try {
      const response = await axios.get(`${API}/training-metrics`);
      setTrainingMetrics(response.data);
    } catch (error) {
      console.error('Error fetching training metrics:', error);
    }
  };

  // Fetch live trade feeds
  const fetchLiveTradeFeeds = async () => {
    try {
      const response = await axios.get(`${API}/live-trade-feed`);
      setLiveTradeFeeds(response.data);
    } catch (error) {
      console.error('Error fetching live trade feeds:', error);
    }
  };

  // Fetch model comparison
  const fetchModelComparison = async () => {
    try {
      const response = await axios.get(`${API}/model-comparison`);
      setModelComparison(response.data);
    } catch (error) {
      console.error('Error fetching model comparison:', error);
    }
  };

  // Fetch mock trades
  const fetchMockTrades = async () => {
    try {
      const response = await axios.get(`${API}/mock-trades`);
      setMockTrades(response.data.trades || []);
    } catch (error) {
      console.error('Error fetching mock trades:', error);
    }
  };

  // Fetch candlestick data
  const fetchCandlestickData = async (symbol) => {
    try {
      const response = await axios.get(`${API}/candlestick-data/${symbol}`, {
        params: { period: '1d', interval: chartInterval }
      });
      setCandlestickData(prev => ({
        ...prev,
        [symbol]: response.data.data
      }));
    } catch (error) {
      console.error('Error fetching candlestick data:', error);
    }
  };

  // Fetch scalping signals
  const fetchScalpingSignals = async (symbol) => {
    try {
      const response = await axios.get(`${API}/scalping-signal/${symbol}`);
      setScalpingSignals(prev => ({
        ...prev,
        [symbol]: response.data
      }));
    } catch (error) {
      console.error('Error fetching scalping signals:', error);
    }
  };

  // Fetch scalping RL performance
  const fetchScalpingRLPerformance = async () => {
    try {
      const response = await axios.get(`${API}/scalping-rl-performance`);
      setScalpingRLPerformance(response.data);
    } catch (error) {
      console.error('Error fetching scalping RL performance:', error);
    }
  };

  // Train ML models
  const trainModels = async () => {
    setIsTraining(true);
    setShowTrainingPanel(true);
    try {
      const response = await axios.post(`${API}/train-models`);
      console.log('Training started:', response.data);
      
      // Start training monitoring
      startTrainingMonitoring();
      
    } catch (error) {
      console.error('Error training models:', error);
      alert('Error training models');
    } finally {
      setIsTraining(false);
    }
  };

  // Start training monitoring
  const startTrainingMonitoring = () => {
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current);
    }
    
    trainingIntervalRef.current = setInterval(async () => {
      await fetchTrainingStatus();
      await fetchTrainingMetrics();
      await fetchLiveTradeFeeds();
      await fetchModelComparison();
      await fetchMockTrades();
      await fetchModelStatus();
      
      // Stop monitoring if training is complete
      if (trainingStatus.stage === 'completed' || trainingStatus.stage === 'stopped') {
        clearInterval(trainingIntervalRef.current);
      }
    }, 1000); // Update every second during training
  };

  // Stop training
  const stopTraining = async () => {
    try {
      await axios.post(`${API}/stop-training`);
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    } catch (error) {
      console.error('Error stopping training:', error);
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
        
        // Fetch candlestick data for scalping
        fetchCandlestickData(selectedSymbol);
        fetchScalpingSignals(selectedSymbol);
        fetchScalpingRLPerformance();
        
        // Fetch training data if training is active
        if (trainingStatus.is_training) {
          fetchTrainingStatus();
          fetchTrainingMetrics();
          fetchLiveTradeFeeds();
        }
      }, autoRefreshInterval * 1000); // Use configurable interval (default 30 seconds)
      
      intervalRef.current = interval;
      return () => clearInterval(interval);
    }
  }, [isAutoRefresh, selectedSymbol, trainingStatus.is_training, autoRefreshInterval, chartInterval]);

  // Initial data fetch
  useEffect(() => {
    fetchMarketData();
    fetchTechnicalIndicators(selectedSymbol);
    fetchTradingSignals(selectedSymbol);
    fetchTradingHistory();
    fetchModelStatus();
    fetchTrainingStatus();
    fetchTrainingMetrics();
    fetchModelComparison();
    
    // Fetch candlestick data for scalping
    fetchCandlestickData(selectedSymbol);
    fetchScalpingSignals(selectedSymbol);
    fetchScalpingRLPerformance();
  }, [selectedSymbol, chartInterval]);

  // Cleanup intervals
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current);
      }
    };
  }, []);

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
    <div className="min-h-screen bg-gray-100">
      {/* Cash-style Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="flex justify-between items-center p-4">
          <div className="flex items-center space-x-4">
            <div className="text-2xl font-bold text-gray-800">
              <Brain className="w-8 h-8 text-blue-600 inline-block mr-2" />
              Cash Trading Bot
            </div>
            <div className="text-sm text-gray-500">Advanced ML Trading System</div>
          </div>
          
          <div className="flex items-center space-x-4">
            <select 
              value={selectedSymbol} 
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="bg-white border border-gray-300 text-gray-700 py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {SYMBOLS.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
            
            <button
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                isAutoRefresh 
                  ? 'bg-green-500 text-white' 
                  : 'bg-gray-200 text-gray-700'
              }`}
            >
              {isAutoRefresh ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isAutoRefresh ? 'Live' : 'Paused'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="px-6">
          <nav className="flex space-x-8">
            <button
              onClick={() => setCurrentView('dashboard')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                currentView === 'dashboard'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              📊 Trading Dashboard
            </button>
            <button
              onClick={() => setCurrentView('trade-history')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                currentView === 'trade-history'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              📈 Enhanced Trade History
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {currentView === 'dashboard' ? (
          <div>
            {/* Statistics Cards Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
              <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Trades</p>
                    <p className="text-2xl font-bold text-gray-900">{performanceMetrics.totalTrades || 0}</p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-full">
                    <BarChart3 className="w-6 h-6 text-blue-600" />
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow p-6 border-l-4 border-green-500">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Win Rate</p>
                    <p className="text-2xl font-bold text-gray-900">{performanceMetrics.winRate || 0}%</p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-full">
                    <Award className="w-6 h-6 text-green-600" />
                  </div>
                </div>
              </div>
          
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Profit</p>
                <p className="text-2xl font-bold text-gray-900">${performanceMetrics.totalProfit || 0}</p>
              </div>
              <div className="p-3 bg-purple-100 rounded-full">
                <DollarSign className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-orange-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Bot Confidence</p>
                <p className="text-2xl font-bold text-gray-900">{performanceMetrics.botConfidence || 0}%</p>
              </div>
              <div className="p-3 bg-orange-100 rounded-full">
                <Brain className="w-6 h-6 text-orange-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Additional Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-red-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Number of Losses</p>
                <p className="text-2xl font-bold text-gray-900">{performanceMetrics.totalLosses || 0}</p>
              </div>
              <div className="p-3 bg-red-100 rounded-full">
                <TrendingDownIcon className="w-6 h-6 text-red-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6 border-l-4 border-gray-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Pips</p>
                <p className="text-2xl font-bold text-gray-900">{performanceMetrics.totalPips || 0}</p>
              </div>
              <div className="p-3 bg-gray-100 rounded-full">
                <Target className="w-6 h-6 text-gray-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Market Data Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
          {SYMBOLS.map(symbol => (
            <div key={symbol} className={`bg-white rounded-lg shadow p-4 cursor-pointer transition-all duration-200 ${
              selectedSymbol === symbol ? 'ring-2 ring-blue-500' : ''
            }`} onClick={() => setSelectedSymbol(symbol)}>
              <h3 className="font-semibold text-gray-700 mb-2">{symbol}</h3>
              {marketData[symbol] && (
                <>
                  <p className="text-xl font-bold text-gray-900">${marketData[symbol].price?.toFixed(4)}</p>
                  <p className={`text-sm ${marketData[symbol].change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {marketData[symbol].change >= 0 ? '+' : ''}{marketData[symbol].change?.toFixed(2)}%
                  </p>
                  <p className="text-xs text-gray-500">Vol: {marketData[symbol].volume?.toLocaleString()}</p>
                </>
              )}
            </div>
          ))}
        </div>

        {/* Training Panel */}
        {showTrainingPanel && (
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800">ML Training Center</h3>
                <button
                  onClick={() => setShowTrainingPanel(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-6">
              {/* Training Status */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Training Progress</span>
                  <span className="text-sm text-gray-500">
                    {trainingStatus.stage || 'Ready'} - Epoch {trainingStatus.epoch || 0}/{trainingStatus.total_epochs || 0}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${trainingStatus.progress_percentage || 0}%` }}
                  ></div>
                </div>
              </div>

              {/* Model Performance Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                {Object.entries(trainingMetrics.model_metrics || {}).map(([modelName, metrics]) => (
                  <div key={modelName} className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-700 mb-2 capitalize">{modelName}</h4>
                    <div className="space-y-1">
                      <p className="text-sm">
                        <span className="text-gray-600">Trades:</span> 
                        <span className="font-bold ml-1">{metrics.trades || 0}</span>
                      </p>
                      <p className="text-sm">
                        <span className="text-gray-600">Win Rate:</span> 
                        <span className={`font-bold ml-1 ${metrics.win_rate >= 60 ? 'text-green-600' : metrics.win_rate >= 40 ? 'text-yellow-600' : 'text-red-600'}`}>
                          {(metrics.win_rate || 0).toFixed(1)}%
                        </span>
                      </p>
                      <p className="text-sm">
                        <span className="text-gray-600">Pips:</span> 
                        <span className={`font-bold ml-1 ${metrics.total_pips >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {(metrics.total_pips || 0).toFixed(1)}
                        </span>
                      </p>
                      <p className="text-sm">
                        <span className="text-gray-600">Streak:</span> 
                        <span className={`font-bold ml-1 ${metrics.current_streak >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {metrics.current_streak || 0}
                        </span>
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Live Trade Feed */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-700 mb-3">Live Training Trades</h4>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {liveTradeFeeds.slice(0, 10).map((trade, idx) => (
                    <div key={idx} className="flex items-center justify-between text-sm bg-white p-2 rounded">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-700">{trade.model}</span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          trade.action === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {trade.action}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`font-bold ${trade.pips >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {trade.pips > 0 ? '+' : ''}{trade.pips.toFixed(1)} pips
                        </span>
                        <span className="text-gray-500">{new Date(trade.timestamp).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Training Controls */}
              <div className="flex justify-center space-x-4 mt-6">
                {trainingStatus.is_training ? (
                  <button
                    onClick={stopTraining}
                    className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600 transition-colors"
                  >
                    Stop Training
                  </button>
                ) : (
                  <button
                    onClick={trainModels}
                    disabled={isTraining}
                    className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-400 transition-colors"
                  >
                    {isTraining ? 'Starting...' : 'Start Training'}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Technical Indicators */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Technical Indicators</h3>
              </div>
              <div className="p-4">
                {technicalIndicators[selectedSymbol] && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">RSI:</span>
                      <span className={`font-bold ${technicalIndicators[selectedSymbol].rsi < 30 ? 'text-green-600' : technicalIndicators[selectedSymbol].rsi > 70 ? 'text-red-600' : 'text-gray-700'}`}>
                        {technicalIndicators[selectedSymbol].rsi?.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">MACD:</span>
                      <span className="font-bold text-gray-700">{technicalIndicators[selectedSymbol].macd?.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">ATR:</span>
                      <span className="font-bold text-gray-700">{technicalIndicators[selectedSymbol].atr?.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Stoch K:</span>
                      <span className="font-bold text-gray-700">{technicalIndicators[selectedSymbol].stoch_k?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Stoch D:</span>
                      <span className="font-bold text-gray-700">{technicalIndicators[selectedSymbol].stoch_d?.toFixed(2)}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Trading Signal */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Trading Signal</h3>
              </div>
              <div className="p-4">
                {tradingSignals[selectedSymbol] && (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <span className={`text-xl font-bold ${getSignalColor(tradingSignals[selectedSymbol].action)}`}>
                        {tradingSignals[selectedSymbol].action}
                      </span>
                      {getSignalIcon(tradingSignals[selectedSymbol].action)}
                    </div>
                    <div className="text-sm text-gray-600">
                      Confidence: {(tradingSignals[selectedSymbol].confidence * 100).toFixed(1)}%
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm font-medium text-gray-700">Reasons:</p>
                      {tradingSignals[selectedSymbol].reasons?.map((reason, idx) => (
                        <p key={idx} className="text-xs text-gray-600">• {reason}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ML Model Status */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">ML Models</h3>
              </div>
              <div className="p-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">XGBoost:</span>
                    <span className={`px-2 py-1 rounded text-xs ${modelStatus.xgboost_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {modelStatus.xgboost_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">CatBoost:</span>
                    <span className={`px-2 py-1 rounded text-xs ${modelStatus.catboost_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {modelStatus.catboost_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Prophet:</span>
                    <span className={`px-2 py-1 rounded text-xs ${modelStatus.prophet_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {modelStatus.prophet_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">TPOT:</span>
                    <span className={`px-2 py-1 rounded text-xs ${modelStatus.tpot_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {modelStatus.tpot_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">RL Agent:</span>
                    <span className={`px-2 py-1 rounded text-xs ${modelStatus.rl_agent_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {modelStatus.rl_agent_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => setShowTrainingPanel(true)}
                  className="w-full mt-4 bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors"
                >
                  Open Training Center
                </button>
              </div>
            </div>
          </div>

          {/* Middle Column */}
          <div className="space-y-6">
            {/* Candlestick Chart */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-semibold text-gray-800">Scalping Chart - {selectedSymbol}</h3>
                  <div className="flex items-center space-x-4">
                    <select 
                      value={chartInterval} 
                      onChange={(e) => setChartInterval(e.target.value)}
                      className="px-3 py-1 border border-gray-300 rounded text-sm"
                    >
                      <option value="1m">1 Min</option>
                      <option value="5m">5 Min</option>
                      <option value="15m">15 Min</option>
                    </select>
                    <select 
                      value={autoRefreshInterval} 
                      onChange={(e) => setAutoRefreshInterval(parseInt(e.target.value))}
                      className="px-3 py-1 border border-gray-300 rounded text-sm"
                    >
                      <option value="10">10s</option>
                      <option value="30">30s</option>
                      <option value="60">1m</option>
                    </select>
                    <button
                      onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                      className={`px-3 py-1 rounded text-sm ${isAutoRefresh ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}
                    >
                      {isAutoRefresh ? 'Auto ON' : 'Auto OFF'}
                    </button>
                  </div>
                </div>
              </div>
              <div className="p-4">
                <ChartErrorBoundary
                  data={candlestickData[selectedSymbol] || []}
                  symbol={selectedSymbol}
                  height={350}
                >
                  <CandlestickChart 
                    data={candlestickData[selectedSymbol] || []}
                    symbol={selectedSymbol}
                    height={350}
                  />
                </ChartErrorBoundary>
              </div>
            </div>

            {/* Scalping Signals */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Scalping Signals - {selectedSymbol}</h3>
              </div>
              <div className="p-4">
                {scalpingSignals[selectedSymbol] && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Signal:</span>
                      <span className={`px-3 py-1 rounded font-semibold ${
                        scalpingSignals[selectedSymbol].action === 'BUY' ? 'bg-green-100 text-green-800' : 
                        scalpingSignals[selectedSymbol].action === 'SELL' ? 'bg-red-100 text-red-800' : 
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {scalpingSignals[selectedSymbol].action}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Entry Price:</span>
                      <span className="font-semibold">{scalpingSignals[selectedSymbol].entry_price?.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Stop Loss:</span>
                      <span className="font-semibold text-red-600">{scalpingSignals[selectedSymbol].stop_loss?.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Take Profit:</span>
                      <span className="font-semibold text-green-600">{scalpingSignals[selectedSymbol].take_profit?.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Confidence:</span>
                      <span className="font-semibold">{(scalpingSignals[selectedSymbol].confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Timeframe:</span>
                      <span className="font-semibold">{scalpingSignals[selectedSymbol].timeframe}</span>
                    </div>
                    <div className="mt-4">
                      <h4 className="font-semibold text-gray-700 mb-2">Reasons:</h4>
                      <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                        {scalpingSignals[selectedSymbol].reasons?.map((reason, index) => (
                          <li key={index}>{reason}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Tweet Analysis */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Expert Tweet Analysis</h3>
              </div>
              <div className="p-4">
                <div className="space-y-4">
                  <textarea
                    value={tweetInput}
                    onChange={(e) => setTweetInput(e.target.value)}
                    placeholder="Enter expert trader tweet for sentiment analysis..."
                    className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="3"
                  />
                  <button
                    onClick={submitTweet}
                    className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition-colors"
                  >
                    Analyze Tweet
                  </button>
                </div>
                
                {/* Recent Tweets */}
                <div className="mt-4 space-y-2 max-h-40 overflow-y-auto">
                  {tweets.map(tweet => (
                    <div key={tweet.id} className="bg-gray-50 p-3 rounded-lg text-sm">
                      <div className="flex justify-between items-start">
                        <span className="font-medium text-gray-700">{tweet.symbol}</span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          tweet.sentiment === 'BULLISH' ? 'bg-green-100 text-green-800' : 
                          tweet.sentiment === 'BEARISH' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {tweet.sentiment}
                        </span>
                      </div>
                      <p className="text-gray-600 mt-1">{tweet.tweet}</p>
                      <p className="text-xs text-gray-500 mt-1">{tweet.timestamp}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Risk Management */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Risk Management</h3>
              </div>
              <div className="p-4">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Max Drawdown (%)</label>
                    <input
                      type="number"
                      value={riskSettings.maxDrawdown}
                      onChange={(e) => setRiskSettings(prev => ({ ...prev, maxDrawdown: e.target.value }))}
                      className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Position Size</label>
                    <input
                      type="number"
                      value={riskSettings.positionSize}
                      onChange={(e) => setRiskSettings(prev => ({ ...prev, positionSize: e.target.value }))}
                      className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Risk per Trade (%)</label>
                    <input
                      type="number"
                      value={riskSettings.riskPercentage}
                      onChange={(e) => setRiskSettings(prev => ({ ...prev, riskPercentage: e.target.value }))}
                      className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Actions</h3>
              </div>
              <div className="p-4">
                <div className="space-y-3">
                  <button
                    onClick={runBacktest}
                    className="w-full bg-purple-500 text-white py-2 px-4 rounded-lg hover:bg-purple-600 transition-colors"
                  >
                    Run Backtest
                  </button>
                  <button
                    onClick={exportTrades}
                    className="w-full bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition-colors"
                  >
                    Export Trades (CSV)
                  </button>
                </div>
              </div>
            </div>

            {/* Scalping RL Performance */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-gray-800">Scalping RL Agent</h3>
              </div>
              <div className="p-4">
                {scalpingRLPerformance.metrics && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Status:</span>
                      <span className={`px-2 py-1 rounded text-xs ${scalpingRLPerformance.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        {scalpingRLPerformance.status}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Trades:</span>
                      <span className="font-semibold">{scalpingRLPerformance.metrics.trades_made}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Win Rate:</span>
                      <span className={`font-semibold ${scalpingRLPerformance.metrics.win_rate >= 60 ? 'text-green-600' : scalpingRLPerformance.metrics.win_rate >= 40 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {scalpingRLPerformance.metrics.win_rate?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Pips:</span>
                      <span className={`font-semibold ${scalpingRLPerformance.metrics.total_pips >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {scalpingRLPerformance.metrics.total_pips?.toFixed(1)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Streak:</span>
                      <span className={`font-semibold ${scalpingRLPerformance.metrics.current_streak >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {scalpingRLPerformance.metrics.current_streak}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Learning Rate:</span>
                      <span className="font-semibold">{scalpingRLPerformance.metrics.epsilon?.toFixed(3)}</span>
                    </div>
                  </div>
                )}
                {scalpingRLPerformance.error && (
                  <p className="text-sm text-red-600">{scalpingRLPerformance.error}</p>
                )}
              </div>
            </div>

            {/* Backtest Results */}
            {backtestResults && (
              <div className="bg-white rounded-lg shadow">
                <div className="p-4 border-b">
                  <h3 className="text-lg font-semibold text-gray-800">Backtest Results</h3>
                </div>
                <div className="p-4">
                  <div className="space-y-2 text-sm">
                    <p><span className="font-medium">Symbol:</span> {backtestResults.symbol}</p>
                    <p><span className="font-medium">Total Trades:</span> {backtestResults.total_trades}</p>
                    <p><span className="font-medium">Win Rate:</span> {backtestResults.win_rate.toFixed(1)}%</p>
                    <p><span className="font-medium">Total Profit:</span> ${backtestResults.total_profit.toFixed(2)}</p>
                    <p><span className="font-medium">Sharpe Ratio:</span> {backtestResults.sharpe_ratio.toFixed(2)}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Trading History Table */}
        <div className="bg-white rounded-lg shadow mt-6">
          <div className="p-4 border-b">
            <h3 className="text-lg font-semibold text-gray-800">Recent Trading History</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left p-4 font-medium text-gray-700">Symbol</th>
                  <th className="text-left p-4 font-medium text-gray-700">Action</th>
                  <th className="text-left p-4 font-medium text-gray-700">Entry Price</th>
                  <th className="text-left p-4 font-medium text-gray-700">Exit Price</th>
                  <th className="text-left p-4 font-medium text-gray-700">Profit</th>
                  <th className="text-left p-4 font-medium text-gray-700">Pips</th>
                  <th className="text-left p-4 font-medium text-gray-700">Bot Strategy</th>
                  <th className="text-left p-4 font-medium text-gray-700">Status</th>
                  <th className="text-left p-4 font-medium text-gray-700">Time</th>
                </tr>
              </thead>
              <tbody>
                {tradingHistory.slice(0, 10).map((trade, idx) => (
                  <tr key={idx} className="border-b hover:bg-gray-50">
                    <td className="p-4 font-medium text-gray-900">{trade.symbol}</td>
                    <td className={`p-4 ${getSignalColor(trade.action)}`}>{trade.action}</td>
                    <td className="p-4 text-gray-700">${trade.entry_price?.toFixed(4)}</td>
                    <td className="p-4 text-gray-700">${trade.exit_price?.toFixed(4) || '-'}</td>
                    <td className={`p-4 ${trade.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${trade.profit?.toFixed(2) || '-'}
                    </td>
                    <td className={`p-4 ${trade.pips >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {trade.pips?.toFixed(1) || '-'}
                    </td>
                    <td className="p-4">
                      <span className="px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                        {trade.bot_strategy || 'Default'}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded text-xs ${trade.is_closed ? 'bg-gray-100 text-gray-800' : 'bg-blue-100 text-blue-800'}`}>
                        {trade.is_closed ? 'Closed' : 'Open'}
                      </span>
                    </td>
                    <td className="p-4 text-gray-500">
                      {new Date(trade.timestamp).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
        ) : currentView === 'trade-history' ? (
          <EnhancedTradeHistory />
        ) : null}
      </div>
    </div>
  );
}

export default App;