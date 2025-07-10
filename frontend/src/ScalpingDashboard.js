import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Activity, TrendingUp, TrendingDown, Zap, Target, AlertCircle, 
  Brain, DollarSign, Award, AlertTriangle, CheckCircle, BarChart3,
  Gauge, Trophy, XCircle
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
const API = `${BACKEND_URL}/api`;

const ScalpingDashboard = () => {
  const [tradeAnalysis, setTradeAnalysis] = useState({});
  const [enhancedAnalysis, setEnhancedAnalysis] = useState({});
  const [botHealth, setBotHealth] = useState({});
  const [enhancedBotHealth, setEnhancedBotHealth] = useState({});
  const [loading, setLoading] = useState(false);

  const fetchTradeAnalysis = async () => {
    try {
      const response = await axios.get(`${API}/trade-analysis`);
      setTradeAnalysis(response.data);
    } catch (error) {
      console.error('Error fetching trade analysis:', error);
    }
  };

  const fetchEnhancedAnalysis = async () => {
    try {
      const response = await axios.get(`${API}/enhanced-trade-analysis`);
      setEnhancedAnalysis(response.data);
    } catch (error) {
      console.error('Error fetching enhanced analysis:', error);
    }
  };

  const fetchBotHealth = async () => {
    try {
      const response = await axios.get(`${API}/bot-health`);
      setBotHealth(response.data);
    } catch (error) {
      console.error('Error fetching bot health:', error);
    }
  };

  const fetchEnhancedBotHealth = async () => {
    try {
      const response = await axios.get(`${API}/enhanced-bot-health`);
      setEnhancedBotHealth(response.data);
    } catch (error) {
      console.error('Error fetching enhanced bot health:', error);
    }
  };

  useEffect(() => {
    const fetchAllData = async () => {
      await fetchTradeAnalysis();
      await fetchEnhancedAnalysis();
      await fetchBotHealth();
      await fetchEnhancedBotHealth();
    };

    fetchAllData();
    const interval = setInterval(fetchAllData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStrategyColor = (strategy) => {
    const stats = enhancedAnalysis.strategy_performance?.[strategy];
    if (!stats) return 'text-gray-600';
    
    if (stats.profit_factor > 1.5 && stats.win_rate > 60) return 'text-green-600';
    if (stats.profit_factor > 1.0 && stats.win_rate > 50) return 'text-blue-600';
    if (stats.profit_factor < 1.0 || stats.win_rate < 30) return 'text-red-600';
    return 'text-yellow-600';
  };

  const getStrategyBg = (strategy) => {
    const stats = enhancedAnalysis.strategy_performance?.[strategy];
    if (!stats) return 'bg-gray-50';
    
    if (stats.profit_factor > 1.5 && stats.win_rate > 60) return 'bg-green-50';
    if (stats.profit_factor > 1.0 && stats.win_rate > 50) return 'bg-blue-50';
    if (stats.profit_factor < 1.0 || stats.win_rate < 30) return 'bg-red-50';
    return 'bg-yellow-50';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Enhanced Scalping Summary with Profit Factor */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
          📈 Enhanced Scalping Summary
        </h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {enhancedAnalysis.profit_factor || '0.00'}
            </div>
            <div className="text-sm text-gray-600">Profit Factor</div>
            <div className={`text-xs ${
              enhancedAnalysis.profit_factor > 1.0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {enhancedAnalysis.profit_factor > 1.0 ? '✅ Profitable' : '❌ Losing'}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              +{enhancedAnalysis.total_profit || '0.00'}%
            </div>
            <div className="text-sm text-gray-600">Total Profit</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">
              -{enhancedAnalysis.total_loss || '0.00'}%
            </div>
            <div className="text-sm text-gray-600">Total Loss</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-800">
              {enhancedAnalysis.total_trades || 0}
            </div>
            <div className="text-sm text-gray-600">Total Trades</div>
          </div>
        </div>
      </div>

      {/* Top Gainers with Strategy + Reason */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
            <Trophy className="w-5 h-5 mr-2 text-green-600" />
            💡 Why Top Gainers Won
          </h3>
          <div className="space-y-3">
            {enhancedAnalysis.top_gainers?.slice(0, 3).map((trade, index) => (
              <div key={index} className="p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="font-semibold text-green-700">
                      {trade.symbol} +{trade.pips} pips
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      <span className="font-medium">Strategy:</span> {trade.strategy}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      <span className="font-medium">Reason:</span> {trade.reason}
                    </div>
                  </div>
                  <div className="text-green-600 font-bold">
                    +{trade.profit_pct}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Losers with Strategy + Reason */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
            <XCircle className="w-5 h-5 mr-2 text-red-600" />
            💡 Why Top Losers Failed
          </h3>
          <div className="space-y-3">
            {enhancedAnalysis.top_losers?.slice(0, 3).map((trade, index) => (
              <div key={index} className="p-3 bg-red-50 rounded-lg border border-red-200">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="font-semibold text-red-700">
                      {trade.symbol} {trade.pips} pips
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      <span className="font-medium">Strategy:</span> {trade.strategy}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      <span className="font-medium">Reason:</span> {trade.reason}
                    </div>
                  </div>
                  <div className="text-red-600 font-bold">
                    {trade.profit_pct}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Weak Strategy Alerts */}
      {enhancedAnalysis.weak_strategies?.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-bold text-red-800 mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2" />
            ⚠️ Underperforming Strategies
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {enhancedAnalysis.weak_strategies.map((strategy, index) => (
              <div key={index} className="bg-white p-4 rounded-lg border border-red-300">
                <div className="font-semibold text-red-700">{strategy.strategy}</div>
                <div className="text-sm text-gray-600 mt-1">
                  Win Rate: {strategy.win_rate}% | Pips: {strategy.total_pips} | Trades: {strategy.trades}
                </div>
                <div className="text-xs text-red-600 mt-1">{strategy.reason}</div>
                <div className="text-xs text-gray-500 mt-2">
                  💡 Consider disabling or improving this strategy
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strategy Performance Grid with Color Coding */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2 text-blue-600" />
          🎯 Strategy Performance (Color-Coded)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(enhancedAnalysis.strategy_performance || {}).map(([strategy, stats]) => (
            <div 
              key={strategy} 
              className={`p-4 rounded-lg border ${getStrategyBg(strategy)} ${
                stats.profit_factor > 1.5 && stats.win_rate > 60 ? 'border-green-300' :
                stats.profit_factor > 1.0 && stats.win_rate > 50 ? 'border-blue-300' :
                stats.profit_factor < 1.0 || stats.win_rate < 30 ? 'border-red-300' : 'border-yellow-300'
              }`}
            >
              <div className={`font-semibold ${getStrategyColor(strategy)}`}>
                {strategy}
              </div>
              <div className="mt-2 space-y-1">
                <div className="flex justify-between text-sm">
                  <span>Win Rate:</span>
                  <span className={getStrategyColor(strategy)}>{stats.win_rate}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Profit Factor:</span>
                  <span className={getStrategyColor(strategy)}>{stats.profit_factor}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Avg Pips:</span>
                  <span className={getStrategyColor(strategy)}>{stats.avg_pips}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Trades:</span>
                  <span className="text-gray-600">{stats.trades}</span>
                </div>
              </div>
              {stats.profit_factor > 1.5 && stats.win_rate > 60 && (
                <div className="mt-2 text-xs text-green-600 font-medium">
                  ⭐ High Performer
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ScalpingDashboard;