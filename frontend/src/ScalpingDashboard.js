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

      {/* Enhanced RL Agent Health Panel */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-600" />
          🧠 RL Agent Evolution & Health
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* RL Metrics */}
          <div className="space-y-3">
            <h3 className="font-semibold text-gray-800 flex items-center">
              <Gauge className="w-4 h-4 mr-2" />
              Learning Metrics
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Last Reward:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.rl_agent_metrics?.last_reward > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {enhancedBotHealth.rl_agent_metrics?.last_reward > 0 ? '+' : ''}
                  {enhancedBotHealth.rl_agent_metrics?.last_reward || '0.0'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Current Streak:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.rl_agent_metrics?.current_streak > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {enhancedBotHealth.rl_agent_metrics?.current_streak || 0} 
                  {enhancedBotHealth.rl_agent_metrics?.current_streak > 0 ? ' wins' : ' losses'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Exploration:</span>
                <span className="text-sm font-medium text-blue-600">
                  {enhancedBotHealth.rl_agent_metrics?.epsilon_progress || 0}% → 
                  {enhancedBotHealth.rl_agent_metrics?.confidence_level || 0}%
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Learning Phase:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.rl_agent_metrics?.learning_phase === 'Expert' ? 'text-green-600' :
                  enhancedBotHealth.rl_agent_metrics?.learning_phase === 'Exploiting' ? 'text-blue-600' : 'text-yellow-600'
                }`}>
                  {enhancedBotHealth.rl_agent_metrics?.learning_phase || 'Unknown'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Account Health */}
          <div className="space-y-3">
            <h3 className="font-semibold text-gray-800 flex items-center">
              <DollarSign className="w-4 h-4 mr-2" />
              Account Health
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Balance:</span>
                <span className="text-sm font-medium text-gray-800">
                  ${enhancedBotHealth.account_health?.current_balance || '200.00'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">P&L:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.account_health?.balance_change_pct > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {enhancedBotHealth.account_health?.balance_change_pct > 0 ? '+' : ''}
                  {enhancedBotHealth.account_health?.balance_change_pct || '0.00'}%
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Risk per Trade:</span>
                <span className="text-sm font-medium text-blue-600">
                  {enhancedBotHealth.account_health?.risk_per_trade || '1.5%'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Max Risk:</span>
                <span className="text-sm font-medium text-gray-600">
                  {enhancedBotHealth.account_health?.max_risk_amount || '$3.00'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Status:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.account_health?.health_status === 'Excellent' ? 'text-green-600' :
                  enhancedBotHealth.account_health?.health_status === 'Good' ? 'text-blue-600' :
                  enhancedBotHealth.account_health?.health_status === 'Caution' ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {enhancedBotHealth.account_health?.health_status || 'Unknown'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Learning Progress */}
          <div className="space-y-3">
            <h3 className="font-semibold text-gray-800 flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              Learning Progress
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Models Active:</span>
                <span className="text-sm font-medium text-green-600">
                  {enhancedBotHealth.total_active_models || 0}/4
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Data Quality:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.learning_progress?.data_quality === 'Excellent' ? 'text-green-600' :
                  enhancedBotHealth.learning_progress?.data_quality === 'Good' ? 'text-blue-600' : 'text-yellow-600'
                }`}>
                  {enhancedBotHealth.learning_progress?.data_quality || 'Unknown'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Memory Size:</span>
                <span className="text-sm font-medium text-gray-600">
                  {enhancedBotHealth.rl_agent_metrics?.memory_size || 0} experiences
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Total Pips:</span>
                <span className={`text-sm font-medium ${
                  enhancedBotHealth.rl_agent_metrics?.total_pips > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {enhancedBotHealth.rl_agent_metrics?.total_pips > 0 ? '+' : ''}
                  {enhancedBotHealth.rl_agent_metrics?.total_pips || '0.0'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTradeAnalysis();
    fetchBotHealth();
    
    const interval = setInterval(() => {
      fetchTradeAnalysis();
      fetchBotHealth();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Scalping Performance */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">🎯 Top Scalping Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Top Gainers */}
          <div>
            <h4 className="font-medium text-green-600 mb-3 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2" />
              Top Gainers
            </h4>
            <div className="space-y-2">
              {tradeAnalysis.top_gainers?.slice(0, 3).map((trade, idx) => (
                <div key={idx} className="flex justify-between items-center p-2 bg-green-50 rounded">
                  <span className="text-sm font-medium">{trade.symbol}</span>
                  <span className="text-sm text-green-600">+{trade.pips?.toFixed(1)} pips</span>
                  <span className="text-xs text-gray-500">{trade.bot_strategy}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Top Losers */}
          <div>
            <h4 className="font-medium text-red-600 mb-3 flex items-center">
              <TrendingDown className="w-4 h-4 mr-2" />
              Top Losers
            </h4>
            <div className="space-y-2">
              {tradeAnalysis.top_losers?.slice(0, 3).map((trade, idx) => (
                <div key={idx} className="flex justify-between items-center p-2 bg-red-50 rounded">
                  <span className="text-sm font-medium">{trade.symbol}</span>
                  <span className="text-sm text-red-600">{trade.pips?.toFixed(1)} pips</span>
                  <span className="text-xs text-gray-500">{trade.bot_strategy}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Strategy Performance */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">📊 Strategy Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Best Strategies */}
          <div>
            <h4 className="font-medium text-blue-600 mb-3">Best Strategies</h4>
            <div className="space-y-2">
              {tradeAnalysis.best_strategies?.map((strategy, idx) => (
                <div key={idx} className="flex justify-between items-center p-2 bg-blue-50 rounded">
                  <span className="text-sm font-medium">{strategy.strategy}</span>
                  <div className="text-right">
                    <div className="text-sm text-blue-600">+{strategy.avg_pips} pips</div>
                    <div className="text-xs text-gray-500">{strategy.win_rate}% win rate</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Worst Strategies */}
          <div>
            <h4 className="font-medium text-orange-600 mb-3">Needs Improvement</h4>
            <div className="space-y-2">
              {tradeAnalysis.worst_strategies?.map((strategy, idx) => (
                <div key={idx} className="flex justify-between items-center p-2 bg-orange-50 rounded">
                  <span className="text-sm font-medium">{strategy.strategy}</span>
                  <div className="text-right">
                    <div className="text-sm text-orange-600">{strategy.avg_pips} pips</div>
                    <div className="text-xs text-gray-500">{strategy.win_rate}% win rate</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bot Health Panel */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">🤖 Bot Health Panel</h3>
        
        {/* Scalping RL Agent */}
        <div className="mb-6">
          <h4 className="font-medium text-gray-700 mb-3">Scalping RL Agent</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Epsilon (Exploration)</div>
              <div className="text-lg font-semibold">{(botHealth.scalping_rl_agent?.epsilon * 100).toFixed(1)}%</div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div 
                  className="bg-blue-600 h-2 rounded-full" 
                  style={{ width: `${(botHealth.scalping_rl_agent?.epsilon || 0) * 100}%` }}
                ></div>
              </div>
            </div>
            
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Memory Size</div>
              <div className="text-lg font-semibold">{botHealth.scalping_rl_agent?.memory_size || 0}</div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div 
                  className="bg-green-600 h-2 rounded-full" 
                  style={{ width: `${Math.min((botHealth.scalping_rl_agent?.memory_size || 0) / 20, 1) * 100}%` }}
                ></div>
              </div>
            </div>
            
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Learning Rate</div>
              <div className="text-lg font-semibold">{botHealth.scalping_rl_agent?.learning_rate || 0}</div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div 
                  className="bg-purple-600 h-2 rounded-full" 
                  style={{ width: `${(botHealth.scalping_rl_agent?.learning_rate || 0) * 1000}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Activation Panel */}
        <div>
          <h4 className="font-medium text-gray-700 mb-3">Model Activation</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {['xgboost', 'catboost', 'prophet', 'tpot'].map((model) => (
              <button
                key={model}
                onClick={() => {
                  // Toggle model activation
                  const isActive = botHealth.system_health?.models_loaded > 0;
                  if (isActive) {
                    deactivateModel(model);
                  } else {
                    activateModel(model);
                  }
                }}
                disabled={loading}
                className={`p-2 rounded text-sm font-medium transition-colors ${
                  botHealth.system_health?.models_loaded > 0
                    ? 'bg-green-100 text-green-800 hover:bg-green-200'
                    : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                }`}
              >
                {model.toUpperCase()}
                <div className="text-xs mt-1">
                  {botHealth.system_health?.models_loaded > 0 ? 'Active' : 'Inactive'}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">📈 Scalping Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{tradeAnalysis.summary?.total_trades || 0}</div>
            <div className="text-sm text-gray-600">Total Scalping Trades</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{tradeAnalysis.summary?.avg_pips || 0}</div>
            <div className="text-sm text-gray-600">Average Pips Per Trade</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">${tradeAnalysis.summary?.avg_profit || 0}</div>
            <div className="text-sm text-gray-600">Average Profit Per Trade</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScalpingDashboard;