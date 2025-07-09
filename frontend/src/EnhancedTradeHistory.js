import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Download, RefreshCw, Filter, Eye, EyeOff } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
const API = `${BACKEND_URL}/api`;

const EnhancedTradeHistory = () => {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('time');
  const [sortOrder, setSortOrder] = useState('desc');
  const [visibleColumns, setVisibleColumns] = useState({
    time: true,
    symbol: true,
    action: true,
    entryPrice: true,
    exitPrice: true,
    pipsGained: true,
    percentagePL: true,
    confidence: true,
    decisionFactors: true,
    tradeType: true,
    forecastTrend: true,
    newsSentiment: true,
    tweetBias: true,
    botStrategy: true,
    mlDecision: true,
    riskLevel: true,
    exitReason: true
  });

  const fetchEnhancedTrades = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/enhanced-trading-history`);
      setTrades(response.data.trades || []);
    } catch (error) {
      console.error('Error fetching enhanced trades:', error);
      // Create sample trades if none exist
      try {
        await axios.post(`${API}/create-sample-trades`);
        const sampleResponse = await axios.get(`${API}/enhanced-trading-history`);
        setTrades(sampleResponse.data.trades || []);
      } catch (sampleError) {
        console.error('Error creating sample trades:', sampleError);
      }
    } finally {
      setLoading(false);
    }
  };

  const exportToCSV = async () => {
    try {
      const response = await axios.get(`${API}/export-enhanced-trades`);
      
      // Create a download link
      const blob = new Blob([response.data.csv_data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `enhanced_trade_history_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      alert(`Exported ${response.data.total_trades} trades to CSV`);
    } catch (error) {
      console.error('Error exporting trades:', error);
      alert('Error exporting trades');
    }
  };

  useEffect(() => {
    fetchEnhancedTrades();
  }, []);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'BUY': return 'text-green-600 bg-green-100';
      case 'SELL': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getProfitColor = (value) => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High': return 'text-red-600 bg-red-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'Low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const filteredTrades = trades.filter(trade => {
    if (filter === 'all') return true;
    if (filter === 'profitable') return trade.pips_gained > 0;
    if (filter === 'losses') return trade.pips_gained < 0;
    if (filter === 'scalping') return trade.trade_type === 'Scalping';
    return true;
  });

  const sortedTrades = [...filteredTrades].sort((a, b) => {
    let aVal, bVal;
    
    switch (sortBy) {
      case 'time':
        aVal = new Date(a.timestamp);
        bVal = new Date(b.timestamp);
        break;
      case 'pips':
        aVal = a.pips_gained || 0;
        bVal = b.pips_gained || 0;
        break;
      case 'profit':
        aVal = a.percentage_pl || 0;
        bVal = b.percentage_pl || 0;
        break;
      default:
        aVal = a[sortBy] || '';
        bVal = b[sortBy] || '';
    }
    
    if (sortOrder === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  const toggleColumn = (column) => {
    setVisibleColumns(prev => ({
      ...prev,
      [column]: !prev[column]
    }));
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-800">📊 Enhanced Trade History</h2>
          <div className="flex space-x-3">
            <button
              onClick={fetchEnhancedTrades}
              disabled={loading}
              className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            <button
              onClick={exportToCSV}
              className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
            >
              <Download className="w-4 h-4 mr-2" />
              Export CSV
            </button>
          </div>
        </div>

        {/* Filters and Controls */}
        <div className="flex flex-wrap gap-4 mb-4">
          <select 
            value={filter} 
            onChange={(e) => setFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="all">All Trades</option>
            <option value="profitable">Profitable Only</option>
            <option value="losses">Losses Only</option>
            <option value="scalping">Scalping Only</option>
          </select>
          
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg"
          >
            <option value="time">Sort by Time</option>
            <option value="pips">Sort by Pips</option>
            <option value="profit">Sort by % P/L</option>
            <option value="symbol">Sort by Symbol</option>
          </select>
          
          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            {sortOrder === 'asc' ? '↑ Ascending' : '↓ Descending'}
          </button>
        </div>

        {/* Column Visibility Controls */}
        <details className="mb-4">
          <summary className="cursor-pointer flex items-center text-gray-700 hover:text-gray-900">
            <Filter className="w-4 h-4 mr-2" />
            Show/Hide Columns
          </summary>
          <div className="grid grid-cols-4 gap-2 mt-2 p-3 bg-gray-50 rounded-lg">
            {Object.keys(visibleColumns).map(column => (
              <label key={column} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={visibleColumns[column]}
                  onChange={() => toggleColumn(column)}
                  className="mr-2"
                />
                {column.charAt(0).toUpperCase() + column.slice(1).replace(/([A-Z])/g, ' $1')}
              </label>
            ))}
          </div>
        </details>
      </div>

      {/* Trade History Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              {visibleColumns.time && <th className="text-left p-3 font-medium text-gray-700">🕒 Time</th>}
              {visibleColumns.symbol && <th className="text-left p-3 font-medium text-gray-700">📈 Symbol</th>}
              {visibleColumns.action && <th className="text-left p-3 font-medium text-gray-700">💰 Action</th>}
              {visibleColumns.entryPrice && <th className="text-left p-3 font-medium text-gray-700">💸 Entry Price</th>}
              {visibleColumns.exitPrice && <th className="text-left p-3 font-medium text-gray-700">⏳ Exit Price</th>}
              {visibleColumns.pipsGained && <th className="text-left p-3 font-medium text-gray-700">📊 Pips Gained</th>}
              {visibleColumns.percentagePL && <th className="text-left p-3 font-medium text-gray-700">💹 % P/L</th>}
              {visibleColumns.confidence && <th className="text-left p-3 font-medium text-gray-700">🤖 Confidence</th>}
              {visibleColumns.decisionFactors && <th className="text-left p-3 font-medium text-gray-700">📋 Decision Factors</th>}
              {visibleColumns.tradeType && <th className="text-left p-3 font-medium text-gray-700">📦 Trade Type</th>}
              {visibleColumns.forecastTrend && <th className="text-left p-3 font-medium text-gray-700">📉 Forecast Trend</th>}
              {visibleColumns.newsSentiment && <th className="text-left p-3 font-medium text-gray-700">📰 News Sentiment</th>}
              {visibleColumns.tweetBias && <th className="text-left p-3 font-medium text-gray-700">🗣️ Tweet Bias</th>}
              {visibleColumns.botStrategy && <th className="text-left p-3 font-medium text-gray-700">💡 Bot Strategy</th>}
              {visibleColumns.mlDecision && <th className="text-left p-3 font-medium text-gray-700">🧠 ML Decision</th>}
              {visibleColumns.riskLevel && <th className="text-left p-3 font-medium text-gray-700">📦 Risk Level</th>}
              {visibleColumns.exitReason && <th className="text-left p-3 font-medium text-gray-700">🧾 Exit Reason</th>}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="17" className="text-center p-8">
                  <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                  Loading trades...
                </td>
              </tr>
            ) : sortedTrades.length === 0 ? (
              <tr>
                <td colSpan="17" className="text-center p-8 text-gray-500">
                  No trades found. Click "Refresh" to load or create sample data.
                </td>
              </tr>
            ) : (
              sortedTrades.map((trade, idx) => (
                <tr key={trade._id || idx} className="border-b hover:bg-gray-50">
                  {visibleColumns.time && (
                    <td className="p-3 text-gray-700">
                      {formatTimestamp(trade.timestamp)}
                    </td>
                  )}
                  {visibleColumns.symbol && (
                    <td className="p-3 font-medium text-gray-900">{trade.symbol}</td>
                  )}
                  {visibleColumns.action && (
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getActionColor(trade.action)}`}>
                        {trade.action}
                      </span>
                    </td>
                  )}
                  {visibleColumns.entryPrice && (
                    <td className="p-3 text-gray-700">{trade.entry_price?.toFixed(4)}</td>
                  )}
                  {visibleColumns.exitPrice && (
                    <td className="p-3 text-gray-700">{trade.exit_price?.toFixed(4) || '-'}</td>
                  )}
                  {visibleColumns.pipsGained && (
                    <td className={`p-3 font-semibold ${getProfitColor(trade.pips_gained)}`}>
                      {trade.pips_gained?.toFixed(1) || '-'}
                    </td>
                  )}
                  {visibleColumns.percentagePL && (
                    <td className={`p-3 font-semibold ${getProfitColor(trade.percentage_pl)}`}>
                      {trade.percentage_pl?.toFixed(2)}%
                    </td>
                  )}
                  {visibleColumns.confidence && (
                    <td className="p-3 text-gray-700">{(trade.confidence * 100).toFixed(1)}%</td>
                  )}
                  {visibleColumns.decisionFactors && (
                    <td className="p-3 text-gray-600 max-w-xs truncate" title={trade.decision_factors}>
                      {trade.decision_factors}
                    </td>
                  )}
                  {visibleColumns.tradeType && (
                    <td className="p-3 text-gray-700">{trade.trade_type}</td>
                  )}
                  {visibleColumns.forecastTrend && (
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded text-xs ${
                        trade.forecast_trend === 'UP' ? 'bg-green-100 text-green-800' :
                        trade.forecast_trend === 'DOWN' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {trade.forecast_trend}
                      </span>
                    </td>
                  )}
                  {visibleColumns.newsSentiment && (
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        trade.news_sentiment > 0.1 ? 'bg-green-100 text-green-800' :
                        trade.news_sentiment < -0.1 ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {trade.news_sentiment?.toFixed(2) || '0.00'}
                      </span>
                    </td>
                  )}
                  {visibleColumns.tweetBias && (
                    <td className="p-3 text-gray-700">{trade.tweet_bias}</td>
                  )}
                  {visibleColumns.botStrategy && (
                    <td className="p-3 text-gray-700">{trade.bot_strategy}</td>
                  )}
                  {visibleColumns.mlDecision && (
                    <td className="p-3 text-gray-700">{trade.ml_decision}</td>
                  )}
                  {visibleColumns.riskLevel && (
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getRiskColor(trade.risk_level)}`}>
                        {trade.risk_level}
                      </span>
                    </td>
                  )}
                  {visibleColumns.exitReason && (
                    <td className="p-3 text-gray-700">{trade.exit_reason}</td>
                  )}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Summary Stats */}
      {sortedTrades.length > 0 && (
        <div className="p-6 bg-gray-50 border-t">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">📈 Summary Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{sortedTrades.length}</div>
              <div className="text-sm text-gray-600">Total Trades</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {sortedTrades.filter(t => t.pips_gained > 0).length}
              </div>
              <div className="text-sm text-gray-600">Winning Trades</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {((sortedTrades.filter(t => t.pips_gained > 0).length / sortedTrades.length) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Win Rate</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${getProfitColor(
                sortedTrades.reduce((sum, t) => sum + (t.pips_gained || 0), 0)
              )}`}>
                {sortedTrades.reduce((sum, t) => sum + (t.pips_gained || 0), 0).toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Total Pips</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedTradeHistory;