import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const FallbackChart = ({ data, symbol, height = 400 }) => {
  if (!data || data.length === 0) {
    return (
      <div className="w-full flex items-center justify-center" style={{ height: `${height}px` }}>
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-2">📊</div>
          <div>Loading chart data...</div>
        </div>
      </div>
    );
  }

  // Convert candlestick data to line chart format
  const chartData = data.map((candle, index) => ({
    time: new Date(candle.timestamp).toLocaleTimeString(),
    price: parseFloat(candle.close),
    high: parseFloat(candle.high),
    low: parseFloat(candle.low),
    volume: parseInt(candle.volume),
  }));

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          {symbol} - Price Chart
        </h3>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">
            {data.length} data points
          </span>
        </div>
      </div>
      <div style={{ height: `${height}px` }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="time" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e0e0e0',
                borderRadius: '4px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              name="Price"
            />
            <Line 
              type="monotone" 
              dataKey="high" 
              stroke="#22c55e" 
              strokeWidth={1}
              dot={false}
              name="High"
              strokeDasharray="3 3"
            />
            <Line 
              type="monotone" 
              dataKey="low" 
              stroke="#ef4444" 
              strokeWidth={1}
              dot={false}
              name="Low"
              strokeDasharray="3 3"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default FallbackChart;