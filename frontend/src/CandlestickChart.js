import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

const CandlestickChart = ({ data, symbol, height = 400 }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const seriesRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    try {
      // Create chart
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: height,
        layout: {
          background: { color: '#ffffff' },
          textColor: '#333333',
        },
        grid: {
          vertLines: { color: '#e0e0e0' },
          horzLines: { color: '#e0e0e0' },
        },
        crosshair: {
          mode: 0,
        },
        rightPriceScale: {
          borderColor: '#cccccc',
        },
        timeScale: {
          borderColor: '#cccccc',
          rightOffset: 12,
          barSpacing: 3,
          fixLeftEdge: true,
          lockVisibleTimeRangeOnResize: true,
          rightBarStaysOnScroll: true,
          visible: true,
          timeVisible: true,
          secondsVisible: false,
        },
      });

      // Add candlestick series - updated API for v5.x
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#4CAF50',
        downColor: '#FF5252',
        borderVisible: false,
        wickUpColor: '#4CAF50',
        wickDownColor: '#FF5252',
      });

      chartRef.current = chart;
      seriesRef.current = candlestickSeries;

      // Handle resize
      const handleResize = () => {
        if (chartRef.current && chartContainerRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
        }
      };
    } catch (error) {
      console.error('Error creating chart:', error);
    }
  }, [height]);

  useEffect(() => {
    if (seriesRef.current && data && data.length > 0) {
      try {
        // Convert data to lightweight-charts format
        const chartData = data.map(candle => {
          // Convert timestamp to proper format
          const timestamp = new Date(candle.timestamp).getTime() / 1000;
          
          return {
            time: timestamp,
            open: parseFloat(candle.open),
            high: parseFloat(candle.high),
            low: parseFloat(candle.low),
            close: parseFloat(candle.close),
          };
        });

        // Sort by time to ensure proper order
        chartData.sort((a, b) => a.time - b.time);

        seriesRef.current.setData(chartData);
      } catch (error) {
        console.error('Error setting chart data:', error);
      }
    }
  }, [data]);

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          {symbol} - Candlestick Chart
        </h3>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">
            {data && data.length > 0 ? `${data.length} candles` : 'No data'}
          </span>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-xs text-gray-600">Up</span>
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span className="text-xs text-gray-600">Down</span>
          </div>
        </div>
      </div>
      <div 
        ref={chartContainerRef}
        className="w-full border border-gray-200 rounded-lg"
        style={{ height: `${height}px` }}
      />
      {(!data || data.length === 0) && (
        <div className="flex items-center justify-center h-64 text-gray-500">
          <div className="text-center">
            <div className="text-lg mb-2">📊</div>
            <div>Loading chart data...</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CandlestickChart;