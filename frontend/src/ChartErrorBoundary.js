import React from 'react';
import FallbackChart from './FallbackChart';

class ChartErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Chart Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // Use fallback chart instead of error message
      return (
        <div className="w-full">
          <div className="mb-2 p-2 bg-yellow-100 border border-yellow-300 rounded text-sm text-yellow-800">
            ⚠️ Candlestick chart unavailable, showing fallback chart
          </div>
          <FallbackChart 
            data={this.props.data}
            symbol={this.props.symbol}
            height={this.props.height}
          />
        </div>
      );
    }

    return this.props.children;
  }
}

export default ChartErrorBoundary;