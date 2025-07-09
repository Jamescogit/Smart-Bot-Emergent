import React from 'react';

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
      return (
        <div className="w-full h-64 bg-gray-100 rounded-lg border border-gray-200 flex items-center justify-center">
          <div className="text-center text-gray-600">
            <div className="text-2xl mb-2">📊</div>
            <div className="text-lg font-semibold mb-2">Chart Error</div>
            <div className="text-sm">Unable to load candlestick chart</div>
            <button 
              onClick={() => this.setState({ hasError: false, error: null })}
              className="mt-3 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ChartErrorBoundary;