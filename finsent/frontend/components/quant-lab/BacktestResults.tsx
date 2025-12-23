'use client';

// =============================================================================
// BACKTEST RESULTS COMPONENT
// =============================================================================
// Displays results from a strategy backtest
//
// Location: frontend/components/quant-lab/BacktestResults.tsx
//
// =============================================================================

import React from 'react';

export interface BacktestTrade {
  id: string;
  type: 'buy' | 'sell';
  date: Date;
  price: number;
  shares: number;
  profit?: number;
}

export interface BacktestResultsData {
  totalReturn: number;
  totalReturnPercent: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitableTrades: number;
  avgWin: number;
  avgLoss: number;
  trades: BacktestTrade[];
  equityCurve: { date: Date; value: number }[];
}

export interface BacktestResultsProps {
  results: BacktestResultsData | null;
  isLoading?: boolean;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

export const BacktestResults: React.FC<BacktestResultsProps> = ({
  results,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <svg className="w-12 h-12 mx-auto mb-4 text-terra-500 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <p className="text-body-sm text-neutral-600">Running backtest...</p>
            <p className="text-caption text-neutral-400 mt-1">This may take a few moments</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <svg className="w-12 h-12 mx-auto mb-4 text-neutral-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-body-sm text-neutral-600">No results yet</p>
            <p className="text-caption text-neutral-400 mt-1">Build a strategy and run a backtest to see results</p>
          </div>
        </div>
      </div>
    );
  }

  const isPositiveReturn = results.totalReturn >= 0;

  // Generate simple chart
  const chartHeight = 120;
  const values = results.equityCurve.map((d) => d.value);
  const minValue = Math.min(...values) * 0.95;
  const maxValue = Math.max(...values) * 1.05;
  const valueRange = maxValue - minValue || 1;

  const generatePath = () => {
    if (results.equityCurve.length === 0) return '';
    const points = results.equityCurve.map((point, index) => {
      const x = (index / (results.equityCurve.length - 1)) * 100;
      const y = chartHeight - ((point.value - minValue) / valueRange) * chartHeight;
      return `${x},${y}`;
    });
    return `M ${points.join(' L ')}`;
  };

  return (
    <div className="bg-white rounded-xl border border-border-light overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-border-light">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Backtest Results
        </h3>
      </div>

      {/* Summary cards */}
      <div className="p-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-cream-50 rounded-xl">
          <p className="text-caption text-neutral-500 mb-1">Total Return</p>
          <p className={`text-body-lg font-bold ${isPositiveReturn ? 'text-success-600' : 'text-error-600'}`}>
            {formatCurrency(results.totalReturn)}
          </p>
          <p className={`text-body-sm ${isPositiveReturn ? 'text-success-600' : 'text-error-600'}`}>
            {formatPercent(results.totalReturnPercent)}
          </p>
        </div>

        <div className="p-4 bg-cream-50 rounded-xl">
          <p className="text-caption text-neutral-500 mb-1">Sharpe Ratio</p>
          <p className="text-body-lg font-bold text-navy-900">
            {results.sharpeRatio.toFixed(2)}
          </p>
          <p className="text-body-sm text-neutral-500">
            {results.sharpeRatio >= 1 ? 'Good' : results.sharpeRatio >= 0.5 ? 'Fair' : 'Poor'}
          </p>
        </div>

        <div className="p-4 bg-cream-50 rounded-xl">
          <p className="text-caption text-neutral-500 mb-1">Max Drawdown</p>
          <p className="text-body-lg font-bold text-error-600">
            {formatPercent(-results.maxDrawdown)}
          </p>
          <p className="text-body-sm text-neutral-500">
            Peak to trough
          </p>
        </div>

        <div className="p-4 bg-cream-50 rounded-xl">
          <p className="text-caption text-neutral-500 mb-1">Win Rate</p>
          <p className="text-body-lg font-bold text-navy-900">
            {results.winRate.toFixed(1)}%
          </p>
          <p className="text-body-sm text-neutral-500">
            {results.profitableTrades}/{results.totalTrades} trades
          </p>
        </div>
      </div>

      {/* Equity curve */}
      <div className="px-6 pb-6">
        <h4 className="text-body-sm font-semibold text-navy-900 mb-4">Equity Curve</h4>
        <div className="h-32 bg-cream-50 rounded-xl p-4">
          <svg
            viewBox={`0 0 100 ${chartHeight}`}
            preserveAspectRatio="none"
            className="w-full h-full"
          >
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isPositiveReturn ? '#22C55E' : '#EF4444'} stopOpacity="0.3" />
                <stop offset="100%" stopColor={isPositiveReturn ? '#22C55E' : '#EF4444'} stopOpacity="0" />
              </linearGradient>
            </defs>

            <path
              d={`${generatePath()} L 100,${chartHeight} L 0,${chartHeight} Z`}
              fill="url(#equityGradient)"
            />
            <path
              d={generatePath()}
              fill="none"
              stroke={isPositiveReturn ? '#22C55E' : '#EF4444'}
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        </div>
      </div>

      {/* Trade statistics */}
      <div className="px-6 pb-6">
        <h4 className="text-body-sm font-semibold text-navy-900 mb-4">Trade Statistics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-body-sm">
          <div>
            <p className="text-neutral-500">Total Trades</p>
            <p className="font-semibold text-navy-900">{results.totalTrades}</p>
          </div>
          <div>
            <p className="text-neutral-500">Profitable</p>
            <p className="font-semibold text-success-600">{results.profitableTrades}</p>
          </div>
          <div>
            <p className="text-neutral-500">Avg Win</p>
            <p className="font-semibold text-success-600">{formatCurrency(results.avgWin)}</p>
          </div>
          <div>
            <p className="text-neutral-500">Avg Loss</p>
            <p className="font-semibold text-error-600">{formatCurrency(results.avgLoss)}</p>
          </div>
        </div>
      </div>

      {/* Recent trades */}
      <div className="px-6 pb-6">
        <h4 className="text-body-sm font-semibold text-navy-900 mb-4">Recent Trades</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-body-sm">
            <thead>
              <tr className="text-left text-caption text-neutral-500 border-b border-border-light">
                <th className="pb-2">Date</th>
                <th className="pb-2">Type</th>
                <th className="pb-2 text-right">Price</th>
                <th className="pb-2 text-right">Shares</th>
                <th className="pb-2 text-right">P/L</th>
              </tr>
            </thead>
            <tbody>
              {results.trades.slice(0, 10).map((trade) => (
                <tr key={trade.id} className="border-b border-border-light">
                  <td className="py-2 text-neutral-600">
                    {trade.date.toLocaleDateString()}
                  </td>
                  <td className="py-2">
                    <span
                      className={`px-2 py-0.5 rounded text-caption font-medium uppercase ${
                        trade.type === 'buy'
                          ? 'bg-success-100 text-success-700'
                          : 'bg-error-100 text-error-700'
                      }`}
                    >
                      {trade.type}
                    </span>
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    ${trade.price.toFixed(2)}
                  </td>
                  <td className="py-2 text-right tabular-nums">
                    {trade.shares}
                  </td>
                  <td className={`py-2 text-right tabular-nums font-medium ${
                    (trade.profit ?? 0) >= 0 ? 'text-success-600' : 'text-error-600'
                  }`}>
                    {trade.profit !== undefined ? formatCurrency(trade.profit) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BacktestResults;
