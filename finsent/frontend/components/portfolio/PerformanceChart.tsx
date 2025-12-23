'use client';

// =============================================================================
// PERFORMANCE CHART COMPONENT
// =============================================================================
// Line chart showing portfolio performance over time
//
// Location: frontend/components/portfolio/PerformanceChart.tsx
//
// =============================================================================

import React, { useState } from 'react';

export interface PerformanceDataPoint {
  date: Date;
  value: number;
}

export interface PerformanceChartProps {
  data: PerformanceDataPoint[];
  isLoading?: boolean;
}

type TimeRange = '1D' | '1W' | '1M' | '3M' | '1Y' | 'ALL';

function formatCurrency(value: number): string {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({
  data,
  isLoading = false,
}) => {
  const [timeRange, setTimeRange] = useState<TimeRange>('1M');

  const timeRanges: TimeRange[] = ['1D', '1W', '1M', '3M', '1Y', 'ALL'];

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex justify-between items-center mb-6">
          <div className="w-40 h-6 rounded bg-cream-100 animate-pulse" />
          <div className="flex gap-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="w-10 h-8 rounded bg-cream-100 animate-pulse" />
            ))}
          </div>
        </div>
        <div className="h-64 rounded bg-cream-100 animate-pulse" />
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
          Performance
        </h3>
        <div className="h-64 flex items-center justify-center text-neutral-500">
          No performance data available
        </div>
      </div>
    );
  }

  // Calculate chart dimensions
  const chartHeight = 200;
  const values = data.map((d) => d.value);
  const minValue = Math.min(...values) * 0.95;
  const maxValue = Math.max(...values) * 1.05;
  const valueRange = maxValue - minValue || 1;

  // Calculate performance
  const startValue = data[0].value;
  const endValue = data[data.length - 1].value;
  const change = endValue - startValue;
  const changePercent = ((endValue - startValue) / startValue) * 100;
  const isPositive = change >= 0;

  // Generate path
  const generatePath = () => {
    if (data.length === 0) return '';

    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = chartHeight - ((point.value - minValue) / valueRange) * chartHeight;
      return `${x},${y}`;
    });

    return `M ${points.join(' L ')}`;
  };

  // Generate area fill
  const generateArea = () => {
    if (data.length === 0) return '';
    const linePath = generatePath();
    return `${linePath} L 100,${chartHeight} L 0,${chartHeight} Z`;
  };

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div>
          <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-1">
            Performance
          </h3>
          <p className={`text-body-sm font-medium ${isPositive ? 'text-success-600' : 'text-error-600'}`}>
            {isPositive ? '+' : ''}{formatCurrency(change)} ({isPositive ? '+' : ''}{changePercent.toFixed(2)}%)
          </p>
        </div>

        {/* Time range selector */}
        <div className="flex gap-1 bg-cream-50 rounded-lg p-1">
          {timeRanges.map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`
                px-3 py-1.5 rounded-md text-body-sm font-medium
                transition-colors duration-fast
                ${timeRange === range
                  ? 'bg-white text-navy-900 shadow-sm'
                  : 'text-neutral-600 hover:text-navy-700'
                }
              `}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="relative h-64">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-8 w-16 flex flex-col justify-between text-caption text-neutral-400">
          <span>{formatCurrency(maxValue)}</span>
          <span>{formatCurrency((maxValue + minValue) / 2)}</span>
          <span>{formatCurrency(minValue)}</span>
        </div>

        {/* Chart area */}
        <div className="ml-20 h-full">
          <svg
            viewBox={`0 0 100 ${chartHeight}`}
            preserveAspectRatio="none"
            className="w-full h-full"
          >
            {/* Gradient fill */}
            <defs>
              <linearGradient id="performanceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isPositive ? '#22C55E' : '#EF4444'} stopOpacity="0.3" />
                <stop offset="100%" stopColor={isPositive ? '#22C55E' : '#EF4444'} stopOpacity="0" />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
              <line
                key={pct}
                x1="0"
                y1={pct * chartHeight}
                x2="100"
                y2={pct * chartHeight}
                stroke="#E5E7EB"
                strokeWidth="0.5"
                strokeDasharray="2 2"
              />
            ))}

            {/* Area fill */}
            <path d={generateArea()} fill="url(#performanceGradient)" />

            {/* Line */}
            <path
              d={generatePath()}
              fill="none"
              stroke={isPositive ? '#22C55E' : '#EF4444'}
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        </div>

        {/* X-axis labels */}
        <div className="ml-20 flex justify-between text-caption text-neutral-400 mt-2">
          {data.length > 0 && (
            <>
              <span>{data[0].date.toLocaleDateString()}</span>
              <span>{data[Math.floor(data.length / 2)]?.date.toLocaleDateString()}</span>
              <span>{data[data.length - 1].date.toLocaleDateString()}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default PerformanceChart;
