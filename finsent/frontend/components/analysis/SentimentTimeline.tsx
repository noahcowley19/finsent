'use client';

// =============================================================================
// SENTIMENT TIMELINE COMPONENT
// =============================================================================
// Chart showing sentiment over time
//
// Location: frontend/components/analysis/SentimentTimeline.tsx
//
// =============================================================================

import React, { useState } from 'react';

export interface TimelineDataPoint {
  date: Date;
  score: number;
  volume: number;
}

export interface SentimentTimelineProps {
  data: TimelineDataPoint[];
  isLoading?: boolean;
}

type TimeRange = '1D' | '1W' | '1M' | '3M';

export const SentimentTimeline: React.FC<SentimentTimelineProps> = ({
  data,
  isLoading = false,
}) => {
  const [timeRange, setTimeRange] = useState<TimeRange>('1W');

  const timeRanges: TimeRange[] = ['1D', '1W', '1M', '3M'];

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex justify-between items-center mb-6">
          <div className="w-40 h-6 rounded bg-cream-100 animate-pulse" />
          <div className="flex gap-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="w-10 h-8 rounded bg-cream-100 animate-pulse" />
            ))}
          </div>
        </div>
        <div className="h-64 rounded bg-cream-100 animate-pulse" />
      </div>
    );
  }

  // Calculate chart dimensions
  const chartHeight = 200;
  const chartWidth = 100; // percentage
  const padding = 40;

  // Find min/max for scaling
  const scores = data.map((d) => d.score);
  const minScore = Math.min(...scores, -50);
  const maxScore = Math.max(...scores, 50);
  const scoreRange = maxScore - minScore || 1;

  // Generate path
  const generatePath = () => {
    if (data.length === 0) return '';

    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = chartHeight - ((point.score - minScore) / scoreRange) * chartHeight;
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
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Sentiment Over Time
        </h3>

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
        <div className="absolute left-0 top-0 bottom-8 w-10 flex flex-col justify-between text-caption text-neutral-400">
          <span>+100</span>
          <span>0</span>
          <span>-100</span>
        </div>

        {/* Chart area */}
        <div className="ml-12 h-full">
          <svg
            viewBox={`0 0 100 ${chartHeight}`}
            preserveAspectRatio="none"
            className="w-full h-full"
          >
            {/* Zero line */}
            <line
              x1="0"
              y1={chartHeight / 2}
              x2="100"
              y2={chartHeight / 2}
              stroke="#E5E7EB"
              strokeWidth="0.5"
              strokeDasharray="2 2"
            />

            {/* Gradient fill */}
            <defs>
              <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22C55E" stopOpacity="0.3" />
                <stop offset="50%" stopColor="#94A3B8" stopOpacity="0.1" />
                <stop offset="100%" stopColor="#EF4444" stopOpacity="0.3" />
              </linearGradient>
            </defs>

            {/* Area fill */}
            <path
              d={generateArea()}
              fill="url(#sentimentGradient)"
            />

            {/* Line */}
            <path
              d={generatePath()}
              fill="none"
              stroke="#131D4F"
              strokeWidth="1.5"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        </div>

        {/* X-axis labels */}
        <div className="ml-12 flex justify-between text-caption text-neutral-400 mt-2">
          {data.length > 0 && (
            <>
              <span>{data[0].date.toLocaleDateString()}</span>
              <span>{data[Math.floor(data.length / 2)]?.date.toLocaleDateString()}</span>
              <span>{data[data.length - 1].date.toLocaleDateString()}</span>
            </>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 text-caption">
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-success-500" />
          Bullish
        </span>
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-neutral-400" />
          Neutral
        </span>
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-error-500" />
          Bearish
        </span>
      </div>
    </div>
  );
};

export default SentimentTimeline;
