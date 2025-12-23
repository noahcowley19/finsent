'use client';

// =============================================================================
// SENTIMENT GAUGE COMPONENT
// =============================================================================
// Visual gauge showing overall sentiment score
//
// Location: frontend/components/analysis/SentimentGauge.tsx
//
// =============================================================================

import React from 'react';

export interface SentimentGaugeProps {
  /** Sentiment score from -100 (bearish) to 100 (bullish) */
  score: number;
  /** Number of data points analyzed */
  dataPoints: number;
  /** Last updated timestamp */
  lastUpdated: Date;
  /** Loading state */
  isLoading?: boolean;
}

function getSentimentLabel(score: number): { label: string; color: string } {
  if (score >= 50) return { label: 'Very Bullish', color: 'text-success-600' };
  if (score >= 20) return { label: 'Bullish', color: 'text-success-500' };
  if (score >= -20) return { label: 'Neutral', color: 'text-neutral-600' };
  if (score >= -50) return { label: 'Bearish', color: 'text-error-500' };
  return { label: 'Very Bearish', color: 'text-error-600' };
}

function getGaugeColor(score: number): string {
  if (score >= 50) return '#22C55E';
  if (score >= 20) return '#4ADE80';
  if (score >= -20) return '#94A3B8';
  if (score >= -50) return '#F87171';
  return '#EF4444';
}

export const SentimentGauge: React.FC<SentimentGaugeProps> = ({
  score,
  dataPoints,
  lastUpdated,
  isLoading = false,
}) => {
  const { label, color } = getSentimentLabel(score);
  const gaugeColor = getGaugeColor(score);
  
  // Convert score (-100 to 100) to angle (-90 to 90 degrees)
  const angle = (score / 100) * 90;
  
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="w-40 h-6 rounded bg-cream-100 animate-pulse mb-6 mx-auto" />
        <div className="w-48 h-48 rounded-full bg-cream-100 animate-pulse mx-auto" />
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6 text-center">
      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-2">
        Overall Sentiment
      </h3>
      <p className="text-body-sm text-neutral-500 mb-6">
        Based on {dataPoints.toLocaleString()} data points
      </p>

      {/* Gauge */}
      <div className="relative w-48 h-24 mx-auto mb-6">
        <svg viewBox="0 0 200 100" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#E5E7EB"
            strokeWidth="16"
            strokeLinecap="round"
          />
          
          {/* Colored arc based on score */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={gaugeColor}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={`${((score + 100) / 200) * 251.2} 251.2`}
          />
          
          {/* Needle */}
          <g transform={`rotate(${angle} 100 100)`}>
            <line
              x1="100"
              y1="100"
              x2="100"
              y2="35"
              stroke="#131D4F"
              strokeWidth="3"
              strokeLinecap="round"
            />
            <circle cx="100" cy="100" r="8" fill="#131D4F" />
          </g>
          
          {/* Labels */}
          <text x="15" y="95" fontSize="10" fill="#94A3B8">-100</text>
          <text x="93" y="25" fontSize="10" fill="#94A3B8">0</text>
          <text x="170" y="95" fontSize="10" fill="#94A3B8">+100</text>
        </svg>
      </div>

      {/* Score display */}
      <div className="mb-4">
        <p className="font-display text-display-sm text-navy-900">
          {score > 0 ? '+' : ''}{score}
        </p>
        <p className={`text-body-lg font-semibold ${color}`}>
          {label}
        </p>
      </div>

      {/* Last updated */}
      <p className="text-caption text-neutral-400">
        Last updated: {lastUpdated.toLocaleTimeString()}
      </p>
    </div>
  );
};

export default SentimentGauge;
