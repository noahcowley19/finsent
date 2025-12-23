'use client';

// =============================================================================
// SOURCE BREAKDOWN COMPONENT
// =============================================================================
// Shows sentiment breakdown by data source
//
// Location: frontend/components/analysis/SourceBreakdown.tsx
//
// =============================================================================

import React from 'react';

export interface SourceData {
  source: 'stocktwits' | 'twitter' | 'reddit' | 'news';
  score: number;
  mentions: number;
  change: number;
}

export interface SourceBreakdownProps {
  sources: SourceData[];
  isLoading?: boolean;
}

const sourceConfig = {
  stocktwits: {
    label: 'StockTwits',
    icon: '📈',
    color: 'bg-blue-500',
  },
  twitter: {
    label: 'X / Twitter',
    icon: '𝕏',
    color: 'bg-neutral-900',
  },
  reddit: {
    label: 'Reddit',
    icon: '🔴',
    color: 'bg-orange-500',
  },
  news: {
    label: 'News',
    icon: '📰',
    color: 'bg-purple-500',
  },
};

function getScoreColor(score: number): string {
  if (score >= 30) return 'text-success-600';
  if (score >= -30) return 'text-neutral-600';
  return 'text-error-600';
}

function getScoreLabel(score: number): string {
  if (score >= 50) return 'Very Bullish';
  if (score >= 20) return 'Bullish';
  if (score >= -20) return 'Neutral';
  if (score >= -50) return 'Bearish';
  return 'Very Bearish';
}

export const SourceBreakdown: React.FC<SourceBreakdownProps> = ({
  sources,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="w-40 h-6 rounded bg-cream-100 animate-pulse mb-6" />
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="flex items-center gap-4 p-4 rounded-lg bg-cream-50 animate-pulse">
              <div className="w-10 h-10 rounded-lg bg-cream-100" />
              <div className="flex-1">
                <div className="w-24 h-4 rounded bg-cream-100 mb-2" />
                <div className="w-32 h-3 rounded bg-cream-100" />
              </div>
              <div className="w-16 h-6 rounded bg-cream-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
        Sentiment by Source
      </h3>

      <div className="space-y-3">
        {sources.map((source) => {
          const config = sourceConfig[source.source];
          const scoreColor = getScoreColor(source.score);
          
          return (
            <div
              key={source.source}
              className="flex items-center gap-4 p-4 rounded-lg bg-cream-50 hover:bg-cream-100 transition-colors"
            >
              {/* Icon */}
              <div className={`w-10 h-10 rounded-lg ${config.color} flex items-center justify-center text-white text-lg`}>
                {config.icon}
              </div>

              {/* Source info */}
              <div className="flex-1 min-w-0">
                <p className="text-body-sm font-medium text-navy-900">
                  {config.label}
                </p>
                <p className="text-caption text-neutral-500">
                  {source.mentions.toLocaleString()} mentions
                </p>
              </div>

              {/* Score */}
              <div className="text-right">
                <p className={`text-body-md font-semibold ${scoreColor}`}>
                  {source.score > 0 ? '+' : ''}{source.score}
                </p>
                <p className="text-caption text-neutral-500">
                  {getScoreLabel(source.score)}
                </p>
              </div>

              {/* Change indicator */}
              <div
                className={`
                  flex items-center gap-1 px-2 py-1 rounded text-caption font-medium
                  ${source.change >= 0
                    ? 'bg-success-100 text-success-700'
                    : 'bg-error-100 text-error-700'
                  }
                `}
              >
                <svg
                  className={`w-3 h-3 ${source.change >= 0 ? '' : 'rotate-180'}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
                {Math.abs(source.change)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SourceBreakdown;
