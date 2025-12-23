'use client';

// =============================================================================
// ANALYSIS HEADER COMPONENT
// =============================================================================
// Shared header for all analysis pages showing stock info
//
// Location: frontend/components/analysis/AnalysisHeader.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export interface AnalysisHeaderProps {
  symbol: string;
  name: string;
  exchange: string;
  price: number;
  change: number;
  changePercent: number;
  onAddToWatchlist?: () => void;
  inWatchlist?: boolean;
}

export const AnalysisHeader: React.FC<AnalysisHeaderProps> = ({
  symbol,
  name,
  exchange,
  price,
  change,
  changePercent,
  onAddToWatchlist,
  inWatchlist = false,
}) => {
  const isPositive = changePercent >= 0;

  return (
    <div className="bg-white border-b border-border-light">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-2 text-body-sm text-neutral-500 mb-4">
          <Link href="/search" className="hover:text-navy-600 transition-colors">
            Search
          </Link>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <span className="text-navy-900 font-medium">{symbol}</span>
        </nav>

        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Stock info */}
          <div className="flex items-center gap-4">
            {/* Symbol badge */}
            <div className="w-14 h-14 rounded-xl bg-navy-100 flex items-center justify-center">
              <span className="text-lg font-bold text-navy-600">
                {symbol.slice(0, 2)}
              </span>
            </div>

            <div>
              <div className="flex items-center gap-3">
                <h1 className="font-display text-display-sm text-navy-900">
                  {symbol}
                </h1>
                <span className="px-2 py-0.5 text-caption font-medium uppercase bg-cream-100 text-neutral-500 rounded">
                  {exchange}
                </span>
              </div>
              <p className="text-body-md text-neutral-600">{name}</p>
            </div>
          </div>

          {/* Price & actions */}
          <div className="flex items-center gap-6">
            {/* Price */}
            <div className="text-right">
              <p className="font-display text-display-sm text-navy-900">
                ${price.toFixed(2)}
              </p>
              <p
                className={`text-body-md font-medium flex items-center justify-end gap-1 ${
                  isPositive ? 'text-success-600' : 'text-error-600'
                }`}
              >
                <svg
                  className={`w-4 h-4 ${isPositive ? '' : 'rotate-180'}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
                {isPositive ? '+' : ''}{change.toFixed(2)} ({isPositive ? '+' : ''}{changePercent.toFixed(2)}%)
              </p>
            </div>

            {/* Watchlist button */}
            {onAddToWatchlist && (
              <button
                onClick={onAddToWatchlist}
                className={`
                  flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-colors
                  ${inWatchlist
                    ? 'bg-warning-100 text-warning-700 hover:bg-warning-200'
                    : 'bg-cream-100 text-navy-700 hover:bg-cream-200'
                  }
                `}
              >
                <svg
                  className="w-5 h-5"
                  fill={inWatchlist ? 'currentColor' : 'none'}
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                  />
                </svg>
                {inWatchlist ? 'Watching' : 'Watch'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisHeader;
