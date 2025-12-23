'use client';

// =============================================================================
// WATCHLIST PREVIEW COMPONENT
// =============================================================================
// Shows a preview of the user's watchlist stocks
//
// Location: frontend/components/dashboard/WatchlistPreview.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export interface WatchlistStock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

export interface WatchlistPreviewProps {
  /** List of stocks */
  stocks: WatchlistStock[];
  /** Loading state */
  isLoading?: boolean;
  /** Max items to show */
  limit?: number;
}

export const WatchlistPreview: React.FC<WatchlistPreviewProps> = ({
  stocks,
  isLoading = false,
  limit = 5,
}) => {
  const displayedStocks = stocks.slice(0, limit);

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="w-24 h-6 rounded bg-cream-100 animate-pulse" />
          <div className="w-16 h-4 rounded bg-cream-100 animate-pulse" />
        </div>
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="flex items-center justify-between py-2 animate-pulse">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-cream-100" />
                <div>
                  <div className="w-16 h-4 rounded bg-cream-100 mb-1" />
                  <div className="w-24 h-3 rounded bg-cream-100" />
                </div>
              </div>
              <div className="text-right">
                <div className="w-16 h-4 rounded bg-cream-100 mb-1" />
                <div className="w-12 h-3 rounded bg-cream-100" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Watchlist
        </h3>
        <Link
          href="/watchlist"
          className="text-body-sm text-navy-500 hover:text-navy-700 transition-colors"
        >
          View all
        </Link>
      </div>

      {/* Stock list */}
      {displayedStocks.length === 0 ? (
        <div className="text-center py-8">
          <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
            <svg className="w-6 h-6 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <p className="text-body-sm text-neutral-500">Your watchlist is empty</p>
          <Link
            href="/search"
            className="inline-block mt-2 text-body-sm text-terra-500 hover:text-terra-600 font-medium"
          >
            Add stocks →
          </Link>
        </div>
      ) : (
        <div className="space-y-1">
          {displayedStocks.map((stock) => (
            <Link
              key={stock.symbol}
              href={`/stock/${stock.symbol}`}
              className="flex items-center justify-between py-3 px-2 -mx-2 rounded-lg hover:bg-cream-50 transition-colors"
            >
              {/* Stock info */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                  <span className="text-body-sm font-semibold text-navy-600">
                    {stock.symbol.slice(0, 2)}
                  </span>
                </div>
                <div>
                  <p className="text-body-sm font-medium text-navy-900">
                    {stock.symbol}
                  </p>
                  <p className="text-caption text-neutral-500 truncate max-w-[120px]">
                    {stock.name}
                  </p>
                </div>
              </div>

              {/* Price info */}
              <div className="text-right">
                <p className="text-body-sm font-medium text-navy-900">
                  ${stock.price.toFixed(2)}
                </p>
                <p
                  className={`text-caption font-medium ${
                    stock.changePercent >= 0 ? 'text-success-600' : 'text-error-600'
                  }`}
                >
                  {stock.changePercent >= 0 ? '+' : ''}
                  {stock.changePercent.toFixed(2)}%
                </p>
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* Add more button */}
      {displayedStocks.length > 0 && displayedStocks.length < stocks.length && (
        <Link
          href="/watchlist"
          className="flex items-center justify-center gap-2 mt-4 py-2 text-body-sm text-navy-500 hover:text-navy-700 transition-colors"
        >
          <span>+{stocks.length - displayedStocks.length} more</span>
        </Link>
      )}
    </div>
  );
};

export default WatchlistPreview;
