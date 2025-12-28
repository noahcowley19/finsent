'use client';

import React from 'react';
import Link from 'next/link';

// =============================================================================
// TYPES
// =============================================================================

export interface WatchlistStock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

export interface WatchlistPreviewProps {
  stocks: WatchlistStock[];
  maxItems?: number;
}

// =============================================================================
// WATCHLIST PREVIEW COMPONENT
// =============================================================================

export const WatchlistPreview: React.FC<WatchlistPreviewProps> = ({
  stocks,
  maxItems = 5,
}) => {
  const displayedStocks = stocks.slice(0, maxItems);

  return (
    <div className="bg-white rounded-xl border border-ink-200/50 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-ink-100">
        <h3 className="font-semibold text-heading-sm text-ink-900 tracking-tight">
          Watchlist
        </h3>
        <Link
          href="/watchlist"
          className="text-body-sm text-ink-500 hover:text-ink-700 transition-colors"
        >
          View all
        </Link>
      </div>

      {/* Stock List */}
      {displayedStocks.length > 0 ? (
        <div className="divide-y divide-ink-100">
          {displayedStocks.map((stock) => {
            const isPositive = stock.changePercent >= 0;

            return (
              <Link
                key={stock.symbol}
                href={`/stock/${stock.symbol.toLowerCase()}`}
                className="flex items-center justify-between px-5 py-3.5 hover:bg-ink-50 transition-colors"
              >
                <div className="min-w-0">
                  <p className="text-body-sm font-medium text-ink-900">
                    {stock.symbol}
                  </p>
                  <p className="text-body-xs text-ink-500 truncate">
                    {stock.name}
                  </p>
                </div>
                <div className="text-right flex-shrink-0 ml-4">
                  <p className="text-body-sm font-medium text-ink-900">
                    ${stock.price.toFixed(2)}
                  </p>
                  <p className={`text-body-xs font-medium ${isPositive ? 'text-success-600' : 'text-error-600'
                    }`}>
                    {isPositive ? '+' : ''}{stock.changePercent.toFixed(2)}%
                  </p>
                </div>
              </Link>
            );
          })}
        </div>
      ) : (
        <div className="px-5 py-12 text-center">
          <div className="w-12 h-12 rounded-xl bg-ink-100 flex items-center justify-center mx-auto mb-3">
            <svg className="w-6 h-6 text-ink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <p className="text-body-sm text-ink-500">No stocks yet</p>
          <Link
            href="/search"
            className="inline-flex items-center gap-1.5 mt-3 text-body-sm font-medium text-accent hover:text-accent-600 transition-colors"
          >
            Add your first stock
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </Link>
        </div>
      )}
    </div>
  );
};

export default WatchlistPreview;
