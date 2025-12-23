'use client';

// =============================================================================
// WATCHLIST TABLE COMPONENT
// =============================================================================
// Displays watchlist stocks with quick actions
//
// Location: frontend/components/watchlist/WatchlistTable.tsx
//
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';

export interface WatchlistItem {
  id: string;
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  sentiment?: number;
  addedAt: Date;
}

export interface WatchlistTableProps {
  items: WatchlistItem[];
  onRemove?: (id: string) => void;
  onAddToPortfolio?: (item: WatchlistItem) => void;
  isLoading?: boolean;
  viewMode?: 'grid' | 'list';
}

function getSentimentLabel(score?: number): { label: string; color: string } | null {
  if (score === undefined) return null;
  if (score >= 30) return { label: 'Bullish', color: 'text-success-600 bg-success-100' };
  if (score >= -30) return { label: 'Neutral', color: 'text-neutral-600 bg-neutral-100' };
  return { label: 'Bearish', color: 'text-error-600 bg-error-100' };
}

export const WatchlistTable: React.FC<WatchlistTableProps> = ({
  items,
  onRemove,
  onAddToPortfolio,
  isLoading = false,
  viewMode = 'list',
}) => {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="bg-white rounded-xl border border-border-light p-4 animate-pulse">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-cream-100" />
              <div className="flex-1">
                <div className="w-20 h-5 rounded bg-cream-100 mb-2" />
                <div className="w-32 h-4 rounded bg-cream-100" />
              </div>
              <div className="w-24 h-6 rounded bg-cream-100" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-12 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        </div>
        <h3 className="font-heading font-semibold text-navy-900 mb-2">Your watchlist is empty</h3>
        <p className="text-body-sm text-neutral-600 mb-4">
          Start tracking stocks by adding them to your watchlist.
        </p>
        <Link
          href="/search"
          className="inline-flex items-center gap-2 px-4 py-2 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          Find Stocks
        </Link>
      </div>
    );
  }

  if (viewMode === 'grid') {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {items.map((item) => {
          const isPositive = item.changePercent >= 0;
          const sentiment = getSentimentLabel(item.sentiment);

          return (
            <div
              key={item.id}
              className="bg-white rounded-xl border border-border-light p-4 hover:border-border-medium hover:shadow-md transition-all"
            >
              <div className="flex items-start justify-between mb-4">
                <Link href={`/stock/${item.symbol}`} className="flex items-center gap-3 group">
                  <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                    <span className="text-body-sm font-bold text-navy-600">
                      {item.symbol.slice(0, 2)}
                    </span>
                  </div>
                  <div>
                    <p className="text-body-sm font-semibold text-navy-900 group-hover:text-navy-700">
                      {item.symbol}
                    </p>
                    <p className="text-caption text-neutral-500 truncate max-w-[100px]">
                      {item.name}
                    </p>
                  </div>
                </Link>

                {onRemove && (
                  <button
                    onClick={() => onRemove(item.id)}
                    className="p-1.5 text-neutral-400 hover:text-error-600 transition-colors"
                    title="Remove from watchlist"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>

              <div className="flex items-end justify-between">
                <div>
                  <p className="text-body-lg font-semibold text-navy-900">
                    ${item.price.toFixed(2)}
                  </p>
                  <p className={`text-body-sm font-medium ${isPositive ? 'text-success-600' : 'text-error-600'}`}>
                    {isPositive ? '+' : ''}{item.changePercent.toFixed(2)}%
                  </p>
                </div>

                {sentiment && (
                  <span className={`px-2 py-1 text-caption font-medium rounded ${sentiment.color}`}>
                    {sentiment.label}
                  </span>
                )}
              </div>

              <div className="flex gap-2 mt-4 pt-4 border-t border-border-light">
                <Link
                  href={`/sentiment/${item.symbol}`}
                  className="flex-1 py-2 text-center text-caption font-medium text-navy-600 bg-cream-50 rounded-lg hover:bg-cream-100 transition-colors"
                >
                  Analyze
                </Link>
                {onAddToPortfolio && (
                  <button
                    onClick={() => onAddToPortfolio(item)}
                    className="flex-1 py-2 text-center text-caption font-medium text-terra-600 bg-terra-50 rounded-lg hover:bg-terra-100 transition-colors"
                  >
                    Add to Portfolio
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // List view
  return (
    <div className="bg-white rounded-xl border border-border-light overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-cream-50">
              <th className="text-left py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Stock
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Price
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Change
              </th>
              <th className="text-center py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Sentiment
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-light">
            {items.map((item) => {
              const isPositive = item.changePercent >= 0;
              const sentiment = getSentimentLabel(item.sentiment);

              return (
                <tr key={item.id} className="hover:bg-cream-50 transition-colors">
                  <td className="py-4 px-4">
                    <Link href={`/stock/${item.symbol}`} className="flex items-center gap-3 group">
                      <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                        <span className="text-body-sm font-bold text-navy-600">
                          {item.symbol.slice(0, 2)}
                        </span>
                      </div>
                      <div>
                        <p className="text-body-sm font-medium text-navy-900 group-hover:text-navy-700">
                          {item.symbol}
                        </p>
                        <p className="text-caption text-neutral-500 truncate max-w-[150px]">
                          {item.name}
                        </p>
                      </div>
                    </Link>
                  </td>
                  <td className="py-4 px-4 text-body-sm font-medium text-navy-900 text-right tabular-nums">
                    ${item.price.toFixed(2)}
                  </td>
                  <td className="py-4 px-4 text-right">
                    <span className={`text-body-sm font-medium tabular-nums ${isPositive ? 'text-success-600' : 'text-error-600'}`}>
                      {isPositive ? '+' : ''}{item.changePercent.toFixed(2)}%
                    </span>
                  </td>
                  <td className="py-4 px-4 text-center">
                    {sentiment ? (
                      <span className={`px-2 py-1 text-caption font-medium rounded ${sentiment.color}`}>
                        {sentiment.label}
                      </span>
                    ) : (
                      <span className="text-caption text-neutral-400">-</span>
                    )}
                  </td>
                  <td className="py-4 px-4">
                    <div className="flex items-center justify-end gap-2">
                      <Link
                        href={`/sentiment/${item.symbol}`}
                        className="px-3 py-1.5 text-caption font-medium text-navy-600 bg-cream-50 rounded hover:bg-cream-100 transition-colors"
                      >
                        Analyze
                      </Link>
                      {onAddToPortfolio && (
                        <button
                          onClick={() => onAddToPortfolio(item)}
                          className="p-1.5 text-neutral-400 hover:text-terra-600 transition-colors"
                          title="Add to portfolio"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        </button>
                      )}
                      {onRemove && (
                        <button
                          onClick={() => onRemove(item.id)}
                          className="p-1.5 text-neutral-400 hover:text-error-600 transition-colors"
                          title="Remove"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default WatchlistTable;
