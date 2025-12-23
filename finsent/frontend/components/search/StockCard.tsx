'use client';

// =============================================================================
// STOCK CARD COMPONENT
// =============================================================================
// Displays a stock with price, change, and quick actions
//
// Location: frontend/components/search/StockCard.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export interface Stock {
  symbol: string;
  name: string;
  exchange: string;
  price?: number;
  change?: number;
  changePercent?: number;
  marketCap?: number;
  volume?: number;
}

export interface StockCardProps {
  /** Stock data */
  stock: Stock;
  /** Show detailed info */
  detailed?: boolean;
  /** On add to watchlist */
  onAddToWatchlist?: (symbol: string) => void;
  /** Is in watchlist */
  inWatchlist?: boolean;
}

function formatMarketCap(value?: number): string {
  if (!value) return '-';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toLocaleString()}`;
}

function formatVolume(value?: number): string {
  if (!value) return '-';
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toLocaleString();
}

export const StockCard: React.FC<StockCardProps> = ({
  stock,
  detailed = false,
  onAddToWatchlist,
  inWatchlist = false,
}) => {
  const hasPrice = stock.price !== undefined;
  const isPositive = (stock.changePercent ?? 0) >= 0;

  return (
    <div className="bg-white rounded-xl border border-border-light p-4 hover:border-border-medium hover:shadow-md transition-all duration-normal group">
      <div className="flex items-start justify-between gap-4">
        {/* Left: Stock info */}
        <Link href={`/stock/${stock.symbol}`} className="flex-1 min-w-0">
          <div className="flex items-center gap-3">
            {/* Symbol badge */}
            <div className="w-12 h-12 rounded-xl bg-navy-100 flex items-center justify-center flex-shrink-0">
              <span className="text-body-sm font-bold text-navy-600">
                {stock.symbol.slice(0, 3)}
              </span>
            </div>

            {/* Name & exchange */}
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-heading font-semibold text-navy-900 group-hover:text-navy-700 transition-colors">
                  {stock.symbol}
                </h3>
                <span className="px-1.5 py-0.5 text-[10px] font-medium uppercase bg-cream-100 text-neutral-500 rounded">
                  {stock.exchange}
                </span>
              </div>
              <p className="text-body-sm text-neutral-500 truncate">
                {stock.name}
              </p>
            </div>
          </div>
        </Link>

        {/* Right: Price & actions */}
        <div className="flex items-center gap-4">
          {/* Price info */}
          {hasPrice && (
            <div className="text-right">
              <p className="font-heading font-semibold text-navy-900">
                ${stock.price?.toFixed(2)}
              </p>
              <p
                className={`text-body-sm font-medium ${
                  isPositive ? 'text-success-600' : 'text-error-600'
                }`}
              >
                {isPositive ? '+' : ''}
                {stock.changePercent?.toFixed(2)}%
              </p>
            </div>
          )}

          {/* Watchlist button */}
          {onAddToWatchlist && (
            <button
              onClick={(e) => {
                e.preventDefault();
                onAddToWatchlist(stock.symbol);
              }}
              className={`
                p-2 rounded-lg transition-colors
                ${inWatchlist
                  ? 'bg-warning-100 text-warning-600'
                  : 'bg-cream-50 text-neutral-400 hover:bg-cream-100 hover:text-neutral-600'
                }
              `}
              title={inWatchlist ? 'Remove from watchlist' : 'Add to watchlist'}
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
            </button>
          )}
        </div>
      </div>

      {/* Detailed info */}
      {detailed && hasPrice && (
        <div className="mt-4 pt-4 border-t border-border-light grid grid-cols-2 gap-4">
          <div>
            <p className="text-caption text-neutral-400">Market Cap</p>
            <p className="text-body-sm font-medium text-navy-700">
              {formatMarketCap(stock.marketCap)}
            </p>
          </div>
          <div>
            <p className="text-caption text-neutral-400">Volume</p>
            <p className="text-body-sm font-medium text-navy-700">
              {formatVolume(stock.volume)}
            </p>
          </div>
        </div>
      )}

      {/* Quick actions */}
      <div className="mt-4 flex items-center gap-2">
        <Link
          href={`/sentiment/${stock.symbol}`}
          className="flex-1 py-2 text-center text-body-sm font-medium text-navy-600 bg-cream-50 rounded-lg hover:bg-cream-100 transition-colors"
        >
          Sentiment
        </Link>
        <Link
          href={`/financials/${stock.symbol}`}
          className="flex-1 py-2 text-center text-body-sm font-medium text-navy-600 bg-cream-50 rounded-lg hover:bg-cream-100 transition-colors"
        >
          Financials
        </Link>
        <Link
          href={`/insider/${stock.symbol}`}
          className="flex-1 py-2 text-center text-body-sm font-medium text-navy-600 bg-cream-50 rounded-lg hover:bg-cream-100 transition-colors"
        >
          Insider
        </Link>
      </div>
    </div>
  );
};

export default StockCard;
