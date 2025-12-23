'use client';

// =============================================================================
// SEARCH RESULTS COMPONENT
// =============================================================================
// Displays search results with loading and empty states
//
// Location: frontend/components/search/SearchResults.tsx
//
// =============================================================================

import React from 'react';
import { StockCard, Stock } from './StockCard';

export interface SearchResultsProps {
  /** Search query */
  query: string;
  /** Search results */
  results: Stock[];
  /** Loading state */
  isLoading?: boolean;
  /** Error message */
  error?: string;
  /** On add to watchlist */
  onAddToWatchlist?: (symbol: string) => void;
  /** Watchlist symbols */
  watchlistSymbols?: string[];
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  query,
  results,
  isLoading = false,
  error,
  onAddToWatchlist,
  watchlistSymbols = [],
}) => {
  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="w-48 h-6 rounded bg-cream-100 animate-pulse" />
        </div>
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className="bg-white rounded-xl border border-border-light p-4 animate-pulse"
          >
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-cream-100" />
              <div className="flex-1">
                <div className="w-24 h-5 rounded bg-cream-100 mb-2" />
                <div className="w-48 h-4 rounded bg-cream-100" />
              </div>
              <div className="text-right">
                <div className="w-20 h-5 rounded bg-cream-100 mb-2" />
                <div className="w-16 h-4 rounded bg-cream-100" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-error-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-error-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h3 className="font-heading font-semibold text-navy-900 mb-2">
          Search Error
        </h3>
        <p className="text-body-md text-neutral-600">
          {error}
        </p>
      </div>
    );
  }

  // Empty query
  if (!query) {
    return null;
  }

  // No results
  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cream-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <h3 className="font-heading font-semibold text-navy-900 mb-2">
          No results found
        </h3>
        <p className="text-body-md text-neutral-600">
          We couldn&apos;t find any stocks matching &quot;{query}&quot;
        </p>
        <p className="text-body-sm text-neutral-400 mt-2">
          Try searching by ticker symbol (e.g., AAPL) or company name
        </p>
      </div>
    );
  }

  // Results
  return (
    <div className="space-y-4">
      {/* Results header */}
      <div className="flex items-center justify-between">
        <p className="text-body-sm text-neutral-600">
          {results.length} result{results.length !== 1 ? 's' : ''} for &quot;{query}&quot;
        </p>
      </div>

      {/* Results list */}
      <div className="space-y-3">
        {results.map((stock) => (
          <StockCard
            key={stock.symbol}
            stock={stock}
            detailed
            onAddToWatchlist={onAddToWatchlist}
            inWatchlist={watchlistSymbols.includes(stock.symbol)}
          />
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
