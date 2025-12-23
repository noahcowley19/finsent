'use client';

// =============================================================================
// ADD STOCK MODAL COMPONENT
// =============================================================================
// Modal for searching and adding stocks to watchlist
//
// Location: frontend/components/watchlist/AddStockModal.tsx
//
// =============================================================================

import React, { useState, useEffect, useRef } from 'react';

export interface SearchResult {
  symbol: string;
  name: string;
  exchange: string;
}

export interface AddStockModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (symbol: string) => void;
  existingSymbols?: string[];
}

// Mock search function - replace with real API
async function searchStocks(query: string): Promise<SearchResult[]> {
  await new Promise((resolve) => setTimeout(resolve, 300));
  
  const mockResults: SearchResult[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ' },
    { symbol: 'AMZN', name: 'Amazon.com, Inc.', exchange: 'NASDAQ' },
    { symbol: 'TSLA', name: 'Tesla, Inc.', exchange: 'NASDAQ' },
    { symbol: 'META', name: 'Meta Platforms, Inc.', exchange: 'NASDAQ' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ' },
    { symbol: 'AMD', name: 'Advanced Micro Devices', exchange: 'NASDAQ' },
  ];

  const queryLower = query.toLowerCase();
  return mockResults.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(queryLower) ||
      stock.name.toLowerCase().includes(queryLower)
  );
}

export const AddStockModal: React.FC<AddStockModalProps> = ({
  isOpen,
  onClose,
  onAdd,
  existingSymbols = [],
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setQuery('');
      setResults([]);
    }
  }, [isOpen]);

  // Search as user types
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const data = await searchStocks(query);
        setResults(data);
      } catch (error) {
        console.error('Search error:', error);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  const handleAdd = (symbol: string) => {
    onAdd(symbol);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-navy-900/50 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="flex min-h-full items-start justify-center p-4 pt-20">
        <div className="relative w-full max-w-lg bg-white rounded-2xl shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-border-light">
            <h2 className="font-heading font-semibold text-heading-md text-navy-900">
              Add to Watchlist
            </h2>
            <button
              onClick={onClose}
              className="p-2 text-neutral-400 hover:text-neutral-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Search input */}
          <div className="p-6 pb-0">
            <div className="relative">
              <svg
                className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by ticker or company name..."
                className="w-full h-12 pl-12 pr-4 bg-cream-50 border border-border-medium rounded-xl text-navy-900 placeholder:text-neutral-400 focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20"
              />
              {isSearching && (
                <svg
                  className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400 animate-spin"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
            </div>
          </div>

          {/* Results */}
          <div className="p-6 max-h-80 overflow-y-auto">
            {query && results.length === 0 && !isSearching && (
              <div className="text-center py-8">
                <p className="text-body-sm text-neutral-500">No stocks found for &quot;{query}&quot;</p>
              </div>
            )}

            {results.length > 0 && (
              <ul className="space-y-2">
                {results.map((stock) => {
                  const isExisting = existingSymbols.includes(stock.symbol);

                  return (
                    <li key={stock.symbol}>
                      <button
                        onClick={() => !isExisting && handleAdd(stock.symbol)}
                        disabled={isExisting}
                        className={`
                          w-full flex items-center justify-between p-3 rounded-xl
                          transition-colors text-left
                          ${isExisting
                            ? 'bg-cream-50 opacity-50 cursor-not-allowed'
                            : 'hover:bg-cream-50'
                          }
                        `}
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                            <span className="text-body-sm font-bold text-navy-600">
                              {stock.symbol.slice(0, 2)}
                            </span>
                          </div>
                          <div>
                            <p className="text-body-sm font-medium text-navy-900">
                              {stock.symbol}
                              <span className="ml-2 text-caption text-neutral-400">
                                {stock.exchange}
                              </span>
                            </p>
                            <p className="text-caption text-neutral-500">
                              {stock.name}
                            </p>
                          </div>
                        </div>

                        {isExisting ? (
                          <span className="text-caption text-neutral-400">Already added</span>
                        ) : (
                          <span className="text-body-sm font-medium text-terra-600">
                            + Add
                          </span>
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}

            {!query && (
              <div className="text-center py-8">
                <svg className="w-12 h-12 mx-auto mb-4 text-neutral-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <p className="text-body-sm text-neutral-500">
                  Search for stocks to add to your watchlist
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AddStockModal;
