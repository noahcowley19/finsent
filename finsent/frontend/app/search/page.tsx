'use client';

// =============================================================================
// SEARCH PAGE
// =============================================================================
// Stock search page
//
// Location: frontend/app/search/page.tsx
//
// =============================================================================

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Section, Container } from '@/components/layout';
import { SearchBar, SearchResults, Stock } from '@/components/search';
import { useMarketMovers, useQuickSearch, useLazyStockSearch } from '@/lib/hooks';

// Recent searches storage
const RECENT_SEARCHES_KEY = 'caveray_recent_searches';

export default function SearchPage() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('q') || '';

  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState<Stock[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  
  // Use market movers for trending stocks
  const { data: moversData, loading: moversLoading } = useMarketMovers();
  const { execute: quickSearchExecute } = useQuickSearch();
  const { execute: stockSearchExecute } = useLazyStockSearch();

  // Load recent searches from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(RECENT_SEARCHES_KEY);
      if (stored) {
        setRecentSearches(JSON.parse(stored));
      }
    } catch (e) {
      console.error('Failed to load recent searches:', e);
    }
  }, []);

  // Save recent search
  const addRecentSearch = (ticker: string) => {
    const updated = [ticker, ...recentSearches.filter(s => s !== ticker)].slice(0, 10);
    setRecentSearches(updated);
    try {
      localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(updated));
    } catch (e) {
      console.error('Failed to save recent search:', e);
    }
  };

  // Convert market movers to Stock format for trending
  const trendingStocks: Stock[] = (moversData?.gainers || []).slice(0, 4).map(stock => ({
    symbol: stock.ticker,
    name: stock.ticker, // API doesn't provide company name
    exchange: 'NASDAQ',
    price: stock.price,
    changePercent: stock.change_percent,
  }));

  const handleSearch = async (searchQuery: string) => {
    setQuery(searchQuery);
    setIsLoading(true);
    setHasSearched(true);

    try {
      // Try quick search first
      const quickResult = await quickSearchExecute(searchQuery.toUpperCase());
      
      if (quickResult?.found) {
        // If found, get full stock data
        const fullData = await stockSearchExecute(searchQuery.toUpperCase());
        
        if (fullData) {
          const stockData: Stock = {
            symbol: fullData.overview.ticker,
            name: fullData.overview.name,
            exchange: fullData.overview.exchange,
            price: fullData.overview.price || undefined,
            change: fullData.overview.change_dollar || undefined,
            changePercent: fullData.overview.change_percent || undefined,
            marketCap: fullData.overview.market_cap || undefined,
            volume: fullData.overview.volume || undefined,
          };
          setResults([stockData]);
          
          // Add to recent searches
          addRecentSearch(searchQuery.toUpperCase());
        }
      } else {
        // Not found
        setResults([]);
      }
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddToWatchlist = (symbol: string) => {
    setWatchlist((prev) =>
      prev.includes(symbol)
        ? prev.filter((s) => s !== symbol)
        : [...prev, symbol]
    );
  };

  return (
    <>
      {/* Hero search section */}
      <Section spacing="xl" background="gradient">
        <Container size="md">
          <div className="text-center mb-8">
            <h1 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-4">
              Search Stocks
            </h1>
            <p className="text-body-lg text-neutral-600">
              Search by ticker symbol or company name to get started
            </p>
          </div>

          <SearchBar
            initialValue={initialQuery}
            onSearch={handleSearch}
            isLoading={isLoading}
            size="large"
            autoFocus
          />
        </Container>
      </Section>

      {/* Results or initial state */}
      <Section spacing="lg" background="default">
        <Container>
          {hasSearched ? (
            <SearchResults
              query={query}
              results={results}
              isLoading={isLoading}
              onAddToWatchlist={handleAddToWatchlist}
              watchlistSymbols={watchlist}
            />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Trending stocks */}
              <div>
                <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                  Trending Stocks
                </h2>
                <div className="space-y-3">
                  {trendingStocks.map((stock) => (
                    <Link
                      key={stock.symbol}
                      href={`/stock/${stock.symbol}`}
                      className="flex items-center justify-between p-4 bg-white rounded-xl border border-border-light hover:border-border-medium hover:shadow-sm transition-all"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                          <span className="text-body-sm font-semibold text-navy-600">
                            {stock.symbol.slice(0, 2)}
                          </span>
                        </div>
                        <div>
                          <p className="font-medium text-navy-900">{stock.symbol}</p>
                          <p className="text-body-sm text-neutral-500">{stock.name}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-navy-900">${stock.price?.toFixed(2)}</p>
                        <p
                          className={`text-body-sm ${
                            (stock.changePercent ?? 0) >= 0
                              ? 'text-success-600'
                              : 'text-error-600'
                          }`}
                        >
                          {(stock.changePercent ?? 0) >= 0 ? '+' : ''}
                          {stock.changePercent?.toFixed(2)}%
                        </p>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>

              {/* Recent searches */}
              <div>
                <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                  Recent Searches
                </h2>
                {recentSearches.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {recentSearches.map((symbol) => (
                      <button
                        key={symbol}
                        onClick={() => handleSearch(symbol)}
                        className="px-4 py-2 bg-white border border-border-light rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 hover:border-border-medium transition-colors"
                      >
                        {symbol}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-body-sm text-neutral-500">
                    Your recent searches will appear here
                  </p>
                )}

                {/* Quick links */}
                <div className="mt-8">
                  <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                    Popular Categories
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    <Link
                      href="/search?q=tech"
                      className="p-4 bg-white border border-border-light rounded-xl hover:border-border-medium transition-colors text-center"
                    >
                      <span className="text-2xl mb-2 block">💻</span>
                      <span className="text-body-sm font-medium text-navy-700">Tech</span>
                    </Link>
                    <Link
                      href="/search?q=finance"
                      className="p-4 bg-white border border-border-light rounded-xl hover:border-border-medium transition-colors text-center"
                    >
                      <span className="text-2xl mb-2 block">🏦</span>
                      <span className="text-body-sm font-medium text-navy-700">Finance</span>
                    </Link>
                    <Link
                      href="/search?q=healthcare"
                      className="p-4 bg-white border border-border-light rounded-xl hover:border-border-medium transition-colors text-center"
                    >
                      <span className="text-2xl mb-2 block">🏥</span>
                      <span className="text-body-sm font-medium text-navy-700">Healthcare</span>
                    </Link>
                    <Link
                      href="/search?q=energy"
                      className="p-4 bg-white border border-border-light rounded-xl hover:border-border-medium transition-colors text-center"
                    >
                      <span className="text-2xl mb-2 block">⚡</span>
                      <span className="text-body-sm font-medium text-navy-700">Energy</span>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Container>
      </Section>
    </>
  );
}
