'use client';

// =============================================================================
// SEARCH PAGE - Enhanced with Data Point Grid
// =============================================================================

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Section, Container } from '@/components/layout';
import { SearchBar, SearchResults, Stock } from '@/components/search';
import { DataPointGrid } from '@/components/charts/DataPointGrid';
import { useMarketMovers, useQuickSearch, useLazyStockSearch } from '@/lib/hooks';
import { ScrollReveal } from '@/components/ui';

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
    name: stock.ticker,
    exchange: 'NASDAQ',
    price: stock.price,
    changePercent: stock.change_percent,
  }));

  const handleSearch = async (searchQuery: string) => {
    setQuery(searchQuery);
    setIsLoading(true);
    setHasSearched(true);

    try {
      const quickResult = await quickSearchExecute(searchQuery.toUpperCase());

      if (quickResult?.found) {
        const fullData = await stockSearchExecute(searchQuery.toUpperCase());

        if (fullData) {
          const stockData: Stock = {
            symbol: fullData.overview.ticker,
            name: fullData.overview.name,
            exchange: fullData.overview.exchange,
            price: fullData.overview.price || undefined,
            change: fullData.overview.change || undefined,
            changePercent: fullData.overview.change_percent || undefined,
            marketCap: fullData.overview.market_cap || undefined,
            volume: fullData.overview.volume || undefined,
          };
          setResults([stockData]);
          addRecentSearch(searchQuery.toUpperCase());
        }
      } else {
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
    <div className="min-h-screen bg-cream-50">
      {/* Hero search section */}
      <Section spacing="xl" background="white">
        <Container size="md">
          <ScrollReveal>
            <div className="text-center mb-8">
              <h1 className="text-3xl lg:text-4xl font-bold text-obsidian-900 tracking-tight mb-4">
                Search Stocks
              </h1>
              <p className="text-lg text-obsidian-500">
                Search by ticker symbol or company name to get started
              </p>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <SearchBar
              initialValue={initialQuery}
              onSearch={handleSearch}
              isLoading={isLoading}
              size="large"
              autoFocus
            />
          </ScrollReveal>
        </Container>
      </Section>

      {/* Results or initial state */}
      <Section spacing="lg" background="default">
        <Container>
          {hasSearched ? (
            <>
              <SearchResults
                query={query}
                results={results}
                isLoading={isLoading}
                onAddToWatchlist={handleAddToWatchlist}
                watchlistSymbols={watchlist}
              />

              {/* Data Point Grid for searched stock */}
              {results.length > 0 && (
                <ScrollReveal delay={200}>
                  <div className="mt-8">
                    <DataPointGrid symbol={results[0].symbol} />
                  </div>
                </ScrollReveal>
              )}
            </>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Trending stocks */}
              <ScrollReveal>
                <div>
                  <h2 className="font-bold text-xl text-obsidian-900 mb-4">
                    Trending Stocks
                  </h2>
                  <div className="space-y-3">
                    {trendingStocks.map((stock, index) => (
                      <ScrollReveal key={stock.symbol} delay={index * 50}>
                        <Link
                          href={`/stock/${stock.symbol}`}
                          className="flex items-center justify-between p-4 bg-white/80 backdrop-blur-lg rounded-xl border border-cream-200/50 shadow-glass hover:shadow-glass-lg hover:-translate-y-0.5 transition-all"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-electric-100 flex items-center justify-center">
                              <span className="text-sm font-semibold text-electric-600">
                                {stock.symbol.slice(0, 2)}
                              </span>
                            </div>
                            <div>
                              <p className="font-medium text-obsidian-900">{stock.symbol}</p>
                              <p className="text-sm text-obsidian-500">{stock.name}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-medium text-obsidian-900">${stock.price?.toFixed(2)}</p>
                            <p
                              className={`text-sm ${(stock.changePercent ?? 0) >= 0
                                ? 'text-success-600'
                                : 'text-coral-600'
                                }`}
                            >
                              {(stock.changePercent ?? 0) >= 0 ? '+' : ''}
                              {stock.changePercent?.toFixed(2)}%
                            </p>
                          </div>
                        </Link>
                      </ScrollReveal>
                    ))}
                  </div>
                </div>
              </ScrollReveal>

              {/* Recent searches */}
              <ScrollReveal delay={100}>
                <div>
                  <h2 className="font-bold text-xl text-obsidian-900 mb-4">
                    Recent Searches
                  </h2>
                  {recentSearches.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {recentSearches.map((symbol) => (
                        <button
                          key={symbol}
                          onClick={() => handleSearch(symbol)}
                          className="px-4 py-2 bg-white/80 backdrop-blur-lg border border-cream-200/50 rounded-xl text-sm font-medium text-obsidian-700 hover:bg-cream-100 shadow-glass hover:shadow-glass-lg transition-all"
                        >
                          {symbol}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-obsidian-400">
                      Your recent searches will appear here
                    </p>
                  )}

                  {/* Quick links */}
                  <div className="mt-8">
                    <h3 className="font-semibold text-lg text-obsidian-900 mb-4">
                      Popular Categories
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { label: 'Tech', emoji: '💻', query: 'tech' },
                        { label: 'Finance', emoji: '🏦', query: 'finance' },
                        { label: 'Healthcare', emoji: '🏥', query: 'healthcare' },
                        { label: 'Energy', emoji: '⚡', query: 'energy' },
                      ].map((category, index) => (
                        <ScrollReveal key={category.label} delay={150 + index * 50}>
                          <Link
                            href={`/search?q=${category.query}`}
                            className="p-4 bg-white/80 backdrop-blur-lg border border-cream-200/50 rounded-xl shadow-glass hover:shadow-glass-lg hover:-translate-y-0.5 transition-all text-center"
                          >
                            <span className="text-2xl mb-2 block">{category.emoji}</span>
                            <span className="text-sm font-medium text-obsidian-700">{category.label}</span>
                          </Link>
                        </ScrollReveal>
                      ))}
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            </div>
          )}
        </Container>
      </Section>

      {/* Feature Data Grid Preview (when no search) */}
      {!hasSearched && (
        <Section spacing="lg" background="white">
          <Container>
            <ScrollReveal>
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-obsidian-900 mb-3">
                  Comprehensive Financial Data
                </h2>
                <p className="text-obsidian-500 max-w-xl mx-auto">
                  Get instant access to key metrics, historical data, and AI-powered insights for any stock.
                </p>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={100}>
              <DataPointGrid />
            </ScrollReveal>
          </Container>
        </Section>
      )}
    </div>
  );
}
