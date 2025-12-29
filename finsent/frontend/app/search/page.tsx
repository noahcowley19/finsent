'use client';

// =============================================================================
// SEARCH PAGE - Professional Stock Analysis
// =============================================================================

import React, { useState, useEffect, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Section, Container } from '@/components/layout';
import { SearchBar, SearchResults, Stock } from '@/components/search';
import { InteractivePriceChart } from '@/components/charts/InteractivePriceChart';
import { FinancialMetricsGrid } from '@/components/charts/FinancialMetricsGrid';
import { useMarketMovers, useQuickSearch, useLazyStockSearch } from '@/lib/hooks';
import { ScrollReveal, Spinner } from '@/components/ui';

// Recent searches storage
const RECENT_SEARCHES_KEY = 'caveray_recent_searches';

// =============================================================================
// MAIN SEARCH PAGE COMPONENT
// =============================================================================

function SearchPageContent() {
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
    const updated = [ticker, ...recentSearches.filter((s: string) => s !== ticker)].slice(0, 10);
    setRecentSearches(updated);
    try {
      localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(updated));
    } catch (e) {
      console.error('Failed to save recent search:', e);
    }
  };

  // Convert market movers to Stock format for trending
  const trendingStocks: Stock[] = (moversData?.gainers || []).slice(0, 6).map((stock: { ticker: string; price?: number; change_percent?: number }) => ({
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
    setWatchlist((prev: string[]) =>
      prev.includes(symbol)
        ? prev.filter((s: string) => s !== symbol)
        : [...prev, symbol]
    );
  };

  // ==========================================================================
  // RENDER: SEARCH RESULTS VIEW
  // ==========================================================================

  if (hasSearched && results.length > 0) {
    const stock = results[0];

    return (
      <div className="min-h-screen bg-cream-50">
        {/* Search Header */}
        <Section spacing="md" background="white">
          <Container>
            <div className="flex flex-col lg:flex-row lg:items-center gap-4">
              <div className="flex-1">
                <SearchBar
                  initialValue={query}
                  onSearch={handleSearch}
                  isLoading={isLoading}
                  size="default"
                />
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleAddToWatchlist(stock.symbol)}
                  className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${watchlist.includes(stock.symbol)
                    ? 'bg-electric-500 text-white'
                    : 'bg-cream-100 text-obsidian-700 hover:bg-cream-200'
                    }`}
                >
                  {watchlist.includes(stock.symbol) ? '★ In Watchlist' : '☆ Add to Watchlist'}
                </button>
                <Link
                  href={`/stock/${stock.symbol}`}
                  className="px-4 py-2.5 bg-obsidian-900 text-white rounded-xl text-sm font-medium hover:bg-obsidian-850 transition-all"
                >
                  Full Analysis →
                </Link>
              </div>
            </div>
          </Container>
        </Section>

        {/* Interactive Price Chart */}
        <Section spacing="md" background="default">
          <Container>
            <ScrollReveal>
              <InteractivePriceChart
                symbol={stock.symbol}
                name={stock.name}
              />
            </ScrollReveal>
          </Container>
        </Section>

        {/* Financial Metrics Grid */}
        <Section spacing="lg" background="white">
          <Container>
            <ScrollReveal delay={100}>
              <FinancialMetricsGrid symbol={stock.symbol} />
            </ScrollReveal>
          </Container>
        </Section>

        {/* Quick Stats Bar */}
        <Section spacing="md" background="default">
          <Container>
            <ScrollReveal delay={150}>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {[
                  { label: 'Market Cap', value: stock.marketCap ? `$${(stock.marketCap / 1e12).toFixed(2)}T` : 'N/A' },
                  { label: 'Volume', value: stock.volume ? `${(stock.volume / 1e6).toFixed(1)}M` : 'N/A' },
                  { label: 'P/E Ratio', value: '28.5' },
                  { label: '52W High', value: '$199.62' },
                  { label: '52W Low', value: '$124.17' },
                  { label: 'Avg Volume', value: '54.2M' },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="bg-white/80 backdrop-blur-lg rounded-xl border border-cream-200/50 p-4 text-center"
                  >
                    <p className="text-xs text-obsidian-400 mb-1">{stat.label}</p>
                    <p className="text-lg font-bold text-obsidian-900">{stat.value}</p>
                  </div>
                ))}
              </div>
            </ScrollReveal>
          </Container>
        </Section>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: NO RESULTS VIEW
  // ==========================================================================

  if (hasSearched && results.length === 0 && !isLoading) {
    return (
      <div className="min-h-screen bg-cream-50">
        <Section spacing="xl" background="white">
          <Container size="md">
            <div className="text-center mb-8">
              <h1 className="text-3xl lg:text-4xl font-bold text-obsidian-900 tracking-tight mb-4">
                Search Stocks
              </h1>
            </div>
            <SearchBar
              initialValue={query}
              onSearch={handleSearch}
              isLoading={isLoading}
              size="large"
              autoFocus
            />
          </Container>
        </Section>

        <Section spacing="lg" background="default">
          <Container size="sm">
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-cream-100 flex items-center justify-center">
                <svg className="w-8 h-8 text-obsidian-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-obsidian-900 mb-2">No results for "{query}"</h2>
              <p className="text-obsidian-500 mb-6">
                Try searching for a different ticker symbol or company name.
              </p>
              <button
                onClick={() => { setHasSearched(false); setQuery(''); }}
                className="px-6 py-2.5 bg-obsidian-900 text-white rounded-xl text-sm font-medium hover:bg-obsidian-850 transition-all"
              >
                Clear Search
              </button>
            </div>
          </Container>
        </Section>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: INITIAL STATE (NO SEARCH YET)
  // ==========================================================================

  return (
    <div className="min-h-screen bg-cream-50">
      {/* Hero Search Section */}
      <Section spacing="xl" background="white">
        <Container size="md">
          <ScrollReveal>
            <div className="text-center mb-8">
              <h1 className="text-4xl lg:text-5xl font-bold text-obsidian-900 tracking-tight mb-4">
                Search Stocks
              </h1>
              <p className="text-lg text-obsidian-500 max-w-lg mx-auto">
                Get instant access to professional-grade charts, financial metrics, and AI-powered insights.
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

          {/* Popular Searches */}
          <ScrollReveal delay={150}>
            <div className="flex items-center justify-center gap-2 mt-6 flex-wrap">
              <span className="text-sm text-obsidian-400">Popular:</span>
              {['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'].map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => handleSearch(symbol)}
                  className="px-3 py-1.5 bg-cream-100 hover:bg-cream-200 rounded-lg text-sm font-medium text-obsidian-700 transition-colors"
                >
                  {symbol}
                </button>
              ))}
            </div>
          </ScrollReveal>
        </Container>
      </Section>

      {/* Trending & Recent */}
      <Section spacing="lg" background="default">
        <Container>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Trending Stocks */}
            <ScrollReveal>
              <div>
                <h2 className="font-bold text-xl text-obsidian-900 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-coral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                  Trending Now
                </h2>
                <div className="grid grid-cols-2 gap-3">
                  {trendingStocks.map((stock, index) => (
                    <button
                      key={stock.symbol}
                      onClick={() => handleSearch(stock.symbol)}
                      className="flex items-center justify-between p-4 bg-white/80 backdrop-blur-lg rounded-xl border border-cream-200/50 shadow-glass hover:shadow-glass-lg hover:-translate-y-0.5 transition-all text-left"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-electric-500 to-electric-600 flex items-center justify-center">
                          <span className="text-sm font-bold text-white">
                            {stock.symbol.slice(0, 2)}
                          </span>
                        </div>
                        <div>
                          <p className="font-semibold text-obsidian-900">{stock.symbol}</p>
                          <p className="text-xs text-obsidian-500">{stock.exchange}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-obsidian-900">${stock.price?.toFixed(2)}</p>
                        <p className={`text-xs font-semibold ${(stock.changePercent ?? 0) >= 0 ? 'text-success-600' : 'text-coral-600'}`}>
                          {(stock.changePercent ?? 0) >= 0 ? '+' : ''}{stock.changePercent?.toFixed(2)}%
                        </p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </ScrollReveal>

            {/* Recent Searches & Categories */}
            <ScrollReveal delay={100}>
              <div>
                <h2 className="font-bold text-xl text-obsidian-900 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-obsidian-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Recent Searches
                </h2>
                {recentSearches.length > 0 ? (
                  <div className="flex flex-wrap gap-2 mb-8">
                    {recentSearches.map((symbol: string) => (
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
                  <p className="text-sm text-obsidian-400 mb-8">
                    Your recent searches will appear here
                  </p>
                )}

                <h3 className="font-semibold text-lg text-obsidian-900 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-electric-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Browse by Sector
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Technology', icon: <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>, tickers: 'AAPL, MSFT, GOOGL', color: 'text-electric-500' },
                    { label: 'Finance', icon: <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" /></svg>, tickers: 'JPM, BAC, GS', color: 'text-amber-500' },
                    { label: 'Healthcare', icon: <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>, tickers: 'JNJ, UNH, PFE', color: 'text-coral-500' },
                    { label: 'Energy', icon: <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>, tickers: 'XOM, CVX, COP', color: 'text-success-500' },
                  ].map((category) => (
                    <div
                      key={category.label}
                      className="p-4 bg-white/80 backdrop-blur-lg border border-cream-200/50 rounded-xl shadow-glass"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className={category.color}>{category.icon}</span>
                        <span className="font-semibold text-obsidian-900">{category.label}</span>
                      </div>
                      <p className="text-xs text-obsidian-400">{category.tickers}</p>
                    </div>
                  ))}
                </div>
              </div>
            </ScrollReveal>
          </div>
        </Container>
      </Section>

      {/* Feature Preview */}
      <Section spacing="lg" background="white">
        <Container>
          <ScrollReveal>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-obsidian-900 mb-3">
                Professional Analysis Tools
              </h2>
              <p className="text-obsidian-500 max-w-xl mx-auto">
                Get access to the same tools used by professional traders and analysts.
              </p>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>,
                  title: 'Interactive Charts',
                  description: 'Candlesticks, moving averages, support/resistance, and more. Full zoom and fullscreen support.',
                  color: 'bg-electric-100 text-electric-600',
                },
                {
                  icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
                  title: 'Financial Metrics',
                  description: '12 key metrics with 7-year historical data. Revenue, margins, cash flow, and returns.',
                  color: 'bg-amber-100 text-amber-600',
                },
                {
                  icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>,
                  title: 'AI Insights',
                  description: 'Sentiment analysis, pattern detection, and AI-powered trading signals.',
                  color: 'bg-coral-100 text-coral-600',
                },
              ].map((feature) => (
                <div
                  key={feature.title}
                  className="p-6 bg-cream-50 rounded-2xl border border-cream-200/50"
                >
                  <div className={`w-12 h-12 rounded-xl ${feature.color} flex items-center justify-center mb-4`}>
                    {feature.icon}
                  </div>
                  <h3 className="font-semibold text-lg text-obsidian-900 mb-2">{feature.title}</h3>
                  <p className="text-sm text-obsidian-500">{feature.description}</p>
                </div>
              ))}
            </div>
          </ScrollReveal>
        </Container>
      </Section>
    </div>
  );
}

// Wrap in Suspense for useSearchParams
export default function SearchPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-cream-50 flex items-center justify-center">
        <Spinner size="lg" />
      </div>
    }>
      <SearchPageContent />
    </Suspense>
  );
}
