// =============================================================================
// CAVERAY REACT HOOKS - State management for API integration
// =============================================================================
// These hooks provide React-friendly wrappers around the API client with:
// - Loading and error states
// - Automatic data fetching
// - Manual refetch capabilities
// - Request cancellation on unmount
// - localStorage persistence for portfolios and watchlists
// =============================================================================

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { api, CaverayApiError } from './api';
import type {
  // Sentiment types
  SentimentAnalysisResponse,
  SocialScreeningResponse,
  // Financials types
  FinancialsResponse,
  // Insider types
  InsiderResponse,
  // Search types
  SearchResponse,
  ChartResponse,
  CompareResponse,
  MarketMoversResponse,
  SectorHeatmapResponse,
  QuickSearchResponse,
  // Portfolio types
  PortfolioAnalyzeResponse,
  PortfolioPositionInput,
  // Quant Lab types
  QuantLabResponse,
  // Local storage types
  LocalPortfolioPosition,
  LocalWatchlistItem,
  SavedStrategy,
  BacktestResult,
} from './types';

// -----------------------------------------------------------------------------
// HOOK TYPES
// -----------------------------------------------------------------------------

/** Base hook state */
interface HookState<T> {
  data: T | null;
  loading: boolean;
  error: CaverayApiError | null;
}

/** Hook return type with refetch */
interface HookResult<T> extends HookState<T> {
  refetch: () => Promise<void>;
}

/** Lazy hook return type with execute */
interface LazyHookResult<T, P extends unknown[]> extends HookState<T> {
  execute: (...params: P) => Promise<T | null>;
  reset: () => void;
}

// -----------------------------------------------------------------------------
// BASE HOOKS
// -----------------------------------------------------------------------------

/**
 * Base hook for API calls with automatic fetching
 * 
 * @param fetchFn - Async function that fetches data
 * @param deps - Dependencies that trigger refetch
 * @param enabled - Whether to fetch automatically (default: true)
 */
export function useApi<T>(
  fetchFn: (signal: AbortSignal) => Promise<T>,
  deps: unknown[] = [],
  enabled: boolean = true
): HookResult<T> {
  const [state, setState] = useState<HookState<T>>({
    data: null,
    loading: enabled,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new controller
    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await fetchFn(signal);
      if (!signal.aborted) {
        setState({ data, loading: false, error: null });
      }
    } catch (error) {
      if (!signal.aborted) {
        const apiError = error instanceof CaverayApiError
          ? error
          : new CaverayApiError(
              error instanceof Error ? error.message : 'Unknown error',
              0,
              'unknown_error'
            );
        setState({ data: null, loading: false, error: apiError });
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (enabled) {
      fetchData();
    }

    // Cleanup on unmount
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchData, enabled]);

  return {
    ...state,
    refetch: fetchData,
  };
}

/**
 * Lazy hook for manual API calls
 * 
 * @param fetchFn - Async function that fetches data with params
 */
export function useLazyApi<T, P extends unknown[]>(
  fetchFn: (...params: [...P, AbortSignal]) => Promise<T>
): LazyHookResult<T, P> {
  const [state, setState] = useState<HookState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(async (...params: P): Promise<T | null> => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    const { signal } = abortControllerRef.current;

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await fetchFn(...params, signal);
      if (!signal.aborted) {
        setState({ data, loading: false, error: null });
        return data;
      }
      return null;
    } catch (error) {
      if (!signal.aborted) {
        const apiError = error instanceof CaverayApiError
          ? error
          : new CaverayApiError(
              error instanceof Error ? error.message : 'Unknown error',
              0,
              'unknown_error'
            );
        setState({ data: null, loading: false, error: apiError });
      }
      return null;
    }
  }, [fetchFn]);

  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setState({ data: null, loading: false, error: null });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

// -----------------------------------------------------------------------------
// SENTIMENT HOOKS
// -----------------------------------------------------------------------------

/**
 * Hook for sentiment analysis
 * 
 * @param ticker - Stock ticker (empty string to disable)
 * @param numArticles - Number of articles to analyze
 * 
 * @example
 * ```tsx
 * const { data, loading, error } = useSentiment('AAPL', 10);
 * if (data) {
 *   console.log(`${data.summary.Positive.percentage}% positive`);
 * }
 * ```
 */
export function useSentiment(ticker: string, numArticles: number = 8) {
  return useApi<SentimentAnalysisResponse>(
    (signal) => api.sentiment.analyze(ticker, numArticles, { signal }),
    [ticker, numArticles],
    ticker.length > 0
  );
}

/**
 * Hook for social screening
 * 
 * @param tickers - Array of tickers (empty to use defaults)
 * 
 * @example
 * ```tsx
 * const { data, loading, refetch } = useSocialScreening(['AAPL', 'MSFT']);
 * ```
 */
export function useSocialScreening(tickers?: string[]) {
  return useApi<SocialScreeningResponse>(
    (signal) => api.sentiment.socialScreening(tickers, { signal }),
    [JSON.stringify(tickers)]
  );
}

/**
 * Lazy hook for sentiment analysis
 */
export function useLazySentiment() {
  return useLazyApi<SentimentAnalysisResponse, [string, number?]>(
    (ticker, numArticles = 8, signal) => 
      api.sentiment.analyze(ticker, numArticles, { signal })
  );
}

// -----------------------------------------------------------------------------
// FINANCIALS HOOKS
// -----------------------------------------------------------------------------

/**
 * Hook for financial analysis
 * 
 * @param ticker - Stock ticker (empty string to disable)
 * 
 * @example
 * ```tsx
 * const { data, loading, error } = useFinancials('AAPL');
 * if (data) {
 *   console.log(`Piotroski: ${data.scores.piotroski.display}`);
 * }
 * ```
 */
export function useFinancials(ticker: string) {
  return useApi<FinancialsResponse>(
    (signal) => api.financials.analyze(ticker, { signal }),
    [ticker],
    ticker.length > 0
  );
}

/**
 * Lazy hook for financial analysis
 */
export function useLazyFinancials() {
  return useLazyApi<FinancialsResponse, [string]>(
    (ticker, signal) => api.financials.analyze(ticker, { signal })
  );
}

// -----------------------------------------------------------------------------
// INSIDER HOOKS
// -----------------------------------------------------------------------------

/**
 * Hook for insider trading analysis
 * 
 * @param ticker - Stock ticker (empty string to disable)
 * @param months - Months of history to analyze
 * 
 * @example
 * ```tsx
 * const { data, loading, error } = useInsider('AAPL', 12);
 * if (data) {
 *   console.log(`Sentiment: ${data.sentiment.sentiment}`);
 * }
 * ```
 */
export function useInsider(ticker: string, months: number = 12) {
  return useApi<InsiderResponse>(
    (signal) => api.insider.analyze(ticker, months, { signal }),
    [ticker, months],
    ticker.length > 0
  );
}

/**
 * Lazy hook for insider analysis
 */
export function useLazyInsider() {
  return useLazyApi<InsiderResponse, [string, number?]>(
    (ticker, months = 12, signal) => 
      api.insider.analyze(ticker, months, { signal })
  );
}

// -----------------------------------------------------------------------------
// SEARCH HOOKS
// -----------------------------------------------------------------------------

/**
 * Hook for stock search
 * 
 * @param ticker - Stock ticker (empty string to disable)
 * 
 * @example
 * ```tsx
 * const { data, loading, error } = useStockSearch('AAPL');
 * if (data) {
 *   console.log(`Price: ${data.overview.price_display}`);
 * }
 * ```
 */
export function useStockSearch(ticker: string) {
  return useApi<SearchResponse>(
    (signal) => api.search.getStock(ticker, { signal }),
    [ticker],
    ticker.length > 0
  );
}

/**
 * Hook for chart data
 * 
 * @param ticker - Stock ticker (empty string to disable)
 * @param period - Time period
 * 
 * @example
 * ```tsx
 * const { data, loading } = useChartData('AAPL', '1y');
 * ```
 */
export function useChartData(
  ticker: string, 
  period: '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' | 'max' = '1y'
) {
  return useApi<ChartResponse>(
    (signal) => api.search.getChart(ticker, period, { signal }),
    [ticker, period],
    ticker.length > 0
  );
}

/**
 * Hook for stock comparison
 * 
 * @param tickers - Array of tickers to compare
 * 
 * @example
 * ```tsx
 * const { data, loading } = useStockCompare(['AAPL', 'MSFT', 'GOOGL']);
 * ```
 */
export function useStockCompare(tickers: string[]) {
  return useApi<CompareResponse>(
    (signal) => api.search.compare(tickers, { signal }),
    [JSON.stringify(tickers)],
    tickers.length >= 2
  );
}

/**
 * Hook for market movers
 * 
 * @example
 * ```tsx
 * const { data, loading, refetch } = useMarketMovers();
 * ```
 */
export function useMarketMovers() {
  return useApi<MarketMoversResponse>(
    (signal) => api.search.getMovers({ signal }),
    []
  );
}

/**
 * Hook for sector heatmap
 * 
 * @example
 * ```tsx
 * const { data, loading } = useSectorHeatmap();
 * ```
 */
export function useSectorHeatmap() {
  return useApi<SectorHeatmapResponse>(
    (signal) => api.search.getSectorHeatmap({ signal }),
    []
  );
}

/**
 * Lazy hook for quick search (autocomplete)
 * 
 * @example
 * ```tsx
 * const { data, loading, execute } = useQuickSearch();
 * 
 * const handleSearch = async (query: string) => {
 *   const result = await execute(query);
 *   if (result?.found) {
 *     console.log(result.name);
 *   }
 * };
 * ```
 */
export function useQuickSearch() {
  return useLazyApi<QuickSearchResponse, [string]>(
    (ticker, signal) => api.search.quick(ticker, { signal })
  );
}

/**
 * Lazy hook for stock search
 */
export function useLazyStockSearch() {
  return useLazyApi<SearchResponse, [string]>(
    (ticker, signal) => api.search.getStock(ticker, { signal })
  );
}

/**
 * Lazy hook for chart data
 */
export function useLazyChartData() {
  return useLazyApi<ChartResponse, [string, '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' | 'max']>(
    (ticker, period, signal) => api.search.getChart(ticker, period, { signal })
  );
}

// -----------------------------------------------------------------------------
// PORTFOLIO HOOKS
// -----------------------------------------------------------------------------

const PORTFOLIO_STORAGE_KEY = 'caveray_portfolio';

/**
 * Hook for portfolio management with localStorage persistence
 * 
 * @example
 * ```tsx
 * const {
 *   positions,
 *   analysis,
 *   loading,
 *   addPosition,
 *   updatePosition,
 *   removePosition,
 *   analyze,
 * } = usePortfolio();
 * ```
 */
export function usePortfolio() {
  // Local positions state
  const [positions, setPositions] = useState<LocalPortfolioPosition[]>([]);
  const [isHydrated, setIsHydrated] = useState(false);
  
  // API analysis state
  const [analysis, setAnalysis] = useState<PortfolioAnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<CaverayApiError | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(PORTFOLIO_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setPositions(parsed);
        }
      }
    } catch (e) {
      console.error('Failed to load portfolio from localStorage:', e);
    }
    setIsHydrated(true);
  }, []);

  // Save to localStorage when positions change
  useEffect(() => {
    if (isHydrated) {
      try {
        localStorage.setItem(PORTFOLIO_STORAGE_KEY, JSON.stringify(positions));
      } catch (e) {
        console.error('Failed to save portfolio to localStorage:', e);
      }
    }
  }, [positions, isHydrated]);

  // Convert local positions to API format
  const toApiFormat = useCallback((pos: LocalPortfolioPosition[]): PortfolioPositionInput[] => {
    return pos.map(p => ({
      ticker: p.ticker,
      shares: p.shares,
      total_cost_basis: p.shares * p.avgCost,
    }));
  }, []);

  // Analyze portfolio
  const analyze = useCallback(async () => {
    if (positions.length === 0) {
      setAnalysis(null);
      return null;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await api.portfolio.analyze(toApiFormat(positions));
      setAnalysis(result);
      return result;
    } catch (e) {
      const apiError = e instanceof CaverayApiError 
        ? e 
        : new CaverayApiError(e instanceof Error ? e.message : 'Unknown error');
      setError(apiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, [positions, toApiFormat]);

  // Add position
  const addPosition = useCallback((ticker: string, shares: number, avgCost: number) => {
    const newPosition: LocalPortfolioPosition = {
      id: `${ticker}-${Date.now()}`,
      ticker: ticker.toUpperCase(),
      shares,
      avgCost,
      dateAdded: new Date().toISOString(),
    };
    setPositions(prev => [...prev, newPosition]);
    return newPosition;
  }, []);

  // Update position
  const updatePosition = useCallback((id: string, updates: Partial<Omit<LocalPortfolioPosition, 'id'>>) => {
    setPositions(prev => 
      prev.map(pos => 
        pos.id === id ? { ...pos, ...updates } : pos
      )
    );
  }, []);

  // Remove position
  const removePosition = useCallback((id: string) => {
    setPositions(prev => prev.filter(pos => pos.id !== id));
  }, []);

  // Clear all positions
  const clearPortfolio = useCallback(() => {
    setPositions([]);
    setAnalysis(null);
  }, []);

  return {
    positions,
    analysis,
    loading,
    error,
    isHydrated,
    addPosition,
    updatePosition,
    removePosition,
    clearPortfolio,
    analyze,
  };
}

// -----------------------------------------------------------------------------
// WATCHLIST HOOKS
// -----------------------------------------------------------------------------

const WATCHLIST_STORAGE_KEY = 'caveray_watchlist';

/**
 * Hook for watchlist management with localStorage persistence
 * 
 * @example
 * ```tsx
 * const {
 *   items,
 *   addItem,
 *   removeItem,
 *   enrichedItems,
 *   loading,
 *   refresh,
 * } = useWatchlist();
 * ```
 */
export function useWatchlist() {
  // Local watchlist state
  const [items, setItems] = useState<LocalWatchlistItem[]>([]);
  const [isHydrated, setIsHydrated] = useState(false);
  
  // Enriched data from API
  const [enrichedData, setEnrichedData] = useState<SocialScreeningResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<CaverayApiError | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(WATCHLIST_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setItems(parsed);
        }
      }
    } catch (e) {
      console.error('Failed to load watchlist from localStorage:', e);
    }
    setIsHydrated(true);
  }, []);

  // Save to localStorage when items change
  useEffect(() => {
    if (isHydrated) {
      try {
        localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(items));
      } catch (e) {
        console.error('Failed to save watchlist to localStorage:', e);
      }
    }
  }, [items, isHydrated]);

  // Refresh enriched data
  const refresh = useCallback(async () => {
    if (items.length === 0) {
      setEnrichedData(null);
      return null;
    }

    setLoading(true);
    setError(null);

    try {
      const tickers = items.map(i => i.ticker);
      const result = await api.sentiment.socialScreening(tickers);
      setEnrichedData(result);
      return result;
    } catch (e) {
      const apiError = e instanceof CaverayApiError 
        ? e 
        : new CaverayApiError(e instanceof Error ? e.message : 'Unknown error');
      setError(apiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, [items]);

  // Add item
  const addItem = useCallback((ticker: string, notes?: string) => {
    // Check if already exists
    if (items.some(i => i.ticker.toUpperCase() === ticker.toUpperCase())) {
      return null;
    }
    
    const newItem: LocalWatchlistItem = {
      id: `${ticker}-${Date.now()}`,
      ticker: ticker.toUpperCase(),
      dateAdded: new Date().toISOString(),
      notes,
    };
    setItems(prev => [...prev, newItem]);
    return newItem;
  }, [items]);

  // Remove item
  const removeItem = useCallback((id: string) => {
    setItems(prev => prev.filter(item => item.id !== id));
  }, []);

  // Update notes
  const updateNotes = useCallback((id: string, notes: string) => {
    setItems(prev => 
      prev.map(item => 
        item.id === id ? { ...item, notes } : item
      )
    );
  }, []);

  // Clear watchlist
  const clearWatchlist = useCallback(() => {
    setItems([]);
    setEnrichedData(null);
  }, []);

  // Combine local items with enriched data
  const enrichedItems = items.map(item => {
    const apiData = enrichedData?.results?.find(
      r => r.ticker.toUpperCase() === item.ticker.toUpperCase()
    );
    return {
      ...item,
      ...apiData,
    };
  });

  return {
    items,
    enrichedItems,
    enrichedData,
    loading,
    error,
    isHydrated,
    addItem,
    removeItem,
    updateNotes,
    clearWatchlist,
    refresh,
  };
}

// -----------------------------------------------------------------------------
// QUANT LAB HOOKS
// -----------------------------------------------------------------------------

/**
 * Hook for Quant Lab analysis
 * 
 * @param ticker - Stock ticker (empty string to disable)
 * 
 * @example
 * ```tsx
 * const { data, loading, error } = useQuantLab('AAPL');
 * if (data) {
 *   console.log(`Alpha Score: ${data.alpha_score.score}`);
 *   console.log(`Signal: ${data.signal.action}`);
 * }
 * ```
 */
export function useQuantLab(ticker: string) {
  return useApi<QuantLabResponse>(
    (signal) => api.quantLab.analyze(ticker, { signal }),
    [ticker],
    ticker.length > 0
  );
}

/**
 * Lazy hook for Quant Lab analysis
 */
export function useLazyQuantLab() {
  return useLazyApi<QuantLabResponse, [string]>(
    (ticker, signal) => api.quantLab.analyze(ticker, { signal })
  );
}

// -----------------------------------------------------------------------------
// STRATEGY MANAGEMENT HOOKS
// -----------------------------------------------------------------------------

const STRATEGIES_STORAGE_KEY = 'caveray_strategies';

/**
 * Hook for managing saved trading strategies
 * 
 * @example
 * ```tsx
 * const { strategies, saveStrategy, deleteStrategy } = useStrategies();
 * ```
 */
export function useStrategies() {
  const [strategies, setStrategies] = useState<SavedStrategy[]>([]);
  const [isHydrated, setIsHydrated] = useState(false);

  // Load from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STRATEGIES_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setStrategies(parsed);
        }
      }
    } catch (e) {
      console.error('Failed to load strategies:', e);
    }
    setIsHydrated(true);
  }, []);

  // Save to localStorage
  useEffect(() => {
    if (isHydrated) {
      try {
        localStorage.setItem(STRATEGIES_STORAGE_KEY, JSON.stringify(strategies));
      } catch (e) {
        console.error('Failed to save strategies:', e);
      }
    }
  }, [strategies, isHydrated]);

  // Save strategy
  const saveStrategy = useCallback((strategy: Omit<SavedStrategy, 'id' | 'createdAt' | 'updatedAt'>) => {
    const newStrategy: SavedStrategy = {
      ...strategy,
      id: `strategy-${Date.now()}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setStrategies(prev => [...prev, newStrategy]);
    return newStrategy;
  }, []);

  // Update strategy
  const updateStrategy = useCallback((id: string, updates: Partial<Omit<SavedStrategy, 'id' | 'createdAt'>>) => {
    setStrategies(prev =>
      prev.map(s =>
        s.id === id
          ? { ...s, ...updates, updatedAt: new Date().toISOString() }
          : s
      )
    );
  }, []);

  // Delete strategy
  const deleteStrategy = useCallback((id: string) => {
    setStrategies(prev => prev.filter(s => s.id !== id));
  }, []);

  return {
    strategies,
    isHydrated,
    saveStrategy,
    updateStrategy,
    deleteStrategy,
  };
}

// -----------------------------------------------------------------------------
// COMBINED DATA HOOKS
// -----------------------------------------------------------------------------

/**
 * Hook that combines stock search and chart data
 * 
 * @param ticker - Stock ticker
 * @param chartPeriod - Chart time period
 * 
 * @example
 * ```tsx
 * const { stockData, chartData, loading, error } = useStockData('AAPL', '1y');
 * ```
 */
export function useStockData(
  ticker: string,
  chartPeriod: '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' | 'max' = '1y'
) {
  const stockResult = useStockSearch(ticker);
  const chartResult = useChartData(ticker, chartPeriod);

  return {
    stockData: stockResult.data,
    chartData: chartResult.data,
    loading: stockResult.loading || chartResult.loading,
    error: stockResult.error || chartResult.error,
    refetch: async () => {
      await Promise.all([stockResult.refetch(), chartResult.refetch()]);
    },
  };
}

/**
 * Hook for comprehensive stock analysis combining multiple endpoints
 * 
 * @param ticker - Stock ticker
 * 
 * @example
 * ```tsx
 * const { search, financials, insider, sentiment, loading } = useComprehensiveAnalysis('AAPL');
 * ```
 */
export function useComprehensiveAnalysis(ticker: string) {
  const searchResult = useStockSearch(ticker);
  const financialsResult = useFinancials(ticker);
  const insiderResult = useInsider(ticker);
  const sentimentResult = useSentiment(ticker);

  return {
    search: searchResult.data,
    financials: financialsResult.data,
    insider: insiderResult.data,
    sentiment: sentimentResult.data,
    loading: 
      searchResult.loading || 
      financialsResult.loading || 
      insiderResult.loading || 
      sentimentResult.loading,
    error: 
      searchResult.error || 
      financialsResult.error || 
      insiderResult.error || 
      sentimentResult.error,
    refetch: async () => {
      await Promise.all([
        searchResult.refetch(),
        financialsResult.refetch(),
        insiderResult.refetch(),
        sentimentResult.refetch(),
      ]);
    },
  };
}

// -----------------------------------------------------------------------------
// DEBOUNCED SEARCH HOOK
// -----------------------------------------------------------------------------

/**
 * Hook for debounced ticker search (useful for autocomplete)
 * 
 * @param delay - Debounce delay in ms (default: 300)
 * 
 * @example
 * ```tsx
 * const { results, loading, search } = useDebouncedSearch();
 * 
 * <input onChange={(e) => search(e.target.value)} />
 * ```
 */
export function useDebouncedSearch(delay: number = 300) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<QuickSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const search = useCallback((value: string) => {
    setQuery(value);
    
    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Cancel previous request
    if (abortRef.current) {
      abortRef.current.abort();
    }

    // Clear results if empty query
    if (!value.trim()) {
      setResults(null);
      setLoading(false);
      return;
    }

    setLoading(true);

    // Debounce the search
    timeoutRef.current = setTimeout(async () => {
      abortRef.current = new AbortController();

      try {
        const result = await api.search.quick(value.trim(), { 
          signal: abortRef.current.signal 
        });
        setResults(result);
      } catch (e) {
        if (!(e instanceof DOMException && e.name === 'AbortError')) {
          setResults(null);
        }
      } finally {
        setLoading(false);
      }
    }, delay);
  }, [delay]);

  const clear = useCallback(() => {
    setQuery('');
    setResults(null);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  return {
    query,
    results,
    loading,
    search,
    clear,
  };
}
