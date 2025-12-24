// =============================================================================
// CAVERAY API CLIENT - Complete API integration for Flask Backend
// =============================================================================
// This client provides full access to all backend endpoints with:
// - Type-safe requests and responses
// - Automatic error handling
// - Request cancellation support
// - Retry logic for transient failures
// - Environment-aware base URL configuration
// =============================================================================

import type {
  // Sentiment types
  SentimentAnalysisResponse,
  SocialScreeningResponse,
  ScreeningDefaultsResponse,
  SentimentHealthResponse,
  // Financials types
  FinancialsResponse,
  // Insider types
  InsiderResponse,
  // Search types
  SearchResponse,
  ChartResponse,
  CompareResponse,
  CompareChartResponse,
  MarketMoversResponse,
  SectorHeatmapResponse,
  QuickSearchResponse,
  // Portfolio types
  PortfolioAnalyzeResponse,
  PortfolioStockResponse,
  PortfolioCAPMResponse,
  PortfolioPositionInput,
  // Quant Lab types
  QuantLabResponse,
  QuantLabHealthResponse,
  // Common types
  ApiError,
} from './types';

// -----------------------------------------------------------------------------
// CONFIGURATION
// -----------------------------------------------------------------------------

/**
 * Get the API base URL from environment or use default
 * 
 * In your .env.local file, set:
 * NEXT_PUBLIC_API_URL=https://finsent-backend.onrender.com
 * 
 * For local development with backend running locally:
 * NEXT_PUBLIC_API_URL=http://localhost:5000
 */
const getBaseUrl = (): string => {
  // Check for environment variable (works in both client and server)
  if (typeof window !== 'undefined') {
    // Client-side: use NEXT_PUBLIC_ prefixed env var
    return process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';
  }
  // Server-side
  return process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';
};

// -----------------------------------------------------------------------------
// ERROR HANDLING
// -----------------------------------------------------------------------------

/**
 * Custom error class for API errors with detailed information
 */
export class CaverayApiError extends Error {
  public readonly statusCode: number;
  public readonly errorType: string;
  public readonly details?: string;
  public readonly originalError?: unknown;

  constructor(
    message: string,
    statusCode: number = 500,
    errorType: string = 'unknown_error',
    details?: string,
    originalError?: unknown
  ) {
    super(message);
    this.name = 'CaverayApiError';
    this.statusCode = statusCode;
    this.errorType = errorType;
    this.details = details;
    this.originalError = originalError;
    
    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, CaverayApiError);
    }
  }

  /** Check if this is a validation error */
  isValidationError(): boolean {
    return this.errorType === 'validation';
  }

  /** Check if this is an invalid ticker error */
  isInvalidTicker(): boolean {
    return this.errorType === 'invalid_ticker' || this.errorType === 'invalid_asset';
  }

  /** Check if this is a network/fetch error */
  isNetworkError(): boolean {
    return this.errorType === 'network_error' || this.errorType === 'fetch_error';
  }

  /** Check if this is a server error */
  isServerError(): boolean {
    return this.statusCode >= 500;
  }
}

// -----------------------------------------------------------------------------
// REQUEST UTILITIES
// -----------------------------------------------------------------------------

/** Request configuration options */
interface RequestOptions {
  /** AbortController signal for cancellation */
  signal?: AbortSignal;
  /** Custom headers to include */
  headers?: Record<string, string>;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Number of retry attempts for failed requests (default: 2) */
  retries?: number;
  /** Delay between retries in milliseconds (default: 1000) */
  retryDelay?: number;
}

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> => 
  new Promise(resolve => setTimeout(resolve, ms));

/**
 * Create a timeout-wrapped fetch with AbortController
 */
const fetchWithTimeout = async (
  url: string,
  options: RequestInit & { timeout?: number }
): Promise<Response> => {
  const { timeout = 30000, ...fetchOptions } = options;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  // Combine signals if one was provided
  const signal = fetchOptions.signal 
    ? anySignal([fetchOptions.signal, controller.signal])
    : controller.signal;

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
};

/**
 * Combine multiple AbortSignals into one
 */
const anySignal = (signals: AbortSignal[]): AbortSignal => {
  const controller = new AbortController();
  
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      return controller.signal;
    }
    signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true });
  }
  
  return controller.signal;
};

/**
 * Parse API error response
 */
const parseErrorResponse = async (response: Response): Promise<ApiError> => {
  try {
    const data = await response.json();
    return {
      error: data.error || `HTTP ${response.status}: ${response.statusText}`,
      error_type: data.error_type || 'server_error',
      details: data.details,
    };
  } catch {
    return {
      error: `HTTP ${response.status}: ${response.statusText}`,
      error_type: 'server_error',
    };
  }
};

// -----------------------------------------------------------------------------
// API CLIENT CLASS
// -----------------------------------------------------------------------------

/**
 * Main API client for Caveray backend
 * 
 * @example
 * ```typescript
 * const api = new CaverayApiClient();
 * 
 * // Simple usage
 * const data = await api.search.getStock('AAPL');
 * 
 * // With cancellation
 * const controller = new AbortController();
 * const data = await api.search.getStock('AAPL', { signal: controller.signal });
 * // Later: controller.abort();
 * ```
 */
export class CaverayApiClient {
  private readonly baseUrl: string;
  private readonly defaultOptions: RequestOptions;

  constructor(baseUrl?: string, defaultOptions?: RequestOptions) {
    this.baseUrl = baseUrl || getBaseUrl();
    this.defaultOptions = {
      timeout: 30000,
      retries: 2,
      retryDelay: 1000,
      ...defaultOptions,
    };
  }

  // ---------------------------------------------------------------------------
  // CORE REQUEST METHOD
  // ---------------------------------------------------------------------------

  /**
   * Make a typed API request with error handling and retries
   */
  private async request<T>(
    method: 'GET' | 'POST',
    endpoint: string,
    body?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const opts = { ...this.defaultOptions, ...options };
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...opts.headers,
    };

    let lastError: unknown;
    const maxAttempts = (opts.retries || 0) + 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const response = await fetchWithTimeout(url, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: opts.signal,
          timeout: opts.timeout,
        });

        // Handle non-OK responses
        if (!response.ok) {
          const errorData = await parseErrorResponse(response);
          throw new CaverayApiError(
            errorData.error,
            response.status,
            errorData.error_type,
            errorData.details
          );
        }

        // Parse and return successful response
        const data = await response.json();
        return data as T;

      } catch (error) {
        lastError = error;

        // Don't retry if request was aborted
        if (error instanceof DOMException && error.name === 'AbortError') {
          throw new CaverayApiError(
            'Request was cancelled',
            0,
            'aborted'
          );
        }

        // Don't retry client errors (4xx)
        if (error instanceof CaverayApiError && error.statusCode >= 400 && error.statusCode < 500) {
          throw error;
        }

        // Retry on network errors or 5xx errors
        if (attempt < maxAttempts) {
          console.warn(`API request failed (attempt ${attempt}/${maxAttempts}), retrying...`, error);
          await sleep(opts.retryDelay || 1000);
          continue;
        }
      }
    }

    // All retries exhausted
    if (lastError instanceof CaverayApiError) {
      throw lastError;
    }

    throw new CaverayApiError(
      lastError instanceof Error ? lastError.message : 'Network request failed',
      0,
      'network_error',
      undefined,
      lastError
    );
  }

  // ---------------------------------------------------------------------------
  // SENTIMENT ENDPOINTS
  // ---------------------------------------------------------------------------

  /** Sentiment analysis endpoints */
  public readonly sentiment = {
    /**
     * Analyze sentiment for a ticker from news articles
     * 
     * @param ticker - Stock ticker symbol (e.g., 'AAPL')
     * @param numArticles - Number of articles to analyze (1-15, default: 8)
     * @param options - Request options
     * @returns Sentiment analysis with articles and summary
     * 
     * @example
     * ```typescript
     * const result = await api.sentiment.analyze('AAPL', 10);
     * console.log(result.summary.Positive.percentage);
     * ```
     */
    analyze: (
      ticker: string,
      numArticles: number = 8,
      options?: RequestOptions
    ): Promise<SentimentAnalysisResponse> => {
      return this.request<SentimentAnalysisResponse>(
        'POST',
        '/api/analyze',
        { ticker, num_articles: numArticles },
        options
      );
    },

    /**
     * Get social sentiment screening for multiple tickers
     * Aggregates StockTwits, X/Twitter, and news sentiment
     * 
     * @param tickers - Array of ticker symbols (max 10)
     * @param options - Request options
     * @returns Social screening results with composite scores
     * 
     * @example
     * ```typescript
     * const result = await api.sentiment.socialScreening(['AAPL', 'MSFT', 'GOOGL']);
     * result.results.forEach(stock => console.log(stock.ticker, stock.composite));
     * ```
     */
    socialScreening: (
      tickers?: string[],
      options?: RequestOptions
    ): Promise<SocialScreeningResponse> => {
      return this.request<SocialScreeningResponse>(
        'POST',
        '/api/social-screening',
        { tickers },
        options
      );
    },

    /**
     * Get default tickers for screening
     * 
     * @param options - Request options
     * @returns Default tickers and max allowed
     */
    getDefaults: (options?: RequestOptions): Promise<ScreeningDefaultsResponse> => {
      return this.request<ScreeningDefaultsResponse>(
        'GET',
        '/api/social-screening/defaults',
        undefined,
        options
      );
    },

    /**
     * Check sentiment service health
     * 
     * @param options - Request options
     * @returns Health status including FinBERT availability
     */
    health: (options?: RequestOptions): Promise<SentimentHealthResponse> => {
      return this.request<SentimentHealthResponse>(
        'GET',
        '/api/sentiment/health',
        undefined,
        options
      );
    },
  };

  // ---------------------------------------------------------------------------
  // FINANCIALS ENDPOINTS
  // ---------------------------------------------------------------------------

  /** Financial analysis endpoints */
  public readonly financials = {
    /**
     * Get comprehensive financial analysis for a ticker
     * Includes Piotroski F-Score, Altman Z-Score, and Beneish M-Score
     * 
     * @param ticker - Stock ticker symbol
     * @param options - Request options
     * @returns Financial scores and metrics
     * 
     * @example
     * ```typescript
     * const result = await api.financials.analyze('AAPL');
     * console.log(`Piotroski: ${result.scores.piotroski.score}/9`);
     * console.log(`Altman Z: ${result.scores.altman.interpretation}`);
     * ```
     */
    analyze: (ticker: string, options?: RequestOptions): Promise<FinancialsResponse> => {
      return this.request<FinancialsResponse>(
        'POST',
        '/api/financials',
        { ticker },
        options
      );
    },
  };

  // ---------------------------------------------------------------------------
  // INSIDER ENDPOINTS
  // ---------------------------------------------------------------------------

  /** Insider trading endpoints */
  public readonly insider = {
    /**
     * Get insider trading analysis for a ticker
     * 
     * @param ticker - Stock ticker symbol
     * @param months - Number of months to analyze (1-24, default: 12)
     * @param options - Request options
     * @returns Insider transactions, sentiment, and institutional holdings
     * 
     * @example
     * ```typescript
     * const result = await api.insider.analyze('AAPL', 6);
     * console.log(`Sentiment: ${result.sentiment.sentiment}`);
     * console.log(`Cluster alerts: ${result.cluster_alerts.length}`);
     * ```
     */
    analyze: (
      ticker: string,
      months: number = 12,
      options?: RequestOptions
    ): Promise<InsiderResponse> => {
      return this.request<InsiderResponse>(
        'POST',
        '/api/insider',
        { ticker, months },
        options
      );
    },

    /**
     * Check insider service health
     * 
     * @param options - Request options
     * @returns Health status
     */
    health: (options?: RequestOptions): Promise<{ status: string; service: string }> => {
      return this.request<{ status: string; service: string }>(
        'GET',
        '/api/insider/health',
        undefined,
        options
      );
    },
  };

  // ---------------------------------------------------------------------------
  // SEARCH ENDPOINTS
  // ---------------------------------------------------------------------------

  /** Stock search and market data endpoints */
  public readonly search = {
    /**
     * Get comprehensive stock data
     * 
     * @param ticker - Stock ticker symbol
     * @param options - Request options
     * @returns Full stock overview, metrics, profile, and news
     * 
     * @example
     * ```typescript
     * const result = await api.search.getStock('AAPL');
     * console.log(`Price: ${result.overview.price_display}`);
     * console.log(`P/E: ${result.valuation[0].display}`);
     * ```
     */
    getStock: (ticker: string, options?: RequestOptions): Promise<SearchResponse> => {
      return this.request<SearchResponse>(
        'POST',
        '/api/search',
        { ticker },
        options
      );
    },

    /**
     * Get historical chart data
     * 
     * @param ticker - Stock ticker symbol
     * @param period - Time period ('1d', '5d', '1m', '3m', '6m', 'ytd', '1y', '2y', '5y', 'max')
     * @param options - Request options
     * @returns OHLCV data arrays
     * 
     * @example
     * ```typescript
     * const result = await api.search.getChart('AAPL', '1y');
     * result.data.dates.forEach((date, i) => {
     *   console.log(date, result.data.prices[i]);
     * });
     * ```
     */
    getChart: (
      ticker: string,
      period: '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' | 'max' = '1y',
      options?: RequestOptions
    ): Promise<ChartResponse> => {
      return this.request<ChartResponse>(
        'POST',
        '/api/search/chart',
        { ticker, period },
        options
      );
    },

    /**
     * Compare multiple stocks
     * 
     * @param tickers - Array of ticker symbols (2-4)
     * @param options - Request options
     * @returns Comparison data for all tickers
     * 
     * @example
     * ```typescript
     * const result = await api.search.compare(['AAPL', 'MSFT', 'GOOGL']);
     * result.results.forEach(stock => {
     *   console.log(stock.ticker, stock.pe_display, stock.roe_display);
     * });
     * ```
     */
    compare: (tickers: string[], options?: RequestOptions): Promise<CompareResponse> => {
      return this.request<CompareResponse>(
        'POST',
        '/api/search/compare',
        { tickers },
        options
      );
    },

    /**
     * Get normalized chart data for comparison
     * 
     * @param tickers - Array of ticker symbols
     * @param period - Time period ('1m', '3m', '6m', 'ytd', '1y', '2y', '5y')
     * @param options - Request options
     * @returns Normalized price series (base 100)
     */
    compareChart: (
      tickers: string[],
      period: '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' = '1y',
      options?: RequestOptions
    ): Promise<CompareChartResponse> => {
      return this.request<CompareChartResponse>(
        'POST',
        '/api/search/compare/chart',
        { tickers, period },
        options
      );
    },

    /**
     * Get market movers (gainers, losers, most active)
     * 
     * @param options - Request options
     * @returns Top gainers, losers, and most active stocks
     * 
     * @example
     * ```typescript
     * const result = await api.search.getMovers();
     * console.log('Top Gainer:', result.gainers[0].ticker, result.gainers[0].change_display);
     * ```
     */
    getMovers: (options?: RequestOptions): Promise<MarketMoversResponse> => {
      return this.request<MarketMoversResponse>(
        'GET',
        '/api/search/movers',
        undefined,
        options
      );
    },

    /**
     * Get sector performance heatmap
     * 
     * @param options - Request options
     * @returns Sector ETF performance data
     * 
     * @example
     * ```typescript
     * const result = await api.search.getSectorHeatmap();
     * result.sectors.forEach(sector => {
     *   console.log(sector.sector, `${sector.change_percent}%`);
     * });
     * ```
     */
    getSectorHeatmap: (options?: RequestOptions): Promise<SectorHeatmapResponse> => {
      return this.request<SectorHeatmapResponse>(
        'GET',
        '/api/search/sector-heatmap',
        undefined,
        options
      );
    },

    /**
     * Quick search for ticker autocomplete
     * 
     * @param ticker - Ticker to look up
     * @param options - Request options
     * @returns Basic stock info or not found
     * 
     * @example
     * ```typescript
     * const result = await api.search.quick('AAPL');
     * if (result.found) {
     *   console.log(result.name, result.price_display);
     * }
     * ```
     */
    quick: (ticker: string, options?: RequestOptions): Promise<QuickSearchResponse> => {
      return this.request<QuickSearchResponse>(
        'POST',
        '/api/search/quick',
        { ticker },
        options
      );
    },

    /**
     * Check search service health
     * 
     * @param options - Request options
     * @returns Health status
     */
    health: (options?: RequestOptions): Promise<{ status: string; service: string; cache_size: number; timestamp: string }> => {
      return this.request<{ status: string; service: string; cache_size: number; timestamp: string }>(
        'GET',
        '/api/search/health',
        undefined,
        options
      );
    },
  };

  // ---------------------------------------------------------------------------
  // PORTFOLIO ENDPOINTS
  // ---------------------------------------------------------------------------

  /** Portfolio analysis endpoints */
  public readonly portfolio = {
    /**
     * Analyze a portfolio of positions
     * 
     * @param positions - Array of portfolio positions
     * @param riskFreeRate - Risk-free rate for CAPM (default: 0.02 / 2%)
     * @param marketReturn - Expected market return (default: 0.10 / 10%)
     * @param options - Request options
     * @returns Portfolio metrics, allocation, and risk analysis
     * 
     * @example
     * ```typescript
     * const positions = [
     *   { ticker: 'AAPL', shares: 100, total_cost_basis: 15000 },
     *   { ticker: 'MSFT', shares: 50, total_cost_basis: 18000 },
     * ];
     * const result = await api.portfolio.analyze(positions);
     * console.log(`Total Value: $${result.portfolio_metrics.total_value}`);
     * console.log(`Beta: ${result.risk_metrics.portfolio_beta}`);
     * ```
     */
    analyze: (
      positions: PortfolioPositionInput[],
      riskFreeRate: number = 0.02,
      marketReturn: number = 0.10,
      options?: RequestOptions
    ): Promise<PortfolioAnalyzeResponse> => {
      return this.request<PortfolioAnalyzeResponse>(
        'POST',
        '/api/portfolio/analyze',
        {
          positions,
          risk_free_rate: riskFreeRate,
          market_return: marketReturn,
        },
        options
      );
    },

    /**
     * Analyze a single stock for portfolio context
     * 
     * @param ticker - Stock ticker symbol
     * @param options - Request options
     * @returns Stock analysis with CAPM and risk metrics
     * 
     * @example
     * ```typescript
     * const result = await api.portfolio.analyzeStock('AAPL');
     * console.log(`Expected Return (CAPM): ${result.capm.expected_return}%`);
     * ```
     */
    analyzeStock: (ticker: string, options?: RequestOptions): Promise<PortfolioStockResponse> => {
      return this.request<PortfolioStockResponse>(
        'POST',
        '/api/portfolio/stock',
        { ticker },
        options
      );
    },

    /**
     * Calculate CAPM expected return
     * 
     * @param beta - Stock/portfolio beta
     * @param riskFreeRate - Risk-free rate (default: 0.02)
     * @param marketReturn - Expected market return (default: 0.10)
     * @param options - Request options
     * @returns CAPM calculation result
     * 
     * @example
     * ```typescript
     * const result = await api.portfolio.calculateCAPM(1.2);
     * console.log(`Expected Return: ${result.expected_return}%`);
     * ```
     */
    calculateCAPM: (
      beta: number,
      riskFreeRate: number = 0.02,
      marketReturn: number = 0.10,
      options?: RequestOptions
    ): Promise<PortfolioCAPMResponse> => {
      return this.request<PortfolioCAPMResponse>(
        'POST',
        '/api/portfolio/capm',
        {
          beta,
          risk_free_rate: riskFreeRate,
          market_return: marketReturn,
        },
        options
      );
    },
  };

  // ---------------------------------------------------------------------------
  // QUANT LAB ENDPOINTS
  // ---------------------------------------------------------------------------

  /** Quant Lab analysis endpoints */
  public readonly quantLab = {
    /**
     * Get comprehensive multi-factor analysis for a ticker
     * 
     * Includes:
     * - 6-factor scoring (momentum, value, quality, growth, volatility, technical)
     * - Composite alpha score
     * - Monte Carlo price forecasting
     * - Risk analysis (VaR, drawdowns, volatility)
     * - Technical pattern recognition
     * - Trading signal synthesis
     * 
     * @param ticker - Stock ticker symbol
     * @param options - Request options
     * @returns Complete quantitative analysis
     * 
     * @example
     * ```typescript
     * const result = await api.quantLab.analyze('AAPL');
     * console.log(`Alpha Score: ${result.alpha_score.score}`);
     * console.log(`Signal: ${result.signal.action} (${result.signal.confidence})`);
     * console.log(`1Y Target: $${result.price_forecast.forecasts['1y']?.median}`);
     * ```
     */
    analyze: (ticker: string, options?: RequestOptions): Promise<QuantLabResponse> => {
      return this.request<QuantLabResponse>(
        'POST',
        '/api/quant-lab',
        { ticker },
        options
      );
    },

    /**
     * Check Quant Lab service health
     * 
     * @param options - Request options
     * @returns Health status and cache info
     */
    health: (options?: RequestOptions): Promise<QuantLabHealthResponse> => {
      return this.request<QuantLabHealthResponse>(
        'GET',
        '/api/quant-lab/health',
        undefined,
        options
      );
    },

    /**
     * Clear the analysis cache (admin function)
     * 
     * @param options - Request options
     * @returns Number of entries cleared
     */
    clearCache: (options?: RequestOptions): Promise<{ success: boolean; message: string; entries_cleared: number }> => {
      return this.request<{ success: boolean; message: string; entries_cleared: number }>(
        'POST',
        '/api/quant-lab/cache/clear',
        undefined,
        options
      );
    },
  };
}

// -----------------------------------------------------------------------------
// SINGLETON INSTANCE
// -----------------------------------------------------------------------------

/**
 * Default API client instance
 * 
 * Use this for simple cases where you don't need custom configuration:
 * ```typescript
 * import { api } from '@/lib/api';
 * const data = await api.search.getStock('AAPL');
 * ```
 */
export const api = new CaverayApiClient();

// -----------------------------------------------------------------------------
// CONVENIENCE EXPORTS
// -----------------------------------------------------------------------------

// Re-export types that consumers commonly need
export type { RequestOptions };

/**
 * Create a new API client with custom configuration
 * 
 * @example
 * ```typescript
 * const customApi = createApiClient('http://localhost:5000', { timeout: 60000 });
 * ```
 */
export const createApiClient = (baseUrl?: string, options?: RequestOptions): CaverayApiClient => {
  return new CaverayApiClient(baseUrl, options);
};

// -----------------------------------------------------------------------------
// CONVENIENCE FUNCTION EXPORTS
// -----------------------------------------------------------------------------
// These functions provide a simpler interface for common operations
// They use the default api singleton instance

/** Analyze sentiment for a ticker */
export const analyzeSentiment = (ticker: string, numArticles?: number, options?: RequestOptions) =>
  api.sentiment.analyze(ticker, numArticles, options);

/** Get social sentiment screening */
export const getSocialScreening = (tickers?: string[], options?: RequestOptions) =>
  api.sentiment.socialScreening(tickers, options);

/** Analyze financials for a ticker */
export const analyzeFinancials = (ticker: string, options?: RequestOptions) =>
  api.financials.analyze(ticker, options);

/** Analyze insider trading for a ticker */
export const analyzeInsider = (ticker: string, months?: number, options?: RequestOptions) =>
  api.insider.analyze(ticker, months, options);

/** Search for stock data */
export const searchStock = (ticker: string, options?: RequestOptions) =>
  api.search.getStock(ticker, options);

/** Get chart data for a ticker */
export const getChartData = (ticker: string, period?: '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' | 'max', options?: RequestOptions) =>
  api.search.getChart(ticker, period, options);

/** Quick search for a ticker */
export const quickSearch = (ticker: string, options?: RequestOptions) =>
  api.search.quick(ticker, options);

/** Compare multiple stocks */
export const compareStocks = (tickers: string[], options?: RequestOptions) =>
  api.search.compare(tickers, options);

/** Get comparison chart data */
export const getCompareChart = (tickers: string[], period?: '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y', options?: RequestOptions) =>
  api.search.compareChart(tickers, period, options);

/** Get market movers */
export const getMarketMovers = (options?: RequestOptions) =>
  api.search.getMovers(options);

/** Get sector heatmap */
export const getSectorHeatmap = (options?: RequestOptions) =>
  api.search.getSectorHeatmap(options);

/** Analyze portfolio */
export const analyzePortfolio = (
  positions: PortfolioPositionInput[],
  riskFreeRate?: number,
  marketReturn?: number,
  options?: RequestOptions
) => api.portfolio.analyze(positions, riskFreeRate, marketReturn, options);

/** Analyze with Quant Lab */
export const analyzeQuantLab = (ticker: string, options?: RequestOptions) =>
  api.quantLab.analyze(ticker, options);
