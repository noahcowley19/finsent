import type {
  SentimentAnalysisResponse,
  ScreeningResponse,
  FinancialsResponse,
  InsiderResponse,
  SearchResponse,
  ChartResponse,
  PortfolioResponse,
  PortfolioPosition,
  MarketMoversResponse,
  CompareResponse,
  SectorPerformance,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Request failed' }));
      throw new Error(error.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // Sentiment endpoints
  async analyzeSentiment(ticker: string, numArticles: number = 8): Promise<SentimentAnalysisResponse> {
    return this.request<SentimentAnalysisResponse>('/api/analyze', {
      method: 'POST',
      body: JSON.stringify({ ticker, num_articles: numArticles }),
    });
  }

  async getSocialScreening(tickers?: string[]): Promise<ScreeningResponse> {
    return this.request<ScreeningResponse>('/api/social-screening', {
      method: 'POST',
      body: JSON.stringify({ tickers }),
    });
  }

  // Financial analysis endpoints
  async analyzeFinancials(ticker: string): Promise<FinancialsResponse> {
    return this.request<FinancialsResponse>('/api/financials', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    });
  }

  // Insider trading endpoints
  async analyzeInsider(ticker: string, months: number = 12): Promise<InsiderResponse> {
    return this.request<InsiderResponse>('/api/insider', {
      method: 'POST',
      body: JSON.stringify({ ticker, months }),
    });
  }

  // Stock search endpoints
  async searchStock(ticker: string): Promise<SearchResponse> {
    return this.request<SearchResponse>('/api/search', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    });
  }

  async getChartData(ticker: string, period: string = '1y'): Promise<ChartResponse> {
    return this.request<ChartResponse>('/api/search/chart', {
      method: 'POST',
      body: JSON.stringify({ ticker, period }),
    });
  }

  async quickSearch(ticker: string): Promise<{
    found: boolean;
    ticker?: string;
    name?: string;
    price?: number;
    price_display?: string;
    change_percent?: number;
    change_status?: string;
    sector?: string;
    market_cap_display?: string;
  }> {
    return this.request('/api/search/quick', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    });
  }

  async compareStocks(tickers: string[]): Promise<CompareResponse> {
    return this.request<CompareResponse>('/api/search/compare', {
      method: 'POST',
      body: JSON.stringify({ tickers }),
    });
  }

  async getCompareChart(tickers: string[], period: string = '1y'): Promise<{
    dates: string[];
    series: Record<string, number[]>;
    period: string;
  }> {
    return this.request('/api/search/compare/chart', {
      method: 'POST',
      body: JSON.stringify({ tickers, period }),
    });
  }

  async getMarketMovers(): Promise<MarketMoversResponse> {
    return this.request<MarketMoversResponse>('/api/search/movers');
  }

  async getSectorHeatmap(): Promise<{ sectors: SectorPerformance[]; timestamp: string }> {
    return this.request('/api/search/sector-heatmap');
  }

  // Portfolio endpoints
  async analyzePortfolio(
    positions: PortfolioPosition[],
    riskFreeRate: number = 0.02,
    marketReturn: number = 0.10
  ): Promise<PortfolioResponse> {
    return this.request<PortfolioResponse>('/api/portfolio/analyze', {
      method: 'POST',
      body: JSON.stringify({
        positions,
        risk_free_rate: riskFreeRate,
        market_return: marketReturn,
      }),
    });
  }

  // Health check
  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.request('/api/health');
  }
}

const api = new ApiClient(API_BASE_URL);

// Export individual functions for convenience
export const analyzeSentiment = (ticker: string, numArticles?: number) => 
  api.analyzeSentiment(ticker, numArticles);

export const getSocialScreening = (tickers?: string[]) => 
  api.getSocialScreening(tickers);

export const analyzeFinancials = (ticker: string) => 
  api.analyzeFinancials(ticker);

export const analyzeInsider = (ticker: string, months?: number) => 
  api.analyzeInsider(ticker, months);

export const searchStock = (ticker: string) => 
  api.searchStock(ticker);

export const getChartData = (ticker: string, period?: string) => 
  api.getChartData(ticker, period);

export const quickSearch = (ticker: string) =>
  api.quickSearch(ticker);

export const compareStocks = (tickers: string[]) =>
  api.compareStocks(tickers);

export const getCompareChart = (tickers: string[], period?: string) =>
  api.getCompareChart(tickers, period);

export const getMarketMovers = () =>
  api.getMarketMovers();

export const getSectorHeatmap = () =>
  api.getSectorHeatmap();

export const analyzePortfolio = (
  positions: PortfolioPosition[],
  riskFreeRate?: number,
  marketReturn?: number
) => api.analyzePortfolio(positions, riskFreeRate, marketReturn);

export const healthCheck = () => api.healthCheck();

export { api };
