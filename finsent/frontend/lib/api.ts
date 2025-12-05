import type {
  SentimentAnalysisResponse,
  ScreeningResponse,
  FinancialsResponse,
  InsiderResponse,
  SearchResponse,
  ChartResponse,
  PortfolioResponse,
  PortfolioPosition,
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

  // Sentiment Analysis
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

  // Financial Analysis
  async analyzeFinancials(ticker: string): Promise<FinancialsResponse> {
    return this.request<FinancialsResponse>('/api/financials', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    });
  }

  // Insider Trading
  async analyzeInsider(ticker: string, months: number = 12): Promise<InsiderResponse> {
    return this.request<InsiderResponse>('/api/insider', {
      method: 'POST',
      body: JSON.stringify({ ticker, months }),
    });
  }

  // Stock Search
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

  // Portfolio
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

  // Health Check
  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.request('/api/health');
  }
}

const api = new ApiClient(API_BASE_URL);

// Export individual functions
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

export const analyzePortfolio = (
  positions: PortfolioPosition[],
  riskFreeRate?: number,
  marketReturn?: number
) => api.analyzePortfolio(positions, riskFreeRate, marketReturn);

export const healthCheck = () => api.healthCheck();

export { api };
