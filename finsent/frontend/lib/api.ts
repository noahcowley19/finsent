import type {
  SentimentResponse,
  ScreeningResponse,
  FinancialResponse,
  InsiderResponse,
  SearchResponse,
  ChartData,
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
  async analyzeSentiment(ticker: string, numArticles: number = 8): Promise<SentimentResponse> {
    return this.request<SentimentResponse>('/api/analyze', {
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
  async analyzeFinancials(ticker: string): Promise<FinancialResponse> {
    return this.request<FinancialResponse>('/api/financials', {
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

  async getChartData(ticker: string, period: string = '1y'): Promise<ChartData> {
    return this.request<ChartData>('/api/search/chart', {
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

export const api = new ApiClient(API_BASE_URL);
