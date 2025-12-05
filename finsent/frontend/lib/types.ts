// Common types
export interface CompanyInfo {
  name: string;
  ticker: string;
  sector: string;
  industry: string;
  price: number | null;
  price_display?: string;
  market_cap: number | null;
  market_cap_display: string;
  currency?: string;
}

// Sentiment Analysis types
export interface SentimentArticle {
  title: string;
  link: string;
  published: string;
  polarity: number;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
}

export interface SentimentSummary {
  Positive: { count: number; percentage: number };
  Negative: { count: number; percentage: number };
  Neutral: { count: number; percentage: number };
}

export interface SentimentResponse {
  ticker: string;
  articles: SentimentArticle[];
  summary: SentimentSummary;
  total_articles: number;
  model: string;
}

// Social Screening types
export interface ScreeningStock {
  ticker: string;
  company: string;
  price: number | null;
  price_display: string;
  change: number | null;
  pct_change: number | null;
  change_display: string;
  volume: number | null;
  volume_display: string;
  stocktwits: number | null;
  stocktwits_display: string;
  stocktwits_detail: string | null;
  x_sentiment: number | null;
  x_display: string;
  x_detail: string | null;
  news_sentiment: number | null;
  news_display: string;
  news_detail: string | null;
  composite: number | null;
  composite_display: string;
  composite_status: 'positive' | 'negative' | 'neutral';
}

export interface ScreeningResponse {
  tickers: string[];
  results: ScreeningStock[];
  sources: {
    market_data: boolean;
    stocktwits: boolean;
    x_sentiment: boolean;
    news: boolean;
  };
  timestamp: string;
}

// Financial Analysis types
export interface FinancialScore {
  score: number | null;
  max_score?: number;
  display: string;
  interpretation: string;
  status: 'positive' | 'negative' | 'neutral';
  components?: any;
}

export interface FinancialMetric {
  name: string;
  value: number | null;
  display: string;
  status: 'positive' | 'negative' | 'neutral';
}

export interface FinancialsResponse {
  ticker: string;
  company: CompanyInfo;
  scores: {
    piotroski: FinancialScore;
    altman: FinancialScore;
    beneish: FinancialScore;
  };
  metrics: {
    valuation: FinancialMetric[];
    profitability: FinancialMetric[];
    leverage: FinancialMetric[];
  };
  timestamp: string;
  disclaimer: string;
}

// Insider Trading types
export interface InsiderTransaction {
  insider: string;
  title: string;
  type: string;
  type_status: 'positive' | 'negative' | 'neutral';
  shares: number;
  shares_display: string;
  value: number;
  value_display: string;
  date: string;
  date_raw: string | null;
}

export interface InsiderSummary {
  total_buys: number;
  total_sells: number;
  buy_value: number;
  sell_value: number;
  net_value: number;
  buy_shares: number;
  sell_shares: number;
  buy_value_display: string;
  sell_value_display: string;
  net_value_display: string;
  net_positive: boolean;
}

export interface InsiderSentiment {
  sentiment: string;
  status: 'positive' | 'negative' | 'neutral';
  score: number;
  description: string;
}

export interface ClusterAlert {
  type: 'cluster_buy' | 'cluster_sell';
  status: 'positive' | 'negative';
  message: string;
  description: string;
  insiders: Array<{ name: string; title: string; value: string }>;
  insider_count: number;
  total_value: number;
  total_value_display: string;
  week: string;
  week_display: string;
}

export interface InsiderResponse {
  ticker: string;
  company: CompanyInfo;
  period_months: number;
  transactions: InsiderTransaction[];
  summary: InsiderSummary;
  monthly_data: Array<{ month: string; label: string; buys: number; sells: number }>;
  cluster_alerts: ClusterAlert[];
  sentiment: InsiderSentiment;
  institutional: any;
  signals: any[];
  has_transaction_data: boolean;
  has_institutional_data: boolean;
  timestamp: string;
  disclaimer: string;
}

// Stock Search types
export interface StockOverview {
  name: string;
  ticker: string;
  exchange: string;
  sector: string;
  industry: string;
  currency: string;
  price: number | null;
  price_display: string;
  change: number | null;
  change_percent: number | null;
  change_display: string;
  change_percent_display: string;
  change_status: 'positive' | 'negative' | 'neutral';
  market_cap: number | null;
  market_cap_display: string;
  volume: number | null;
  volume_display: string;
  avg_volume: number | null;
  avg_volume_display: string;
  fifty_two_high: number | null;
  fifty_two_high_display: string;
  fifty_two_low: number | null;
  fifty_two_low_display: string;
  range_position: number | null;
  day_high: number | null;
  day_low: number | null;
  open: number | null;
  prev_close: number | null;
  beta: number | null;
}

export interface SearchResponse {
  ticker: string;
  overview: StockOverview;
  key_stats: Array<{ label: string; value: string }>;
  valuation: FinancialMetric[];
  profitability: FinancialMetric[];
  financial_health: FinancialMetric[];
  growth: FinancialMetric[];
  dividend: any;
  analyst: any;
  trading: any;
  profile: any;
  news: Array<{ title: string; source: string; link: string; published: string; published_relative: string }>;
  timestamp: string;
}

export interface ChartData {
  ticker: string;
  period: string;
  data: {
    dates: string[];
    prices: number[];
    volumes: number[];
    highs: number[];
    lows: number[];
    opens: number[];
  };
}

// Portfolio types
export interface PortfolioPosition {
  ticker: string;
  shares: number;
  total_cost_basis: number;
}

export interface StockAnalysis {
  ticker: string;
  name: string;
  current_price: number;
  previous_close: number | null;
  beta: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  capm: {
    expected_return: number;
    beta: number;
    risk_free_rate: number;
    market_return: number;
    risk_premium: number;
  };
  pe_ratio: number | null;
  market_cap: number | null;
  market_cap_display: string;
  dividend_yield: number | null;
  sector: string;
  industry: string;
  shares?: number;
  cost_basis?: number;
  current_value?: number;
  gain_loss?: number;
  gain_loss_percent?: number;
}

export interface PortfolioMetrics {
  total_value: number;
  total_cost: number;
  total_gain_loss: number;
  total_gain_loss_percent: number;
  positions_count: number;
  positions?: Array<{
    ticker: string;
    shares: number;
    cost_basis: number;
    cost_basis_total: number;
    current_price: number;
    current_value: number;
    gain_loss: number;
    gain_loss_percent: number;
    beta: number;
    sector: string;
    industry: string;
  }>;
}

export interface AllocationData {
  sector: Array<{ name: string; value: number; percentage: number }>;
  industry: Array<{ name: string; value: number; percentage: number }>;
  ticker: Array<{ ticker: string; value: number; percentage: number }>;
}

export interface PortfolioResponse {
  portfolio_metrics: PortfolioMetrics;
  allocation: AllocationData;
  risk_metrics: {
    portfolio_beta: number;
    diversification_score: number;
    concentration_risk: string;
    num_positions: number;
    num_sectors: number;
    max_position_pct: number;
  };
  portfolio_capm: {
    expected_return: number;
    beta: number;
    risk_free_rate: number;
    market_return: number;
    risk_premium: number;
  };
  stock_analyses: StockAnalysis[];
  timestamp: string;
}
