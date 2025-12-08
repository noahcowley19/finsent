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
export interface Score {
  score: number | null;
  max_score?: number;
  display: string;
  interpretation: string;
  status: 'positive' | 'negative' | 'neutral';
  components?: Record<string, unknown>;
}

export interface FinancialScore {
  score: number | null;
  max_score?: number;
  display: string;
  interpretation: string;
  status: 'positive' | 'negative' | 'neutral';
  components?: Record<string, unknown>;
}

export interface FinancialMetric {
  name: string;
  value: number | null;
  display: string;
  status: 'positive' | 'negative' | 'neutral';
  description?: string;
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
  institutional: {
    holders: Array<{
      name: string;
      shares: number;
      shares_display: string;
      value: number;
      value_display: string;
      percent: number;
      percent_display: string;
    }>;
  };
  signals: Signal[];
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
  distance_from_high?: number | null;
  distance_from_low?: number | null;
  day_high: number | null;
  day_low: number | null;
  open: number | null;
  prev_close: number | null;
  prev_close_display?: string;
  beta: number | null;
  ma50?: number | null;
  ma50_display?: string;
  ma200?: number | null;
  ma200_display?: string;
  above_ma50?: boolean | null;
  above_ma200?: boolean | null;
}

export interface SearchSignal {
  type: string;
  status: 'positive' | 'negative' | 'neutral' | 'warning';
  title: string;
  description: string;
}

export interface PeerData {
  sector: string;
  industry: string;
  market_cap_tier: string;
  peer_tickers: string[];
}

export interface SearchResponse {
  ticker: string;
  overview: StockOverview;
  key_stats: Array<{ label: string; value: string; category?: string }>;
  valuation: FinancialMetric[];
  profitability: FinancialMetric[];
  financial_health: FinancialMetric[];
  growth: FinancialMetric[];
  dividend: {
    has_dividend: boolean;
    yield: number | null;
    yield_display: string;
    rate: number | null;
    rate_display: string;
    payout_ratio: number | null;
    payout_ratio_display: string;
    payout_status: 'positive' | 'negative' | 'neutral';
    ex_date: string | null;
    five_yr_avg?: number | null;
    five_yr_avg_display?: string;
    trailing_annual?: number | null;
    trailing_annual_display?: string;
  };
  analyst: {
    has_data: boolean;
    target_mean: number | null;
    target_mean_display: string;
    target_high: number | null;
    target_high_display: string;
    target_low: number | null;
    target_low_display: string;
    target_median?: number | null;
    target_median_display?: string;
    upside: number | null;
    upside_display: string;
    upside_high?: number | null;
    upside_high_display?: string;
    upside_low?: number | null;
    upside_low_display?: string;
    upside_status: 'positive' | 'negative' | 'neutral';
    num_analysts: number | null;
    num_analysts_display: string;
    recommendation: string;
    recommendation_display: string;
    recommendation_status: 'positive' | 'negative' | 'neutral';
    recommendation_mean?: number | null;
  };
  trading: {
    avg_volume_10d?: number | null;
    avg_volume_10d_display?: string;
    avg_volume_3m?: number | null;
    avg_volume_3m_display?: string;
    shares_outstanding?: number | null;
    shares_outstanding_display?: string;
    float_shares?: number | null;
    float_shares_display?: string;
    shares_short?: number | null;
    shares_short_display?: string;
    short_ratio?: number | null;
    short_percent?: number | null;
    short_percent_display?: string;
    insider_percent?: number | null;
    insider_percent_display?: string;
    institution_percent?: number | null;
    institution_percent_display?: string;
  };
  profile: {
    description: string;
    employees: string;
    employees_raw?: number | null;
    headquarters: string;
    website: string | null;
    city?: string;
    state?: string;
    country?: string;
    phone?: string;
    address?: string;
    zip?: string;
  };
  peers?: PeerData;
  signals?: SearchSignal[];
  news: NewsItem[];
  timestamp: string;
  data_freshness?: string;
}

export interface ChartData {
  ticker: string;
  period: string;
  data: {
    dates: string[];
    prices: (number | null)[];
    volumes: number[];
    highs: number[];
    lows: number[];
    opens: number[];
    period_change?: number | null;
    period_change_percent?: number | null;
  };
}

// Market data types
export interface MarketMover {
  ticker: string;
  price: number;
  price_display: string;
  change: number;
  change_percent: number;
  change_display: string;
  change_status: 'positive' | 'negative';
  volume: number;
  volume_display: string;
}

export interface SectorPerformance {
  sector: string;
  etf: string;
  change_percent: number;
  status: 'positive' | 'negative';
}

export interface MarketMoversResponse {
  gainers: MarketMover[];
  losers: MarketMover[];
  most_active: MarketMover[];
  timestamp: string;
}

export interface CompareStock {
  ticker: string;
  name: string;
  price: number;
  price_display: string;
  change_percent: number;
  change_status: string;
  market_cap: number;
  market_cap_display: string;
  sector: string;
  pe: number | null;
  pe_display: string;
  ps: number | null;
  pb: number | null;
  roe: number | null;
  roe_display: string;
  revenue_growth: number | null;
  gross_margin: number | null;
  operating_margin: number | null;
  dividend_yield: number | null;
  beta: number | null;
  fifty_two_high: number | null;
  fifty_two_low: number | null;
  range_position: number | null;
}

export interface CompareResponse {
  tickers: string[];
  results: CompareStock[];
  timestamp: string;
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

// Type aliases
export type SentimentAnalysisResponse = SentimentResponse;
export type ChartResponse = ChartData;
export type NewsItem = { title: string; source: string; link: string; published: string; published_relative: string };
export type Signal = { type: string; title: string; description: string; status: 'positive' | 'negative' | 'neutral' };
