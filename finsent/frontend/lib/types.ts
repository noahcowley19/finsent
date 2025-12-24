// =============================================================================
// CAVERAY API TYPES - Complete TypeScript definitions for Flask Backend
// =============================================================================
// These types exactly match the response structures from:
// - sentiment.py (sentiment analysis, social screening)
// - financials.py (Piotroski, Altman Z, Beneish M scores)
// - insider.py (insider transactions, institutional holdings)
// - search.py (stock search, charts, market data)
// - portfolio.py (portfolio analysis, CAPM)
// - quant_lab.py (multi-factor analysis, forecasting)
// =============================================================================

// -----------------------------------------------------------------------------
// COMMON TYPES
// -----------------------------------------------------------------------------

/** Status indicator for UI coloring */
export type StatusType = 'positive' | 'negative' | 'neutral' | 'warning';

/** Generic API error response */
export interface ApiError {
  error: string;
  error_type: 'validation' | 'invalid_ticker' | 'invalid_asset' | 'fetch_error' | 'server_error' | 'data_error' | 'analysis_error';
  details?: string;
}

/** Base metric with display formatting */
export interface DisplayMetric {
  name: string;
  value: number | null;
  display: string;
  status: StatusType;
  description?: string;
}

// -----------------------------------------------------------------------------
// SENTIMENT ANALYSIS TYPES (sentiment.py)
// -----------------------------------------------------------------------------

/** Individual analyzed article from /api/analyze */
export interface SentimentArticle {
  title: string;
  link: string;
  published: string;
  polarity: number;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
}

/** Sentiment summary statistics */
export interface SentimentSummary {
  Positive: { count: number; percentage: number };
  Negative: { count: number; percentage: number };
  Neutral: { count: number; percentage: number };
}

/** Response from POST /api/analyze */
export interface SentimentAnalysisResponse {
  ticker: string;
  articles: SentimentArticle[];
  summary: SentimentSummary;
  total_articles: number;
  model: string;
}

/** StockTwits sentiment data */
export interface StockTwitsSentiment {
  score: number;
  bullish: number;
  bearish: number;
  total_posts: number;
  labeled_posts: number;
  watchlist_count: number;
  source: 'StockTwits';
}

/** X/Twitter sentiment data */
export interface XSentiment {
  score: number;
  positive: number;
  negative: number;
  neutral: number;
  total_analyzed: number;
  source: 'X/Social';
}

/** News sentiment data */
export interface NewsSentiment {
  score: number;
  positive: number;
  negative: number;
  neutral: number;
  total_articles: number;
  source: 'News';
}

/** Individual stock result in social screening */
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
  composite_status: StatusType;
}

/** Response from POST /api/social-screening */
export interface SocialScreeningResponse {
  tickers: string[];
  results: ScreeningStock[];
  sources: {
    market_data: boolean;
    stocktwits: boolean;
    x_sentiment: boolean;
    news: boolean;
  };
  timestamp: string;
  cache_ttl: number;
}

/** Response from GET /api/social-screening/defaults */
export interface ScreeningDefaultsResponse {
  tickers: string[];
  max_tickers: number;
}

/** Response from GET /api/sentiment/health */
export interface SentimentHealthResponse {
  status: string;
  finbert_available: boolean;
  fallback: string;
  hf_token_set: boolean;
  features: string[];
}

// -----------------------------------------------------------------------------
// FINANCIAL ANALYSIS TYPES (financials.py)
// -----------------------------------------------------------------------------

/** Company info from financials endpoint */
export interface FinancialsCompanyInfo {
  name: string;
  ticker: string;
  sector: string;
  industry: string;
  price: number | null;
  market_cap: number | null;
  market_cap_display: string;
  currency: string;
}

/** Piotroski F-Score components */
export interface PiotroskiComponents {
  net_income_positive: boolean | null;
  ocf_positive: boolean | null;
  roa_increasing: boolean | null;
  ocf_exceeds_net_income: boolean | null;
  leverage_decreasing: boolean | null;
  current_ratio_increasing: boolean | null;
  no_dilution: boolean | null;
  gross_margin_increasing: boolean | null;
  asset_turnover_increasing: boolean | null;
}

/** Piotroski F-Score result */
export interface PiotroskiScore {
  score: number | null;
  max_score: number;
  display: string;
  interpretation: 'Strong' | 'Neutral' | 'Weak' | 'Insufficient Data';
  status: StatusType;
  components: PiotroskiComponents;
  error?: string;
}

/** Altman Z-Score components */
export interface AltmanComponents {
  working_capital_to_assets: number | null;
  retained_earnings_to_assets: number | null;
  ebit_to_assets: number | null;
  market_cap_to_liabilities: number | null;
  revenue_to_assets: number | null;
}

/** Altman Z-Score result */
export interface AltmanZScore {
  score: number | null;
  display: string;
  interpretation: 'Safe Zone' | 'Grey Zone' | 'Distress Zone' | 'Insufficient Data';
  status: StatusType;
  components: AltmanComponents;
  error?: string;
}

/** Beneish M-Score components */
export interface BeneishComponents {
  dsri: number | null;
  gmi: number | null;
  aqi: number | null;
  sgi: number | null;
  depi: number | null;
  sgai: number | null;
  tata: number | null;
  lvgi: number | null;
}

/** Beneish M-Score result */
export interface BeneishMScore {
  score: number | null;
  display: string;
  interpretation: 'Unlikely Manipulator' | 'Possible Manipulator' | 'Insufficient Data';
  status: StatusType;
  components: BeneishComponents;
  error?: string;
}

/** Financial metric for display */
export interface FinancialMetric {
  name: string;
  value: number | null;
  display: string;
  status: StatusType;
}

/** Categorized financial metrics */
export interface FinancialMetrics {
  valuation: FinancialMetric[];
  profitability: FinancialMetric[];
  leverage: FinancialMetric[];
  error?: string;
}

/** Response from POST /api/financials */
export interface FinancialsResponse {
  ticker: string;
  company: FinancialsCompanyInfo;
  scores: {
    piotroski: PiotroskiScore;
    altman: AltmanZScore;
    beneish: BeneishMScore;
  };
  metrics: FinancialMetrics;
  timestamp: string;
  disclaimer: string;
}

// -----------------------------------------------------------------------------
// INSIDER TRADING TYPES (insider.py)
// -----------------------------------------------------------------------------

/** Company info from insider endpoint */
export interface InsiderCompanyInfo {
  name: string;
  ticker: string;
  sector: string;
  industry: string;
  price: number | null;
  price_display: string;
  market_cap: number | null;
  market_cap_display: string;
  currency: string;
}

/** Individual insider transaction */
export interface InsiderTransaction {
  insider: string;
  title: string;
  type: string;
  type_status: StatusType;
  shares: number;
  shares_display: string;
  value: number;
  value_display: string;
  date: string;
  date_raw: string | null;
}

/** Transaction summary statistics */
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

/** Monthly transaction data for charts */
export interface MonthlyTransactionData {
  month: string;
  label: string;
  buys: number;
  sells: number;
}

/** Cluster alert for coordinated insider activity */
export interface ClusterAlert {
  type: 'cluster_buy' | 'cluster_sell';
  status: StatusType;
  message: string;
  description: string;
  insiders: Array<{ name: string; title: string; value: string }>;
  insider_count: number;
  total_value: number;
  total_value_display: string;
  week: string;
  week_display: string;
}

/** Insider sentiment analysis */
export interface InsiderSentiment {
  sentiment: 'Very Bullish' | 'Bullish' | 'Neutral' | 'Bearish' | 'Very Bearish' | 'No Data';
  status: StatusType;
  score: number;
  description: string;
}

/** Institutional holder info */
export interface InstitutionalHolder {
  name: string;
  shares: number;
  shares_display: string;
  value: number;
  value_display: string;
  percent: number;
  percent_display: string;
  date_reported: string;
}

/** Institutional holdings summary */
export interface InstitutionalData {
  holders: InstitutionalHolder[];
  summary: {
    total_institutional: number | null;
    total_insider: number | null;
    top_holders_percent: number;
  };
  has_data: boolean;
}

/** Key signal/alert */
export interface InsiderSignal {
  type: string;
  status: StatusType;
  title: string;
  description: string;
}

/** Response from POST /api/insider */
export interface InsiderResponse {
  ticker: string;
  company: InsiderCompanyInfo;
  period_months: number;
  transactions: InsiderTransaction[];
  summary: InsiderSummary;
  monthly_data: MonthlyTransactionData[];
  cluster_alerts: ClusterAlert[];
  sentiment: InsiderSentiment;
  institutional: InstitutionalData;
  signals: InsiderSignal[];
  has_transaction_data: boolean;
  has_institutional_data: boolean;
  timestamp: string;
  disclaimer: string;
}

// -----------------------------------------------------------------------------
// STOCK SEARCH TYPES (search.py)
// -----------------------------------------------------------------------------

/** Stock overview/summary */
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
  change_status: StatusType;
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
  distance_from_high: number | null;
  distance_from_low: number | null;
  day_high: number | null;
  day_low: number | null;
  open: number | null;
  prev_close: number | null;
  prev_close_display: string;
  beta: number | null;
  ma50: number | null;
  ma50_display: string;
  ma200: number | null;
  ma200_display: string;
  above_ma50: boolean | null;
  above_ma200: boolean | null;
  bid: number | null;
  ask: number | null;
  bid_size: number | null;
  ask_size: number | null;
}

/** Key stat for grid display */
export interface KeyStat {
  label: string;
  value: string;
  category: string;
}

/** Dividend information */
export interface DividendInfo {
  has_dividend: boolean;
  yield: number | null;
  yield_display: string;
  rate: number | null;
  rate_display: string;
  payout_ratio: number | null;
  payout_ratio_display: string;
  payout_status: StatusType;
  ex_date: string;
  five_yr_avg: number | null;
  five_yr_avg_display: string;
  trailing_annual: number | null;
  trailing_annual_display: string;
}

/** Analyst ratings and targets */
export interface AnalystData {
  has_data: boolean;
  target_high: number | null;
  target_high_display: string;
  target_low: number | null;
  target_low_display: string;
  target_mean: number | null;
  target_mean_display: string;
  target_median: number | null;
  target_median_display: string;
  upside: number | null;
  upside_display: string;
  upside_high: number | null;
  upside_high_display: string;
  upside_low: number | null;
  upside_low_display: string;
  upside_status: StatusType;
  recommendation: string;
  recommendation_display: string;
  recommendation_status: StatusType;
  recommendation_mean: number | null;
  num_analysts: number | null;
  num_analysts_display: string;
}

/** Trading information */
export interface TradingInfo {
  avg_volume_10d: number | null;
  avg_volume_10d_display: string;
  avg_volume_3m: number | null;
  avg_volume_3m_display: string;
  shares_outstanding: number | null;
  shares_outstanding_display: string;
  float_shares: number | null;
  float_shares_display: string;
  shares_short: number | null;
  shares_short_display: string;
  short_ratio: number | null;
  short_percent: number | null;
  short_percent_display: string;
  insider_percent: number | null;
  insider_percent_display: string;
  institution_percent: number | null;
  institution_percent_display: string;
}

/** Company profile */
export interface CompanyProfile {
  description: string;
  website: string;
  employees: string;
  employees_raw: number | null;
  city: string;
  state: string;
  country: string;
  headquarters: string;
  phone: string;
  address: string;
  zip: string;
}

/** Peer companies */
export interface PeerData {
  sector: string;
  industry: string;
  market_cap_tier: 'mega' | 'large' | 'mid' | 'small';
  peer_tickers: string[];
}

/** Trading signal/alert */
export interface TradingSignal {
  type: string;
  status: StatusType;
  title: string;
  description: string;
}

/** News article */
export interface NewsArticle {
  title: string;
  source: string;
  link: string;
  published: string;
  published_relative: string;
}

/** Response from POST /api/search */
export interface SearchResponse {
  ticker: string;
  overview: StockOverview;
  key_stats: KeyStat[];
  valuation: DisplayMetric[];
  profitability: DisplayMetric[];
  financial_health: DisplayMetric[];
  growth: DisplayMetric[];
  dividend: DividendInfo;
  analyst: AnalystData;
  trading: TradingInfo;
  profile: CompanyProfile;
  peers: PeerData;
  signals: TradingSignal[];
  news: NewsArticle[];
  timestamp: string;
  data_freshness: 'real-time' | 'delayed';
}

/** Historical chart data point arrays */
export interface ChartData {
  dates: string[];
  prices: (number | null)[];
  volumes: number[];
  highs: (number | null)[];
  lows: (number | null)[];
  opens: (number | null)[];
  period_change: number | null;
  period_change_percent: number | null;
}

/** Response from POST /api/search/chart */
export interface ChartResponse {
  ticker: string;
  period: string;
  data: ChartData;
}

/** Stock comparison result */
export interface CompareStock {
  ticker: string;
  name: string;
  price: number | null;
  price_display: string;
  change_percent: number | null;
  change_status: StatusType;
  market_cap: number | null;
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

/** Response from POST /api/search/compare */
export interface CompareResponse {
  tickers: string[];
  results: CompareStock[];
  timestamp: string;
}

/** Response from POST /api/search/compare/chart */
export interface CompareChartResponse {
  dates: string[];
  series: Record<string, number[]>;
  period: string;
}

/** Market mover stock */
export interface MarketMover {
  ticker: string;
  price: number;
  price_display: string;
  change: number;
  change_percent: number;
  change_display: string;
  change_status: StatusType;
  volume: number;
  volume_display: string;
}

/** Response from GET /api/search/movers */
export interface MarketMoversResponse {
  gainers: MarketMover[];
  losers: MarketMover[];
  most_active: MarketMover[];
  timestamp: string;
}

/** Sector performance */
export interface SectorPerformance {
  sector: string;
  etf: string;
  change_percent: number;
  status: StatusType;
}

/** Response from GET /api/search/sector-heatmap */
export interface SectorHeatmapResponse {
  sectors: SectorPerformance[];
  timestamp: string;
}

/** Response from POST /api/search/quick */
export interface QuickSearchResponse {
  found: boolean;
  ticker?: string;
  name?: string;
  price?: number;
  price_display?: string;
  change_percent?: number;
  change_status?: StatusType;
  sector?: string;
  market_cap?: number;
  market_cap_display?: string;
  error?: string;
}

// -----------------------------------------------------------------------------
// PORTFOLIO ANALYSIS TYPES (portfolio.py)
// -----------------------------------------------------------------------------

/** Portfolio position input */
export interface PortfolioPositionInput {
  ticker: string;
  shares: number;
  total_cost_basis: number;
}

/** Analyzed portfolio position */
export interface PortfolioPosition {
  ticker: string;
  name: string;
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
}

/** Portfolio metrics summary */
export interface PortfolioMetrics {
  total_value: number;
  total_cost: number;
  total_gain_loss: number;
  total_gain_loss_percent: number;
  positions_count: number;
  positions: PortfolioPosition[];
}

/** Allocation breakdown */
export interface AllocationItem {
  name?: string;
  ticker?: string;
  value: number;
  percentage: number;
}

/** Portfolio allocation data */
export interface PortfolioAllocation {
  sector: AllocationItem[];
  industry: AllocationItem[];
  ticker: AllocationItem[];
}

/** Portfolio risk metrics */
export interface PortfolioRiskMetrics {
  portfolio_beta: number;
  diversification_score: number;
  concentration_risk: 'Very High' | 'High' | 'Moderate' | 'Low' | 'N/A';
  num_positions: number;
  num_sectors: number;
  max_position_pct: number;
}

/** CAPM calculation result */
export interface CAPMResult {
  expected_return: number;
  beta: number;
  risk_free_rate: number;
  market_return: number;
  risk_premium: number;
}

/** Individual stock analysis in portfolio context */
export interface PortfolioStockAnalysis {
  ticker: string;
  name: string;
  current_price: number;
  previous_close: number | null;
  beta: number;
  volatility: number | null;
  sharpe_ratio: number | null;
  capm: CAPMResult;
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

/** Response from POST /api/portfolio/analyze */
export interface PortfolioAnalyzeResponse {
  portfolio_metrics: PortfolioMetrics;
  allocation: PortfolioAllocation;
  risk_metrics: PortfolioRiskMetrics;
  portfolio_capm: CAPMResult;
  stock_analyses: PortfolioStockAnalysis[];
  timestamp: string;
}

/** Response from POST /api/portfolio/stock */
export interface PortfolioStockResponse extends PortfolioStockAnalysis {}

/** Response from POST /api/portfolio/capm */
export interface PortfolioCAPMResponse extends CAPMResult {}

// -----------------------------------------------------------------------------
// QUANT LAB TYPES (quant_lab.py)
// -----------------------------------------------------------------------------

/** Company info from quant lab */
export interface QuantLabCompany {
  name: string;
  ticker: string;
  sector: string;
  industry: string;
  price: number | null;
  price_display: string;
  change_percent: number | null;
  market_cap: number | null;
  market_cap_display: string;
}

/** Alpha score result */
export interface AlphaScore {
  score: number | null;
  display: string;
  status: StatusType;
  percentile: number | null;
}

/** Individual factor score with details */
export interface FactorScore {
  score: number | null;
  factors: Record<string, number | boolean | null>;
  status: StatusType;
}

/** Factor contribution to alpha score */
export interface FactorContribution {
  score: number;
  weight: number;
  contribution: number;
  status: StatusType;
}

/** All factor scores */
export interface FactorScores {
  momentum: FactorScore;
  value: FactorScore;
  quality: FactorScore;
  growth: FactorScore;
  volatility: FactorScore;
  technical: FactorScore;
}

/** Individual forecast period */
export interface ForecastPeriod {
  days: number;
  current_price: number;
  p10: number;
  p25: number;
  median: number;
  p75: number;
  p90: number;
  expected_return: number;
  upside_potential: number;
  downside_risk: number;
}

/** Price scenario */
export interface PriceScenario {
  label: string;
  description: string;
  probability: string;
  price_30d: number | null;
  price_90d: number | null;
  price_1y: number | null;
}

/** Trend analysis */
export interface TrendAnalysis {
  annual_expected_return: number;
  annual_volatility: number;
  sharpe_estimate: number | null;
  trend_direction: 'bullish' | 'bearish';
  volatility_regime: 'high' | 'low' | 'normal';
}

/** Price forecast data */
export interface PriceForecast {
  has_forecast: boolean;
  forecasts: {
    '30d'?: ForecastPeriod;
    '90d'?: ForecastPeriod;
    '1y'?: ForecastPeriod;
  };
  scenarios: {
    bull: PriceScenario;
    base: PriceScenario;
    bear: PriceScenario;
  };
  trend: TrendAnalysis;
  methodology: string;
}

/** Volatility metrics */
export interface VolatilityMetrics {
  annual: number;
  daily: number;
  upside: number | null;
  downside: number | null;
}

/** Value at Risk metrics */
export interface VaRMetrics {
  var_95_1d: number;
  var_99_1d: number;
  var_95_30d: number;
  var_99_30d: number;
  cvar_95: number;
}

/** Drawdown metrics */
export interface DrawdownMetrics {
  max_drawdown: number;
  current_drawdown: number;
  days_in_drawdown: number;
}

/** Return distribution analysis */
export interface DistributionAnalysis {
  skewness: number;
  kurtosis: number;
  fat_tails: boolean;
  skew_direction: 'left' | 'right' | 'symmetric';
}

/** Risk analysis data */
export interface RiskAnalysis {
  has_risk_data: boolean;
  risk_score: number;
  risk_level: 'Low' | 'Moderate' | 'High' | 'Extreme';
  risk_status: StatusType;
  volatility: VolatilityMetrics;
  var: VaRMetrics;
  drawdown: DrawdownMetrics;
  distribution: DistributionAnalysis;
  beta: number | null;
}

/** Technical pattern */
export interface TechnicalPattern {
  name: string;
  type: string;
  confidence: number;
  status: StatusType;
  description: string;
  target?: number;
  stop_loss?: number;
}

/** Support and resistance levels */
export interface SupportResistance {
  resistance_levels: number[];
  support_levels: number[];
  nearest_resistance: number | null;
  nearest_support: number | null;
  distance_to_resistance: number | null;
  distance_to_support: number | null;
}

/** Technical analysis data */
export interface TechnicalAnalysis {
  patterns: TechnicalPattern[];
  support_resistance: SupportResistance;
}

/** Position guidance */
export interface PositionGuidance {
  stop_loss: number;
  target_short: number | null;
  target_medium: number | null;
  target_long: number | null;
  risk_reward_ratio: number | null;
}

/** Trading signal/recommendation */
export interface QuantSignal {
  action: 'Strong Buy' | 'Buy' | 'Hold' | 'Sell' | 'Strong Sell';
  action_status: StatusType;
  confidence: 'High' | 'Medium' | 'Low';
  thesis: string[];
  bull_case: string[];
  bear_case: string[];
  position_guidance: PositionGuidance;
}

/** Response from POST /api/quant-lab */
export interface QuantLabResponse {
  ticker: string;
  company: QuantLabCompany;
  alpha_score: AlphaScore;
  factor_scores: FactorScores;
  factor_contributions: Record<string, FactorContribution>;
  price_forecast: PriceForecast;
  risk_analysis: RiskAnalysis;
  technical_analysis: TechnicalAnalysis;
  signal: QuantSignal;
  timestamp: string;
  disclaimer: string;
}

/** Response from GET /api/quant-lab/health */
export interface QuantLabHealthResponse {
  status: string;
  service: string;
  version: string;
  features: string[];
  cache_ttl_seconds: number;
  cache_entries: number;
}

// -----------------------------------------------------------------------------
// LOCAL STORAGE TYPES (client-side persistence)
// -----------------------------------------------------------------------------

/** Local portfolio position (before API enrichment) */
export interface LocalPortfolioPosition {
  id: string;
  ticker: string;
  shares: number;
  avgCost: number;
  dateAdded: string;
}

/** Local watchlist item (before API enrichment) */
export interface LocalWatchlistItem {
  id: string;
  ticker: string;
  dateAdded: string;
  notes?: string;
}

/** Saved quant strategy */
export interface SavedStrategy {
  id: string;
  name: string;
  conditions: StrategyCondition[];
  createdAt: string;
  updatedAt: string;
}

/** Strategy condition */
export interface StrategyCondition {
  id: string;
  type: 'entry' | 'exit';
  indicator: string;
  operator: string;
  value: number;
}

/** Backtest result */
export interface BacktestResult {
  strategyId: string;
  ticker: string;
  startDate: string;
  endDate: string;
  totalReturn: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  avgWin: number;
  avgLoss: number;
  totalTrades: number;
  equityCurve: { date: string; value: number }[];
}
