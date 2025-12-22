// Quant Lab TypeScript Type Definitions

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

export interface AlphaScore {
  score: number | null;
  display: string;
  status: 'positive' | 'negative' | 'neutral';
  percentile: number | null;
}

export interface FactorData {
  score: number | null;
  factors: Record<string, number | string | boolean | null>;
  status: 'positive' | 'negative' | 'neutral';
}

export interface FactorScores {
  momentum: FactorData;
  value: FactorData;
  quality: FactorData;
  growth: FactorData;
  volatility: FactorData;
  technical: FactorData;
}

export interface FactorContribution {
  score: number;
  weight: number;
  contribution: number;
  status: 'positive' | 'negative' | 'neutral';
}

export interface PriceForecastPeriod {
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

export interface ForecastScenario {
  label: string;
  description: string;
  probability: string;
  price_30d: number | null;
  price_90d: number | null;
  price_1y: number | null;
}

export interface TrendData {
  annual_expected_return: number;
  annual_volatility: number;
  sharpe_estimate: number | null;
  trend_direction: 'bullish' | 'bearish';
  volatility_regime: 'high' | 'normal' | 'low';
}

export interface PriceForecast {
  has_forecast: boolean;
  forecasts: {
    '30d'?: PriceForecastPeriod;
    '90d'?: PriceForecastPeriod;
    '1y'?: PriceForecastPeriod;
  };
  scenarios: {
    bull: ForecastScenario;
    base: ForecastScenario;
    bear: ForecastScenario;
  };
  trend?: TrendData;
  methodology?: string;
}

export interface VolatilityMetrics {
  annual: number;
  daily: number;
  upside: number | null;
  downside: number | null;
}

export interface VaRMetrics {
  var_95_1d: number;
  var_99_1d: number;
  var_95_30d: number;
  var_99_30d: number;
  cvar_95: number;
}

export interface DrawdownMetrics {
  max_drawdown: number;
  current_drawdown: number;
  days_in_drawdown: number;
}

export interface DistributionMetrics {
  skewness: number;
  kurtosis: number;
  fat_tails: boolean;
  skew_direction: 'left' | 'right' | 'symmetric';
}

export interface RiskAnalysis {
  has_risk_data: boolean;
  risk_score: number;
  risk_level: 'Low' | 'Moderate' | 'High' | 'Extreme';
  risk_status: 'positive' | 'neutral' | 'warning' | 'negative';
  volatility: VolatilityMetrics;
  var: VaRMetrics;
  drawdown: DrawdownMetrics;
  distribution: DistributionMetrics;
  beta: number | null;
}

export interface ChartPattern {
  name: string;
  type: string;
  confidence: number;
  status: 'positive' | 'negative' | 'neutral';
  description: string;
  target?: number;
  stop_loss?: number;
}

export interface SupportResistance {
  resistance_levels: number[];
  support_levels: number[];
  nearest_resistance: number | null;
  nearest_support: number | null;
  distance_to_resistance: number | null;
  distance_to_support: number | null;
}

export interface TechnicalAnalysis {
  patterns: ChartPattern[];
  support_resistance: SupportResistance;
}

export interface PositionGuidance {
  stop_loss: number;
  target_short: number | null;
  target_medium: number | null;
  target_long: number | null;
  risk_reward_ratio: number | null;
}

export interface Signal {
  action: 'Strong Buy' | 'Buy' | 'Hold' | 'Sell' | 'Strong Sell';
  action_status: 'positive' | 'negative' | 'neutral';
  confidence: 'High' | 'Medium' | 'Low';
  thesis: string[];
  bull_case: string[];
  bear_case: string[];
  position_guidance?: PositionGuidance;
}

export interface QuantLabResponse {
  ticker: string;
  company: QuantLabCompany;
  alpha_score: AlphaScore;
  factor_scores: FactorScores;
  factor_contributions: Record<string, FactorContribution>;
  price_forecast: PriceForecast;
  risk_analysis: RiskAnalysis;
  technical_analysis: TechnicalAnalysis;
  signal: Signal;
  timestamp: string;
  disclaimer: string;
}

// Helper type for factor names
export type FactorName = 'momentum' | 'value' | 'quality' | 'growth' | 'volatility' | 'technical';

// Display configuration for factors
export const FACTOR_CONFIG: Record<FactorName, { label: string; icon: string; description: string }> = {
  momentum: {
    label: 'Momentum',
    icon: '📈',
    description: 'Price momentum and relative strength indicators',
  },
  value: {
    label: 'Value',
    icon: '💰',
    description: 'Valuation metrics like P/E, P/B, and PEG ratios',
  },
  quality: {
    label: 'Quality',
    icon: '⭐',
    description: 'Profitability, margins, and financial health',
  },
  growth: {
    label: 'Growth',
    icon: '🌱',
    description: 'Revenue and earnings growth rates',
  },
  volatility: {
    label: 'Volatility',
    icon: '📊',
    description: 'Risk-adjusted metrics and volatility measures',
  },
  technical: {
    label: 'Technical',
    icon: '📉',
    description: 'Moving averages, MACD, and chart signals',
  },
};

// Status colors
export const STATUS_COLORS = {
  positive: {
    bg: 'var(--positive-light)',
    color: 'var(--positive)',
    border: 'rgba(0, 229, 160, 0.3)',
  },
  negative: {
    bg: 'var(--negative-light)',
    color: 'var(--negative)',
    border: 'rgba(255, 107, 107, 0.3)',
  },
  neutral: {
    bg: 'var(--neutral-light)',
    color: 'var(--text-secondary)',
    border: 'var(--border)',
  },
  warning: {
    bg: 'var(--warning-light)',
    color: 'var(--warning)',
    border: 'rgba(251, 191, 36, 0.3)',
  },
};
