// =============================================================================
// CAVERAY INTEGRATION - Main Export Index
// =============================================================================
// This file provides a clean API for importing all integration utilities
// 
// Usage examples:
// 
// // Import everything from lib
// import { api, useStockSearch, type SearchResponse } from '@/lib';
// 
// // Or import from specific modules
// import { api } from '@/lib/api';
// import { useStockSearch } from '@/lib/hooks';
// import type { SearchResponse } from '@/lib/types';
// =============================================================================

// -----------------------------------------------------------------------------
// API CLIENT EXPORTS
// -----------------------------------------------------------------------------

export {
  // Main client class
  CaverayApiClient,
  // Default singleton instance
  api,
  // Factory function
  createApiClient,
  // Error class
  CaverayApiError,
  // Types
  type RequestOptions,
} from './api';

// -----------------------------------------------------------------------------
// HOOK EXPORTS
// -----------------------------------------------------------------------------

export {
  // Base hooks
  useApi,
  useLazyApi,
  
  // Sentiment hooks
  useSentiment,
  useSocialScreening,
  useLazySentiment,
  
  // Financials hooks
  useFinancials,
  useLazyFinancials,
  
  // Insider hooks
  useInsider,
  useLazyInsider,
  
  // Search hooks
  useStockSearch,
  useChartData,
  useStockCompare,
  useMarketMovers,
  useSectorHeatmap,
  useQuickSearch,
  useLazyStockSearch,
  useLazyChartData,
  
  // Portfolio hooks
  usePortfolio,
  
  // Watchlist hooks
  useWatchlist,
  
  // Quant Lab hooks
  useQuantLab,
  useLazyQuantLab,
  
  // Strategy hooks
  useStrategies,
  
  // Combined hooks
  useStockData,
  useComprehensiveAnalysis,
  useDebouncedSearch,
} from './hooks';

// -----------------------------------------------------------------------------
// TYPE EXPORTS
// -----------------------------------------------------------------------------

export type {
  // Common types
  StatusType,
  ApiError,
  DisplayMetric,
  
  // Sentiment types
  SentimentArticle,
  SentimentSummary,
  SentimentAnalysisResponse,
  StockTwitsSentiment,
  XSentiment,
  NewsSentiment,
  ScreeningStock,
  SocialScreeningResponse,
  ScreeningDefaultsResponse,
  SentimentHealthResponse,
  
  // Financials types
  FinancialsCompanyInfo,
  PiotroskiComponents,
  PiotroskiScore,
  AltmanComponents,
  AltmanZScore,
  BeneishComponents,
  BeneishMScore,
  FinancialMetric,
  FinancialMetrics,
  FinancialsResponse,
  
  // Insider types
  InsiderCompanyInfo,
  InsiderTransaction,
  InsiderSummary,
  MonthlyTransactionData,
  ClusterAlert,
  InsiderSentiment,
  InstitutionalHolder,
  InstitutionalData,
  InsiderSignal,
  InsiderResponse,
  
  // Search types
  StockOverview,
  KeyStat,
  DividendInfo,
  AnalystData,
  TradingInfo,
  CompanyProfile,
  PeerData,
  TradingSignal,
  NewsArticle,
  SearchResponse,
  ChartData,
  ChartResponse,
  CompareStock,
  CompareResponse,
  CompareChartResponse,
  MarketMover,
  MarketMoversResponse,
  SectorPerformance,
  SectorHeatmapResponse,
  QuickSearchResponse,
  
  // Portfolio types
  PortfolioPositionInput,
  PortfolioPosition,
  PortfolioMetrics,
  AllocationItem,
  PortfolioAllocation,
  PortfolioRiskMetrics,
  CAPMResult,
  PortfolioStockAnalysis,
  PortfolioAnalyzeResponse,
  PortfolioStockResponse,
  PortfolioCAPMResponse,
  
  // Quant Lab types
  QuantLabCompany,
  AlphaScore,
  FactorScore,
  FactorContribution,
  FactorScores,
  ForecastPeriod,
  PriceScenario,
  TrendAnalysis,
  PriceForecast,
  VolatilityMetrics,
  VaRMetrics,
  DrawdownMetrics,
  DistributionAnalysis,
  RiskAnalysis,
  TechnicalPattern,
  SupportResistance,
  TechnicalAnalysis,
  PositionGuidance,
  QuantSignal,
  QuantLabResponse,
  QuantLabHealthResponse,
  
  // Local storage types
  LocalPortfolioPosition,
  LocalWatchlistItem,
  SavedStrategy,
  StrategyCondition,
  BacktestResult,
} from './types';

// -----------------------------------------------------------------------------
// UTILITY EXPORTS
// -----------------------------------------------------------------------------

/**
 * Format a number as currency
 */
export const formatCurrency = (value: number | null | undefined, currency = 'USD'): string => {
  if (value === null || value === undefined) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

/**
 * Format a number as percentage
 */
export const formatPercent = (value: number | null | undefined, decimals = 2): string => {
  if (value === null || value === undefined) return 'N/A';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
};

/**
 * Format large numbers with suffixes (K, M, B, T)
 */
export const formatLargeNumber = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return 'N/A';
  const absValue = Math.abs(value);
  if (absValue >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (absValue >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (absValue >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (absValue >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toLocaleString();
};

/**
 * Get status color class based on status type
 */
export const getStatusColor = (status: string): string => {
  switch (status) {
    case 'positive':
      return 'text-green-600';
    case 'negative':
      return 'text-red-600';
    case 'warning':
      return 'text-yellow-600';
    default:
      return 'text-gray-600';
  }
};

/**
 * Get status background color class based on status type
 */
export const getStatusBgColor = (status: string): string => {
  switch (status) {
    case 'positive':
      return 'bg-green-100';
    case 'negative':
      return 'bg-red-100';
    case 'warning':
      return 'bg-yellow-100';
    default:
      return 'bg-gray-100';
  }
};
