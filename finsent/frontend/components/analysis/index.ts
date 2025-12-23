// =============================================================================
// ANALYSIS COMPONENTS
// =============================================================================
// Barrel export for analysis components
//
// Location: frontend/components/analysis/index.ts
//
// =============================================================================

// Shared components
export { AnalysisHeader, type AnalysisHeaderProps } from './AnalysisHeader';
export { AnalysisTabs, type AnalysisTabsProps } from './AnalysisTabs';
export { StockOverviewCard, type StockOverviewCardProps, type StockStats } from './StockOverviewCard';

// Sentiment components
export { SentimentGauge, type SentimentGaugeProps } from './SentimentGauge';
export { SentimentTimeline, type SentimentTimelineProps, type TimelineDataPoint } from './SentimentTimeline';
export { SourceBreakdown, type SourceBreakdownProps, type SourceData } from './SourceBreakdown';

// Financial components
export { FinancialMetrics, type FinancialMetricsProps, type Metric } from './FinancialMetrics';
export { FinancialChart, type FinancialChartProps, type FinancialDataPoint } from './FinancialChart';

// Insider components
export { InsiderSummary, type InsiderSummaryProps, type InsiderSummaryData } from './InsiderSummary';
export { InsiderTable, type InsiderTableProps, type InsiderTransaction } from './InsiderTable';
