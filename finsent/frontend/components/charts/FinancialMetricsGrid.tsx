'use client';

import React, { useMemo } from 'react';

// =============================================================================
// FINANCIAL METRICS GRID
// =============================================================================
// 12 bar charts × 7 historical periods showing key financial metrics
// Professional-grade visualization inspired by Bloomberg/Qualtrim
// =============================================================================

// -----------------------------------------------------------------------------
// TYPES
// -----------------------------------------------------------------------------

export interface MetricDataPoint {
    period: string;
    value: number;
}

export interface MetricConfig {
    title: string;
    key: string;
    format: 'currency' | 'percent' | 'ratio' | 'number';
    color: string;
    description?: string;
}

export interface FinancialMetricsGridProps {
    symbol?: string;
    data?: Record<string, MetricDataPoint[]>;
}

// -----------------------------------------------------------------------------
// METRIC CONFIGURATIONS
// -----------------------------------------------------------------------------

const metricConfigs: MetricConfig[] = [
    // Row 1: Income Statement
    { title: 'Revenue', key: 'revenue', format: 'currency', color: '#3B82F6', description: 'Total revenue' },
    { title: 'Net Income', key: 'netIncome', format: 'currency', color: '#22C55E', description: 'Net profit' },
    { title: 'EPS', key: 'eps', format: 'currency', color: '#8B5CF6', description: 'Earnings per share' },

    // Row 2: Margins
    { title: 'Gross Margin', key: 'grossMargin', format: 'percent', color: '#F59E0B', description: 'Gross profit %' },
    { title: 'Operating Margin', key: 'operatingMargin', format: 'percent', color: '#EC4899', description: 'Operating profit %' },
    { title: 'Net Margin', key: 'netMargin', format: 'percent', color: '#06B6D4', description: 'Net profit %' },

    // Row 3: Cash Flow
    { title: 'Free Cash Flow', key: 'fcf', format: 'currency', color: '#10B981', description: 'FCF' },
    { title: 'Operating Cash Flow', key: 'ocf', format: 'currency', color: '#6366F1', description: 'OCF' },
    { title: 'CapEx', key: 'capex', format: 'currency', color: '#EF4444', description: 'Capital expenditures' },

    // Row 4: Returns & Leverage
    { title: 'ROE', key: 'roe', format: 'percent', color: '#14B8A6', description: 'Return on equity' },
    { title: 'ROA', key: 'roa', format: 'percent', color: '#A855F7', description: 'Return on assets' },
    { title: 'Debt/Equity', key: 'debtEquity', format: 'ratio', color: '#F97316', description: 'D/E ratio' },
];

// -----------------------------------------------------------------------------
// SAMPLE DATA GENERATOR
// -----------------------------------------------------------------------------

const generateSampleData = (): Record<string, MetricDataPoint[]> => {
    const years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024'];

    const generateTrend = (base: number, growth: number, volatility: number = 0.1): MetricDataPoint[] => {
        let value = base;
        return years.map(year => {
            value = value * (1 + growth + (Math.random() - 0.5) * volatility);
            return { period: year, value };
        });
    };

    return {
        revenue: generateTrend(250e9, 0.08, 0.05),
        netIncome: generateTrend(55e9, 0.10, 0.08),
        eps: generateTrend(3.5, 0.12, 0.10),
        grossMargin: generateTrend(38, 0.02, 0.03),
        operatingMargin: generateTrend(25, 0.01, 0.04),
        netMargin: generateTrend(22, 0.015, 0.05),
        fcf: generateTrend(70e9, 0.09, 0.12),
        ocf: generateTrend(95e9, 0.07, 0.08),
        capex: generateTrend(12e9, 0.05, 0.15),
        roe: generateTrend(45, -0.02, 0.08),
        roa: generateTrend(18, 0.01, 0.06),
        debtEquity: generateTrend(1.2, 0.03, 0.1),
    };
};

// -----------------------------------------------------------------------------
// FORMAT VALUE HELPER
// -----------------------------------------------------------------------------

const formatValue = (value: number, format: string): string => {
    switch (format) {
        case 'currency':
            if (Math.abs(value) >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
            if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
            if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
            if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
            return `$${value.toFixed(2)}`;
        case 'percent':
            return `${value.toFixed(1)}%`;
        case 'ratio':
            return value.toFixed(2);
        default:
            if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
            if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
            return value.toFixed(1);
    }
};

// -----------------------------------------------------------------------------
// METRIC CARD COMPONENT
// -----------------------------------------------------------------------------

interface MetricCardProps {
    config: MetricConfig;
    data: MetricDataPoint[];
    delay?: number;
}

const MetricCard: React.FC<MetricCardProps> = ({ config, data, delay = 0 }) => {
    const latestValue = data[data.length - 1]?.value || 0;
    const previousValue = data[data.length - 2]?.value || latestValue;
    const change = previousValue !== 0 ? ((latestValue - previousValue) / Math.abs(previousValue)) * 100 : 0;
    const isPositive = change >= 0;

    // Calculate bar heights
    const maxValue = Math.max(...data.map(d => Math.abs(d.value)));
    const minValue = Math.min(...data.map(d => d.value));
    const hasNegatives = minValue < 0;

    return (
        <div
            className="bg-white/90 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-5 hover:shadow-glass-lg hover:-translate-y-1 transition-all duration-300"
            style={{ animationDelay: `${delay}ms` }}
        >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
                <div>
                    <h3 className="text-sm font-semibold text-obsidian-700 mb-0.5">{config.title}</h3>
                    {config.description && (
                        <p className="text-xs text-obsidian-400">{config.description}</p>
                    )}
                </div>
                <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: config.color }}
                />
            </div>

            {/* Current Value */}
            <div className="mb-4">
                <p className="text-2xl font-bold text-obsidian-900 mb-1">
                    {formatValue(latestValue, config.format)}
                </p>
                <p className={`text-xs font-medium ${isPositive ? 'text-success-600' : 'text-coral-600'}`}>
                    {isPositive ? '▲' : '▼'} {Math.abs(change).toFixed(1)}% YoY
                </p>
            </div>

            {/* Bar Chart */}
            <div className="relative h-24 flex items-end gap-1">
                {data.map((point, index) => {
                    const normalizedHeight = maxValue > 0 ? (Math.abs(point.value) / maxValue) * 100 : 0;
                    const isNegative = point.value < 0;
                    const isLatest = index === data.length - 1;

                    return (
                        <div
                            key={point.period}
                            className="flex-1 flex flex-col items-center justify-end group relative"
                        >
                            {/* Tooltip */}
                            <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-obsidian-900/95 text-white text-xs px-2 py-1 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                                <p className="font-medium">{point.period}</p>
                                <p>{formatValue(point.value, config.format)}</p>
                            </div>

                            {/* Bar */}
                            <div
                                className="w-full rounded-t transition-all duration-500 group-hover:opacity-80"
                                style={{
                                    height: `${Math.max(normalizedHeight, 4)}%`,
                                    backgroundColor: isNegative ? '#EF4444' : config.color,
                                    opacity: 0.4 + (index / data.length) * 0.6,
                                    boxShadow: isLatest ? `0 0 0 1px white, 0 0 0 3px ${config.color}` : 'none',
                                }}
                            />

                            {/* X-axis label (only show first, middle, last) */}
                            {(index === 0 || index === data.length - 1 || index === 3) && (
                                <span className="text-[10px] text-obsidian-400 mt-1.5 font-medium">
                                    {point.period.slice(-2)}
                                </span>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Min/Max Range */}
            <div className="flex justify-between mt-3 text-[10px] text-obsidian-400">
                <span>Min: {formatValue(Math.min(...data.map(d => d.value)), config.format)}</span>
                <span>Max: {formatValue(Math.max(...data.map(d => d.value)), config.format)}</span>
            </div>
        </div>
    );
};

// -----------------------------------------------------------------------------
// MAIN GRID COMPONENT
// -----------------------------------------------------------------------------

export const FinancialMetricsGrid: React.FC<FinancialMetricsGridProps> = ({
    symbol,
    data: providedData,
}) => {
    const data = useMemo(() => providedData || generateSampleData(), [providedData]);

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold text-obsidian-900">
                        Financial Metrics{symbol ? ` · ${symbol}` : ''}
                    </h2>
                    <p className="text-sm text-obsidian-500 mt-1">
                        7-year historical data across 12 key metrics
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <select className="h-9 px-3 rounded-xl border border-cream-300 text-sm text-obsidian-700 bg-white/80 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-electric-500/20 focus:border-electric-500 transition-all">
                        <option>Annual</option>
                        <option>Quarterly</option>
                    </select>
                    <button className="h-9 px-4 rounded-xl border border-cream-300 text-sm font-medium text-obsidian-700 bg-white/80 backdrop-blur-sm hover:bg-cream-100 transition-all flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Export
                    </button>
                </div>
            </div>

            {/* Grid of 12 Metric Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {metricConfigs.map((config, index) => (
                    <MetricCard
                        key={config.key}
                        config={config}
                        data={data[config.key] || []}
                        delay={index * 50}
                    />
                ))}
            </div>

            {/* Legend */}
            <div className="flex flex-wrap items-center gap-4 pt-4 border-t border-cream-200/50">
                <p className="text-xs text-obsidian-400 font-medium">Categories:</p>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-electric-500" />
                    <span className="text-xs text-obsidian-500">Income Statement</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-amber-500" />
                    <span className="text-xs text-obsidian-500">Margins</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-emerald-500" />
                    <span className="text-xs text-obsidian-500">Cash Flow</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-orange-500" />
                    <span className="text-xs text-obsidian-500">Returns & Leverage</span>
                </div>
            </div>
        </div>
    );
};

export default FinancialMetricsGrid;
