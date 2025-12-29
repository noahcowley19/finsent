'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// DATA POINT GRID - Qualtrim-style Bar Charts
// =============================================================================
// Grid of color-coded bar charts showing various financial metrics over time.
// =============================================================================

export interface DataPoint {
    label: string;
    value: number;
}

export interface DataPointCardProps {
    title: string;
    data: DataPoint[];
    color: string;
    format?: 'currency' | 'percent' | 'number' | 'ratio';
    unit?: string;
}

const formatValue = (value: number, format?: string, unit?: string): string => {
    switch (format) {
        case 'currency':
            if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
            if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
            if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
            return `$${value.toFixed(2)}`;
        case 'percent':
            return `${value.toFixed(1)}%`;
        case 'ratio':
            return value.toFixed(2);
        default:
            if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
            if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
            if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
            return value.toFixed(1);
    }
};

export const DataPointCard: React.FC<DataPointCardProps> = ({
    title,
    data,
    color,
    format = 'number',
    unit,
}) => {
    const maxValue = Math.max(...data.map(d => Math.abs(d.value)));
    const latestValue = data[data.length - 1]?.value || 0;

    return (
        <div className="p-5 rounded-2xl bg-white/80 backdrop-blur-lg border border-cream-200/50 shadow-glass hover:shadow-glass-lg transition-all duration-300 hover:-translate-y-1">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-obsidian-500">{title}</h3>
                <span className="text-xs text-obsidian-400">{unit}</span>
            </div>

            {/* Current Value */}
            <p className="text-2xl font-bold text-obsidian-900 mb-4">
                {formatValue(latestValue, format)}
            </p>

            {/* Bar Chart */}
            <div className="flex items-end gap-1 h-20">
                {data.map((point, index) => {
                    const height = maxValue > 0 ? (Math.abs(point.value) / maxValue) * 100 : 0;
                    const isNegative = point.value < 0;

                    return (
                        <div
                            key={point.label}
                            className="flex-1 flex flex-col items-center justify-end group cursor-pointer"
                        >
                            <div
                                className="w-full rounded-t transition-all duration-300 group-hover:opacity-80"
                                style={{
                                    height: `${Math.max(height, 4)}%`,
                                    backgroundColor: isNegative ? '#FF5C5C' : color,
                                    opacity: 0.2 + (index / data.length) * 0.8,
                                }}
                            />
                            {/* Tooltip on hover - shows only on last items */}
                            {index === data.length - 1 && (
                                <span className="text-xs text-obsidian-400 mt-1 whitespace-nowrap">
                                    {point.label}
                                </span>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* X-axis labels */}
            <div className="flex justify-between text-xs text-obsidian-300 mt-2">
                <span>{data[0]?.label}</span>
                <span>{data[data.length - 1]?.label}</span>
            </div>
        </div>
    );
};

// =============================================================================
// DATA POINT GRID
// =============================================================================

export interface DataPointGridProps {
    symbol?: string;
}

// Sample data generator for demo purposes
const generateSampleData = (baseValue: number, trend: 'up' | 'down' | 'flat' = 'up'): DataPoint[] => {
    const quarters = ['Q1 22', 'Q2 22', 'Q3 22', 'Q4 22', 'Q1 23', 'Q2 23', 'Q3 23', 'Q4 23'];
    let value = baseValue;

    return quarters.map((label) => {
        const change = trend === 'up'
            ? Math.random() * 0.1 + 0.02
            : trend === 'down'
                ? -(Math.random() * 0.1 + 0.02)
                : (Math.random() - 0.5) * 0.1;
        value = value * (1 + change);
        return { label, value };
    });
};

const metricConfigs = [
    { title: 'Revenue', color: '#3B82F6', format: 'currency' as const, base: 50e9 },
    { title: 'Net Income', color: '#22C55E', format: 'currency' as const, base: 12e9 },
    { title: 'Gross Margin', color: '#8B5CF6', format: 'percent' as const, base: 42 },
    { title: 'EPS', color: '#F59E0B', format: 'currency' as const, base: 5.5 },
    { title: 'P/E Ratio', color: '#EC4899', format: 'ratio' as const, base: 28 },
    { title: 'Free Cash Flow', color: '#06B6D4', format: 'currency' as const, base: 8e9 },
    { title: 'Debt/Equity', color: '#FF5C5C', format: 'ratio' as const, base: 0.8 },
    { title: 'ROE', color: '#10B981', format: 'percent' as const, base: 45 },
];

export const DataPointGrid: React.FC<DataPointGridProps> = ({ symbol }) => {
    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-obsidian-900">
                    Key Metrics{symbol ? ` for ${symbol}` : ''}
                </h2>
                <div className="flex items-center gap-2">
                    <select className="h-9 px-3 rounded-lg border border-cream-300 text-sm text-obsidian-700 bg-white focus:outline-none focus:ring-2 focus:ring-electric-500/20 focus:border-electric-500">
                        <option>Quarterly</option>
                        <option>Annual</option>
                    </select>
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {metricConfigs.map((config, index) => (
                    <ScrollReveal key={config.title} delay={index * 50}>
                        <DataPointCard
                            title={config.title}
                            data={generateSampleData(
                                config.base,
                                index % 3 === 0 ? 'up' : index % 3 === 1 ? 'flat' : 'down'
                            )}
                            color={config.color}
                            format={config.format}
                        />
                    </ScrollReveal>
                ))}
            </div>
        </div>
    );
};

export default DataPointGrid;
