'use client';

// =============================================================================
// CHART CARD
// =============================================================================
// Premium wrapper component for charts with loading states and glass styling
//
// Location: frontend/components/charts/ChartCard.tsx
// =============================================================================

import React from 'react';
import { HiDotsHorizontal } from 'react-icons/hi';

interface ChartCardProps {
    /** Card title */
    title: string;
    /** Optional subtitle or description */
    subtitle?: string;
    /** Whether data is loading */
    loading?: boolean;
    /** Optional action button handler */
    onAction?: () => void;
    /** Card children (chart content) */
    children: React.ReactNode;
    /** Additional CSS classes */
    className?: string;
    /** Use glass effect */
    glass?: boolean;
}

export const ChartCard: React.FC<ChartCardProps> = ({
    title,
    subtitle,
    loading = false,
    onAction,
    children,
    className = '',
    glass = false,
}) => {
    return (
        <div
            className={`
        rounded-2xl overflow-hidden
        ${glass
                    ? 'bg-white/80 backdrop-blur-xl border border-white/50 shadow-xl'
                    : 'bg-white border border-navy-100/50 shadow-lg shadow-navy-900/5'
                }
        ${className}
      `}
        >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-navy-100/30">
                <div>
                    <h3 className="font-heading font-semibold text-lg text-navy-900">
                        {title}
                    </h3>
                    {subtitle && (
                        <p className="text-body-sm text-navy-500/70 mt-0.5">{subtitle}</p>
                    )}
                </div>
                {onAction && (
                    <button
                        onClick={onAction}
                        className="p-2 rounded-lg text-navy-400 hover:text-navy-600 hover:bg-navy-50 transition-colors"
                    >
                        <HiDotsHorizontal className="w-5 h-5" />
                    </button>
                )}
            </div>

            {/* Content */}
            <div className="relative p-6">
                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <div className="flex flex-col items-center gap-4">
                            {/* Animated loading skeleton */}
                            <div className="w-full max-w-md space-y-3">
                                <div className="h-4 bg-navy-100 rounded-full animate-pulse w-full" />
                                <div className="h-4 bg-navy-100 rounded-full animate-pulse w-4/5" />
                                <div className="h-4 bg-navy-100 rounded-full animate-pulse w-3/5" />
                                <div className="h-32 bg-navy-50 rounded-xl animate-pulse mt-4" />
                            </div>
                            <p className="text-body-sm text-navy-400 animate-pulse">
                                Loading chart data...
                            </p>
                        </div>
                    </div>
                ) : (
                    children
                )}
            </div>
        </div>
    );
};

// =============================================================================
// STAT CARD
// =============================================================================
// Beautiful stat display card for dashboards

interface StatCardProps {
    label: string;
    value: string | number;
    change?: number;
    changeLabel?: string;
    icon?: React.ReactNode;
    className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
    label,
    value,
    change,
    changeLabel,
    icon,
    className = '',
}) => {
    const isPositive = change && change >= 0;

    return (
        <div
            className={`
        p-6 rounded-2xl
        bg-white border border-navy-100/50
        shadow-lg shadow-navy-900/5
        hover:shadow-xl hover:-translate-y-0.5
        transition-all duration-300
        ${className}
      `}
        >
            <div className="flex items-start justify-between mb-4">
                <p className="text-body-sm font-medium text-navy-500 uppercase tracking-wide">
                    {label}
                </p>
                {icon && (
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-navy-100 to-navy-50 flex items-center justify-center text-navy-600">
                        {icon}
                    </div>
                )}
            </div>

            <p className="font-display text-3xl lg:text-4xl text-navy-900 mb-2">
                {value}
            </p>

            {change !== undefined && (
                <div className="flex items-center gap-2">
                    <span
                        className={`
              inline-flex items-center gap-1 px-2 py-1 rounded-full text-caption font-semibold
              ${isPositive
                                ? 'bg-green-100 text-green-700'
                                : 'bg-red-100 text-red-700'
                            }
            `}
                    >
                        {isPositive ? '↑' : '↓'} {Math.abs(change).toFixed(2)}%
                    </span>
                    {changeLabel && (
                        <span className="text-body-sm text-navy-400">{changeLabel}</span>
                    )}
                </div>
            )}
        </div>
    );
};

export default ChartCard;
