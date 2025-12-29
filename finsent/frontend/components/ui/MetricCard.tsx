'use client';

import React from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface MetricCardProps {
    /** Metric label */
    label: string;
    /** Metric value */
    value: string | number;
    /** Optional trend indicator */
    trend?: {
        value: number;
        direction: 'up' | 'down';
    };
    /** Additional CSS classes */
    className?: string;
    /** Icon to display */
    icon?: React.ReactNode;
}

// =============================================================================
// METRIC CARD COMPONENT - Editorial Finance Dashboard Stats
// =============================================================================

export const MetricCard: React.FC<MetricCardProps> = ({
    label,
    value,
    trend,
    className = '',
    icon,
}) => {
    const getTrendColor = (direction: 'up' | 'down') => {
        return direction === 'up' ? 'text-success-600' : 'text-terracotta-500';
    };

    const getTrendIcon = (direction: 'up' | 'down') => {
        return direction === 'up' ? '↑' : '↓';
    };

    return (
        <div className={`bg-white rounded-xl border border-gray-200 p-6 ${className}`}>
            <div className="flex items-start justify-between mb-3">
                <p className="text-sm text-gray-500 uppercase tracking-wide font-medium">
                    {label}
                </p>
                {icon && (
                    <div className="text-navy-900 opacity-60">
                        {icon}
                    </div>
                )}
            </div>

            <div className="flex items-end justify-between">
                <h3 className="text-3xl font-bold text-ink-black-900">
                    {value}
                </h3>

                {trend && (
                    <div className={`flex items-center gap-1 text-sm font-medium ${getTrendColor(trend.direction)}`}>
                        <span>{getTrendIcon(trend.direction)}</span>
                        <span>{Math.abs(trend.value)}%</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MetricCard;
