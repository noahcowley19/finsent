'use client';

import React, { ReactNode } from 'react';

// =============================================================================
// STATS CARD - Dashboard metric cards
// =============================================================================

export interface StatsCardProps {
  /** Card title */
  title: string;
  /** Main value */
  value: string | number;
  /** Subtitle text */
  subtitle?: string;
  /** Change indicator */
  change?: {
    value: number;
    label?: string;
  };
  /** Icon */
  icon?: ReactNode;
  /** Icon color */
  iconColor?: 'default' | 'accent' | 'success' | 'warning' | 'error';
}

const iconColorStyles = {
  default: 'bg-ink-100 text-ink-600',
  accent: 'bg-accent/10 text-accent',
  success: 'bg-success-50 text-success-600',
  warning: 'bg-warning-50 text-warning-600',
  error: 'bg-error-50 text-error-600',
};

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  subtitle,
  change,
  icon,
  iconColor = 'default',
}) => {
  const isPositive = change ? change.value >= 0 : false;

  return (
    <div className="bg-white rounded-xl border border-ink-200/50 p-5 transition-all duration-200 hover:border-ink-300/50 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-body-sm text-ink-500 mb-1 truncate">{title}</p>
          <p className="text-display-sm text-ink-900 font-semibold tracking-tight truncate">
            {value}
          </p>

          {change !== undefined && (
            <p className={`mt-1.5 text-body-sm font-medium flex items-center gap-1 ${isPositive ? 'text-success-600' : 'text-error-600'
              }`}>
              <span className="text-xs">{isPositive ? '↑' : '↓'}</span>
              {Math.abs(change.value).toFixed(2)}%
              {change.label && (
                <span className="text-ink-400 font-normal ml-1">{change.label}</span>
              )}
            </p>
          )}

          {subtitle && !change && (
            <p className="mt-1.5 text-body-xs text-ink-400">{subtitle}</p>
          )}
        </div>

        {icon && (
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${iconColorStyles[iconColor]}`}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};

export default StatsCard;
