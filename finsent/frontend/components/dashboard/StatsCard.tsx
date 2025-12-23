'use client';

// =============================================================================
// STATS CARD COMPONENT
// =============================================================================
// Displays a single stat with icon, value, and optional change indicator
//
// Location: frontend/components/dashboard/StatsCard.tsx
//
// =============================================================================

import React, { ReactNode } from 'react';

export interface StatsCardProps {
  /** Card title */
  title: string;
  /** Main value to display */
  value: string | number;
  /** Optional subtitle/description */
  subtitle?: string;
  /** Icon to display */
  icon: ReactNode;
  /** Icon background color */
  iconColor?: 'terra' | 'navy' | 'success' | 'warning';
  /** Change from previous period */
  change?: {
    value: number;
    label: string;
  };
  /** Loading state */
  isLoading?: boolean;
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  iconColor = 'navy',
  change,
  isLoading = false,
}) => {
  const iconColorStyles = {
    terra: 'bg-terra-100 text-terra-600',
    navy: 'bg-navy-100 text-navy-600',
    success: 'bg-success-100 text-success-600',
    warning: 'bg-warning-100 text-warning-600',
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 rounded-xl bg-cream-100" />
            <div className="w-20 h-4 rounded bg-cream-100" />
          </div>
          <div className="w-24 h-8 rounded bg-cream-100 mb-2" />
          <div className="w-32 h-4 rounded bg-cream-100" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6 hover:shadow-md transition-shadow duration-normal">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${iconColorStyles[iconColor]}`}>
          {icon}
        </div>
        {change && (
          <div
            className={`
              flex items-center gap-1 px-2 py-1 rounded-full text-caption font-medium
              ${change.value >= 0 
                ? 'bg-success-100 text-success-700' 
                : 'bg-error-100 text-error-700'
              }
            `}
          >
            <svg
              className={`w-3 h-3 ${change.value >= 0 ? '' : 'rotate-180'}`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
            {Math.abs(change.value)}%
          </div>
        )}
      </div>

      {/* Value */}
      <p className="font-display text-display-sm text-navy-900 mb-1">
        {value}
      </p>

      {/* Title & subtitle */}
      <p className="text-body-sm text-neutral-600">{title}</p>
      {subtitle && (
        <p className="text-caption text-neutral-400 mt-1">{subtitle}</p>
      )}
      {change && (
        <p className="text-caption text-neutral-400 mt-1">{change.label}</p>
      )}
    </div>
  );
};

export default StatsCard;
