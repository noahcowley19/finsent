'use client';

// =============================================================================
// FINANCIAL METRICS COMPONENT
// =============================================================================
// Grid displaying key financial metrics with comparisons
//
// Location: frontend/components/analysis/FinancialMetrics.tsx
//
// =============================================================================

import React from 'react';

export interface Metric {
  label: string;
  value: number | string | null;
  sectorAvg?: number | string | null;
  format?: 'number' | 'currency' | 'percent' | 'ratio';
  description?: string;
}

export interface FinancialMetricsProps {
  metrics: Metric[];
  isLoading?: boolean;
}

function formatValue(value: number | string | null, format?: string): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'string') return value;

  switch (format) {
    case 'currency':
      return `$${value.toFixed(2)}`;
    case 'percent':
      return `${value.toFixed(2)}%`;
    case 'ratio':
      return value.toFixed(2);
    default:
      return value.toLocaleString();
  }
}

function getComparisonColor(value: number | null, sectorAvg: number | null, higherIsBetter = true): string {
  if (value === null || sectorAvg === null) return 'text-neutral-600';
  
  const isAbove = value > sectorAvg;
  const isGood = higherIsBetter ? isAbove : !isAbove;
  
  if (Math.abs(value - sectorAvg) / sectorAvg < 0.05) return 'text-neutral-600';
  return isGood ? 'text-success-600' : 'text-error-600';
}

export const FinancialMetrics: React.FC<FinancialMetricsProps> = ({
  metrics,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="w-40 h-6 rounded bg-cream-100 animate-pulse mb-6" />
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="w-24 h-4 rounded bg-cream-100 mb-2" />
              <div className="w-16 h-8 rounded bg-cream-100 mb-1" />
              <div className="w-20 h-3 rounded bg-cream-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
        Key Financial Metrics
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {metrics.map((metric) => (
          <div key={metric.label} className="group relative">
            <p className="text-caption text-neutral-500 mb-1 flex items-center gap-1">
              {metric.label}
              {metric.description && (
                <span className="cursor-help" title={metric.description}>
                  <svg className="w-3.5 h-3.5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </span>
              )}
            </p>
            <p className="text-body-lg font-semibold text-navy-900">
              {formatValue(metric.value, metric.format)}
            </p>
            {metric.sectorAvg !== undefined && (
              <p className="text-caption text-neutral-400">
                Sector: {formatValue(metric.sectorAvg, metric.format)}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default FinancialMetrics;
