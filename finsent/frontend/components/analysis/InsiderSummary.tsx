'use client';

// =============================================================================
// INSIDER SUMMARY COMPONENT
// =============================================================================
// Summary cards showing insider buying/selling activity
//
// Location: frontend/components/analysis/InsiderSummary.tsx
//
// =============================================================================

import React from 'react';

export interface InsiderSummaryData {
  buyCount: number;
  sellCount: number;
  buyValue: number;
  sellValue: number;
  netShares: number;
  period: string;
}

export interface InsiderSummaryProps {
  data: InsiderSummaryData;
  isLoading?: boolean;
}

function formatCurrency(value: number): string {
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
  return `$${value.toLocaleString()}`;
}

function formatShares(value: number): string {
  const absValue = Math.abs(value);
  if (absValue >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (absValue >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toLocaleString();
}

export const InsiderSummary: React.FC<InsiderSummaryProps> = ({
  data,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="bg-white rounded-xl border border-border-light p-6 animate-pulse">
            <div className="w-20 h-4 rounded bg-cream-100 mb-4" />
            <div className="w-32 h-8 rounded bg-cream-100 mb-2" />
            <div className="w-24 h-4 rounded bg-cream-100" />
          </div>
        ))}
      </div>
    );
  }

  const netValue = data.buyValue - data.sellValue;
  const isNetPositive = netValue > 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Buys */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-success-100 flex items-center justify-center">
            <svg className="w-4 h-4 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          </div>
          <span className="text-body-sm text-neutral-600">Insider Buys</span>
        </div>
        <p className="font-display text-display-sm text-navy-900 mb-1">
          {formatCurrency(data.buyValue)}
        </p>
        <p className="text-body-sm text-neutral-500">
          {data.buyCount} transaction{data.buyCount !== 1 ? 's' : ''} in {data.period}
        </p>
      </div>

      {/* Sells */}
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-error-100 flex items-center justify-center">
            <svg className="w-4 h-4 text-error-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
          <span className="text-body-sm text-neutral-600">Insider Sells</span>
        </div>
        <p className="font-display text-display-sm text-navy-900 mb-1">
          {formatCurrency(data.sellValue)}
        </p>
        <p className="text-body-sm text-neutral-500">
          {data.sellCount} transaction{data.sellCount !== 1 ? 's' : ''} in {data.period}
        </p>
      </div>

      {/* Net activity */}
      <div className={`rounded-xl border p-6 ${isNetPositive ? 'bg-success-50 border-success-200' : 'bg-error-50 border-error-200'}`}>
        <div className="flex items-center gap-2 mb-4">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isNetPositive ? 'bg-success-200' : 'bg-error-200'}`}>
            <svg className={`w-4 h-4 ${isNetPositive ? 'text-success-700' : 'text-error-700'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <span className={`text-body-sm ${isNetPositive ? 'text-success-700' : 'text-error-700'}`}>
            Net Activity
          </span>
        </div>
        <p className={`font-display text-display-sm mb-1 ${isNetPositive ? 'text-success-700' : 'text-error-700'}`}>
          {isNetPositive ? '+' : ''}{formatCurrency(netValue)}
        </p>
        <p className={`text-body-sm ${isNetPositive ? 'text-success-600' : 'text-error-600'}`}>
          {formatShares(data.netShares)} shares net {isNetPositive ? 'bought' : 'sold'}
        </p>
      </div>
    </div>
  );
};

export default InsiderSummary;
