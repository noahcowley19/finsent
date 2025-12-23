'use client';

// =============================================================================
// PORTFOLIO SUMMARY COMPONENT
// =============================================================================
// Displays total portfolio value and key metrics
//
// Location: frontend/components/portfolio/PortfolioSummary.tsx
//
// =============================================================================

import React from 'react';

export interface PortfolioSummaryData {
  totalValue: number;
  totalCost: number;
  dayChange: number;
  dayChangePercent: number;
  totalGain: number;
  totalGainPercent: number;
  cashBalance: number;
}

export interface PortfolioSummaryProps {
  data: PortfolioSummaryData;
  isLoading?: boolean;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({
  data,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="animate-pulse">
          <div className="w-32 h-4 rounded bg-cream-100 mb-4" />
          <div className="w-48 h-10 rounded bg-cream-100 mb-2" />
          <div className="w-24 h-6 rounded bg-cream-100 mb-6" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i}>
                <div className="w-20 h-3 rounded bg-cream-100 mb-2" />
                <div className="w-24 h-6 rounded bg-cream-100" />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const isDayPositive = data.dayChange >= 0;
  const isTotalPositive = data.totalGain >= 0;

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      {/* Total value */}
      <div className="mb-6">
        <p className="text-body-sm text-neutral-500 mb-1">Total Portfolio Value</p>
        <div className="flex items-baseline gap-4">
          <h2 className="font-display text-display-md text-navy-900">
            {formatCurrency(data.totalValue)}
          </h2>
          <div
            className={`
              flex items-center gap-1 px-2 py-1 rounded-lg text-body-sm font-medium
              ${isDayPositive ? 'bg-success-100 text-success-700' : 'bg-error-100 text-error-700'}
            `}
          >
            <svg
              className={`w-4 h-4 ${isDayPositive ? '' : 'rotate-180'}`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
            {isDayPositive ? '+' : ''}{formatCurrency(data.dayChange)} ({isDayPositive ? '+' : ''}{data.dayChangePercent.toFixed(2)}%) today
          </div>
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 pt-6 border-t border-border-light">
        <div>
          <p className="text-caption text-neutral-500 mb-1">Total Cost</p>
          <p className="text-body-lg font-semibold text-navy-900">
            {formatCurrency(data.totalCost)}
          </p>
        </div>
        <div>
          <p className="text-caption text-neutral-500 mb-1">Total Gain/Loss</p>
          <p className={`text-body-lg font-semibold ${isTotalPositive ? 'text-success-600' : 'text-error-600'}`}>
            {isTotalPositive ? '+' : ''}{formatCurrency(data.totalGain)}
            <span className="text-body-sm ml-1">
              ({isTotalPositive ? '+' : ''}{data.totalGainPercent.toFixed(2)}%)
            </span>
          </p>
        </div>
        <div>
          <p className="text-caption text-neutral-500 mb-1">Cash Balance</p>
          <p className="text-body-lg font-semibold text-navy-900">
            {formatCurrency(data.cashBalance)}
          </p>
        </div>
        <div>
          <p className="text-caption text-neutral-500 mb-1">Invested</p>
          <p className="text-body-lg font-semibold text-navy-900">
            {formatCurrency(data.totalValue - data.cashBalance)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary;
