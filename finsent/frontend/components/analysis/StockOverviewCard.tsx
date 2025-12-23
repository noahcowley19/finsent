'use client';

// =============================================================================
// STOCK OVERVIEW CARD COMPONENT
// =============================================================================
// Card displaying stock overview stats
//
// Location: frontend/components/analysis/StockOverviewCard.tsx
//
// =============================================================================

import React from 'react';

export interface StockStats {
  marketCap: number;
  volume: number;
  avgVolume: number;
  high52w: number;
  low52w: number;
  peRatio: number | null;
  eps: number | null;
  dividend: number | null;
  beta: number | null;
}

export interface StockOverviewCardProps {
  stats: StockStats;
  isLoading?: boolean;
}

function formatLargeNumber(value: number): string {
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toLocaleString()}`;
}

function formatVolume(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toLocaleString();
}

export const StockOverviewCard: React.FC<StockOverviewCardProps> = ({
  stats,
  isLoading = false,
}) => {
  const metrics = [
    { label: 'Market Cap', value: formatLargeNumber(stats.marketCap) },
    { label: 'Volume', value: formatVolume(stats.volume) },
    { label: 'Avg Volume', value: formatVolume(stats.avgVolume) },
    { label: '52W High', value: `$${stats.high52w.toFixed(2)}` },
    { label: '52W Low', value: `$${stats.low52w.toFixed(2)}` },
    { label: 'P/E Ratio', value: stats.peRatio?.toFixed(2) ?? '-' },
    { label: 'EPS', value: stats.eps ? `$${stats.eps.toFixed(2)}` : '-' },
    { label: 'Dividend Yield', value: stats.dividend ? `${stats.dividend.toFixed(2)}%` : '-' },
    { label: 'Beta', value: stats.beta?.toFixed(2) ?? '-' },
  ];

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="w-32 h-6 rounded bg-cream-100 animate-pulse mb-6" />
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          {[...Array(9)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="w-20 h-4 rounded bg-cream-100 mb-2" />
              <div className="w-24 h-6 rounded bg-cream-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
        Key Statistics
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
        {metrics.map((metric) => (
          <div key={metric.label}>
            <p className="text-caption text-neutral-500 mb-1">{metric.label}</p>
            <p className="text-body-md font-semibold text-navy-900">{metric.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StockOverviewCard;
