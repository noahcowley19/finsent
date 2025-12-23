'use client';

// =============================================================================
// FINANCIAL CHART COMPONENT
// =============================================================================
// Bar/line chart for revenue and earnings
//
// Location: frontend/components/analysis/FinancialChart.tsx
//
// =============================================================================

import React, { useState } from 'react';

export interface FinancialDataPoint {
  period: string;
  revenue: number;
  earnings: number;
}

export interface FinancialChartProps {
  data: FinancialDataPoint[];
  isLoading?: boolean;
}

type ChartType = 'revenue' | 'earnings' | 'both';

function formatBillions(value: number): string {
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toLocaleString()}`;
}

export const FinancialChart: React.FC<FinancialChartProps> = ({
  data,
  isLoading = false,
}) => {
  const [chartType, setChartType] = useState<ChartType>('both');

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="flex justify-between items-center mb-6">
          <div className="w-48 h-6 rounded bg-cream-100 animate-pulse" />
          <div className="flex gap-2">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="w-20 h-8 rounded bg-cream-100 animate-pulse" />
            ))}
          </div>
        </div>
        <div className="h-64 rounded bg-cream-100 animate-pulse" />
      </div>
    );
  }

  // Calculate max value for scaling
  const allValues = data.flatMap((d) => [d.revenue, d.earnings]);
  const maxValue = Math.max(...allValues);
  const chartHeight = 200;

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Revenue & Earnings
        </h3>

        {/* Chart type selector */}
        <div className="flex gap-1 bg-cream-50 rounded-lg p-1">
          {(['both', 'revenue', 'earnings'] as ChartType[]).map((type) => (
            <button
              key={type}
              onClick={() => setChartType(type)}
              className={`
                px-3 py-1.5 rounded-md text-body-sm font-medium capitalize
                transition-colors duration-fast
                ${chartType === type
                  ? 'bg-white text-navy-900 shadow-sm'
                  : 'text-neutral-600 hover:text-navy-700'
                }
              `}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-8 w-16 flex flex-col justify-between text-caption text-neutral-400">
          <span>{formatBillions(maxValue)}</span>
          <span>{formatBillions(maxValue / 2)}</span>
          <span>$0</span>
        </div>

        {/* Chart area */}
        <div className="ml-20">
          <div className="flex items-end justify-around gap-2 h-52">
            {data.map((point, index) => {
              const revenueHeight = (point.revenue / maxValue) * chartHeight;
              const earningsHeight = (point.earnings / maxValue) * chartHeight;

              return (
                <div key={point.period} className="flex-1 flex flex-col items-center">
                  <div className="flex items-end gap-1 h-52">
                    {/* Revenue bar */}
                    {(chartType === 'both' || chartType === 'revenue') && (
                      <div
                        className="w-6 bg-navy-500 rounded-t transition-all duration-normal hover:bg-navy-600 cursor-pointer group relative"
                        style={{ height: `${revenueHeight}px` }}
                      >
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-navy-900 text-white text-caption rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                          Revenue: {formatBillions(point.revenue)}
                        </div>
                      </div>
                    )}

                    {/* Earnings bar */}
                    {(chartType === 'both' || chartType === 'earnings') && (
                      <div
                        className={`w-6 rounded-t transition-all duration-normal cursor-pointer group relative ${
                          point.earnings >= 0 ? 'bg-terra-500 hover:bg-terra-600' : 'bg-error-500 hover:bg-error-600'
                        }`}
                        style={{ height: `${Math.abs(earningsHeight)}px` }}
                      >
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-navy-900 text-white text-caption rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                          Earnings: {formatBillions(point.earnings)}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Period label */}
                  <p className="mt-2 text-caption text-neutral-500">{point.period}</p>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-6 text-caption">
        {(chartType === 'both' || chartType === 'revenue') && (
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-navy-500" />
            Revenue
          </span>
        )}
        {(chartType === 'both' || chartType === 'earnings') && (
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-terra-500" />
            Net Income
          </span>
        )}
      </div>
    </div>
  );
};

export default FinancialChart;
