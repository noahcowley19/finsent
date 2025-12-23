'use client';

// =============================================================================
// FINANCIAL ANALYSIS PAGE
// =============================================================================
// Detailed financial analysis for a stock
//
// Location: frontend/app/financials/[symbol]/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import { useParams } from 'next/navigation';
import { Section, Grid } from '@/components/layout';
import {
  AnalysisHeader,
  AnalysisTabs,
  FinancialMetrics,
  FinancialChart,
} from '@/components/analysis';
import type { Metric, FinancialDataPoint } from '@/components/analysis';

// Mock data - replace with real API calls
const mockStockData = {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  exchange: 'NASDAQ',
  price: 178.72,
  change: 2.34,
  changePercent: 1.33,
};

const mockMetrics: Metric[] = [
  { label: 'P/E Ratio', value: 28.5, sectorAvg: 32.1, format: 'ratio', description: 'Price to Earnings ratio' },
  { label: 'P/B Ratio', value: 45.2, sectorAvg: 38.5, format: 'ratio', description: 'Price to Book ratio' },
  { label: 'P/S Ratio', value: 7.2, sectorAvg: 5.8, format: 'ratio', description: 'Price to Sales ratio' },
  { label: 'EV/EBITDA', value: 22.1, sectorAvg: 24.3, format: 'ratio', description: 'Enterprise Value to EBITDA' },
  { label: 'ROE', value: 147.2, sectorAvg: 28.5, format: 'percent', description: 'Return on Equity' },
  { label: 'ROA', value: 28.3, sectorAvg: 12.4, format: 'percent', description: 'Return on Assets' },
  { label: 'Profit Margin', value: 25.3, sectorAvg: 18.2, format: 'percent', description: 'Net Profit Margin' },
  { label: 'Operating Margin', value: 29.8, sectorAvg: 22.1, format: 'percent', description: 'Operating Profit Margin' },
  { label: 'Current Ratio', value: 0.98, sectorAvg: 1.5, format: 'ratio', description: 'Current Assets / Current Liabilities' },
  { label: 'Quick Ratio', value: 0.94, sectorAvg: 1.2, format: 'ratio', description: 'Liquid Assets / Current Liabilities' },
  { label: 'Debt/Equity', value: 1.81, sectorAvg: 0.85, format: 'ratio', description: 'Total Debt to Equity' },
  { label: 'Revenue Growth', value: 8.2, sectorAvg: 12.5, format: 'percent', description: 'Year-over-Year Revenue Growth' },
];

const mockChartData: FinancialDataPoint[] = [
  { period: 'Q1 23', revenue: 94836000000, earnings: 24160000000 },
  { period: 'Q2 23', revenue: 81797000000, earnings: 19881000000 },
  { period: 'Q3 23', revenue: 89500000000, earnings: 22956000000 },
  { period: 'Q4 23', revenue: 119575000000, earnings: 33916000000 },
  { period: 'Q1 24', revenue: 90753000000, earnings: 23636000000 },
  { period: 'Q2 24', revenue: 85777000000, earnings: 21448000000 },
];

export default function FinancialsPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || 'AAPL';
  
  const [inWatchlist, setInWatchlist] = useState(false);

  // In real app, fetch data based on symbol
  const stockData = { ...mockStockData, symbol };

  const handleWatchlist = () => {
    setInWatchlist(!inWatchlist);
  };

  // Calculate financial health score
  const healthScore = 'A-';
  const healthColor = 'text-success-600';

  return (
    <>
      {/* Header */}
      <AnalysisHeader
        symbol={stockData.symbol}
        name={stockData.name}
        exchange={stockData.exchange}
        price={stockData.price}
        change={stockData.change}
        changePercent={stockData.changePercent}
        onAddToWatchlist={handleWatchlist}
        inWatchlist={inWatchlist}
      />

      {/* Tabs */}
      <AnalysisTabs symbol={symbol} />

      {/* Content */}
      <Section spacing="lg" background="default">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <Grid cols={1} colsLg={3} gap="lg">
            {/* Main content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Metrics grid */}
              <FinancialMetrics metrics={mockMetrics} />

              {/* Revenue chart */}
              <FinancialChart data={mockChartData} />
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Financial health score */}
              <div className="bg-white rounded-xl border border-border-light p-6 text-center">
                <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-2">
                  Financial Health
                </h3>
                <p className="text-body-sm text-neutral-500 mb-6">
                  Overall assessment based on key ratios
                </p>
                <div className="w-24 h-24 mx-auto rounded-full bg-success-100 flex items-center justify-center mb-4">
                  <span className={`font-display text-display-md ${healthColor}`}>
                    {healthScore}
                  </span>
                </div>
                <p className="text-body-md font-medium text-success-600">
                  Strong
                </p>
              </div>

              {/* Key highlights */}
              <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                  Key Highlights
                </h3>
                <ul className="space-y-4">
                  <li className="flex items-start gap-3">
                    <span className="w-2 h-2 rounded-full bg-success-500 mt-2" />
                    <div>
                      <p className="text-body-sm font-medium text-navy-900">
                        Exceptional Profitability
                      </p>
                      <p className="text-caption text-neutral-500">
                        ROE of 147% significantly exceeds sector average
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-2 h-2 rounded-full bg-success-500 mt-2" />
                    <div>
                      <p className="text-body-sm font-medium text-navy-900">
                        Strong Margins
                      </p>
                      <p className="text-caption text-neutral-500">
                        25% net margin beats sector by 7 points
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-2 h-2 rounded-full bg-warning-500 mt-2" />
                    <div>
                      <p className="text-body-sm font-medium text-navy-900">
                        Moderate Valuation
                      </p>
                      <p className="text-caption text-neutral-500">
                        P/E ratio slightly below sector average
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-2 h-2 rounded-full bg-error-500 mt-2" />
                    <div>
                      <p className="text-body-sm font-medium text-navy-900">
                        Higher Leverage
                      </p>
                      <p className="text-caption text-neutral-500">
                        Debt/Equity ratio above sector norm
                      </p>
                    </div>
                  </li>
                </ul>
              </div>

              {/* Data note */}
              <div className="bg-cream-50 rounded-xl p-6">
                <h4 className="text-body-sm font-medium text-navy-700 mb-2">
                  Data Sources
                </h4>
                <p className="text-caption text-neutral-500">
                  Financial data sourced from SEC filings and updated quarterly. Sector averages based on Technology sector peers.
                </p>
              </div>
            </div>
          </Grid>
        </div>
      </Section>
    </>
  );
}
