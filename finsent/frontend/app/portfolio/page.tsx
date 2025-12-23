'use client';

// =============================================================================
// PORTFOLIO PAGE
// =============================================================================
// Portfolio management page
//
// Location: frontend/app/portfolio/page.tsx
//
// =============================================================================

import React from 'react';
import { Section, Grid, AuthGuard } from '@/components/layout';
import {
  PortfolioSummary,
  HoldingsTable,
  PerformanceChart,
  AllocationChart,
} from '@/components/portfolio';
import type {
  PortfolioSummaryData,
  Holding,
  PerformanceDataPoint,
  AllocationItem,
} from '@/components/portfolio';

// Mock data - replace with real API calls
const mockSummary: PortfolioSummaryData = {
  totalValue: 48632.50,
  totalCost: 42500.00,
  dayChange: 523.45,
  dayChangePercent: 1.09,
  totalGain: 6132.50,
  totalGainPercent: 14.43,
  cashBalance: 5240.00,
};

const mockHoldings: Holding[] = [
  {
    id: '1',
    symbol: 'AAPL',
    name: 'Apple Inc.',
    shares: 50,
    avgCost: 165.00,
    currentPrice: 178.72,
    value: 8936.00,
    gain: 686.00,
    gainPercent: 8.32,
    dayChange: 2.34,
    dayChangePercent: 1.33,
  },
  {
    id: '2',
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    shares: 30,
    avgCost: 350.00,
    currentPrice: 378.91,
    value: 11367.30,
    gain: 867.30,
    gainPercent: 8.26,
    dayChange: 4.12,
    dayChangePercent: 1.10,
  },
  {
    id: '3',
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    shares: 40,
    avgCost: 135.00,
    currentPrice: 141.80,
    value: 5672.00,
    gain: 272.00,
    gainPercent: 5.04,
    dayChange: -0.92,
    dayChangePercent: -0.64,
  },
  {
    id: '4',
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    shares: 15,
    avgCost: 750.00,
    currentPrice: 875.28,
    value: 13129.20,
    gain: 1879.20,
    gainPercent: 16.71,
    dayChange: 12.45,
    dayChangePercent: 1.44,
  },
  {
    id: '5',
    symbol: 'TSLA',
    name: 'Tesla, Inc.',
    shares: 20,
    avgCost: 280.00,
    currentPrice: 248.50,
    value: 4970.00,
    gain: -630.00,
    gainPercent: -11.25,
    dayChange: -5.20,
    dayChangePercent: -2.05,
  },
];

// Generate mock performance data
const generatePerformanceData = (): PerformanceDataPoint[] => {
  const data: PerformanceDataPoint[] = [];
  const now = new Date();
  let value = 40000;

  for (let i = 90; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    value = value * (1 + (Math.random() - 0.48) * 0.02);
    data.push({ date, value });
  }

  // End at current value
  data[data.length - 1].value = mockSummary.totalValue;
  return data;
};

const mockAllocation: AllocationItem[] = [
  { label: 'NVDA', value: 13129.20, color: '#131D4F' },
  { label: 'MSFT', value: 11367.30, color: '#954C2E' },
  { label: 'AAPL', value: 8936.00, color: '#22C55E' },
  { label: 'GOOGL', value: 5672.00, color: '#6366F1' },
  { label: 'TSLA', value: 4970.00, color: '#F59E0B' },
  { label: 'Cash', value: 5240.00, color: '#94A3B8' },
];

function PortfolioContent() {
  const performanceData = generatePerformanceData();

  const handleEditHolding = (holding: Holding) => {
    console.log('Edit holding:', holding);
    // TODO: Open edit modal
  };

  const handleDeleteHolding = (holdingId: string) => {
    console.log('Delete holding:', holdingId);
    // TODO: Confirm and delete
  };

  return (
    <>
      {/* Header */}
      <Section spacing="md" background="gradient">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
              Portfolio
            </h1>
            <p className="text-body-md text-neutral-600">
              Track your investments and performance
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2.5 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors">
              Export
            </button>
            <button className="px-4 py-2.5 bg-terra-500 rounded-lg text-body-sm font-medium text-white hover:bg-terra-600 transition-colors">
              + Add Position
            </button>
          </div>
        </div>
      </Section>

      {/* Summary */}
      <Section spacing="md" background="default">
        <PortfolioSummary data={mockSummary} />
      </Section>

      {/* Charts */}
      <Section spacing="md" background="default">
        <Grid cols={1} colsLg={3} gap="lg">
          <div className="lg:col-span-2">
            <PerformanceChart data={performanceData} />
          </div>
          <div>
            <AllocationChart data={mockAllocation} />
          </div>
        </Grid>
      </Section>

      {/* Holdings */}
      <Section spacing="lg" background="default">
        <HoldingsTable
          holdings={mockHoldings}
          onEdit={handleEditHolding}
          onDelete={handleDeleteHolding}
        />
      </Section>
    </>
  );
}

export default function PortfolioPage() {
  return (
    <AuthGuard>
      <PortfolioContent />
    </AuthGuard>
  );
}
