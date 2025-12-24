'use client';

// =============================================================================
// PORTFOLIO PAGE
// =============================================================================
// Portfolio management page
//
// Location: frontend/app/portfolio/page.tsx
//
// =============================================================================

import React, { useEffect, useState } from 'react';
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
import { usePortfolio } from '@/lib/hooks';

function PortfolioContent() {
  const { positions, analysis, loading, error, addPosition, removePosition, updatePosition, analyze } = usePortfolio();
  
  const [summary, setSummary] = useState<PortfolioSummaryData>({
    totalValue: 0,
    totalCost: 0,
    dayChange: 0,
    dayChangePercent: 0,
    totalGain: 0,
    totalGainPercent: 0,
    cashBalance: 0,
  });
  
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [allocationData, setAllocationData] = useState<AllocationItem[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceDataPoint[]>([]);

  // Analyze portfolio when positions change
  useEffect(() => {
    if (positions.length > 0) {
      analyze();
    }
  }, [positions.length]);

  // Update display data when analysis completes
  useEffect(() => {
    if (analysis) {
      // Update summary
      const metrics = analysis.portfolio_metrics;
      setSummary({
        totalValue: metrics.total_value,
        totalCost: metrics.total_cost_basis,
        dayChange: metrics.total_gain_loss, // Using total gain as day change (API doesn't provide daily change)
        dayChangePercent: metrics.total_gain_loss_percent,
        totalGain: metrics.total_gain_loss,
        totalGainPercent: metrics.total_gain_loss_percent,
        cashBalance: 0, // Not provided by API
      });

      // Update holdings
      const holdingsData: Holding[] = analysis.positions.map((pos, index) => {
        const localPos = positions.find(p => p.ticker === pos.ticker);
        return {
          id: localPos?.id || `${pos.ticker}-${index}`,
          symbol: pos.ticker,
          name: pos.ticker, // API doesn't provide name
          shares: pos.shares,
          avgCost: pos.avg_cost,
          currentPrice: pos.current_price,
          value: pos.current_value,
          gain: pos.gain_loss,
          gainPercent: pos.gain_loss_percent,
          dayChange: 0, // Not provided by API
          dayChangePercent: 0, // Not provided by API
        };
      });
      setHoldings(holdingsData);

      // Update allocation
      const colors = ['#131D4F', '#954C2E', '#22C55E', '#6366F1', '#F59E0B', '#94A3B8', '#EC4899', '#06B6D4'];
      const allocation: AllocationItem[] = analysis.positions.map((pos, index) => ({
        label: pos.ticker,
        value: pos.current_value,
        color: colors[index % colors.length],
      }));
      setAllocationData(allocation);

      // Generate performance data (simplified - using current value as endpoint)
      const perfData: PerformanceDataPoint[] = [];
      const now = new Date();
      let value = metrics.total_cost_basis;
      
      for (let i = 90; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        // Simple linear interpolation from cost basis to current value
        const progress = (90 - i) / 90;
        value = metrics.total_cost_basis + (metrics.total_gain_loss * progress);
        perfData.push({ date, value });
      }
      setPerformanceData(perfData);
    }
  }, [analysis, positions]);

  const handleEditHolding = (holding: Holding) => {
    console.log('Edit holding:', holding);
    // TODO: Open edit modal
  };

  const handleDeleteHolding = (holdingId: string) => {
    removePosition(holdingId);
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
        <PortfolioSummary data={summary} />
      </Section>

      {/* Charts */}
      <Section spacing="md" background="default">
        <Grid cols={1} colsLg={3} gap="lg">
          <div className="lg:col-span-2">
            <PerformanceChart data={performanceData} />
          </div>
          <div>
            <AllocationChart data={allocationData} />
          </div>
        </Grid>
      </Section>

      {/* Holdings */}
      <Section spacing="lg" background="default">
        <HoldingsTable
          holdings={holdings}
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
