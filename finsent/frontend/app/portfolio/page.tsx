'use client';

// =============================================================================
// PORTFOLIO PAGE - Enhanced with Add Position Modal
// =============================================================================

import React, { useEffect, useState } from 'react';
import { Section, Grid } from '@/components/layout';
import {
  PortfolioSummary,
  HoldingsTable,
  PerformanceChart,
  AllocationChart,
  AddPositionModal,
} from '@/components/portfolio';
import type {
  PortfolioSummaryData,
  Holding,
  PerformanceDataPoint,
  AllocationItem,
} from '@/components/portfolio';
import { usePortfolio } from '@/lib/hooks';
import { ScrollReveal } from '@/components/ui';

function PortfolioContent() {
  const { positions, analysis, loading, error, addPosition, removePosition, updatePosition, analyze } = usePortfolio();
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

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
      const metrics = analysis.portfolio_metrics;
      setSummary({
        totalValue: metrics.total_value,
        totalCost: metrics.total_cost,
        dayChange: metrics.total_gain_loss,
        dayChangePercent: metrics.total_gain_loss_percent,
        totalGain: metrics.total_gain_loss,
        totalGainPercent: metrics.total_gain_loss_percent,
        cashBalance: 0,
      });

      const holdingsData: Holding[] = metrics.positions.map((pos, index) => {
        const localPos = positions.find(p => p.ticker === pos.ticker);
        return {
          id: localPos?.id || `${pos.ticker}-${index}`,
          symbol: pos.ticker,
          name: pos.name || pos.ticker,
          shares: pos.shares,
          avgCost: pos.cost_basis,
          currentPrice: pos.current_price,
          value: pos.current_value,
          gain: pos.gain_loss,
          gainPercent: pos.gain_loss_percent,
          dayChange: 0,
          dayChangePercent: 0,
        };
      });
      setHoldings(holdingsData);

      const colors = ['#3B82F6', '#FF5C5C', '#22C55E', '#F59E0B', '#8B5CF6', '#06B6D4', '#EC4899', '#94A3B8'];
      const allocation: AllocationItem[] = metrics.positions.map((pos, index) => ({
        label: pos.ticker,
        value: pos.current_value,
        color: colors[index % colors.length],
      }));
      setAllocationData(allocation);

      const perfData: PerformanceDataPoint[] = [];
      const now = new Date();
      let value = metrics.total_cost;

      for (let i = 90; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        const progress = (90 - i) / 90;
        value = metrics.total_cost + (metrics.total_gain_loss * progress);
        perfData.push({ date, value });
      }
      setPerformanceData(perfData);
    }
  }, [analysis, positions]);

  const handleAddPosition = (position: {
    ticker: string;
    shares: number;
    cost_basis: number;
    name?: string;
  }) => {
    addPosition(position.ticker, position.shares, position.cost_basis);
  };

  const handleEditHolding = (holding: Holding) => {
    console.log('Edit holding:', holding);
  };

  const handleDeleteHolding = (holdingId: string) => {
    removePosition(holdingId);
  };

  return (
    <div className="min-h-screen bg-cream-50">
      {/* Header */}
      <Section spacing="md" background="white">
        <ScrollReveal>
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl lg:text-4xl font-bold text-obsidian-900 tracking-tight mb-2">
                Portfolio
              </h1>
              <p className="text-lg text-obsidian-500">
                Track your investments and performance
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button className="px-4 py-2.5 bg-white border border-cream-300 rounded-xl text-sm font-medium text-obsidian-700 hover:bg-cream-100 transition-colors">
                Export
              </button>
              <button
                onClick={() => setIsAddModalOpen(true)}
                className="px-5 py-2.5 bg-obsidian-900 rounded-xl text-sm font-medium text-white hover:bg-obsidian-850 transition-all hover:-translate-y-0.5 hover:shadow-lg"
              >
                + Add Position
              </button>
            </div>
          </div>
        </ScrollReveal>
      </Section>

      {/* Empty State */}
      {positions.length === 0 && (
        <Section spacing="xl" background="default">
          <ScrollReveal>
            <div className="text-center py-16">
              <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-cream-100 flex items-center justify-center">
                <svg className="w-10 h-10 text-obsidian-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-obsidian-900 mb-3">No positions yet</h2>
              <p className="text-obsidian-500 mb-8 max-w-md mx-auto">
                Add your first stock position to start tracking your portfolio performance and get personalized insights.
              </p>
              <button
                onClick={() => setIsAddModalOpen(true)}
                className="inline-flex items-center gap-2 px-6 py-3 bg-obsidian-900 rounded-xl text-base font-medium text-white hover:bg-obsidian-850 transition-all hover:-translate-y-0.5 hover:shadow-lg"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Add Your First Position
              </button>
            </div>
          </ScrollReveal>
        </Section>
      )}

      {/* Portfolio Content */}
      {positions.length > 0 && (
        <>
          {/* Summary */}
          <Section spacing="md" background="default">
            <ScrollReveal>
              <PortfolioSummary data={summary} />
            </ScrollReveal>
          </Section>

          {/* Charts */}
          <Section spacing="md" background="default">
            <Grid cols={1} colsLg={3} gap="lg">
              <div className="lg:col-span-2">
                <ScrollReveal>
                  <PerformanceChart data={performanceData} />
                </ScrollReveal>
              </div>
              <div>
                <ScrollReveal delay={100}>
                  <AllocationChart data={allocationData} />
                </ScrollReveal>
              </div>
            </Grid>
          </Section>

          {/* Holdings */}
          <Section spacing="lg" background="default">
            <ScrollReveal>
              <HoldingsTable
                holdings={holdings}
                onEdit={handleEditHolding}
                onDelete={handleDeleteHolding}
              />
            </ScrollReveal>
          </Section>
        </>
      )}

      {/* Add Position Modal */}
      <AddPositionModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAddPosition={handleAddPosition}
      />
    </div>
  );
}

export default function PortfolioPage() {
  return <PortfolioContent />;
}
