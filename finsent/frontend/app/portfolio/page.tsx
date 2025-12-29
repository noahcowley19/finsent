'use client';

import React, { useEffect, useState } from 'react';
import { Section, Grid } from '@/components/layout';
import {
  PortfolioSummary,
  HoldingsTable,
  PerformanceChart,
  AllocationChart,
  AddPositionModal,
} from '@/components/portfolio';
//
import {
  CorrelationMatrix,
  MonteCarloSimulation,
  VaRCard,
  OptimizationPanel,
  WhatIfSimulator
} from '@/components/portfolio/AdvancedPortfolio';
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
  
  // Transform positions for the advanced tools
  const advancedPositions = positions.map(p => ({
     ticker: p.ticker,
     shares: p.shares,
     total_cost_basis: p.cost_basis * p.shares
  }));
  const tickers = positions.map(p => p.ticker);

  const [summary, setSummary] = useState<PortfolioSummaryData>({
    totalValue: 0, totalCost: 0, dayChange: 0, dayChangePercent: 0,
    totalGain: 0, totalGainPercent: 0, cashBalance: 0,
  });

  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [allocationData, setAllocationData] = useState<AllocationItem[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceDataPoint[]>([]);

  useEffect(() => {
    if (positions.length > 0) analyze();
  }, [positions.length]);

  useEffect(() => {
    if (analysis) {
      const metrics = analysis.portfolio_metrics;
      setSummary({
        totalValue: metrics.total_value,
        totalCost: metrics.total_cost,
        dayChange: metrics.total_gain_loss, // fallback
        dayChangePercent: metrics.total_gain_loss_percent,
        totalGain: metrics.total_gain_loss,
        totalGainPercent: metrics.total_gain_loss_percent,
        cashBalance: 0,
      });

      const holdingsData: Holding[] = metrics.positions.map((pos: any, index: number) => {
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

      const colors = ['#3B82F6', '#FF5C5C', '#22C55E', '#F59E0B', '#8B5CF6', '#06B6D4', '#EC4899'];
      const allocation: AllocationItem[] = metrics.positions.map((pos: any, index: number) => ({
        label: pos.ticker,
        value: pos.current_value,
        color: colors[index % colors.length],
      }));
      setAllocationData(allocation);

      // Simple mock performance data generator
      const perfData: PerformanceDataPoint[] = [];
      const now = new Date();
      for (let i = 90; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        const progress = (90 - i) / 90;
        const val = metrics.total_cost + (metrics.total_gain_loss * progress);
        perfData.push({ date, value: val });
      }
      setPerformanceData(perfData);
    }
  }, [analysis, positions]);

  const handleAddPosition = (pos: { ticker: string; shares: number; cost_basis: number }) => {
    addPosition(pos.ticker, pos.shares, pos.cost_basis);
  };

  return (
    <div className="min-h-screen bg-cream-50 pb-20">
      {/* Header */}
      <Section spacing="md" background="white">
        <ScrollReveal>
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl lg:text-4xl font-bold text-obsidian-900 tracking-tight mb-2">Portfolio</h1>
              <p className="text-lg text-obsidian-500">Track your investments and performance</p>
            </div>
            <button
              onClick={() => setIsAddModalOpen(true)}
              className="px-5 py-2.5 bg-obsidian-900 rounded-xl text-sm font-medium text-white hover:bg-obsidian-850"
            >
              + Add Position
            </button>
          </div>
        </ScrollReveal>
      </Section>

      {/* Main Content */}
      {positions.length > 0 && (
        <>
          <Section spacing="md" background="default">
            <ScrollReveal>
              <PortfolioSummary data={summary} />
            </ScrollReveal>
          </Section>

          <Section spacing="md" background="default">
            <Grid cols={1} colsLg={3} gap="lg">
              <div className="lg:col-span-2 space-y-8">
                <ScrollReveal>
                  <PerformanceChart data={performanceData} />
                </ScrollReveal>
                
                {/* --- NEW V2 MODULES --- */}
                <ScrollReveal delay={100}>
                  <WhatIfSimulator positions={advancedPositions} />
                </ScrollReveal>
                
                <ScrollReveal delay={200}>
                   <MonteCarloSimulation positions={advancedPositions} />
                </ScrollReveal>
              </div>
              
              <div className="space-y-8">
                <ScrollReveal delay={100}>
                  <AllocationChart data={allocationData} />
                </ScrollReveal>

                {/* --- NEW V2 MODULES --- */}
                <ScrollReveal delay={200}>
                   <VaRCard positions={advancedPositions} />
                </ScrollReveal>
                
                <ScrollReveal delay={300}>
                  <CorrelationMatrix tickers={tickers} />
                </ScrollReveal>
                
                <ScrollReveal delay={400}>
                   <OptimizationPanel positions={advancedPositions} />
                </ScrollReveal>
              </div>
            </Grid>
          </Section>

          <Section spacing="lg" background="default">
            <ScrollReveal>
              <HoldingsTable
                holdings={holdings}
                onEdit={() => {}}
                onDelete={(id) => removePosition(id)}
              />
            </ScrollReveal>
          </Section>
        </>
      )}

      {/* Empty State */}
      {positions.length === 0 && (
        <Section spacing="xl" background="default">
            <div className="text-center py-16">
              <h2 className="text-2xl font-bold text-obsidian-900 mb-3">No positions yet</h2>
              <button
                onClick={() => setIsAddModalOpen(true)}
                className="px-6 py-3 bg-obsidian-900 rounded-xl text-white font-medium"
              >
                Add Your First Position
              </button>
            </div>
        </Section>
      )}

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
