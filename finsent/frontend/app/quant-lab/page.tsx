'use client';

// =============================================================================
// QUANT LAB PAGE
// =============================================================================
// Strategy backtesting page (Pro only)
//
// Location: frontend/app/quant-lab/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';
import { Section, Grid, AuthGuard } from '@/components/layout';
import { StrategyBuilder, BacktestResults, StrategyList } from '@/components/quant-lab';
import type { Strategy, BacktestResultsData, SavedStrategy } from '@/components/quant-lab';

// Mock saved strategies
const mockStrategies: SavedStrategy[] = [
  {
    id: '1',
    name: 'Golden Cross',
    symbol: 'AAPL',
    createdAt: new Date('2024-11-15'),
    lastRun: new Date('2024-12-10'),
    performance: { totalReturn: 12.5, winRate: 58 },
  },
  {
    id: '2',
    name: 'RSI Reversal',
    symbol: 'MSFT',
    createdAt: new Date('2024-11-20'),
    lastRun: new Date('2024-12-08'),
    performance: { totalReturn: -3.2, winRate: 42 },
  },
  {
    id: '3',
    name: 'MACD Momentum',
    symbol: 'GOOGL',
    createdAt: new Date('2024-12-01'),
    performance: { totalReturn: 8.7, winRate: 55 },
  },
];

// Mock backtest result generator
function generateMockResults(): BacktestResultsData {
  const trades = [];
  let equity = 10000;
  const equityCurve = [];
  let profitableTrades = 0;
  let totalWins = 0;
  let totalLosses = 0;

  for (let i = 0; i < 20; i++) {
    const date = new Date();
    date.setDate(date.getDate() - (20 - i) * 7);
    const type = i % 2 === 0 ? 'buy' : 'sell';
    const price = 150 + Math.random() * 50;
    const shares = Math.floor(equity * 0.1 / price);
    const profit = type === 'sell' ? (Math.random() - 0.4) * 500 : undefined;

    if (profit !== undefined) {
      equity += profit;
      if (profit > 0) {
        profitableTrades++;
        totalWins += profit;
      } else {
        totalLosses += Math.abs(profit);
      }
    }

    trades.push({ id: i.toString(), type, date, price, shares, profit } as any);
    equityCurve.push({ date, value: equity });
  }

  const totalReturn = equity - 10000;
  const sellTrades = trades.filter((t) => t.type === 'sell').length;

  return {
    totalReturn,
    totalReturnPercent: (totalReturn / 10000) * 100,
    annualizedReturn: ((totalReturn / 10000) * 100) * 4,
    sharpeRatio: 0.8 + Math.random() * 0.8,
    maxDrawdown: 5 + Math.random() * 10,
    winRate: sellTrades > 0 ? (profitableTrades / sellTrades) * 100 : 0,
    totalTrades: sellTrades,
    profitableTrades,
    avgWin: profitableTrades > 0 ? totalWins / profitableTrades : 0,
    avgLoss: (sellTrades - profitableTrades) > 0 ? totalLosses / (sellTrades - profitableTrades) : 0,
    trades,
    equityCurve,
  };
}

// Pro upgrade prompt component
function ProUpgradePrompt() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="max-w-lg text-center px-4">
        <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-terra-100 flex items-center justify-center">
          <svg className="w-10 h-10 text-terra-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
        
        <h1 className="font-display text-display-sm text-navy-900 mb-4">
          Quant Lab is a Pro Feature
        </h1>
        
        <p className="text-body-lg text-neutral-600 mb-8">
          Build, backtest, and optimize trading strategies with our powerful quantitative analysis tools. Upgrade to Pro to unlock this feature.
        </p>

        <div className="bg-cream-50 rounded-xl p-6 mb-8 text-left">
          <h3 className="font-heading font-semibold text-navy-900 mb-4">
            What you&apos;ll get:
          </h3>
          <ul className="space-y-3">
            {[
              'Build custom strategies with technical indicators',
              'Backtest against historical data',
              'Analyze performance metrics and risk',
              'Save and compare multiple strategies',
              'Export results and trade logs',
            ].map((feature, i) => (
              <li key={i} className="flex items-center gap-3 text-body-sm text-neutral-600">
                <svg className="w-5 h-5 text-success-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                {feature}
              </li>
            ))}
          </ul>
        </div>

        <Link
          href="/pricing"
          className="inline-flex items-center gap-2 px-8 py-4 bg-terra-500 text-white font-semibold rounded-xl hover:bg-terra-600 transition-colors"
        >
          Upgrade to Pro
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </Link>

        <p className="text-caption text-neutral-500 mt-4">
          Starting at $7.99/month with annual billing
        </p>
      </div>
    </div>
  );
}

function QuantLabContent() {
  const { user } = useAuth();
  const isPro = user?.tier === 'pro';

  const [strategies] = useState<SavedStrategy[]>(mockStrategies);
  const [activeStrategyId, setActiveStrategyId] = useState<string | undefined>();
  const [results, setResults] = useState<BacktestResultsData | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  // If not Pro, show upgrade prompt
  if (!isPro) {
    return <ProUpgradePrompt />;
  }

  const handleRunBacktest = async (strategy: Strategy) => {
    setIsRunning(true);
    setResults(null);

    // Simulate backtest delay
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const mockResults = generateMockResults();
    setResults(mockResults);
    setIsRunning(false);
  };

  const handleSaveStrategy = (strategy: Strategy) => {
    console.log('Save strategy:', strategy);
    // TODO: Save to backend
  };

  const handleLoadStrategy = (strategyId: string) => {
    setActiveStrategyId(strategyId);
    // TODO: Load strategy into builder
  };

  const handleDeleteStrategy = (strategyId: string) => {
    console.log('Delete strategy:', strategyId);
    // TODO: Delete from backend
  };

  return (
    <>
      {/* Header */}
      <Section spacing="md" background="gradient">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="font-display text-display-sm lg:text-display-md text-navy-900">
                Quant Lab
              </h1>
              <span className="px-2 py-1 bg-terra-100 text-terra-700 text-caption font-semibold rounded-full">
                Pro
              </span>
            </div>
            <p className="text-body-md text-neutral-600">
              Build and backtest trading strategies
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2.5 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors">
              Documentation
            </button>
          </div>
        </div>
      </Section>

      {/* Main content */}
      <Section spacing="lg" background="default">
        <Grid cols={1} colsLg={4} gap="lg">
          {/* Strategy list sidebar */}
          <div className="lg:col-span-1">
            <StrategyList
              strategies={strategies}
              onLoad={handleLoadStrategy}
              onDelete={handleDeleteStrategy}
              activeStrategyId={activeStrategyId}
            />
          </div>

          {/* Builder and results */}
          <div className="lg:col-span-3 space-y-6">
            <StrategyBuilder
              onRunBacktest={handleRunBacktest}
              onSave={handleSaveStrategy}
              isRunning={isRunning}
            />

            <BacktestResults results={results} isLoading={isRunning} />
          </div>
        </Grid>
      </Section>
    </>
  );
}

export default function QuantLabPage() {
  return (
    <AuthGuard>
      <QuantLabContent />
    </AuthGuard>
  );
}
