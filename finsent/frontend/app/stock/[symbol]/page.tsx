'use client';

// =============================================================================
// STOCK OVERVIEW PAGE
// =============================================================================
// Hub page for a stock with links to all analysis types
//
// Location: frontend/app/stock/[symbol]/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { Section, Grid } from '@/components/layout';
import { AnalysisHeader, AnalysisTabs, StockOverviewCard } from '@/components/analysis';

// Mock data - replace with real API calls
const mockStockData = {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  exchange: 'NASDAQ',
  price: 178.72,
  change: 2.34,
  changePercent: 1.33,
  stats: {
    marketCap: 2800000000000,
    volume: 52000000,
    avgVolume: 58000000,
    high52w: 199.62,
    low52w: 124.17,
    peRatio: 28.5,
    eps: 6.27,
    dividend: 0.52,
    beta: 1.28,
  },
};

const analysisCards = [
  {
    title: 'Sentiment Analysis',
    description: 'AI-powered analysis of social media, news, and forums',
    href: (symbol: string) => `/sentiment/${symbol}`,
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    color: 'terra',
    score: '+42',
    scoreLabel: 'Bullish',
  },
  {
    title: 'Financial Analysis',
    description: 'Deep-dive into company financials and ratios',
    href: (symbol: string) => `/financials/${symbol}`,
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    color: 'navy',
    score: 'A-',
    scoreLabel: 'Strong',
  },
  {
    title: 'Insider Trading',
    description: 'Track what executives and insiders are doing',
    href: (symbol: string) => `/insider/${symbol}`,
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
    color: 'success',
    score: 'Net Buy',
    scoreLabel: '$2.4M',
  },
];

export default function StockOverviewPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || 'AAPL';
  
  const [inWatchlist, setInWatchlist] = useState(false);

  // In real app, fetch data based on symbol
  const stockData = { ...mockStockData, symbol };

  const handleWatchlist = () => {
    setInWatchlist(!inWatchlist);
  };

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
            {/* Analysis cards */}
            <div className="lg:col-span-2">
              <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                Analysis
              </h2>
              <div className="space-y-4">
                {analysisCards.map((card) => (
                  <Link
                    key={card.title}
                    href={card.href(symbol)}
                    className="flex items-center gap-6 p-6 bg-white rounded-xl border border-border-light hover:border-border-medium hover:shadow-md transition-all group"
                  >
                    {/* Icon */}
                    <div
                      className={`
                        w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0
                        transition-transform group-hover:scale-105
                        ${card.color === 'terra' ? 'bg-terra-100 text-terra-600' : ''}
                        ${card.color === 'navy' ? 'bg-navy-100 text-navy-600' : ''}
                        ${card.color === 'success' ? 'bg-success-100 text-success-600' : ''}
                      `}
                    >
                      {card.icon}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-1 group-hover:text-navy-700">
                        {card.title}
                      </h3>
                      <p className="text-body-sm text-neutral-600">
                        {card.description}
                      </p>
                    </div>

                    {/* Score */}
                    <div className="text-right">
                      <p
                        className={`
                          font-display text-xl font-bold
                          ${card.color === 'terra' ? 'text-terra-600' : ''}
                          ${card.color === 'navy' ? 'text-navy-600' : ''}
                          ${card.color === 'success' ? 'text-success-600' : ''}
                        `}
                      >
                        {card.score}
                      </p>
                      <p className="text-caption text-neutral-500">
                        {card.scoreLabel}
                      </p>
                    </div>

                    {/* Arrow */}
                    <svg
                      className="w-5 h-5 text-neutral-400 group-hover:text-navy-500 group-hover:translate-x-1 transition-all"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                ))}
              </div>
            </div>

            {/* Stats sidebar */}
            <div>
              <StockOverviewCard stats={stockData.stats} />
            </div>
          </Grid>
        </div>
      </Section>
    </>
  );
}
