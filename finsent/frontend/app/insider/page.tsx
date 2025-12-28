'use client';

// =============================================================================
// INSIDER TRADING PAGE
// =============================================================================
// Detailed insider trading analysis for a stock
//
// Location: frontend/app/insider/[symbol]/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import { useParams } from 'next/navigation';
import { Section, Grid } from '@/components/layout';
import {
  AnalysisHeader,
  AnalysisTabs,
  InsiderSummary,
  InsiderTable,
} from '@/components/analysis';
import type { InsiderSummaryData, InsiderTransaction } from '@/components/analysis';

// Mock data - replace with real API calls
const mockStockData = {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  exchange: 'NASDAQ',
  price: 178.72,
  change: 2.34,
  changePercent: 1.33,
};

const mockSummary: InsiderSummaryData = {
  buyCount: 8,
  sellCount: 12,
  buyValue: 3250000,
  sellValue: 850000,
  netShares: 45000,
  period: 'Last 90 days',
};

const mockTransactions: InsiderTransaction[] = [
  {
    id: '1',
    date: new Date('2024-12-15'),
    insider: 'Tim Cook',
    title: 'CEO',
    type: 'sell',
    shares: 50000,
    price: 178.50,
    value: 8925000,
  },
  {
    id: '2',
    date: new Date('2024-12-10'),
    insider: 'Luca Maestri',
    title: 'CFO',
    type: 'buy',
    shares: 10000,
    price: 175.20,
    value: 1752000,
  },
  {
    id: '3',
    date: new Date('2024-12-05'),
    insider: 'Jeff Williams',
    title: 'COO',
    type: 'buy',
    shares: 5000,
    price: 176.80,
    value: 884000,
  },
  {
    id: '4',
    date: new Date('2024-11-28'),
    insider: 'Katherine Adams',
    title: 'General Counsel',
    type: 'sell',
    shares: 8000,
    price: 174.25,
    value: 1394000,
  },
  {
    id: '5',
    date: new Date('2024-11-20'),
    insider: 'Deirdre O\'Brien',
    title: 'SVP Retail',
    type: 'buy',
    shares: 3000,
    price: 172.50,
    value: 517500,
  },
  {
    id: '6',
    date: new Date('2024-11-15'),
    insider: 'Craig Federighi',
    title: 'SVP Software',
    type: 'option',
    shares: 25000,
    price: 171.00,
    value: 4275000,
  },
  {
    id: '7',
    date: new Date('2024-11-10'),
    insider: 'Tim Cook',
    title: 'CEO',
    type: 'sell',
    shares: 75000,
    price: 169.80,
    value: 12735000,
  },
  {
    id: '8',
    date: new Date('2024-11-01'),
    insider: 'Luca Maestri',
    title: 'CFO',
    type: 'buy',
    shares: 15000,
    price: 168.25,
    value: 2523750,
  },
];

export default function InsiderPage() {
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
          {/* Summary cards */}
          <div className="mb-8">
            <InsiderSummary data={mockSummary} />
          </div>

          <Grid cols={1} colsLg={3} gap="lg">
            {/* Transaction table */}
            <div className="lg:col-span-2">
              <InsiderTable transactions={mockTransactions} />
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Notable insiders */}
              <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                  Notable Insiders
                </h3>
                <div className="space-y-4">
                  {[
                    { name: 'Tim Cook', title: 'CEO', activity: 'Net Seller', color: 'error' },
                    { name: 'Luca Maestri', title: 'CFO', activity: 'Net Buyer', color: 'success' },
                    { name: 'Jeff Williams', title: 'COO', activity: 'Net Buyer', color: 'success' },
                  ].map((insider) => (
                    <div
                      key={insider.name}
                      className="flex items-center justify-between py-2"
                    >
                      <div>
                        <p className="text-body-sm font-medium text-navy-900">
                          {insider.name}
                        </p>
                        <p className="text-caption text-neutral-500">
                          {insider.title}
                        </p>
                      </div>
                      <span
                        className={`
                          px-2 py-1 text-caption font-medium rounded
                          ${insider.color === 'success' ? 'bg-success-100 text-success-700' : ''}
                          ${insider.color === 'error' ? 'bg-error-100 text-error-700' : ''}
                        `}
                      >
                        {insider.activity}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Insights */}
              <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                  Insights
                </h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-full bg-success-100 text-success-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                      </svg>
                    </span>
                    <p className="text-body-sm text-neutral-600">
                      Net insider buying of <strong className="text-success-600">$2.4M</strong> in last 90 days
                    </p>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-full bg-navy-100 text-navy-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                      </svg>
                    </span>
                    <p className="text-body-sm text-neutral-600">
                      <strong className="text-navy-900">3 of 5</strong> top executives are net buyers
                    </p>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-full bg-warning-100 text-warning-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </span>
                    <p className="text-body-sm text-neutral-600">
                      CEO Tim Cook sells primarily for <strong className="text-navy-900">tax purposes</strong>
                    </p>
                  </li>
                </ul>
              </div>

              {/* Data note */}
              <div className="bg-cream-50 rounded-xl p-6">
                <h4 className="text-body-sm font-medium text-navy-700 mb-2">
                  Data Sources
                </h4>
                <p className="text-caption text-neutral-500">
                  Insider trading data sourced from SEC Form 4 filings. Transactions are typically reported within 2 business days of execution.
                </p>
              </div>
            </div>
          </Grid>
        </div>
      </Section>
    </>
  );
}
