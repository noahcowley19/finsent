'use client';

// =============================================================================
// SENTIMENT ANALYSIS PAGE
// =============================================================================
// Detailed sentiment analysis for a stock
//
// Location: frontend/app/sentiment/[symbol]/page.tsx
//
// =============================================================================

import React, { useState } from 'react';
import { useParams } from 'next/navigation';
import { Section, Grid } from '@/components/layout';
import {
  AnalysisHeader,
  AnalysisTabs,
  SentimentGauge,
  SentimentTimeline,
  SourceBreakdown,
} from '@/components/analysis';
import type { TimelineDataPoint, SourceData } from '@/components/analysis';

// Mock data - replace with real API calls
const mockStockData = {
  symbol: 'AAPL',
  name: 'Apple Inc.',
  exchange: 'NASDAQ',
  price: 178.72,
  change: 2.34,
  changePercent: 1.33,
};

// Generate mock timeline data
const generateTimelineData = (): TimelineDataPoint[] => {
  const data: TimelineDataPoint[] = [];
  const now = new Date();
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    data.push({
      date,
      score: Math.round((Math.random() - 0.3) * 100),
      volume: Math.round(Math.random() * 10000),
    });
  }
  
  return data;
};

const mockSources: SourceData[] = [
  { source: 'stocktwits', score: 48, mentions: 2450, change: 12 },
  { source: 'twitter', score: 35, mentions: 8920, change: -5 },
  { source: 'reddit', score: 52, mentions: 1280, change: 18 },
  { source: 'news', score: 28, mentions: 342, change: 3 },
];

export default function SentimentPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || 'AAPL';
  
  const [inWatchlist, setInWatchlist] = useState(false);

  // In real app, fetch data based on symbol
  const stockData = { ...mockStockData, symbol };
  const timelineData = generateTimelineData();
  
  // Calculate overall sentiment
  const overallScore = Math.round(
    mockSources.reduce((sum, s) => sum + s.score * s.mentions, 0) /
    mockSources.reduce((sum, s) => sum + s.mentions, 0)
  );
  const totalDataPoints = mockSources.reduce((sum, s) => sum + s.mentions, 0);

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
            {/* Main content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Timeline */}
              <SentimentTimeline data={timelineData} />

              {/* Source breakdown */}
              <SourceBreakdown sources={mockSources} />
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Gauge */}
              <SentimentGauge
                score={overallScore}
                dataPoints={totalDataPoints}
                lastUpdated={new Date()}
              />

              {/* Key insights */}
              <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                  Key Insights
                </h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-full bg-success-100 text-success-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                      </svg>
                    </span>
                    <p className="text-body-sm text-neutral-600">
                      Sentiment up <strong className="text-success-600">15%</strong> from last week
                    </p>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-full bg-terra-100 text-terra-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    </span>
                    <p className="text-body-sm text-neutral-600">
                      Reddit mentions increased <strong className="text-terra-600">18%</strong> today
                    </p>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="w-6 h-6 rounded-full bg-navy-100 text-navy-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                      </svg>
                    </span>
                    <p className="text-body-sm text-neutral-600">
                      News sentiment mostly <strong className="text-navy-600">neutral</strong> to positive
                    </p>
                  </li>
                </ul>
              </div>

              {/* Data sources */}
              <div className="bg-cream-50 rounded-xl p-6">
                <h4 className="text-body-sm font-medium text-navy-700 mb-2">
                  Data Sources
                </h4>
                <p className="text-caption text-neutral-500">
                  Sentiment is calculated using FinBERT and VADER models analyzing data from StockTwits, X/Twitter, Reddit, and major financial news outlets. Updated every 15 minutes.
                </p>
              </div>
            </div>
          </Grid>
        </div>
      </Section>
    </>
  );
}
