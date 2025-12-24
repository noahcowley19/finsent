'use client';

// =============================================================================
// SENTIMENT ANALYSIS PAGE
// =============================================================================
// Detailed sentiment analysis for a stock
//
// Location: frontend/app/sentiment/[symbol]/page.tsx
//
// =============================================================================

import React, { useState, useEffect } from 'react';
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
import { useSentiment, useStockSearch } from '@/lib/hooks';

export default function SentimentPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || 'AAPL';
  
  const [inWatchlist, setInWatchlist] = useState(false);
  
  // Fetch real sentiment and stock data
  const { data: sentimentData, loading: sentimentLoading, error: sentimentError } = useSentiment(symbol, 10);
  const { data: stockData, loading: stockLoading } = useStockSearch(symbol);

  // Process sentiment data for display
  const [timelineData, setTimelineData] = useState<TimelineDataPoint[]>([]);
  const [sourcesData, setSourcesData] = useState<SourceData[]>([]);
  const [overallScore, setOverallScore] = useState(0);
  const [totalDataPoints, setTotalDataPoints] = useState(0);

  useEffect(() => {
    if (sentimentData) {
      // Calculate overall score from summary
      const summary = sentimentData.summary;
      const positivePercent = summary.Positive?.percentage || 0;
      const negativePercent = summary.Negative?.percentage || 0;
      const neutralPercent = summary.Neutral?.percentage || 0;
      
      // Score from -100 to 100
      const score = positivePercent - negativePercent;
      setOverallScore(Math.round(score));
      
      // Total data points
      setTotalDataPoints(sentimentData.articles.length);

      // Generate timeline data from articles
      const timeline: TimelineDataPoint[] = [];
      const articlesByDate = new Map<string, { scores: number[], count: number }>();
      
      sentimentData.articles.forEach(article => {
        const date = new Date(article.published_date);
        const dateKey = date.toISOString().split('T')[0];
        
        if (!articlesByDate.has(dateKey)) {
          articlesByDate.set(dateKey, { scores: [], count: 0 });
        }
        
        const entry = articlesByDate.get(dateKey)!;
        // Convert sentiment label to score
        const sentScore = article.sentiment === 'positive' ? 50 : 
                         article.sentiment === 'negative' ? -50 : 0;
        entry.scores.push(sentScore);
        entry.count++;
      });

      // Convert to timeline format
      articlesByDate.forEach((value, dateKey) => {
        const avgScore = value.scores.reduce((sum, s) => sum + s, 0) / value.scores.length;
        timeline.push({
          date: new Date(dateKey),
          score: Math.round(avgScore),
          volume: value.count * 100, // Scale volume for display
        });
      });
      
      // Sort by date
      timeline.sort((a, b) => a.date.getTime() - b.date.getTime());
      setTimelineData(timeline);

      // Mock source breakdown (API doesn't provide source-level breakdown)
      // In a real implementation, this would aggregate by news source
      setSourcesData([
        { source: 'news', score: Math.round(score), mentions: sentimentData.articles.length, change: 0 },
      ]);
    }
  }, [sentimentData]);

  const handleWatchlist = () => {
    setInWatchlist(!inWatchlist);
  };

  // Show loading state
  if (sentimentLoading || stockLoading) {
    return (
      <Section spacing="lg" background="default">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-terra-500 mx-auto mb-4"></div>
            <p className="text-neutral-600">Loading sentiment analysis...</p>
          </div>
        </div>
      </Section>
    );
  }

  // Show error state
  if (sentimentError) {
    return (
      <Section spacing="lg" background="default">
        <div className="max-w-2xl mx-auto text-center">
          <div className="bg-error-50 border border-error-200 rounded-xl p-6">
            <h2 className="text-heading-md font-semibold text-error-900 mb-2">
              Failed to Load Sentiment Data
            </h2>
            <p className="text-body-sm text-error-700">
              {sentimentError.message}
            </p>
          </div>
        </div>
      </Section>
    );
  }

  // Use stock data if available, otherwise use defaults
  const displayData = stockData ? {
    symbol: stockData.overview.ticker,
    name: stockData.overview.name,
    exchange: stockData.overview.exchange,
    price: stockData.overview.price,
    change: stockData.overview.change_dollar,
    changePercent: stockData.overview.change_percent,
  } : {
    symbol,
    name: symbol,
    exchange: 'UNKNOWN',
    price: 0,
    change: 0,
    changePercent: 0,
  };

  return (
    <>
      {/* Header */}
      <AnalysisHeader
        symbol={displayData.symbol}
        name={displayData.name}
        exchange={displayData.exchange}
        price={displayData.price}
        change={displayData.change}
        changePercent={displayData.changePercent}
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
              {timelineData.length > 0 && <SentimentTimeline data={timelineData} />}

              {/* Source breakdown */}
              {sourcesData.length > 0 && <SourceBreakdown sources={sourcesData} />}

              {/* Articles list */}
              {sentimentData && (
                <div className="bg-white rounded-xl border border-border-light p-6">
                  <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                    Recent Articles ({sentimentData.articles.length})
                  </h3>
                  <div className="space-y-4">
                    {sentimentData.articles.slice(0, 5).map((article, index) => (
                      <div key={index} className="border-b border-border-light pb-4 last:border-0 last:pb-0">
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-body-md font-medium text-navy-900 hover:text-terra-600 transition-colors"
                        >
                          {article.title}
                        </a>
                        <div className="flex items-center gap-3 mt-2">
                          <span className="text-caption text-neutral-500">{article.source}</span>
                          <span className="text-caption text-neutral-400">•</span>
                          <span className="text-caption text-neutral-500">
                            {new Date(article.published_date).toLocaleDateString()}
                          </span>
                          <span className={`px-2 py-0.5 rounded-full text-caption font-medium ${
                            article.sentiment === 'positive' ? 'bg-success-100 text-success-700' :
                            article.sentiment === 'negative' ? 'bg-error-100 text-error-700' :
                            'bg-neutral-100 text-neutral-700'
                          }`}>
                            {article.sentiment}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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
              {sentimentData && (
                <div className="bg-white rounded-xl border border-border-light p-6">
                  <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                    Key Insights
                  </h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-full bg-success-100 text-success-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </span>
                      <p className="text-body-sm text-neutral-600">
                        <strong className="text-success-600">{sentimentData.summary.Positive?.percentage.toFixed(1)}%</strong> positive sentiment
                      </p>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-full bg-error-100 text-error-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      </span>
                      <p className="text-body-sm text-neutral-600">
                        <strong className="text-error-600">{sentimentData.summary.Negative?.percentage.toFixed(1)}%</strong> negative sentiment
                      </p>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-full bg-neutral-100 text-neutral-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                        </svg>
                      </span>
                      <p className="text-body-sm text-neutral-600">
                        <strong className="text-neutral-600">{sentimentData.summary.Neutral?.percentage.toFixed(1)}%</strong> neutral sentiment
                      </p>
                    </li>
                  </ul>
                </div>
              )}

              {/* Data sources */}
              <div className="bg-cream-50 rounded-xl p-6">
                <h4 className="text-body-sm font-medium text-navy-700 mb-2">
                  Data Sources
                </h4>
                <p className="text-caption text-neutral-500">
                  Sentiment is calculated using FinBERT model analyzing data from major financial news outlets.
                </p>
              </div>
            </div>
          </Grid>
        </div>
      </Section>
    </>
  );
}
