'use client';

import { useState } from 'react';
import {
  Card,
  CardHeader,
  TickerInput,
  MultiTickerInput,
  LoadingOverlay,
  ErrorMessage,
  MetricCard,
  MetricGrid,
  Badge,
  DataTable,
  SentimentChart,
} from '@/components';
import { analyzeSentiment, getSocialScreening } from '@/lib/api';
import type { SentimentAnalysisResponse, ScreeningResponse, SentimentArticle, ScreeningStock } from '@/lib/types';

type TabType = 'news' | 'social';

export default function SentimentPage() {
  const [activeTab, setActiveTab] = useState<TabType>('news');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [sentimentData, setSentimentData] = useState<SentimentAnalysisResponse | null>(null);
  const [screeningData, setScreeningData] = useState<ScreeningResponse | null>(null);

  const handleSentimentAnalysis = async (ticker: string) => {
    setLoading(true);
    setError(null);
    setSentimentData(null);
    
    try {
      const data = await analyzeSentiment(ticker);
      setSentimentData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  const handleSocialScreening = async (tickers: string[]) => {
    setLoading(true);
    setError(null);
    setScreeningData(null);
    
    try {
      const data = await getSocialScreening(tickers);
      setScreeningData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to screen stocks');
    } finally {
      setLoading(false);
    }
  };

  const getScoreStatus = (score: number | null): 'positive' | 'negative' | 'neutral' => {
    if (score === null) return 'neutral';
    if (score >= 0.1) return 'positive';
    if (score <= -0.1) return 'negative';
    return 'neutral';
  };

  const articleColumns = [
    { key: 'title', header: 'Title', render: (row: SentimentArticle) => (
      <a href={row.link} target="_blank" rel="noopener noreferrer" className="news-title">
        {row.title}
      </a>
    )},
    { key: 'published', header: 'Published' },
    { key: 'sentiment', header: 'Sentiment', render: (row: SentimentArticle) => (
      <Badge variant={row.sentiment.toLowerCase() as 'positive' | 'negative' | 'neutral'}>
        {row.sentiment}
      </Badge>
    )},
    { key: 'polarity', header: 'Polarity', align: 'right' as const, render: (row: SentimentArticle) => 
      row.polarity.toFixed(3)
    },
  ];

  const screeningColumns = [
    { key: 'ticker', header: 'Ticker', render: (row: ScreeningStock) => (
      <span className="font-semibold">{row.ticker}</span>
    )},
    { key: 'company', header: 'Company' },
    { key: 'price', header: 'Price', align: 'right' as const, render: (row: ScreeningStock) => 
      row.price_display
    },
    { key: 'stocktwits', header: 'StockTwits', align: 'right' as const, render: (row: ScreeningStock) => (
      <span className={`font-medium ${row.stocktwits !== null ? (row.stocktwits >= 0 ? 'text-positive' : 'text-negative') : 'text-neutral'}`}>
        {row.stocktwits_display}
      </span>
    )},
    { key: 'x_sentiment', header: 'X Sentiment', align: 'right' as const, render: (row: ScreeningStock) => (
      <span className={`font-medium ${row.x_sentiment !== null ? (row.x_sentiment >= 0 ? 'text-positive' : 'text-negative') : 'text-neutral'}`}>
        {row.x_display}
      </span>
    )},
    { key: 'news_sentiment', header: 'News', align: 'right' as const, render: (row: ScreeningStock) => (
      <span className={`font-medium ${row.news_sentiment !== null ? (row.news_sentiment >= 0 ? 'text-positive' : 'text-negative') : 'text-neutral'}`}>
        {row.news_display}
      </span>
    )},
    { key: 'composite', header: 'Composite', align: 'right' as const, render: (row: ScreeningStock) => (
      <span className={`font-bold ${row.composite !== null ? (row.composite >= 0 ? 'text-positive' : 'text-negative') : 'text-neutral'}`}>
        {row.composite_display}
      </span>
    )},
  ];

  return (
    <div className="container">
      {loading && <LoadingOverlay message="Analyzing sentiment..." />}
      
      <div className="hero">
        <h1 style={{ fontSize: '3rem' }}>Sentiment Analysis</h1>
        <p className="subtitle">Real-time market sentiment using transformer models and social data</p>
      </div>

      <div className="main-layout" style={{ maxWidth: '1200px' }}>
        <div className="period-tabs mb-6" style={{ width: 'fit-content' }}>
          <button
            onClick={() => setActiveTab('news')}
            className={`period-tab ${activeTab === 'news' ? 'active' : ''}`}
          >
            News Sentiment
          </button>
          <button
            onClick={() => setActiveTab('social')}
            className={`period-tab ${activeTab === 'social' ? 'active' : ''}`}
          >
            Social Screening
          </button>
        </div>

        {activeTab === 'news' ? (
          <Card className="mb-6">
            <CardHeader 
              title="News Sentiment Analysis" 
              subtitle="Analyze news articles for a single stock"
            />
            <TickerInput 
              onSubmit={handleSentimentAnalysis} 
              loading={loading}
              buttonText="Analyze"
            />
          </Card>
        ) : (
          <Card className="mb-6">
            <CardHeader 
              title="Social Media Screening" 
              subtitle="Screen multiple stocks using social sentiment"
            />
            <MultiTickerInput 
              onSubmit={handleSocialScreening} 
              loading={loading}
              buttonText="Screen"
            />
          </Card>
        )}

        {error && (
          <ErrorMessage 
            message={error} 
            onRetry={() => setError(null)} 
            className="mb-6"
          />
        )}

        {activeTab === 'news' && sentimentData && (
          <div className="space-y-6">
            <MetricGrid cols={4}>
              <MetricCard 
                label="Total Articles" 
                value={sentimentData.total_articles}
                status="neutral"
              />
              <MetricCard 
                label="Positive" 
                value={sentimentData.summary.Positive.count}
                subtitle={`${sentimentData.summary.Positive.percentage.toFixed(1)}%`}
                status="positive"
              />
              <MetricCard 
                label="Negative" 
                value={sentimentData.summary.Negative.count}
                subtitle={`${sentimentData.summary.Negative.percentage.toFixed(1)}%`}
                status="negative"
              />
              <MetricCard 
                label="Neutral" 
                value={sentimentData.summary.Neutral.count}
                subtitle={`${sentimentData.summary.Neutral.percentage.toFixed(1)}%`}
                status="neutral"
              />
            </MetricGrid>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2">
                <CardHeader title="Articles" subtitle={`Model: ${sentimentData.model}`} />
                <DataTable
                  columns={articleColumns}
                  data={sentimentData.articles}
                  keyExtractor={(_, i) => i}
                  emptyMessage="No articles found"
                />
              </Card>
              
              <Card>
                <CardHeader title="Sentiment Distribution" />
                <SentimentChart
                  positive={sentimentData.summary.Positive.count}
                  negative={sentimentData.summary.Negative.count}
                  neutral={sentimentData.summary.Neutral.count}
                />
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'social' && screeningData && (
          <div className="space-y-6">
            <MetricCard 
              label="Stocks Screened" 
              value={screeningData.results.length}
              status="neutral"
              className="max-w-xs"
            />

            <Card>
              <CardHeader title="Screening Results" subtitle="Sorted by composite score" />
              <DataTable
                columns={screeningColumns}
                data={[...screeningData.results].sort((a, b) => 
                  (b.composite ?? -999) - (a.composite ?? -999)
                )}
                keyExtractor={(row) => row.ticker}
                emptyMessage="No results found"
              />
            </Card>

            {screeningData.results.length > 0 && (
              <Card>
                <CardHeader title="Score Comparison" />
                <div className="space-y-4">
                  {screeningData.results.map((result) => (
                    <div key={result.ticker} className="flex items-center gap-4">
                      <span className="w-16 font-semibold text-primary">{result.ticker}</span>
                      <div className="flex-1 h-6 bg-neutral-light rounded-full overflow-hidden relative">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-px h-full bg-border" />
                        </div>
                        {result.composite !== null && (
                          <div 
                            className={`absolute top-0 h-full ${result.composite >= 0 ? 'bg-positive' : 'bg-negative'}`}
                            style={{ 
                              width: `${Math.min(Math.abs(result.composite) * 50, 50)}%`,
                              left: result.composite >= 0 ? '50%' : undefined,
                              right: result.composite < 0 ? '50%' : undefined,
                            }}
                          />
                        )}
                      </div>
                      <span className={`w-16 text-right font-medium ${getScoreStatus(result.composite) === 'positive' ? 'text-positive' : getScoreStatus(result.composite) === 'negative' ? 'text-negative' : 'text-neutral'}`}>
                        {result.composite_display}
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
