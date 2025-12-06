'use client';

import { useState } from 'react';
import {
  Card,
  CardHeader,
  TickerInput,
  LoadingOverlay,
  ErrorMessage,
  MetricCard,
  MetricGrid,
  MetricRow,
  PriceChart,
  DataTable,
} from '@/components';
import { searchStock, getChartData } from '@/lib/api';
import type { SearchResponse, ChartResponse, NewsItem } from '@/lib/types';

type TimePeriod = '1m' | '3m' | '6m' | '1y' | '5y';

export default function SearchPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [period, setPeriod] = useState<TimePeriod>('1y');
  const [currentTicker, setCurrentTicker] = useState<string>('');

  const handleSearch = async (ticker: string) => {
    setLoading(true);
    setError(null);
    setData(null);
    setChartData(null);
    setCurrentTicker(ticker);
    
    try {
      const [searchResult, chart] = await Promise.all([
        searchStock(ticker),
        getChartData(ticker, period),
      ]);
      setData(searchResult);
      setChartData(chart);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search stock');
    } finally {
      setLoading(false);
    }
  };

  const handlePeriodChange = async (newPeriod: TimePeriod) => {
    setPeriod(newPeriod);
    if (currentTicker) {
      try {
        const chart = await getChartData(currentTicker, newPeriod);
        setChartData(chart);
      } catch (err) {
        console.error('Failed to update chart:', err);
      }
    }
  };

  const newsColumns = [
    { key: 'title', header: 'Title', render: (row: NewsItem) => (
      <a href={row.link} target="_blank" rel="noopener noreferrer" className="news-title">
        {row.title}
      </a>
    )},
    { key: 'source', header: 'Source' },
    { key: 'published_relative', header: 'Date' },
  ];

  return (
    <div className="container">
      {loading && <LoadingOverlay message="Searching..." />}
      
      <div className="hero">
        <h1 style={{ fontSize: '3rem' }}>Stock Search</h1>
        <p className="subtitle">Get comprehensive stock information, charts, and news</p>
      </div>

      <div className="main-layout" style={{ maxWidth: '1200px' }}>
        <Card className="mb-6">
          <CardHeader 
            title="Search Stock" 
            subtitle="Enter a ticker symbol to get started"
          />
          <TickerInput 
            onSubmit={handleSearch} 
            loading={loading}
            buttonText="Search"
          />
        </Card>

        {error && (
          <ErrorMessage 
            message={error} 
            onRetry={() => setError(null)} 
            className="mb-6"
          />
        )}

        {data && (
          <div className="space-y-6">
            <Card>
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-primary" style={{ letterSpacing: '-0.02em' }}>
                    {data.overview.name || currentTicker}
                  </h2>
                  <p className="text-secondary">
                    {data.overview.sector} • {data.overview.industry}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold text-primary">
                    {data.overview.price_display}
                  </p>
                  <p className={`text-lg font-semibold ${
                    data.overview.change_status === 'positive' ? 'text-positive' :
                    data.overview.change_status === 'negative' ? 'text-negative' : 'text-neutral'
                  }`}>
                    {data.overview.change_display} ({data.overview.change_percent_display})
                  </p>
                </div>
              </div>
            </Card>

            <MetricGrid cols={4}>
              <MetricCard label="Market Cap" value={data.overview.market_cap_display} />
              <MetricCard label="52W High" value={data.overview.fifty_two_high_display} />
              <MetricCard label="52W Low" value={data.overview.fifty_two_low_display} />
              <MetricCard label="Avg Volume" value={data.overview.avg_volume_display} />
            </MetricGrid>

            {chartData && chartData.data.prices.length > 0 && (
              <Card>
                <CardHeader title="Price Chart" />
                <div className="period-tabs mb-4" style={{ width: 'fit-content' }}>
                  {(['1m', '3m', '6m', '1y', '5y'] as TimePeriod[]).map((p) => (
                    <button
                      key={p}
                      onClick={() => handlePeriodChange(p)}
                      className={`period-tab ${period === p ? 'active' : ''}`}
                    >
                      {p.toUpperCase()}
                    </button>
                  ))}
                </div>
                <PriceChart
                  dates={chartData.data.dates}
                  prices={chartData.data.prices.filter((p): p is number => p !== null)}
                />
              </Card>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader title="Valuation Metrics" />
                <div className="space-y-2">
                  {data.valuation.map((metric, i) => (
                    <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                  ))}
                </div>
              </Card>

              <Card>
                <CardHeader title="Profitability Metrics" />
                <div className="space-y-2">
                  {data.profitability.map((metric, i) => (
                    <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                  ))}
                </div>
              </Card>

              <Card>
                <CardHeader title="Financial Health" />
                <div className="space-y-2">
                  {data.financial_health.map((metric, i) => (
                    <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                  ))}
                </div>
              </Card>

              <Card>
                <CardHeader title="Growth Metrics" />
                <div className="space-y-2">
                  {data.growth.map((metric, i) => (
                    <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                  ))}
                </div>
              </Card>
            </div>

            {data.dividend.has_dividend && (
              <Card>
                <CardHeader title="Dividend Information" />
                <MetricGrid cols={4}>
                  <MetricCard label="Dividend Yield" value={data.dividend.yield_display} />
                  <MetricCard label="Annual Rate" value={data.dividend.rate_display} />
                  <MetricCard label="Payout Ratio" value={data.dividend.payout_ratio_display} status={data.dividend.payout_status} />
                  <MetricCard label="Ex-Dividend Date" value={data.dividend.ex_date || 'N/A'} />
                </MetricGrid>
              </Card>
            )}

            {data.analyst.has_data && (
              <Card>
                <CardHeader title="Analyst Ratings" />
                <MetricGrid cols={4}>
                  <MetricCard 
                    label="Target Price" 
                    value={data.analyst.target_mean_display}
                    subtitle={data.analyst.num_analysts_display}
                  />
                  <MetricCard 
                    label="High Target" 
                    value={data.analyst.target_high_display}
                    status="positive"
                  />
                  <MetricCard 
                    label="Low Target" 
                    value={data.analyst.target_low_display}
                    status="negative"
                  />
                  <MetricCard 
                    label="Recommendation" 
                    value={data.analyst.recommendation_display}
                    status={data.analyst.recommendation_status}
                  />
                </MetricGrid>
              </Card>
            )}

            {data.profile.description && (
              <Card>
                <CardHeader title="Company Profile" />
                <p className="text-secondary leading-relaxed mb-4">
                  {data.profile.description}
                </p>
                <div className="flex flex-wrap gap-6 text-sm">
                  <span><strong className="text-primary">Employees:</strong> <span className="text-secondary">{data.profile.employees}</span></span>
                  <span><strong className="text-primary">HQ:</strong> <span className="text-secondary">{data.profile.headquarters}</span></span>
                  {data.profile.website && (
                    <a href={data.profile.website} target="_blank" rel="noopener noreferrer" className="text-positive hover:underline font-medium">
                      Website →
                    </a>
                  )}
                </div>
              </Card>
            )}

            {data.news && data.news.length > 0 && (
              <Card>
                <CardHeader title="Latest News" />
                <DataTable
                  columns={newsColumns}
                  data={data.news}
                  keyExtractor={(_, i) => i}
                />
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
