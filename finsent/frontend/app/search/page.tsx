'use client';

import { useState } from 'react';
import { LoadingOverlay, PriceChart } from '@/components';
import { searchStock, getChartData } from '@/lib/api';
import type { SearchResponse, ChartResponse, FinancialMetric, NewsItem } from '@/lib/types';

type TimePeriod = '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '5y';

export default function SearchPage() {
  const [loading, setLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('AAPL');
  const [period, setPeriod] = useState<TimePeriod>('1y');

  const handleSearch = async () => {
    if (!tickerInput.trim()) return;
    
    const ticker = tickerInput.trim().toUpperCase();
    setLoading(true);
    setError(null);
    
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
    if (data) {
      setChartLoading(true);
      try {
        const chart = await getChartData(data.ticker, newPeriod);
        setChartData(chart);
      } catch (err) {
        console.error('Failed to update chart:', err);
      } finally {
        setChartLoading(false);
      }
    }
  };

  const renderMetricItem = (metric: FinancialMetric, index: number) => (
    <div key={index} style={{ background: 'var(--background)', borderRadius: '10px', padding: '16px', position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: metric.status === 'positive' ? 'var(--positive)' : metric.status === 'negative' ? 'var(--negative)' : 'var(--neutral)' }} />
      <div style={{ fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.03em', color: 'var(--secondary)', marginBottom: '6px' }}>{metric.name}</div>
      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: metric.status === 'positive' ? 'var(--positive)' : metric.status === 'negative' ? 'var(--negative)' : 'var(--primary)' }}>{metric.display}</div>
    </div>
  );

  return (
    <div className="container" style={{ maxWidth: '1400px' }}>
      {loading && <LoadingOverlay message="Fetching stock data..." />}
      
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '3rem', fontWeight: 700, marginBottom: '12px' }}>Stock Search</h1>
        <p className="subtitle" style={{ maxWidth: '700px', margin: '0 auto' }}>
          A comprehensive stock search with charts, metrics, ratings, financial health indicators, and real-time news, sourced from yfinance.
        </p>
      </header>

      <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', maxWidth: '600px', margin: '0 auto' }}>
          <div style={{ flex: 1 }}>
            <label className="input-label">Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)</label>
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Enter ticker symbol"
              className="input-field"
            />
          </div>
          <button onClick={handleSearch} className="btn-primary" disabled={loading}>Search</button>
        </div>
        {error && <div className="error-message" style={{ marginTop: '20px' }}>{error}</div>}
      </div>

      {data && (
        <div id="results">
          {/* Stock Header */}
          <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px', marginBottom: '16px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                  <span style={{ fontSize: '1.5rem', fontWeight: 700 }}>{data.overview.name}</span>
                  <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--secondary)', background: 'var(--neutral-light)', padding: '4px 10px', borderRadius: '6px' }}>{data.overview.ticker}</span>
                  <span style={{ fontSize: '12px', color: 'var(--secondary)' }}>{data.overview.exchange}</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', fontSize: '13px', color: 'var(--secondary)' }}>
                  <span>{data.overview.sector}</span>
                  <span>•</span>
                  <span>{data.overview.industry}</span>
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '2.25rem', fontWeight: 700, marginBottom: '4px' }}>{data.overview.price_display}</div>
                <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                  <span style={{ padding: '4px 12px', borderRadius: '6px', fontSize: '14px', fontWeight: 600, background: data.overview.change_status === 'positive' ? 'var(--positive-light)' : data.overview.change_status === 'negative' ? 'var(--negative-light)' : 'var(--neutral-light)', color: data.overview.change_status === 'positive' ? 'var(--positive-dark)' : data.overview.change_status === 'negative' ? 'var(--negative-dark)' : 'var(--neutral-dark)' }}>
                    {data.overview.change_display}
                  </span>
                  <span style={{ padding: '4px 12px', borderRadius: '6px', fontSize: '14px', fontWeight: 600, background: data.overview.change_status === 'positive' ? 'var(--positive-light)' : data.overview.change_status === 'negative' ? 'var(--negative-light)' : 'var(--neutral-light)', color: data.overview.change_status === 'positive' ? 'var(--positive-dark)' : data.overview.change_status === 'negative' ? 'var(--negative-dark)' : 'var(--neutral-dark)' }}>
                    {data.overview.change_percent_display}
                  </span>
                </div>
              </div>
            </div>
            {/* 52-Week Range */}
            <div style={{ marginTop: '16px' }}>
              <div style={{ fontSize: '12px', color: 'var(--secondary)', marginBottom: '8px' }}>52-Week Range</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '14px', fontWeight: 600 }}>{data.overview.fifty_two_low_display}</span>
                <div style={{ flex: 1, height: '6px', background: 'var(--neutral-light)', borderRadius: '3px', position: 'relative' }}>
                  <div style={{ position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)', width: '12px', height: '12px', background: 'var(--primary)', borderRadius: '50%', left: `${(data.overview.range_position ?? 50)}%` }} />
                </div>
                <span style={{ fontSize: '14px', fontWeight: 600 }}>{data.overview.fifty_two_high_display}</span>
              </div>
            </div>
          </div>

          {/* Chart Section */}
          <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)' }}>Price History</div>
              <div className="period-tabs">
                {(['1d', '5d', '1m', '3m', '6m', 'ytd', '1y', '5y'] as TimePeriod[]).map((p) => (
                  <button key={p} onClick={() => handlePeriodChange(p)} className={`period-tab ${period === p ? 'active' : ''}`}>{p.toUpperCase()}</button>
                ))}
              </div>
            </div>
            <div style={{ height: '350px', position: 'relative' }}>
              {chartLoading && <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.8)', zIndex: 10 }}><div className="spinner" /></div>}
              {chartData && chartData.data.prices.length > 0 && (
                <PriceChart dates={chartData.data.dates} prices={chartData.data.prices.filter((p): p is number => p !== null)} />
              )}
            </div>
          </div>

          {/* Key Statistics */}
          <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
            <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Key Statistics</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: 'var(--border)' }}>
              {data.key_stats.map((stat, i) => (
                <div key={i} style={{ padding: '16px', background: 'var(--card-bg)' }}>
                  <div style={{ fontSize: '12px', color: 'var(--secondary)', marginBottom: '4px' }}>{stat.label}</div>
                  <div style={{ fontSize: '15px', fontWeight: 600 }}>{stat.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Metrics Row 1: Valuation & Profitability */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '24px', marginBottom: '24px' }}>
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Valuation</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>{data.valuation.map(renderMetricItem)}</div>
            </div>
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Profitability</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>{data.profitability.map(renderMetricItem)}</div>
            </div>
          </div>

          {/* Metrics Row 2: Financial Health & Growth */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '24px', marginBottom: '24px' }}>
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Financial Health</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>{data.financial_health.map(renderMetricItem)}</div>
            </div>
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Growth</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>{data.growth.map(renderMetricItem)}</div>
            </div>
          </div>

          {/* Analyst & Dividend Row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '24px', marginBottom: '24px' }}>
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Analyst Ratings</div>
              {data.analyst.has_data ? (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                    <div>
                      <div style={{ fontSize: '2rem', fontWeight: 700 }}>{data.analyst.target_mean_display}</div>
                      <div style={{ fontSize: '13px', color: 'var(--secondary)' }}>{data.analyst.num_analysts_display}</div>
                    </div>
                    <div style={{ padding: '8px 16px', borderRadius: '8px', fontSize: '14px', fontWeight: 600, background: data.analyst.recommendation_status === 'positive' ? 'var(--positive-light)' : data.analyst.recommendation_status === 'negative' ? 'var(--negative-light)' : 'var(--neutral-light)', color: data.analyst.recommendation_status === 'positive' ? 'var(--positive-dark)' : data.analyst.recommendation_status === 'negative' ? 'var(--negative-dark)' : 'var(--neutral-dark)' }}>
                      {data.analyst.recommendation_display}
                    </div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                    <div><span style={{ color: 'var(--secondary)' }}>Low: </span><span style={{ fontWeight: 600 }}>{data.analyst.target_low_display}</span></div>
                    <div><span style={{ color: 'var(--secondary)' }}>High: </span><span style={{ fontWeight: 600 }}>{data.analyst.target_high_display}</span></div>
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--secondary)', fontSize: '14px' }}>No analyst data available</div>
              )}
            </div>
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Dividend Information</div>
              {data.dividend.has_dividend ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                  <div><div style={{ fontSize: '12px', color: 'var(--secondary)' }}>Yield</div><div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data.dividend.yield_display}</div></div>
                  <div><div style={{ fontSize: '12px', color: 'var(--secondary)' }}>Annual Rate</div><div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data.dividend.rate_display}</div></div>
                  <div><div style={{ fontSize: '12px', color: 'var(--secondary)' }}>Payout Ratio</div><div style={{ fontSize: '1.25rem', fontWeight: 700, color: data.dividend.payout_status === 'positive' ? 'var(--positive)' : data.dividend.payout_status === 'negative' ? 'var(--negative)' : 'var(--primary)' }}>{data.dividend.payout_ratio_display}</div></div>
                  <div><div style={{ fontSize: '12px', color: 'var(--secondary)' }}>Ex-Dividend Date</div><div style={{ fontSize: '1rem', fontWeight: 600 }}>{data.dividend.ex_date || 'N/A'}</div></div>
                </div>
              ) : (
                <div style={{ color: 'var(--secondary)', fontSize: '14px' }}>This stock does not pay dividends</div>
              )}
            </div>
          </div>

          {/* Company Profile */}
          {data.profile.description && (
            <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Company Profile</div>
              <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', marginBottom: '16px' }}>
                <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Employees: </span><span style={{ fontWeight: 600 }}>{data.profile.employees}</span></div>
                <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>HQ: </span><span style={{ fontWeight: 600 }}>{data.profile.headquarters}</span></div>
              </div>
              <p style={{ fontSize: '14px', color: 'var(--secondary)', lineHeight: 1.7 }}>{data.profile.description}</p>
              {data.profile.website && (
                <div style={{ marginTop: '16px' }}>
                  <a href={data.profile.website} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', fontWeight: 500, fontSize: '14px', textDecoration: 'none' }}>Visit Website →</a>
                </div>
              )}
            </div>
          )}

          {/* News Section */}
          {data.news && data.news.length > 0 && (
            <div className="card" style={{ padding: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Recent News</div>
              {data.news.map((news: NewsItem, i) => (
                <div key={i} style={{ padding: '16px 0', borderBottom: i < data.news.length - 1 ? '1px solid var(--border)' : 'none' }}>
                  <a href={news.link} target="_blank" rel="noopener noreferrer" style={{ fontSize: '15px', fontWeight: 500, color: 'var(--primary)', textDecoration: 'none', lineHeight: 1.5, display: 'block', marginBottom: '8px' }}>{news.title}</a>
                  <div style={{ display: 'flex', gap: '12px', fontSize: '13px', color: 'var(--secondary)' }}>
                    <span style={{ fontWeight: 500 }}>{news.source}</span>
                    <span>{news.published_relative}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
