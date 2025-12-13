'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { LoadingOverlay } from '@/components';
import AdvancedChart from '@/components/AdvancedChart';
import { searchStock, getChartData } from '@/lib/api';
import type { SearchResponse, ChartResponse, FinancialMetric, NewsItem } from '@/lib/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

type TimePeriod = '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y';
type ChartType = 'line' | 'candlestick';

interface Signal {
  type: string;
  status: 'positive' | 'negative' | 'neutral' | 'warning';
  title: string;
  description: string;
}

// Icons
const SearchIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>
);

const TrendUpIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
);

const TrendDownIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6" /><polyline points="17 18 23 18 23 12" /></svg>
);

const ExternalLinkIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" /></svg>
);

const SettingsIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M12 1v6m0 6v6m0 0a9 9 0 0 0 9-9 9 9 0 0 0-9-9 9 9 0 0 0-9 9 9 9 0 0 0 9 9z" /><path d="M19.4 15c.7-1.2 1-2.5 1.1-3.9M4.6 9c-.7 1.2-1 2.5-1.1 3.9M9 4.6c1.2-.7 2.5-1 3.9-1.1M15 19.4c-1.2.7-2.5 1-3.9 1.1" /></svg>
);

const CloseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
);

const CandlestickIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="6" y1="4" x2="6" y2="20" /><rect x="4" y="8" width="4" height="8" /><line x1="12" y1="2" x2="12" y2="22" /><rect x="10" y="10" width="4" height="4" /><line x1="18" y1="6" x2="18" y2="18" /><rect x="16" y="9" width="4" height="6" /></svg>
);

const LineChartIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
);

const formatPercent = (value: number | null | undefined, showSign = true): string => {
  if (value === null || value === undefined) return 'N/A';
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

// Metric Card Component
function MetricCard({ metric, compact = false }: { metric: FinancialMetric; compact?: boolean }) {
  return (
    <div style={{
      background: 'var(--bg-secondary)',
      borderRadius: compact ? '12px' : '16px',
      padding: compact ? '14px' : '20px',
      position: 'relative',
      overflow: 'hidden',
      border: '1px solid var(--border)',
      transition: 'all 0.3s var(--ease-out-expo)',
    }}>
      <div style={{
        position: 'absolute',
        top: 0, left: 0, right: 0,
        height: '3px',
        background: metric.status === 'positive' ? 'var(--positive)' : 
                    metric.status === 'negative' ? 'var(--negative)' : 'var(--neutral)',
      }} />
      <div style={{
        fontSize: '11px',
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        color: 'var(--text-tertiary)',
        marginBottom: '8px',
      }}>
        {metric.name}
      </div>
      <div style={{
        fontSize: compact ? '1.125rem' : '1.375rem',
        fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        color: metric.status === 'positive' ? 'var(--positive)' : 
               metric.status === 'negative' ? 'var(--negative)' : 'var(--text-primary)'
      }}>
        {metric.display}
      </div>
    </div>
  );
}

// Signal Badge
function SignalBadge({ signal }: { signal: Signal }) {
  const statusColors: Record<string, { bg: string; color: string; border: string }> = {
    positive: { bg: 'var(--positive-light)', color: 'var(--positive)', border: 'rgba(0, 229, 160, 0.3)' },
    negative: { bg: 'var(--negative-light)', color: 'var(--negative)', border: 'rgba(255, 107, 107, 0.3)' },
    warning: { bg: 'var(--warning-light)', color: 'var(--warning)', border: 'rgba(251, 191, 36, 0.3)' },
    neutral: { bg: 'var(--neutral-light)', color: 'var(--text-secondary)', border: 'var(--border)' }
  };
  const colors = statusColors[signal.status] || statusColors.neutral;
  
  return (
    <div style={{
      background: colors.bg,
      border: `1px solid ${colors.border}`,
      borderRadius: '12px',
      padding: '12px 16px',
      flex: '1 1 280px',
      animation: 'fadeIn 0.5s ease-out',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <div style={{
          width: '8px', height: '8px', borderRadius: '50%',
          background: colors.color, boxShadow: `0 0 8px ${colors.color}`
        }} />
        <span style={{ fontSize: '13px', fontWeight: 600, color: colors.color }}>{signal.title}</span>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
        {signal.description}
      </p>
    </div>
  );
}

// Main Component
export default function EnhancedSearchPage() {
  // Search state
  const [loading, setLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('');
  const [period, setPeriod] = useState<TimePeriod>('1y');
  
  // Chart customization
  const [chartType, setChartType] = useState<ChartType>('line');
  const [showVolume, setShowVolume] = useState(true);
  const [showMA, setShowMA] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [chartHeight, setChartHeight] = useState(400);
  
  // Watchlist
  const [watchlist, setWatchlist] = useState<string[]>([]);
  
  // UI state
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'news'>('overview');
  const [showAllStats, setShowAllStats] = useState(false);
  
  const inputRef = useRef<HTMLInputElement>(null);

  // Calculate moving averages
  const calculateMA = (data: number[], period: number): number[] => {
    const ma: number[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        ma.push(NaN);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        ma.push(sum / period);
      }
    }
    return ma;
  };

  // Load watchlist from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('stock_watchlist');
    if (saved) setWatchlist(JSON.parse(saved));
  }, []);

  // Save watchlist
  useEffect(() => {
    localStorage.setItem('stock_watchlist', JSON.stringify(watchlist));
  }, [watchlist]);

  const handleSearch = async (ticker?: string) => {
    const searchTicker = ticker || tickerInput.trim().toUpperCase();
    if (!searchTicker) return;
    
    setLoading(true);
    setError(null);
    setTickerInput(searchTicker);
    
    try {
      const [searchResult, chart] = await Promise.all([
        searchStock(searchTicker),
        getChartData(searchTicker, period),
      ]);
      setData(searchResult);
      setChartData(chart);
      setActiveTab('overview');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search stock');
      setData(null);
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

  const addToWatchlist = () => {
    if (data && !watchlist.includes(data.ticker)) {
      setWatchlist([...watchlist, data.ticker]);
    }
  };

  const removeFromWatchlist = (ticker: string) => {
    setWatchlist(watchlist.filter(t => t !== ticker));
  };

  // Prepare chart data for AdvancedChart component
  const prepareChartData = () => {
    if (!chartData || !chartData.data) return [];
    
    const { dates, prices, volumes, highs, lows, opens } = chartData.data;
    
    return dates.map((date, i) => ({
      date,
      open: opens[i] || prices[i] || 0,
      high: highs[i] || prices[i] || 0,
      low: lows[i] || prices[i] || 0,
      close: prices[i] || 0,
      volume: volumes[i] || 0,
    }));
  };

  const chartDataPoints = prepareChartData();
  const ma20 = showMA && chartDataPoints.length > 0 ? calculateMA(chartDataPoints.map(d => d.close), 20) : [];
  const ma50 = showMA && chartDataPoints.length > 0 ? calculateMA(chartDataPoints.map(d => d.close), 50) : [];
  const ma200 = showMA && chartDataPoints.length > 0 ? calculateMA(chartDataPoints.map(d => d.close), 200) : [];

  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      {/* Background */}
      <div className="mesh-gradient-bg" />
      <div className="grid-overlay" />
      
      {loading && <LoadingOverlay message="Fetching stock data..." />}
      
      <div className="container" style={{ maxWidth: '1600px', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <header style={{ marginBottom: '32px', animation: 'fadeInUp 0.6s ease-out' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <h1 style={{ fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '8px' }}>
                Stock <span style={{ background: 'var(--gradient-accent)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Search</span>
              </h1>
              <p style={{ fontSize: '15px', color: 'var(--text-secondary)', maxWidth: '500px' }}>
                Advanced stock analysis with interactive charts and real-time data
              </p>
            </div>
          </div>
        </header>

        {/* Search Bar */}
        <div className="card" style={{ padding: '24px', marginBottom: '24px', animation: 'fadeInUp 0.6s ease-out 0.1s both' }}>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <div style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}>
                <SearchIcon />
              </div>
              <input
                ref={inputRef}
                type="text"
                value={tickerInput}
                onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search by ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                className="input-field"
                style={{ paddingLeft: '48px', fontSize: '15px' }}
              />
              <div style={{ position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', fontSize: '11px', color: 'var(--text-muted)', background: 'var(--bg-tertiary)', padding: '2px 8px', borderRadius: '4px' }}>
                Press /
              </div>
            </div>
            <button onClick={() => handleSearch()} className="btn-primary" disabled={loading} style={{ minWidth: '120px' }}>
              Search
            </button>
          </div>
          
          {/* Watchlist */}
          {watchlist.length > 0 && (
            <div style={{ marginTop: '16px', display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', animation: 'fadeIn 0.4s ease-out' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Watchlist:</span>
              {watchlist.map(t => (
                <div 
                  key={t}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 12px',
                    background: 'var(--accent-subtle)',
                    borderRadius: '20px',
                    fontSize: '13px',
                    fontWeight: 600,
                    color: 'var(--accent)',
                    border: '1px solid var(--border-accent)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'var(--accent)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'var(--accent-subtle)'}
                >
                  <span onClick={() => handleSearch(t)}>{t}</span>
                  <button 
                    onClick={(e) => { e.stopPropagation(); removeFromWatchlist(t); }}
                    style={{
                      background: 'transparent', border: 'none', cursor: 'pointer',
                      color: 'currentColor', display: 'flex', padding: 0,
                    }}
                  >
                    <CloseIcon />
                  </button>
                </div>
              ))}
            </div>
          )}
          
          {error && <div className="error-message" style={{ marginTop: '16px' }}>{error}</div>}
        </div>

        {/* Main Content */}
        <div>
          {data ? (
            <>
              {/* Stock Header */}
              <div className="card" style={{ padding: '28px', marginBottom: '24px', animation: 'fadeInUp 0.6s ease-out 0.2s both' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                      <h2 style={{ fontSize: '1.75rem', fontWeight: 800, letterSpacing: '-0.02em' }}>{data.overview.name}</h2>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--accent)', background: 'var(--accent-subtle)', padding: '4px 12px', borderRadius: '8px' }}>{data.overview.ticker}</span>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{data.overview.exchange}</span>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', fontSize: '13px', color: 'var(--text-tertiary)' }}>
                      <span style={{ background: 'var(--bg-secondary)', padding: '4px 10px', borderRadius: '6px' }}>{data.overview.sector}</span>
                      <span style={{ background: 'var(--bg-secondary)', padding: '4px 10px', borderRadius: '6px' }}>{data.overview.industry}</span>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: 800, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '-0.02em' }}>{data.overview.price_display}</div>
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', marginTop: '4px' }}>
                      <span style={{ 
                        padding: '6px 14px', borderRadius: '8px', fontSize: '14px', fontWeight: 600,
                        display: 'flex', alignItems: 'center', gap: '6px',
                        background: data.overview.change_status === 'positive' ? 'var(--positive-light)' : 'var(--negative-light)',
                        color: data.overview.change_status === 'positive' ? 'var(--positive)' : 'var(--negative)'
                      }}>
                        {data.overview.change_status === 'positive' ? <TrendUpIcon /> : <TrendDownIcon />}
                        {data.overview.change_display} ({data.overview.change_percent_display})
                      </span>
                    </div>
                  </div>
                </div>
                
                {/* Quick Actions */}
                <div style={{ display: 'flex', gap: '8px', marginTop: '20px', paddingTop: '20px', borderTop: '1px solid var(--border)', flexWrap: 'wrap' }}>
                  <button onClick={addToWatchlist} className="btn-secondary" style={{ fontSize: '13px', padding: '8px 16px' }} disabled={watchlist.includes(data.ticker)}>
                    {watchlist.includes(data.ticker) ? '✓ In Watchlist' : '+ Add to Watchlist'}
                  </button>
                  <Link href={`/sentiment?ticker=${data.ticker}`} className="btn-secondary" style={{ fontSize: '13px', padding: '8px 16px', textDecoration: 'none' }}>
                    Analyze Sentiment
                  </Link>
                  <Link href={`/financials?ticker=${data.ticker}`} className="btn-secondary" style={{ fontSize: '13px', padding: '8px 16px', textDecoration: 'none' }}>
                    Deep Financials
                  </Link>
                  <Link href={`/insider?ticker=${data.ticker}`} className="btn-secondary" style={{ fontSize: '13px', padding: '8px 16px', textDecoration: 'none' }}>
                    Insider Activity
                  </Link>
                </div>
              </div>

              {/* Signals */}
              {data.signals && (data.signals as Signal[]).length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '24px' }}>
                  {(data.signals as Signal[]).map((signal, i) => <SignalBadge key={i} signal={signal} />)}
                </div>
              )}

              {/* Tab Navigation */}
              <div style={{ display: 'flex', gap: '4px', marginBottom: '20px', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '12px', width: 'fit-content', animation: 'fadeInUp 0.6s ease-out 0.3s both' }}>
                {(['overview', 'metrics', 'news'] as const).map(tab => (
                  <button key={tab} onClick={() => setActiveTab(tab)} style={{
                    padding: '10px 20px', fontSize: '13px', fontWeight: 600, border: 'none', borderRadius: '8px', cursor: 'pointer',
                    background: activeTab === tab ? 'var(--bg-elevated)' : 'transparent',
                    color: activeTab === tab ? 'var(--accent)' : 'var(--text-tertiary)',
                    transition: 'all 0.2s ease'
                  }}>
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>

              {/* Overview Tab */}
              {activeTab === 'overview' && (
                <>
                  {/* Advanced Chart */}
                  <div className="card" style={{ padding: '24px', marginBottom: '24px', animation: 'fadeInUp 0.6s ease-out 0.4s both' }}>
                    {/* Chart Controls */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Interactive Price Chart
                      </div>
                      
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                        {/* Chart Type Toggle */}
                        <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border)' }}>
                          <button
                            onClick={() => setChartType('line')}
                            style={{
                              padding: '6px 12px',
                              background: chartType === 'line' ? 'var(--accent-subtle)' : 'transparent',
                              border: chartType === 'line' ? '1px solid var(--border-accent)' : 'none',
                              borderRadius: '6px',
                              cursor: 'pointer',
                              color: chartType === 'line' ? 'var(--accent)' : 'var(--text-muted)',
                              fontSize: '12px',
                              fontWeight: 600,
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              transition: 'all 0.2s ease',
                            }}
                          >
                            <LineChartIcon />
                            Line
                          </button>
                          <button
                            onClick={() => setChartType('candlestick')}
                            style={{
                              padding: '6px 12px',
                              background: chartType === 'candlestick' ? 'var(--accent-subtle)' : 'transparent',
                              border: chartType === 'candlestick' ? '1px solid var(--border-accent)' : 'none',
                              borderRadius: '6px',
                              cursor: 'pointer',
                              color: chartType === 'candlestick' ? 'var(--accent)' : 'var(--text-muted)',
                              fontSize: '12px',
                              fontWeight: 600,
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              transition: 'all 0.2s ease',
                            }}
                          >
                            <CandlestickIcon />
                            Candles
                          </button>
                        </div>
                        
                        {/* Settings Button */}
                        <button
                          onClick={() => setShowSettings(!showSettings)}
                          style={{
                            padding: '6px 12px',
                            background: showSettings ? 'var(--accent-subtle)' : 'var(--bg-secondary)',
                            border: `1px solid ${showSettings ? 'var(--border-accent)' : 'var(--border)'}`,
                            borderRadius: '10px',
                            cursor: 'pointer',
                            color: showSettings ? 'var(--accent)' : 'var(--text-muted)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            fontSize: '12px',
                            fontWeight: 600,
                            transition: 'all 0.2s ease',
                          }}
                        >
                          <SettingsIcon />
                          Settings
                        </button>
                        
                        {/* Period Tabs */}
                        <div className="period-tabs">
                          {(['1d', '5d', '1m', '3m', '6m', 'ytd', '1y', '5y'] as TimePeriod[]).map(p => (
                            <button key={p} onClick={() => handlePeriodChange(p)} className={`period-tab ${period === p ? 'active' : ''}`}>{p.toUpperCase()}</button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Chart Settings Panel */}
                    {showSettings && (
                      <div style={{ 
                        marginBottom: '16px', 
                        padding: '16px', 
                        background: 'var(--bg-secondary)', 
                        borderRadius: '12px',
                        border: '1px solid var(--border)',
                        animation: 'fadeIn 0.3s ease-out'
                      }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={showVolume}
                              onChange={(e) => setShowVolume(e.target.checked)}
                              style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                            />
                            <span style={{ fontSize: '13px', fontWeight: 500 }}>Show Volume</span>
                          </label>
                          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={showMA}
                              onChange={(e) => setShowMA(e.target.checked)}
                              style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                            />
                            <span style={{ fontSize: '13px', fontWeight: 500 }}>Show Moving Averages</span>
                          </label>
                          <div>
                            <label style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '6px', display: 'block' }}>
                              Chart Height
                            </label>
                            <input
                              type="range"
                              min="300"
                              max="600"
                              value={chartHeight}
                              onChange={(e) => setChartHeight(parseInt(e.target.value))}
                              style={{ width: '100%' }}
                            />
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>{chartHeight}px</div>
                          </div>
                        </div>
                        
                        {showMA && (
                          <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid var(--border)', display: 'flex', gap: '16px', fontSize: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <div style={{ width: '20px', height: '2px', background: '#fbbf24' }} />
                              <span>MA20</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <div style={{ width: '20px', height: '2px', background: '#3b82f6' }} />
                              <span>MA50</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <div style={{ width: '20px', height: '2px', background: '#a855f7' }} />
                              <span>MA200</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Chart */}
                    <div style={{ position: 'relative' }}>
                      {chartLoading && (
                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(6,8,13,0.8)', zIndex: 10, borderRadius: '12px' }}>
                          <div className="spinner" />
                        </div>
                      )}
                      {chartDataPoints.length > 0 && (
                        <AdvancedChart
                          data={chartDataPoints}
                          type={chartType}
                          showVolume={showVolume}
                          showMA={showMA}
                          ma20={ma20}
                          ma50={ma50}
                          ma200={ma200}
                          height={chartHeight}
                        />
                      )}
                    </div>
                    
                    {chartData?.data.period_change_percent !== undefined && (
                      <div style={{ marginTop: '12px', fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>
                          Period return: <span style={{ fontWeight: 600, color: (chartData.data.period_change_percent || 0) >= 0 ? 'var(--positive)' : 'var(--negative)' }}>{formatPercent(chartData.data.period_change_percent)}</span>
                        </span>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          Scroll to zoom • Drag to pan • Hover for details
                        </span>
                      </div>
                    )}
                  </div>

                  {/* 52-Week Range */}
                  <div className="card" style={{ padding: '20px', marginBottom: '24px', animation: 'fadeInUp 0.6s ease-out 0.5s both' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: '12px' }}>52-Week Range</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{data.overview.fifty_two_low_display}</span>
                      <div style={{ flex: 1, height: '8px', background: 'var(--bg-secondary)', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${data.overview.range_position || 50}%`, background: 'linear-gradient(90deg, var(--negative), var(--warning), var(--positive))', borderRadius: '4px', transition: 'width 0.8s ease-out' }} />
                        <div style={{ position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)', left: `${data.overview.range_position || 50}%`, width: '14px', height: '14px', background: 'var(--text-primary)', borderRadius: '50%', border: '2px solid var(--bg-primary)', boxShadow: 'var(--shadow-md)' }} />
                      </div>
                      <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{data.overview.fifty_two_high_display}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>Low</span>
                      <span>Current: {data.overview.range_position?.toFixed(0)}% of range</span>
                      <span>High</span>
                    </div>
                  </div>

                  {/* Key Stats */}
                  <div className="card" style={{ padding: '24px', marginBottom: '24px', animation: 'fadeInUp 0.6s ease-out 0.6s both' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Key Statistics</div>
                      <button onClick={() => setShowAllStats(!showAllStats)} style={{ fontSize: '12px', color: 'var(--accent)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 500 }}>
                        {showAllStats ? 'Show Less' : 'Show All'}
                      </button>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '1px', background: 'var(--border)' }}>
                      {(showAllStats ? data.key_stats : data.key_stats.slice(0, 12)).map((stat, i) => (
                        <div key={i} style={{ padding: '14px', background: 'var(--bg-card)' }}>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>{stat.label}</div>
                          <div style={{ fontSize: '14px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{stat.value}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Analyst Ratings */}
                  {data.analyst.has_data && (
                    <div className="card" style={{ padding: '24px', marginBottom: '24px', animation: 'fadeInUp 0.6s ease-out 0.7s both' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '16px' }}>Analyst Ratings</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                        <div>
                          <div style={{ marginBottom: '16px' }}>
                            <span style={{ 
                              display: 'inline-block', padding: '8px 20px', borderRadius: '10px', fontSize: '16px', fontWeight: 700,
                              background: data.analyst.recommendation_status === 'positive' ? 'var(--positive-light)' : data.analyst.recommendation_status === 'negative' ? 'var(--negative-light)' : 'var(--neutral-light)',
                              color: data.analyst.recommendation_status === 'positive' ? 'var(--positive)' : data.analyst.recommendation_status === 'negative' ? 'var(--negative)' : 'var(--text-secondary)'
                            }}>
                              {data.analyst.recommendation_display}
                            </span>
                          </div>
                          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{data.analyst.num_analysts_display} analysts</div>
                        </div>
                        <div>
                          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Price Target</div>
                          <div style={{ fontSize: '1.75rem', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{data.analyst.target_mean_display}</div>
                          <div style={{ fontSize: '13px', marginTop: '4px', color: data.analyst.upside_status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>
                            {data.analyst.upside_display} upside
                          </div>
                          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' }}>
                            Range: {data.analyst.target_low_display} - {data.analyst.target_high_display}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Metrics Tab */}
              {activeTab === 'metrics' && (
                <>
                  <div className="card" style={{ padding: '24px', marginBottom: '20px', animation: 'fadeInUp 0.6s ease-out 0.4s both' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Valuation</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
                      {data.valuation.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                    </div>
                  </div>
                  <div className="card" style={{ padding: '24px', marginBottom: '20px', animation: 'fadeInUp 0.6s ease-out 0.5s both' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Profitability</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
                      {data.profitability.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                    </div>
                  </div>
                  <div className="card" style={{ padding: '24px', marginBottom: '20px', animation: 'fadeInUp 0.6s ease-out 0.6s both' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Financial Health</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
                      {data.financial_health.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                    </div>
                  </div>
                  <div className="card" style={{ padding: '24px', marginBottom: '20px', animation: 'fadeInUp 0.6s ease-out 0.7s both' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Growth</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
                      {data.growth.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                    </div>
                  </div>
                  {data.dividend.has_dividend && (
                    <div className="card" style={{ padding: '24px', animation: 'fadeInUp 0.6s ease-out 0.8s both' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Dividend</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '16px' }}>
                        <div><div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Yield</div><div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data.dividend.yield_display}</div></div>
                        <div><div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Annual Rate</div><div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data.dividend.rate_display}</div></div>
                        <div><div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Payout Ratio</div><div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data.dividend.payout_ratio_display}</div></div>
                        <div><div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Ex-Date</div><div style={{ fontSize: '1rem', fontWeight: 600 }}>{data.dividend.ex_date}</div></div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* News Tab */}
              {activeTab === 'news' && (
                <div className="card" style={{ padding: '24px', animation: 'fadeInUp 0.6s ease-out 0.4s both' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '20px' }}>Recent News</div>
                  {data.news && data.news.length > 0 ? (
                    data.news.map((news: NewsItem, i: number) => (
                      <a key={i} href={news.link} target="_blank" rel="noopener noreferrer" style={{ display: 'block', padding: '16px 0', borderBottom: i < data.news.length - 1 ? '1px solid var(--border)' : 'none', textDecoration: 'none', transition: 'all 0.2s ease' }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-secondary)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                      >
                        <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.5, marginBottom: '8px', display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                          {news.title}
                          <ExternalLinkIcon />
                        </div>
                        <div style={{ display: 'flex', gap: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>
                          <span style={{ fontWeight: 500 }}>{news.source}</span>
                          <span>{news.published_relative}</span>
                        </div>
                      </a>
                    ))
                  ) : (
                    <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>No recent news found</div>
                  )}
                </div>
              )}
            </>
          ) : (
            /* Discovery Mode */
            <div className="card" style={{ padding: '32px', textAlign: 'center', animation: 'fadeInUp 0.6s ease-out 0.2s both' }}>
              <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.3 }}>📈</div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>Search for a Stock</h3>
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto' }}>
                Enter a ticker symbol above to view comprehensive analysis with interactive charts and real-time data
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
