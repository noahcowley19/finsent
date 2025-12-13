'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { LoadingOverlay } from '@/components';
import UltraAdvancedChart from '@/components/UltraAdvancedChart';
import { searchStock, getChartData } from '@/lib/api';
import type { SearchResponse, ChartResponse } from '@/lib/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

type TimePeriod = '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y' | 'max';
type ChartType = 'line' | 'candlestick' | 'area' | 'ohlc' | 'heikin-ashi';
type Indicator = 'sma20' | 'sma50' | 'sma200' | 'ema12' | 'ema26' | 'bb' | 'rsi' | 'macd' | 'vwap';

interface CompareStock {
  ticker: string;
  name: string;
  price: string;
  change_percent: number;
  color: string;
}

// Icons
const SearchIcon = () => <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>;
const TrendUpIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>;
const TrendDownIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>;
const SettingsIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6"/></svg>;
const CompareIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>;
const LayersIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>;
const PlusIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>;
const XIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>;
const ChevronDownIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9"/></svg>;
const StarIcon = ({ filled }: { filled?: boolean }) => filled ? <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg> : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>;
const BellIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>;
const DownloadIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>;
const RefreshIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>;
const ExpandIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>;

export default function UltraEnhancedSearchPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('');
  const [period, setPeriod] = useState<TimePeriod>('1y');
  
  const [chartType, setChartType] = useState<ChartType>('candlestick');
  const [indicators, setIndicators] = useState<Indicator[]>([]);
  const [showVolume, setShowVolume] = useState(true);
  const [chartHeight, setChartHeight] = useState(500);
  const [chartTheme, setChartTheme] = useState<'dark' | 'light'>('dark');
  
  const [showSettings, setShowSettings] = useState(false);
  const [showIndicators, setShowIndicators] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'news' | 'technicals'>('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [fullscreenChart, setFullscreenChart] = useState(false);
  
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [compareStocks, setCompareStocks] = useState<CompareStock[]>([]);
  const [compareInput, setCompareInput] = useState('');
  const [marketMovers, setMarketMovers] = useState<any>(null);
  const [sectorData, setSectorData] = useState<any[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(60);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const refreshTimerRef = useRef<NodeJS.Timeout>();

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
      try {
        const chart = await getChartData(data.ticker, newPeriod);
        setChartData(chart);
      } catch (err) {
        console.error('Failed to update chart:', err);
      }
    }
  };

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        const [movers, sectors] = await Promise.all([
          fetch(`${API_BASE}/api/search/movers`).then(r => r.json()),
          fetch(`${API_BASE}/api/search/sector-heatmap`).then(r => r.json())
        ]);
        setMarketMovers(movers);
        setSectorData(sectors.sectors || []);
      } catch (err) {
        console.error('Failed to fetch market data');
      }
    };
    fetchMarketData();
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem('watchlist_v2');
    if (saved) setWatchlist(JSON.parse(saved));
  }, []);

  useEffect(() => {
    localStorage.setItem('watchlist_v2', JSON.stringify(watchlist));
  }, [watchlist]);

  useEffect(() => {
    if (autoRefresh && data) {
      refreshTimerRef.current = setInterval(() => {
        handleSearch(data.ticker);
      }, refreshInterval * 1000);
    }
    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current);
    };
  }, [autoRefresh, refreshInterval, data]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        inputRef.current?.focus();
      } else if (e.key === 'Escape') {
        setShowSettings(false);
        setShowIndicators(false);
        setShowComparison(false);
        setFullscreenChart(false);
      } else if (e.ctrlKey && e.key === 'f') {
        e.preventDefault();
        setFullscreenChart(!fullscreenChart);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fullscreenChart]);

  const toggleIndicator = (indicator: Indicator) => {
    setIndicators(prev => 
      prev.includes(indicator) 
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator]
    );
  };

  const toggleWatchlist = (ticker: string) => {
    setWatchlist(prev => 
      prev.includes(ticker)
        ? prev.filter(t => t !== ticker)
        : [...prev, ticker]
    );
  };

  const addCompareStock = async () => {
    if (!compareInput.trim() || compareStocks.length >= 4) return;
    const ticker = compareInput.trim().toUpperCase();
    if (compareStocks.some(s => s.ticker === ticker)) return;
    
    try {
      const result = await searchStock(ticker);
      setCompareStocks([...compareStocks, {
        ticker,
        name: result.overview.name,
        price: result.overview.price_display,
        change_percent: result.overview.change_percent || 0,
        color: `hsl(${Math.random() * 360}, 70%, 60%)`,
      }]);
      setCompareInput('');
    } catch (err) {
      console.error('Failed to add stock');
    }
  };

  const removeCompareStock = (ticker: string) => {
    setCompareStocks(prev => prev.filter(s => s.ticker !== ticker));
  };

  const prepareChartData = () => {
    if (!chartData?.data) return [];
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

  return (
    <>
      <div className="mesh-gradient-bg" />
      <div className="grid-overlay" />
      
      {loading && <LoadingOverlay message="Fetching stock data..." />}
      
      <div className="container" style={{ 
        maxWidth: fullscreenChart ? '100%' : '1800px', 
        position: 'relative', 
        zIndex: 1,
        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
      }}>
        
        {!fullscreenChart && (
          <header style={{ marginBottom: '28px', animation: 'fadeInUp 0.6s ease-out' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
              <div>
                <h1 style={{ 
                  fontSize: '2.75rem', 
                  fontWeight: 900, 
                  letterSpacing: '-0.04em', 
                  marginBottom: '6px',
                  background: 'linear-gradient(135deg, #ffffff 0%, #00d4aa 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                  Advanced Stock Analysis
                </h1>
                <p style={{ fontSize: '15px', color: 'var(--text-secondary)' }}>
                  Professional-grade charting with technical indicators & real-time data
                </p>
              </div>
              
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={autoRefresh ? 'btn-primary' : 'btn-secondary'}
                  style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', padding: '10px 18px' }}
                >
                  <RefreshIcon />
                  {autoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
                </button>
              </div>
            </div>
          </header>
        )}

        {!fullscreenChart && (
          <div className="card" style={{ 
            padding: '24px', 
            marginBottom: '24px', 
            animation: 'fadeInUp 0.6s ease-out 0.1s both',
            border: '1px solid var(--border)',
            background: 'var(--bg-card)',
          }}>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ position: 'relative', flex: '1 1 400px' }}>
                <div style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}>
                  <SearchIcon />
                </div>
                <input
                  ref={inputRef}
                  type="text"
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Enter ticker symbol (AAPL, TSLA, NVDA...)"
                  className="input-field"
                  style={{ paddingLeft: '48px', fontSize: '15px', fontWeight: 500 }}
                />
                <kbd style={{ 
                  position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', 
                  fontSize: '11px', color: 'var(--text-muted)', background: 'var(--bg-tertiary)', 
                  padding: '3px 8px', borderRadius: '6px', border: '1px solid var(--border)', fontFamily: 'monospace',
                }}>/</kbd>
              </div>
              
              <button onClick={() => handleSearch()} className="btn-primary" disabled={loading} style={{ minWidth: '120px', height: '48px' }}>
                {loading ? 'Searching...' : 'Search'}
              </button>
              
              <button
                onClick={() => setShowComparison(!showComparison)}
                className={showComparison ? 'btn-primary' : 'btn-secondary'}
                style={{ display: 'flex', alignItems: 'center', gap: '8px', height: '48px', padding: '0 20px' }}
              >
                <CompareIcon /> Compare
              </button>
            </div>
            
            {watchlist.length > 0 && (
              <div style={{ marginTop: '18px', paddingTop: '18px', borderTop: '1px solid var(--border)', animation: 'fadeIn 0.4s' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
                  <StarIcon filled />
                  <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Watchlist ({watchlist.length})
                  </span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {watchlist.map(ticker => (
                    <button
                      key={ticker}
                      onClick={() => handleSearch(ticker)}
                      style={{
                        display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '8px 14px',
                        background: 'linear-gradient(135deg, var(--accent-subtle) 0%, rgba(0, 212, 170, 0.05) 100%)',
                        borderRadius: '10px', fontSize: '13px', fontWeight: 600, color: 'var(--accent)',
                        border: '1px solid var(--border-accent)', cursor: 'pointer',
                        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 8px 20px rgba(0, 212, 170, 0.2)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = 'none';
                      }}
                    >
                      <span>{ticker}</span>
                      <button 
                        onClick={(e) => { e.stopPropagation(); toggleWatchlist(ticker); }}
                        style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'currentColor', display: 'flex', padding: 0, opacity: 0.6, transition: 'opacity 0.2s' }}
                        onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                        onMouseLeave={(e) => e.currentTarget.style.opacity = '0.6'}
                      >
                        <XIcon />
                      </button>
                    </button>
                  ))}
                </div>
              </div>
            )}
            
            {error && <div className="error-message" style={{ marginTop: '16px', animation: 'shake 0.5s' }}>{error}</div>}
          </div>
        )}

        {showComparison && !fullscreenChart && (
          <div className="card" style={{ padding: '20px', marginBottom: '24px', animation: 'slideDown 0.3s ease-out', background: 'var(--bg-elevated)' }}>
            <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 700 }}>Stock Comparison</h3>
              <button onClick={() => setShowComparison(false)} className="btn-ghost" style={{ padding: '6px' }}><XIcon /></button>
            </div>
            
            <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
              <input
                type="text"
                value={compareInput}
                onChange={(e) => setCompareInput(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && addCompareStock()}
                placeholder="Add ticker to compare..."
                className="input-field"
                style={{ flex: 1 }}
              />
              <button onClick={addCompareStock} className="btn-primary" disabled={compareStocks.length >= 4}>
                <PlusIcon /> Add
              </button>
            </div>
            
            {compareStocks.length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
                {compareStocks.map(stock => (
                  <div key={stock.ticker} style={{ padding: '14px', background: 'var(--bg-card)', borderRadius: '12px', border: `2px solid ${stock.color}`, position: 'relative' }}>
                    <button onClick={() => removeCompareStock(stock.ticker)} style={{ position: 'absolute', top: '8px', right: '8px', background: 'transparent', border: 'none', cursor: 'pointer', opacity: 0.5 }}>
                      <XIcon />
                    </button>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: stock.color, marginBottom: '4px' }}>{stock.ticker}</div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>{stock.name.slice(0, 30)}</div>
                    <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>{stock.price}</div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: stock.change_percent >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                      {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: fullscreenChart ? '1fr' : (sidebarCollapsed ? '1fr 60px' : '1fr 340px'), gap: '24px', transition: 'grid-template-columns 0.4s cubic-bezier(0.4, 0, 0.2, 1)' }}>
          <div>
            {data ? (
              <>
                {!fullscreenChart && (
                  <div className="card" style={{ padding: '28px', marginBottom: '20px', animation: 'fadeInUp 0.6s ease-out 0.2s both', background: 'var(--bg-card)', border: '1px solid var(--border)', position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: data.overview.change_status === 'positive' ? 'linear-gradient(90deg, var(--positive), #00ff9d)' : 'linear-gradient(90deg, var(--negative), #ff4757)', animation: 'shimmer 2s infinite' }} />
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '24px' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px', flexWrap: 'wrap' }}>
                          <h2 style={{ fontSize: '2rem', fontWeight: 900, letterSpacing: '-0.03em', margin: 0 }}>{data.overview.name}</h2>
                          <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--accent)', background: 'var(--accent-subtle)', padding: '6px 14px', borderRadius: '10px', border: '1px solid var(--border-accent)' }}>{data.overview.ticker}</span>
                          <button onClick={() => toggleWatchlist(data.ticker)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: watchlist.includes(data.ticker) ? '#fbbf24' : 'var(--text-muted)', transition: 'all 0.2s', display: 'flex', padding: '6px' }}>
                            <StarIcon filled={watchlist.includes(data.ticker)} />
                          </button>
                        </div>
                        
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', fontSize: '13px' }}>
                          <span style={{ background: 'var(--bg-secondary)', padding: '6px 12px', borderRadius: '8px', color: 'var(--text-secondary)', fontWeight: 500 }}>{data.overview.exchange}</span>
                          <span style={{ background: 'var(--bg-secondary)', padding: '6px 12px', borderRadius: '8px', color: 'var(--text-secondary)', fontWeight: 500 }}>{data.overview.sector}</span>
                          <span style={{ background: 'var(--bg-secondary)', padding: '6px 12px', borderRadius: '8px', color: 'var(--text-secondary)', fontWeight: 500 }}>{data.overview.industry}</span>
                        </div>
                      </div>
                      
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '3rem', fontWeight: 900, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '-0.03em', lineHeight: 1, marginBottom: '8px' }}>{data.overview.price_display}</div>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '10px', fontSize: '15px', fontWeight: 700, background: data.overview.change_status === 'positive' ? 'var(--positive-light)' : 'var(--negative-light)', color: data.overview.change_status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>
                          {data.overview.change_status === 'positive' ? <TrendUpIcon /> : <TrendDownIcon />}
                          {data.overview.change_display} ({data.overview.change_percent_display})
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '10px', marginTop: '24px', paddingTop: '24px', borderTop: '1px solid var(--border)', flexWrap: 'wrap' }}>
                      <Link href={`/sentiment?ticker=${data.ticker}`} className="btn-secondary" style={{ fontSize: '13px', padding: '10px 18px', textDecoration: 'none' }}>📊 Sentiment Analysis</Link>
                      <Link href={`/financials?ticker=${data.ticker}`} className="btn-secondary" style={{ fontSize: '13px', padding: '10px 18px', textDecoration: 'none' }}>💰 Financial Reports</Link>
                      <Link href={`/insider?ticker=${data.ticker}`} className="btn-secondary" style={{ fontSize: '13px', padding: '10px 18px', textDecoration: 'none' }}>👥 Insider Trading</Link>
                      <button className="btn-secondary" style={{ fontSize: '13px', padding: '10px 18px', display: 'flex', alignItems: 'center', gap: '8px' }}><BellIcon /> Set Alert</button>
                    </div>
                  </div>
                )}

                <div className="card" style={{ padding: fullscreenChart ? '32px' : '24px', marginBottom: '24px', animation: fullscreenChart ? 'none' : 'fadeInUp 0.6s ease-out 0.3s both', background: 'var(--bg-card)', border: '1px solid var(--border)', position: fullscreenChart ? 'fixed' : 'relative', top: fullscreenChart ? 0 : 'auto', left: fullscreenChart ? 0 : 'auto', right: fullscreenChart ? 0 : 'auto', bottom: fullscreenChart ? 0 : 'auto', zIndex: fullscreenChart ? 9999 : 'auto', height: fullscreenChart ? '100vh' : 'auto', width: fullscreenChart ? '100vw' : 'auto' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em', display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <LayersIcon /> Advanced Chart
                    </div>
                    
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                      <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '12px', border: '1px solid var(--border)' }}>
                        {(['candlestick', 'line', 'area', 'ohlc', 'heikin-ashi'] as ChartType[]).map(type => (
                          <button key={type} onClick={() => setChartType(type)} style={{ padding: '7px 14px', background: chartType === type ? 'var(--accent-subtle)' : 'transparent', border: chartType === type ? '1px solid var(--border-accent)' : 'none', borderRadius: '8px', cursor: 'pointer', color: chartType === type ? 'var(--accent)' : 'var(--text-muted)', fontSize: '12px', fontWeight: 600, transition: 'all 0.2s', textTransform: 'capitalize' }}>
                            {type === 'heikin-ashi' ? 'H-Ashi' : type}
                          </button>
                        ))}
                      </div>
                      
                      <button onClick={() => setShowIndicators(!showIndicators)} className={showIndicators ? 'btn-primary' : 'btn-secondary'} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', padding: '8px 14px' }}>
                        <LayersIcon /> Indicators ({indicators.length})
                      </button>
                      
                      <button onClick={() => setShowSettings(!showSettings)} className={showSettings ? 'btn-primary' : 'btn-secondary'} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', padding: '8px 14px' }}>
                        <SettingsIcon /> Settings
                      </button>
                      
                      <button onClick={() => setFullscreenChart(!fullscreenChart)} className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', padding: '8px 14px' }}>
                        <ExpandIcon />
                      </button>
                      
                      <button className="btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', padding: '8px 14px' }}>
                        <DownloadIcon />
                      </button>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '6px', marginBottom: '20px', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '12px', width: 'fit-content' }}>
                    {(['1d', '5d', '1m', '3m', '6m', 'ytd', '1y', '2y', '5y', 'max'] as TimePeriod[]).map(p => (
                      <button key={p} onClick={() => handlePeriodChange(p)} style={{ padding: '8px 16px', background: period === p ? 'var(--accent-subtle)' : 'transparent', border: period === p ? '1px solid var(--border-accent)' : 'none', borderRadius: '8px', cursor: 'pointer', color: period === p ? 'var(--accent)' : 'var(--text-muted)', fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', transition: 'all 0.2s' }}>
                        {p}
                      </button>
                    ))}
                  </div>

                  {showIndicators && (
                    <div style={{ marginBottom: '20px', padding: '20px', background: 'var(--bg-elevated)', borderRadius: '16px', border: '1px solid var(--border)', animation: 'slideDown 0.3s ease-out' }}>
                      <h4 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '14px' }}>Technical Indicators</h4>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '10px' }}>
                        {[
                          { id: 'sma20' as Indicator, label: 'SMA 20', color: '#fbbf24' },
                          { id: 'sma50' as Indicator, label: 'SMA 50', color: '#3b82f6' },
                          { id: 'sma200' as Indicator, label: 'SMA 200', color: '#a855f7' },
                          { id: 'ema12' as Indicator, label: 'EMA 12', color: '#22d3ee' },
                          { id: 'ema26' as Indicator, label: 'EMA 26', color: '#f472b6' },
                          { id: 'bb' as Indicator, label: 'Bollinger Bands', color: '#8b5cf6' },
                          { id: 'vwap' as Indicator, label: 'VWAP', color: '#10b981' },
                          { id: 'rsi' as Indicator, label: 'RSI', color: '#f59e0b' },
                          { id: 'macd' as Indicator, label: 'MACD', color: '#3b82f6' },
                        ].map(ind => (
                          <button key={ind.id} onClick={() => toggleIndicator(ind.id)} style={{ padding: '10px 14px', background: indicators.includes(ind.id) ? `${ind.color}22` : 'var(--bg-secondary)', border: `2px solid ${indicators.includes(ind.id) ? ind.color : 'var(--border)'}`, borderRadius: '10px', cursor: 'pointer', color: indicators.includes(ind.id) ? ind.color : 'var(--text-secondary)', fontSize: '12px', fontWeight: 600, transition: 'all 0.2s', display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center' }}>
                            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: ind.color, opacity: indicators.includes(ind.id) ? 1 : 0.3 }} />
                            {ind.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {showSettings && (
                    <div style={{ marginBottom: '20px', padding: '20px', background: 'var(--bg-elevated)', borderRadius: '16px', border: '1px solid var(--border)', animation: 'slideDown 0.3s ease-out' }}>
                      <h4 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '16px' }}>Chart Settings</h4>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
                        <div>
                          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
                            <input type="checkbox" checked={showVolume} onChange={(e) => setShowVolume(e.target.checked)} style={{ width: '18px', height: '18px', cursor: 'pointer' }} />
                            <span style={{ fontSize: '13px', fontWeight: 500 }}>Show Volume Bars</span>
                          </label>
                        </div>
                        <div>
                          <label style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', display: 'block', fontWeight: 600 }}>Chart Height</label>
                          <input type="range" min="400" max="800" value={chartHeight} onChange={(e) => setChartHeight(parseInt(e.target.value))} style={{ width: '100%' }} />
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>{chartHeight}px</div>
                        </div>
                        <div>
                          <label style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', display: 'block', fontWeight: 600 }}>Auto-Refresh Interval</label>
                          <select value={refreshInterval} onChange={(e) => setRefreshInterval(parseInt(e.target.value))} style={{ width: '100%', padding: '8px 12px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px', color: 'var(--text-primary)', fontSize: '13px' }}>
                            <option value={30}>30 seconds</option>
                            <option value={60}>1 minute</option>
                            <option value={300}>5 minutes</option>
                            <option value={600}>10 minutes</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}

                  {chartDataPoints.length > 0 && (
                    <UltraAdvancedChart
                      data={chartDataPoints}
                      type={chartType}
                      indicators={indicators}
                      height={fullscreenChart ? window.innerHeight - 200 : chartHeight}
                      theme={chartTheme}
                      showVolume={showVolume}
                      enableDrawing={true}
                    />
                  )}
                  
                  <div style={{ marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                    <span>💡 Scroll to zoom • Drag to pan • Hover for details</span>
                    <span style={{ fontFamily: 'JetBrains Mono' }}>Data points: {chartDataPoints.length}</span>
                  </div>
                </div>

                {!fullscreenChart && (
                  <div className="card" style={{ padding: '20px', marginBottom: '20px', animation: 'fadeInUp 0.6s ease-out 0.4s both' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: '14px' }}>52-Week Range</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <span style={{ fontSize: '14px', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--negative)' }}>{data.overview.fifty_two_low_display}</span>
                      <div style={{ flex: 1, height: '10px', background: 'var(--bg-secondary)', borderRadius: '6px', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${data.overview.range_position || 50}%`, background: 'linear-gradient(90deg, var(--negative) 0%, var(--warning) 50%, var(--positive) 100%)', borderRadius: '6px', transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)' }} />
                        <div style={{ position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)', left: `${data.overview.range_position || 50}%`, width: '16px', height: '16px', background: 'var(--text-primary)', borderRadius: '50%', border: '3px solid var(--bg-primary)', boxShadow: '0 0 12px rgba(0, 212, 170, 0.5)' }} />
                      </div>
                      <span style={{ fontSize: '14px', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--positive)' }}>{data.overview.fifty_two_high_display}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', marginTop: '10px', fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>
                      Current: {data.overview.range_position?.toFixed(1)}% of 52-week range
                    </div>
                  </div>
                )}
              </>
            ) : (
              !fullscreenChart && (
                <div className="card" style={{ padding: '80px 40px', textAlign: 'center', animation: 'fadeInUp 0.6s ease-out 0.2s both' }}>
                  <div style={{ fontSize: '5rem', marginBottom: '20px', opacity: 0.2 }}>📈</div>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '12px' }}>Start Your Analysis</h3>
                  <p style={{ fontSize: '15px', color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto' }}>
                    Enter a ticker symbol above to unlock professional-grade charting with advanced technical indicators and real-time market data.
                  </p>
                </div>
              )
            )}
          </div>

          {!fullscreenChart && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)' }}>
              {!sidebarCollapsed && marketMovers && (
                <>
                  <div className="card" style={{ padding: '20px', animation: 'fadeInUp 0.6s ease-out 0.4s both' }}>
                    <h3 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>📊 Market Movers</h3>
                    {marketMovers.gainers?.slice(0, 5).map((stock: any, i: number) => (
                      <button key={i} onClick={() => handleSearch(stock.ticker)} style={{ width: '100%', padding: '10px', marginBottom: '8px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px', cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s' }} onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.transform = 'translateX(4px)'; }} onMouseLeave={(e) => { e.currentTarget.style.background = 'var(--bg-secondary)'; e.currentTarget.style.transform = 'translateX(0)'; }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <div style={{ fontSize: '13px', fontWeight: 700 }}>{stock.ticker}</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{stock.price_display}</div>
                          </div>
                          <div style={{ fontSize: '12px', fontWeight: 700, color: stock.change_status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>{stock.change_display}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                  
                  {sectorData.length > 0 && (
                    <div className="card" style={{ padding: '20px', animation: 'fadeInUp 0.6s ease-out 0.5s both' }}>
                      <h3 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>🎯 Sector Performance</h3>
                      {sectorData.slice(0, 8).map((sector: any, i: number) => (
                        <div key={i} style={{ marginBottom: '12px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                            <span style={{ fontSize: '12px', fontWeight: 500 }}>{sector.sector}</span>
                            <span style={{ fontSize: '12px', fontWeight: 700, color: sector.status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>
                              {sector.change_percent >= 0 ? '+' : ''}{sector.change_percent.toFixed(2)}%
                            </span>
                          </div>
                          <div style={{ height: '4px', background: 'var(--bg-secondary)', borderRadius: '2px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${Math.min(100, Math.abs(sector.change_percent) * 20)}%`, background: sector.status === 'positive' ? 'var(--positive)' : 'var(--negative)', borderRadius: '2px', transition: 'width 0.8s ease-out' }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
              
              <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} style={{ padding: '12px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '12px', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '12px', fontWeight: 600, transition: 'all 0.2s' }}>
                <ChevronDownIcon />
              </button>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
          0% { background-position: -1000px 0; }
          100% { background-position: 1000px 0; }
        }
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-10px); }
          75% { transform: translateX(10px); }
        }
      `}</style>
    </>
  );
}
