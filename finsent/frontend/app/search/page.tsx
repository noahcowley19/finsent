'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { LoadingOverlay, PriceChart } from '@/components';
import { searchStock, getChartData } from '@/lib/api';
import type { SearchResponse, ChartResponse, FinancialMetric, NewsItem } from '@/lib/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

type TimePeriod = '1d' | '5d' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '2y' | '5y';

interface Signal {
  type: string;
  status: 'positive' | 'negative' | 'neutral' | 'warning';
  title: string;
  description: string;
}

interface MarketMover {
  ticker: string;
  price: number;
  price_display: string;
  change_percent: number;
  change_display: string;
  change_status: 'positive' | 'negative';
  volume: number;
  volume_display: string;
}

interface SectorData {
  sector: string;
  etf: string;
  change_percent: number;
  status: 'positive' | 'negative';
}

interface CompareStock {
  ticker: string;
  name: string;
  price: number;
  price_display: string;
  change_percent: number;
  change_status: string;
  market_cap_display: string;
  sector: string;
  pe_display: string;
  roe_display: string;
  range_position: number | null;
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

const CompareIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /></svg>
);

const ExternalLinkIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" /></svg>
);

const InfoIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
);

const CloseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
);

const PlusIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
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

// Market Mover Row
function MoverRow({ mover, rank, onClick }: { mover: MarketMover; rank: number; onClick: () => void }) {
  return (
    <div 
      onClick={onClick}
      style={{
        display: 'grid',
        gridTemplateColumns: '28px 60px 1fr 90px 80px',
        alignItems: 'center',
        padding: '10px 12px',
        borderBottom: '1px solid var(--border)',
        cursor: 'pointer',
        transition: 'background 0.15s ease',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-elevated)')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>{rank}</span>
      <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent)' }}>{mover.ticker}</span>
      <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{mover.price_display}</span>
      <span style={{ 
        fontSize: '12px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px',
        color: mover.change_status === 'positive' ? 'var(--positive)' : 'var(--negative)'
      }}>
        {mover.change_status === 'positive' ? <TrendUpIcon /> : <TrendDownIcon />}
        {mover.change_display}
      </span>
      <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', textAlign: 'right' }}>{mover.volume_display}</span>
    </div>
  );
}

// Sector Cell
function SectorCell({ sector }: { sector: SectorData }) {
  const intensity = Math.min(Math.abs(sector.change_percent) / 3, 1);
  const bgColor = sector.status === 'positive' 
    ? `rgba(0, 229, 160, ${0.1 + intensity * 0.25})`
    : `rgba(255, 107, 107, ${0.1 + intensity * 0.25})`;
  
  return (
    <div style={{
      background: bgColor,
      borderRadius: '10px',
      padding: '12px 8px',
      textAlign: 'center',
      border: '1px solid var(--border)',
      cursor: 'pointer',
      transition: 'transform 0.15s ease',
    }}
    onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.03)')}
    onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
    >
      <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {sector.sector.replace(' Services', '').replace('Consumer ', '')}
      </div>
      <div style={{ fontSize: '14px', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: sector.status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>
        {formatPercent(sector.change_percent)}
      </div>
    </div>
  );
}

// Watchlist chip
function WatchlistChip({ ticker, onRemove, onClick }: { ticker: string; onRemove: () => void; onClick: () => void }) {
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      padding: '6px 10px 6px 12px',
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      borderRadius: '20px',
      fontSize: '13px',
      fontWeight: 600,
      color: 'var(--accent)',
      cursor: 'pointer',
    }}>
      <span onClick={onClick}>{ticker}</span>
      <button
        onClick={(e) => { e.stopPropagation(); onRemove(); }}
        style={{
          background: 'transparent', border: 'none', cursor: 'pointer',
          color: 'var(--text-muted)', display: 'flex', padding: 0,
        }}
      >
        <CloseIcon />
      </button>
    </div>
  );
}

// Main Component
export default function SearchPage() {
  // Search state
  const [loading, setLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('');
  const [period, setPeriod] = useState<TimePeriod>('1y');
  
  // Market data state
  const [movers, setMovers] = useState<{ gainers: MarketMover[]; losers: MarketMover[]; most_active: MarketMover[] } | null>(null);
  const [sectors, setSectors] = useState<SectorData[]>([]);
  const [moversLoading, setMoversLoading] = useState(true);
  
  // Compare state
  const [compareMode, setCompareMode] = useState(false);
  const [compareTickers, setCompareTickers] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<CompareStock[]>([]);
  const [compareLoading, setCompareLoading] = useState(false);
  
  // Watchlist
  const [watchlist, setWatchlist] = useState<string[]>([]);
  
  // UI state
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'news'>('overview');
  const [moverTab, setMoverTab] = useState<'gainers' | 'losers' | 'active'>('gainers');
  const [showAllStats, setShowAllStats] = useState(false);
  
  const inputRef = useRef<HTMLInputElement>(null);

  // Load watchlist from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('stock_watchlist');
    if (saved) setWatchlist(JSON.parse(saved));
  }, []);

  // Save watchlist
  useEffect(() => {
    localStorage.setItem('stock_watchlist', JSON.stringify(watchlist));
  }, [watchlist]);

  // Fetch market movers on mount
  useEffect(() => {
    const fetchMarketData = async () => {
      setMoversLoading(true);
      try {
        const [moversRes, sectorsRes] = await Promise.all([
          fetch(`${API_BASE}/api/search/movers`).then(r => r.json()),
          fetch(`${API_BASE}/api/search/sector-heatmap`).then(r => r.json())
        ]);
        if (moversRes.gainers) setMovers(moversRes);
        if (sectorsRes.sectors) setSectors(sectorsRes.sectors);
      } catch (err) {
        console.error('Failed to fetch market data:', err);
      } finally {
        setMoversLoading(false);
      }
    };
    fetchMarketData();
  }, []);

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

  const handleCompare = async () => {
    if (compareTickers.length < 2) return;
    setCompareLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/search/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers: compareTickers })
      });
      const result = await res.json();
      if (result.results) setCompareData(result.results);
    } catch (err) {
      console.error('Compare failed:', err);
    } finally {
      setCompareLoading(false);
    }
  };

  const addToCompare = (ticker: string) => {
    if (compareTickers.length < 4 && !compareTickers.includes(ticker)) {
      setCompareTickers([...compareTickers, ticker]);
    }
  };

  const removeFromCompare = (ticker: string) => {
    setCompareTickers(compareTickers.filter(t => t !== ticker));
    setCompareData(compareData.filter(d => d.ticker !== ticker));
  };

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
      
      <div className="container" style={{ maxWidth: '1500px', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <header style={{ marginBottom: '32px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <h1 style={{ fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '8px' }}>
                Stock <span style={{ background: 'var(--gradient-accent)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Search</span>
              </h1>
              <p style={{ fontSize: '15px', color: 'var(--text-secondary)', maxWidth: '500px' }}>
                Comprehensive stock analysis with real-time data, metrics, and insights
              </p>
            </div>
            
            {/* Compare Mode Toggle */}
            <button
              onClick={() => setCompareMode(!compareMode)}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '10px 20px',
                background: compareMode ? 'var(--accent)' : 'var(--bg-secondary)',
                color: compareMode ? 'var(--bg-primary)' : 'var(--text-secondary)',
                border: `1px solid ${compareMode ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius: '12px',
                fontSize: '14px', fontWeight: 600, cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              <CompareIcon />
              Compare Mode
            </button>
          </div>
        </header>

        {/* Search Bar */}
        <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
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
            {compareMode && tickerInput && (
              <button onClick={() => { addToCompare(tickerInput); setTickerInput(''); }} className="btn-secondary" style={{ minWidth: '100px' }}>
                <PlusIcon /> Add
              </button>
            )}
          </div>
          
          {/* Watchlist */}
          {watchlist.length > 0 && (
            <div style={{ marginTop: '16px', display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Watchlist:</span>
              {watchlist.map(t => (
                <WatchlistChip key={t} ticker={t} onClick={() => handleSearch(t)} onRemove={() => removeFromWatchlist(t)} />
              ))}
            </div>
          )}
          
          {/* Compare chips */}
          {compareMode && compareTickers.length > 0 && (
            <div style={{ marginTop: '16px', display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Comparing:</span>
              {compareTickers.map(t => (
                <WatchlistChip key={t} ticker={t} onClick={() => handleSearch(t)} onRemove={() => removeFromCompare(t)} />
              ))}
              {compareTickers.length >= 2 && (
                <button onClick={handleCompare} className="btn-primary" style={{ padding: '6px 16px', fontSize: '13px' }} disabled={compareLoading}>
                  Compare
                </button>
              )}
            </div>
          )}
          
          {error && <div className="error-message" style={{ marginTop: '16px' }}>{error}</div>}
        </div>

        {/* Compare Results */}
        {compareMode && compareData.length >= 2 && (
          <div className="card" style={{ padding: '24px', marginBottom: '24px', overflowX: 'auto' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '20px' }}>Stock Comparison</h3>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {compareData.map(s => <th key={s.ticker} style={{ textAlign: 'center' }}>{s.ticker}</th>)}
                </tr>
              </thead>
              <tbody>
                <tr><td>Company</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center', fontSize: '12px' }}>{s.name}</td>)}</tr>
                <tr><td>Price</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center', fontWeight: 600 }}>{s.price_display}</td>)}</tr>
                <tr><td>Change</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center', fontWeight: 600, color: s.change_status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>{formatPercent(s.change_percent)}</td>)}</tr>
                <tr><td>Market Cap</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center' }}>{s.market_cap_display}</td>)}</tr>
                <tr><td>P/E</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center' }}>{s.pe_display}</td>)}</tr>
                <tr><td>ROE</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center' }}>{s.roe_display}</td>)}</tr>
                <tr><td>Sector</td>{compareData.map(s => <td key={s.ticker} style={{ textAlign: 'center', fontSize: '12px' }}>{s.sector}</td>)}</tr>
              </tbody>
            </table>
          </div>
        )}

        {/* Main Content Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: data ? '1fr 360px' : '1fr', gap: '24px' }}>
          {/* Left Column - Results or Discovery */}
          <div>
            {data ? (
              <>
                {/* Stock Header */}
                <div className="card" style={{ padding: '28px', marginBottom: '24px' }}>
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
                    {!compareMode && (
                      <button onClick={() => { setCompareMode(true); addToCompare(data.ticker); }} className="btn-secondary" style={{ fontSize: '13px', padding: '8px 16px' }}>
                        <CompareIcon /> Compare
                      </button>
                    )}
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
                <div style={{ display: 'flex', gap: '4px', marginBottom: '20px', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '12px', width: 'fit-content' }}>
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
                    {/* Chart */}
                    <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Price History</div>
                        <div className="period-tabs">
                          {(['1d', '5d', '1m', '3m', '6m', 'ytd', '1y', '5y'] as TimePeriod[]).map(p => (
                            <button key={p} onClick={() => handlePeriodChange(p)} className={`period-tab ${period === p ? 'active' : ''}`}>{p.toUpperCase()}</button>
                          ))}
                        </div>
                      </div>
                      <div style={{ height: '320px', position: 'relative' }}>
                        {chartLoading && <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(6,8,13,0.8)', zIndex: 10, borderRadius: '12px' }}><div className="spinner" /></div>}
                        {chartData && chartData.data.prices.length > 0 && (
                          <PriceChart dates={chartData.data.dates} prices={chartData.data.prices.filter((p): p is number => p !== null)} />
                        )}
                      </div>
                      {chartData?.data.period_change_percent && (
                        <div style={{ marginTop: '12px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                          Period return: <span style={{ fontWeight: 600, color: (chartData.data.period_change_percent || 0) >= 0 ? 'var(--positive)' : 'var(--negative)' }}>{formatPercent(chartData.data.period_change_percent)}</span>
                        </div>
                      )}
                    </div>

                    {/* 52-Week Range */}
                    <div className="card" style={{ padding: '20px', marginBottom: '24px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: '12px' }}>52-Week Range</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{data.overview.fifty_two_low_display}</span>
                        <div style={{ flex: 1, height: '8px', background: 'var(--bg-secondary)', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                          <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${data.overview.range_position || 50}%`, background: 'linear-gradient(90deg, var(--negative), var(--warning), var(--positive))', borderRadius: '4px' }} />
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
                    <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Key Statistics</div>
                        <button onClick={() => setShowAllStats(!showAllStats)} style={{ fontSize: '12px', color: 'var(--accent)', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 500 }}>
                          {showAllStats ? 'Show Less' : 'Show All'}
                        </button>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1px', background: 'var(--border)' }}>
                        {(showAllStats ? data.key_stats : data.key_stats.slice(0, 16)).map((stat, i) => (
                          <div key={i} style={{ padding: '14px', background: 'var(--bg-card)' }}>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>{stat.label}</div>
                            <div style={{ fontSize: '14px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>{stat.value}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Analyst Ratings */}
                    {data.analyst.has_data && (
                      <div className="card" style={{ padding: '24px', marginBottom: '24px' }}>
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
                    <div className="card" style={{ padding: '24px', marginBottom: '20px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Valuation</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                        {data.valuation.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                      </div>
                    </div>
                    <div className="card" style={{ padding: '24px', marginBottom: '20px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Profitability</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                        {data.profitability.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                      </div>
                    </div>
                    <div className="card" style={{ padding: '24px', marginBottom: '20px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Financial Health</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                        {data.financial_health.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                      </div>
                    </div>
                    <div className="card" style={{ padding: '24px', marginBottom: '20px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Growth</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                        {data.growth.map((m, i) => <MetricCard key={i} metric={m} compact />)}
                      </div>
                    </div>
                    {data.dividend.has_dividend && (
                      <div className="card" style={{ padding: '24px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '16px' }}>Dividend</div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
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
                  <div className="card" style={{ padding: '24px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '20px' }}>Recent News</div>
                    {data.news && data.news.length > 0 ? (
                      data.news.map((news: NewsItem, i: number) => (
                        <a key={i} href={news.link} target="_blank" rel="noopener noreferrer" style={{ display: 'block', padding: '16px 0', borderBottom: i < data.news.length - 1 ? '1px solid var(--border)' : 'none', textDecoration: 'none' }}>
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
              /* Discovery Mode - No Search Yet */
              <div className="card" style={{ padding: '32px', textAlign: 'center' }}>
                <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.3 }}>📈</div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>Search for a Stock</h3>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '400px', margin: '0 auto' }}>
                  Enter a ticker symbol above to view comprehensive analysis, charts, metrics, and news
                </p>
              </div>
            )}
          </div>

          {/* Right Column - Market Overview */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Sector Heatmap */}
            <div className="card" style={{ padding: '20px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>Sector Performance</div>
              {sectors.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                  {sectors.slice(0, 9).map((s, i) => <SectorCell key={i} sector={s} />)}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-muted)', fontSize: '13px' }}>
                  {moversLoading ? 'Loading sectors...' : 'Unable to load sector data'}
                </div>
              )}
            </div>

            {/* Market Movers */}
            <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
              <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
                {(['gainers', 'losers', 'active'] as const).map(tab => (
                  <button key={tab} onClick={() => setMoverTab(tab)} style={{
                    flex: 1, padding: '12px', fontSize: '12px', fontWeight: 600, border: 'none', cursor: 'pointer',
                    background: moverTab === tab ? 'var(--bg-elevated)' : 'transparent',
                    color: moverTab === tab ? (tab === 'gainers' ? 'var(--positive)' : tab === 'losers' ? 'var(--negative)' : 'var(--accent)') : 'var(--text-muted)',
                    borderBottom: moverTab === tab ? `2px solid ${tab === 'gainers' ? 'var(--positive)' : tab === 'losers' ? 'var(--negative)' : 'var(--accent)'}` : '2px solid transparent',
                  }}>
                    {tab === 'gainers' ? '▲ Gainers' : tab === 'losers' ? '▼ Losers' : '◉ Active'}
                  </button>
                ))}
              </div>
              <div style={{ maxHeight: '320px', overflowY: 'auto' }}>
                {moversLoading ? (
                  <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}><div className="spinner" style={{ margin: '0 auto' }} /></div>
                ) : movers ? (
                  (moverTab === 'gainers' ? movers.gainers : moverTab === 'losers' ? movers.losers : movers.most_active).map((m, i) => (
                    <MoverRow key={m.ticker} mover={m} rank={i + 1} onClick={() => handleSearch(m.ticker)} />
                  ))
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)', fontSize: '13px' }}>Unable to load data</div>
                )}
              </div>
            </div>

            {/* Company Profile (when data available) */}
            {data && data.profile.description && (
              <div className="card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>About</div>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: '16px' }}>
                  {data.profile.description.slice(0, 300)}{data.profile.description.length > 300 && '...'}
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '12px' }}>
                  <div><span style={{ color: 'var(--text-muted)' }}>Employees:</span> <span style={{ fontWeight: 600 }}>{data.profile.employees}</span></div>
                  <div><span style={{ color: 'var(--text-muted)' }}>HQ:</span> <span style={{ fontWeight: 600 }}>{data.profile.headquarters}</span></div>
                </div>
                {data.profile.website && data.profile.website !== 'N/A' && (
                  <a href={data.profile.website} target="_blank" rel="noopener noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginTop: '12px', fontSize: '13px', color: 'var(--accent)', textDecoration: 'none', fontWeight: 500 }}>
                    Visit Website <ExternalLinkIcon />
                  </a>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
