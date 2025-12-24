'use client';

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { LoadingOverlay, Badge, BarChart } from '@/components';
import { analyzeSentiment, getSocialScreening } from '@/lib/api';
import type { SentimentAnalysisResponse, SocialScreeningResponse, SentimentArticle, ScreeningStock } from '@/lib/types';

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'];
const MAX_TICKERS = 15;
const MAX_HISTORY_SNAPSHOTS = 1000;
const HISTORY_RETENTION_DAYS = 30;

// Sentiment thresholds
const SENTIMENT_THRESHOLDS = {
  BULLISH: 55,
  BEARISH: 45,
  STRONG_BULLISH: 70,
  STRONG_BEARISH: 30,
};

const DIVERGENCE_THRESHOLD = 30;

// ============================================================================
// TYPES
// ============================================================================

interface SentimentSnapshot {
  timestamp: number;
  ticker: string;
  composite: number | null;
  stocktwits: number | null;
  twitter: number | null;
  news: number | null;
}

interface Insight {
  type: 'success' | 'danger' | 'warning' | 'info';
  message: string;
}

type SortField = 'ticker' | 'composite' | 'stocktwits' | 'x_sentiment' | 'news_sentiment';
type SortDirection = 'asc' | 'desc' | null;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const getSentimentColor = (value: number | null): string => {
  if (value === null) return '#64748b'; // neutral color
  if (value >= SENTIMENT_THRESHOLDS.BULLISH) return '#00e5a0'; // positive
  if (value <= SENTIMENT_THRESHOLDS.BEARISH) return '#ff6b6b'; // negative
  return '#64748b'; // neutral
};

const formatSentimentValue = (value: number | null): string => {
  if (value === null) return 'N/A';
  return value.toFixed(1);
};

const sentimentToVariant = (sentiment: string): 'success' | 'danger' | 'default' => {
  const lower = sentiment.toLowerCase();
  if (lower === 'positive') return 'success';
  if (lower === 'negative') return 'danger';
  return 'default';
};

// ============================================================================
// LOCAL STORAGE HELPERS
// ============================================================================

const STORAGE_KEY = 'caveray_sentiment_history';

const loadHistory = (): SentimentSnapshot[] => {
  if (typeof window === 'undefined') return [];
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const history = JSON.parse(stored) as SentimentSnapshot[];
    
    // Clean up old data
    const cutoff = Date.now() - (HISTORY_RETENTION_DAYS * 24 * 60 * 60 * 1000);
    return history.filter(snap => snap.timestamp > cutoff);
  } catch (err) {
    console.error('Failed to load sentiment history:', err);
    return [];
  }
};

const saveHistory = (history: SentimentSnapshot[]) => {
  if (typeof window === 'undefined') return;
  try {
    // Keep only the most recent MAX_HISTORY_SNAPSHOTS
    const trimmed = history.slice(-MAX_HISTORY_SNAPSHOTS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch (err) {
    console.error('Failed to save sentiment history:', err);
  }
};

const addSnapshot = (data: SocialScreeningResponse) => {
  const history = loadHistory();
  const timestamp = Date.now();
  
  const newSnapshots: SentimentSnapshot[] = data.results.map(stock => ({
    timestamp,
    ticker: stock.ticker,
    composite: stock.composite,
    stocktwits: stock.stocktwits,
    twitter: stock.x_sentiment,
    news: stock.news_sentiment,
  }));
  
  saveHistory([...history, ...newSnapshots]);
};

const getTickerHistory = (ticker: string, days: number = 7): SentimentSnapshot[] => {
  const history = loadHistory();
  const cutoff = Date.now() - (days * 24 * 60 * 60 * 1000);
  
  return history
    .filter(snap => snap.ticker === ticker && snap.timestamp > cutoff)
    .sort((a, b) => a.timestamp - b.timestamp);
};

// ============================================================================
// CANVAS CHART COMPONENTS
// ============================================================================

const SentimentGauge: React.FC<{ value: number | null; size?: number }> = ({ value, size = 120 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = (size / 2) - 10;
    
    // Clear canvas
    ctx.clearRect(0, 0, size, size);
    
    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 0.25 * Math.PI);
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 12;
    ctx.stroke();
    
    if (value !== null) {
      // Value arc
      const angle = 0.75 * Math.PI + (1.5 * Math.PI * (value / 100));
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, angle);
      ctx.strokeStyle = getSentimentColor(value);
      ctx.lineWidth = 12;
      ctx.lineCap = 'round';
      ctx.stroke();
      
      // Value text
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${size / 4}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(value.toFixed(0), centerX, centerY);
    } else {
      // N/A text
      ctx.fillStyle = '#666';
      ctx.font = `${size / 5}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('N/A', centerX, centerY);
    }
  }, [value, size]);
  
  return <canvas ref={canvasRef} width={size} height={size} style={{ display: 'block' }} />;
};

const MiniTrendChart: React.FC<{ history: SentimentSnapshot[]; width?: number; height?: number }> = ({ 
  history, 
  width = 60, 
  height = 24 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, width, height);
    
    const values = history.map(h => h.composite).filter((v): v is number => v !== null);
    if (values.length === 0) return;
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    
    const xStep = width / (values.length - 1 || 1);
    
    ctx.beginPath();
    values.forEach((value, i) => {
      const x = i * xStep;
      const y = height - ((value - min) / range) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    const latestValue = values[values.length - 1];
    const firstValue = values[0];
    const trend = latestValue >= firstValue ? 'up' : 'down';
    
    ctx.strokeStyle = trend === 'up' ? 'var(--positive)' : 'var(--negative)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
  }, [history, width, height]);
  
  return <canvas ref={canvasRef} width={width} height={height} style={{ display: 'block' }} />;
};

const HistoricalBarChart: React.FC<{ history: SentimentSnapshot[] }> = ({ history }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = 400;
    const height = 200;
    const padding = 40;
    
    ctx.clearRect(0, 0, width, height);
    
    const values = history.map(h => h.composite).filter((v): v is number => v !== null);
    if (values.length === 0) return;
    
    const barWidth = (width - 2 * padding) / values.length;
    
    // Draw bars
    values.forEach((value, i) => {
      const x = padding + i * barWidth;
      const barHeight = (value / 100) * (height - 2 * padding);
      const y = height - padding - barHeight;
      
      ctx.fillStyle = getSentimentColor(value);
      ctx.fillRect(x, y, barWidth - 2, barHeight);
    });
    
    // Draw axes
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
  }, [history]);
  
  return <canvas ref={canvasRef} width={400} height={200} style={{ display: 'block' }} />;
};

// ============================================================================
// INSIGHTS GENERATION
// ============================================================================

const generateInsights = (stocks: ScreeningStock[]): Insight[] => {
  const insights: Insight[] = [];
  
  if (stocks.length === 0) return insights;
  
  // Find most bullish stocks
  const bullish = stocks.filter(s => s.composite !== null && s.composite >= SENTIMENT_THRESHOLDS.STRONG_BULLISH);
  if (bullish.length > 0) {
    const top = bullish.sort((a, b) => (b.composite ?? 0) - (a.composite ?? 0))[0];
    insights.push({
      type: 'success',
      message: `🚀 ${top.ticker} shows strong bullish sentiment with a composite score of ${formatSentimentValue(top.composite)}.`
    });
  }
  
  // Find most bearish stocks
  const bearish = stocks.filter(s => s.composite !== null && s.composite <= SENTIMENT_THRESHOLDS.STRONG_BEARISH);
  if (bearish.length > 0) {
    const bottom = bearish.sort((a, b) => (a.composite ?? 0) - (b.composite ?? 0))[0];
    insights.push({
      type: 'danger',
      message: `⚠️ ${bottom.ticker} shows strong bearish sentiment with a composite score of ${formatSentimentValue(bottom.composite)}.`
    });
  }
  
  // Detect divergences
  stocks.forEach(stock => {
    const sources = [
      stock.stocktwits,
      stock.x_sentiment,
      stock.news_sentiment
    ].filter((v): v is number => v !== null);
    
    if (sources.length >= 2) {
      const min = Math.min(...sources);
      const max = Math.max(...sources);
      
      if (max - min > DIVERGENCE_THRESHOLD) {
        insights.push({
          type: 'warning',
          message: `⚡ ${stock.ticker} shows major divergence: sources disagree by ${(max - min).toFixed(0)} points.`
        });
      }
    }
  });
  
  // Market overview
  const validScores = stocks
    .map(s => s.composite)
    .filter((v): v is number => v !== null);
  
  if (validScores.length > 0) {
    const avg = validScores.reduce((a, b) => a + b, 0) / validScores.length;
    const sentiment = avg >= SENTIMENT_THRESHOLDS.BULLISH ? 'bullish' : 
                     avg <= SENTIMENT_THRESHOLDS.BEARISH ? 'bearish' : 'neutral';
    
    insights.push({
      type: 'info',
      message: `📊 Market average sentiment is ${avg.toFixed(1)} (${sentiment}) across ${stocks.length} tracked stocks.`
    });
  }
  
  return insights.slice(0, 5); // Limit to 5 insights
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function SentimentPage() {
  // ========== STATE ==========
  
  // Screener state
  const [screenerLoading, setScreenerLoading] = useState(true);
  const [screenerData, setScreenerData] = useState<SocialScreeningResponse | null>(null);
  const [selectedTickers, setSelectedTickers] = useState<string[]>(DEFAULT_TICKERS);
  
  // News analyzer state
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsData, setNewsData] = useState<SentimentAnalysisResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('gold');
  const [numArticles, setNumArticles] = useState(5);
  const [error, setError] = useState<string | null>(null);
  
  // UI state
  const [showScreenerExplainer, setShowScreenerExplainer] = useState(false);
  const [showInsights, setShowInsights] = useState(true);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  
  // Filter & sort state
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<SortField>('composite');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [sentimentMin, setSentimentMin] = useState(0);
  const [sentimentMax, setSentimentMax] = useState(100);
  const [showFilters, setShowFilters] = useState(false);
  
  // Auto-refresh state
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(60); // seconds
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // History state
  const [sentimentHistory, setSentimentHistory] = useState<SentimentSnapshot[]>([]);
  
  // ========== EFFECTS ==========
  
  // Load history on mount
  useEffect(() => {
    setSentimentHistory(loadHistory());
  }, []);
  
  // Load screening data on mount
  useEffect(() => {
    loadScreeningData();
  }, [selectedTickers]);
  
  // Auto-refresh timer
  useEffect(() => {
    if (autoRefresh) {
      refreshTimerRef.current = setInterval(() => {
        loadScreeningData();
      }, refreshInterval * 1000);
      
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    } else {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    }
  }, [autoRefresh, refreshInterval]);
  
  // ========== CALLBACKS ==========
  
  const loadScreeningData = useCallback(async () => {
    setScreenerLoading(true);
    try {
      const data = await getSocialScreening(selectedTickers);
      setScreenerData(data);
      
      // Save snapshot to history
      addSnapshot(data);
      setSentimentHistory(loadHistory());
      
    } catch (err) {
      console.error('Failed to load screening data:', err);
    } finally {
      setScreenerLoading(false);
    }
  }, [selectedTickers]);
  
  const handleAnalyzeSentiment = async () => {
    if (!tickerInput.trim()) return;
    
    setNewsLoading(true);
    setError(null);
    setNewsData(null);
    
    try {
      const data = await analyzeSentiment(tickerInput.trim(), numArticles);
      setNewsData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze sentiment');
    } finally {
      setNewsLoading(false);
    }
  };
  
  const removeTicker = useCallback((ticker: string) => {
    setSelectedTickers(prev => prev.filter(t => t !== ticker));
  }, []);
  
  const toggleRow = useCallback((ticker: string) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(ticker)) {
        next.delete(ticker);
      } else {
        next.add(ticker);
      }
      return next;
    });
  }, []);
  
  const handleSort = useCallback((field: SortField) => {
    if (sortField === field) {
      // Cycle through: asc -> desc -> null
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortDirection(null);
        setSortField('composite'); // Reset to default
      } else {
        setSortDirection('asc');
      }
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  }, [sortField, sortDirection]);
  
  const exportToCSV = useCallback(() => {
    if (!screenerData) return;
    
    const headers = ['Ticker', 'Composite', 'StockTwits', 'Twitter', 'News'];
    const rows = screenerData.results.map(stock => [
      stock.ticker,
      formatSentimentValue(stock.composite),
      formatSentimentValue(stock.stocktwits),
      formatSentimentValue(stock.x_sentiment),
      formatSentimentValue(stock.news_sentiment),
    ]);
    
    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [screenerData]);
  
  // ========== MEMOIZED VALUES ==========
  
  const filteredAndSortedStocks = useMemo(() => {
    if (!screenerData) return [];
    
    let filtered = screenerData.results;
    
    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(stock => 
        stock.ticker.toLowerCase().includes(query) ||
        stock.company?.toLowerCase().includes(query)
      );
    }
    
    // Apply sentiment range filter
    filtered = filtered.filter(stock => {
      const score = stock.composite;
      if (score === null) return true;
      return score >= sentimentMin && score <= sentimentMax;
    });
    
    // Apply sorting
    if (sortDirection !== null && sortField) {
      filtered = [...filtered].sort((a, b) => {
        let aVal: number | null = null;
        let bVal: number | null = null;
        
        switch (sortField) {
          case 'ticker':
            return sortDirection === 'asc' 
              ? a.ticker.localeCompare(b.ticker)
              : b.ticker.localeCompare(a.ticker);
          case 'composite':
            aVal = a.composite;
            bVal = b.composite;
            break;
          case 'stocktwits':
            aVal = a.stocktwits;
            bVal = b.stocktwits;
            break;
          case 'x_sentiment':
            aVal = a.x_sentiment;
            bVal = b.x_sentiment;
            break;
          case 'news_sentiment':
            aVal = a.news_sentiment;
            bVal = b.news_sentiment;
            break;
        }
        
        // Handle nulls
        if (aVal === null && bVal === null) return 0;
        if (aVal === null) return 1;
        if (bVal === null) return -1;
        
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      });
    }
    
    return filtered;
  }, [screenerData, searchQuery, sentimentMin, sentimentMax, sortField, sortDirection]);
  
  const insights = useMemo(() => {
    if (!screenerData) return [];
    return generateInsights(screenerData.results);
  }, [screenerData]);
  
  // ========== RENDER ==========
  
  return (
    <div className="container" style={{ maxWidth: '1400px' }}>
      
      {/* Header */}
      <div style={{ marginBottom: '2rem', animation: 'fadeIn 0.6s ease-out' }}>
        <h1 style={{ 
          fontSize: '2.5rem', 
          marginBottom: '0.5rem',
          background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          display: 'inline-block'
        }}>
          📊 Sentiment Intelligence
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
          Real-time market sentiment analysis powered by AI
        </p>
      </div>
      
      {/* Insights Panel */}
      {showInsights && insights.length > 0 && (
        <div style={{ 
          marginBottom: '2rem',
          padding: '1.5rem',
          background: 'linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 212, 170, 0.05) 100%)',
          borderRadius: '12px',
          border: '1px solid rgba(0, 212, 170, 0.2)',
          animation: 'slideDown 0.6s ease-out'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.3rem', margin: 0 }}>💡 Smart Insights</h2>
            <button
              onClick={() => setShowInsights(false)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '1.2rem'
              }}
            >
              ✕
            </button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {insights.map((insight, i) => (
              <div
                key={i}
                style={{
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  background: insight.type === 'success' ? 'rgba(0, 229, 160, 0.15)' :
                             insight.type === 'danger' ? 'rgba(255, 107, 107, 0.15)' :
                             insight.type === 'warning' ? 'rgba(255, 193, 7, 0.15)' :
                             'rgba(0, 212, 170, 0.15)',
                  border: `1px solid ${
                    insight.type === 'success' ? 'var(--positive)' :
                    insight.type === 'danger' ? 'var(--negative)' :
                    insight.type === 'warning' ? '#ffc107' :
                    'var(--accent)'
                  }`,
                  fontSize: '0.95rem',
                  animation: `fadeIn 0.6s ease-out ${i * 0.1}s both`
                }}
              >
                {insight.message}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* ===== SOCIAL SCREENER SECTION ===== */}
      <section style={{ marginBottom: '3rem' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '1.5rem',
          flexWrap: 'wrap',
          gap: '1rem'
        }}>
          <div>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '0.3rem' }}>
              🔍 Social Sentiment Screener
            </h2>
            <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
              Track real-time sentiment across StockTwits, X (Twitter), and news
            </p>
          </div>
          
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              onClick={() => setShowFilters(!showFilters)}
              style={{
                padding: '0.6rem 1.2rem',
                background: showFilters ? 'var(--accent)' : 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              🎛️ Filters {showFilters && '✓'}
            </button>
            
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              style={{
                padding: '0.6rem 1.2rem',
                background: autoRefresh ? 'var(--positive)' : 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              {autoRefresh ? '⏸️' : '▶️'} Auto-Refresh {autoRefresh && `(${refreshInterval}s)`}
            </button>
            
            <button
              onClick={loadScreeningData}
              disabled={screenerLoading}
              style={{
                padding: '0.6rem 1.2rem',
                background: 'var(--accent)',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                cursor: screenerLoading ? 'not-allowed' : 'pointer',
                opacity: screenerLoading ? 0.6 : 1
              }}
            >
              {screenerLoading ? '⏳ Loading...' : '🔄 Refresh'}
            </button>
            
            <button
              onClick={exportToCSV}
              disabled={!screenerData}
              style={{
                padding: '0.6rem 1.2rem',
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text)',
                cursor: screenerData ? 'pointer' : 'not-allowed',
                opacity: screenerData ? 1 : 0.6
              }}
            >
              📥 Export CSV
            </button>
          </div>
        </div>
        
        {/* Filter Panel */}
        {showFilters && (
          <div style={{
            padding: '1.5rem',
            background: 'var(--surface)',
            borderRadius: '8px',
            marginBottom: '1rem',
            border: '1px solid var(--border)',
            animation: 'slideDown 0.3s ease-out'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
              {/* Search */}
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  🔎 Search Ticker / Company
                </label>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="e.g., AAPL or Apple"
                  style={{
                    width: '100%',
                    padding: '0.6rem',
                    background: 'var(--background)',
                    border: '1px solid var(--border)',
                    borderRadius: '6px',
                    color: 'var(--text)',
                    fontSize: '1rem'
                  }}
                />
              </div>
              
              {/* Sentiment Range */}
              <div>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  📊 Sentiment Range: {sentimentMin} - {sentimentMax}
                </label>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={sentimentMin}
                    onChange={(e) => setSentimentMin(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={sentimentMax}
                    onChange={(e) => setSentimentMax(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                </div>
              </div>
              
              {/* Auto-refresh interval */}
              {autoRefresh && (
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    ⏱️ Refresh Interval: {refreshInterval}s
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="300"
                    step="10"
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Ticker Management */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            Tracked Tickers ({selectedTickers.length}/{MAX_TICKERS})
          </label>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {selectedTickers.map(ticker => (
              <span
                key={ticker}
                style={{
                  padding: '0.4rem 0.8rem',
                  background: 'var(--accent)',
                  borderRadius: '6px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  fontSize: '0.9rem'
                }}
              >
                {ticker}
                <button
                  onClick={() => removeTicker(ticker)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'white',
                    cursor: 'pointer',
                    fontSize: '1rem',
                    padding: 0,
                    lineHeight: 1
                  }}
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        </div>
        
        {screenerLoading ? (
          <LoadingOverlay message="Analyzing market sentiment..." />
        ) : screenerData ? (
          <>
            <div style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Showing {filteredAndSortedStocks.length} / {screenerData.results.length} stocks
            </div>
            
            <div style={{ overflowX: 'auto' }}>
              <table style={{ 
                width: '100%', 
                borderCollapse: 'collapse',
                background: 'var(--surface)',
                borderRadius: '8px',
                overflow: 'hidden'
              }}>
                <thead>
                  <tr style={{ background: 'var(--background)' }}>
                    <th style={{ padding: '1rem', textAlign: 'left', borderBottom: '2px solid var(--border)' }}>
                      <button
                        onClick={() => handleSort('ticker')}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: sortField === 'ticker' ? 'var(--accent)' : 'var(--text)',
                          cursor: 'pointer',
                          fontSize: '1rem',
                          fontWeight: 'bold',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.3rem'
                        }}
                      >
                        Ticker
                        {sortField === 'ticker' && (sortDirection === 'asc' ? ' ↑' : ' ↓')}
                      </button>
                    </th>
                    <th style={{ padding: '1rem', textAlign: 'center', borderBottom: '2px solid var(--border)' }}>
                      <button
                        onClick={() => handleSort('composite')}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: sortField === 'composite' ? 'var(--accent)' : 'var(--text)',
                          cursor: 'pointer',
                          fontSize: '1rem',
                          fontWeight: 'bold'
                        }}
                      >
                        Composite
                        {sortField === 'composite' && (sortDirection === 'asc' ? ' ↑' : ' ↓')}
                      </button>
                    </th>
                    <th style={{ padding: '1rem', textAlign: 'center', borderBottom: '2px solid var(--border)' }}>
                      <button
                        onClick={() => handleSort('stocktwits')}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: sortField === 'stocktwits' ? 'var(--accent)' : 'var(--text)',
                          cursor: 'pointer',
                          fontSize: '1rem',
                          fontWeight: 'bold'
                        }}
                      >
                        StockTwits
                        {sortField === 'stocktwits' && (sortDirection === 'asc' ? ' ↑' : ' ↓')}
                      </button>
                    </th>
                    <th style={{ padding: '1rem', textAlign: 'center', borderBottom: '2px solid var(--border)' }}>
                      <button
                        onClick={() => handleSort('x_sentiment')}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: sortField === 'x_sentiment' ? 'var(--accent)' : 'var(--text)',
                          cursor: 'pointer',
                          fontSize: '1rem',
                          fontWeight: 'bold'
                        }}
                      >
                        Twitter
                        {sortField === 'x_sentiment' && (sortDirection === 'asc' ? ' ↑' : ' ↓')}
                      </button>
                    </th>
                    <th style={{ padding: '1rem', textAlign: 'center', borderBottom: '2px solid var(--border)' }}>
                      <button
                        onClick={() => handleSort('news_sentiment')}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: sortField === 'news_sentiment' ? 'var(--accent)' : 'var(--text)',
                          cursor: 'pointer',
                          fontSize: '1rem',
                          fontWeight: 'bold'
                        }}
                      >
                        News
                        {sortField === 'news_sentiment' && (sortDirection === 'asc' ? ' ↑' : ' ↓')}
                      </button>
                    </th>
                    <th style={{ padding: '1rem', textAlign: 'center', borderBottom: '2px solid var(--border)' }}>
                      7-Day Trend
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAndSortedStocks.map((stock, index) => {
                    const history = getTickerHistory(stock.ticker, 7);
                    const isExpanded = expandedRows.has(stock.ticker);
                    
                    return (
                      <React.Fragment key={stock.ticker}>
                        <tr 
                          style={{ 
                            borderBottom: '1px solid var(--border)',
                            cursor: 'pointer',
                            transition: 'background 0.2s',
                            animation: `fadeIn 0.6s ease-out ${index * 0.05}s both`
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'rgba(0, 212, 170, 0.05)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'transparent';
                          }}
                          onClick={() => toggleRow(stock.ticker)}
                        >
                          <td style={{ padding: '1rem', fontWeight: 'bold' }}>
                            {isExpanded ? '▼' : '▶'} {stock.ticker}
                          </td>
                          <td style={{ padding: '1rem', textAlign: 'center' }}>
                            <Badge
                              variant={
                                stock.composite === null ? 'default' :
                                stock.composite >= SENTIMENT_THRESHOLDS.BULLISH ? 'success' :
                                stock.composite <= SENTIMENT_THRESHOLDS.BEARISH ? 'danger' :
                                'default'
                              }
                            >
                              {formatSentimentValue(stock.composite)}
                            </Badge>
                          </td>
                          <td style={{ padding: '1rem', textAlign: 'center', color: getSentimentColor(stock.stocktwits) }}>
                            {formatSentimentValue(stock.stocktwits)}
                          </td>
                          <td style={{ padding: '1rem', textAlign: 'center', color: getSentimentColor(stock.x_sentiment) }}>
                            {formatSentimentValue(stock.x_sentiment)}
                          </td>
                          <td style={{ padding: '1rem', textAlign: 'center', color: getSentimentColor(stock.news_sentiment) }}>
                            {formatSentimentValue(stock.news_sentiment)}
                          </td>
                          <td style={{ padding: '1rem', textAlign: 'center' }}>
                            {history.length > 0 ? (
                              <div style={{ display: 'flex', justifyContent: 'center' }}>
                                <MiniTrendChart history={history} />
                              </div>
                            ) : (
                              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No data</span>
                            )}
                          </td>
                        </tr>
                        
                        {/* Expanded Row Details */}
                        {isExpanded && (
                          <tr style={{ background: 'var(--background)' }}>
                            <td colSpan={6} style={{ padding: '1.5rem' }}>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
                                {/* Composite Gauge */}
                                <div style={{ textAlign: 'center' }}>
                                  <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Composite Score</h4>
                                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                                    <SentimentGauge value={stock.composite} />
                                  </div>
                                </div>
                                
                                {/* Source Breakdown */}
                                <div>
                                  <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Source Breakdown</h4>
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'var(--surface)', borderRadius: '4px' }}>
                                      <span>StockTwits:</span>
                                      <span style={{ color: getSentimentColor(stock.stocktwits), fontWeight: 'bold' }}>
                                        {formatSentimentValue(stock.stocktwits)}
                                      </span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'var(--surface)', borderRadius: '4px' }}>
                                      <span>Twitter:</span>
                                      <span style={{ color: getSentimentColor(stock.x_sentiment), fontWeight: 'bold' }}>
                                        {formatSentimentValue(stock.x_sentiment)}
                                      </span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'var(--surface)', borderRadius: '4px' }}>
                                      <span>News:</span>
                                      <span style={{ color: getSentimentColor(stock.news_sentiment), fontWeight: 'bold' }}>
                                        {formatSentimentValue(stock.news_sentiment)}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                                
                                {/* 7-Day History Chart */}
                                {history.length > 0 && (
                                  <div style={{ gridColumn: 'span 2' }}>
                                    <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>7-Day Sentiment History</h4>
                                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                                      <HistoricalBarChart history={history} />
                                    </div>
                                  </div>
                                )}
                                
                                {/* Market Data (if available) */}
                                {stock.company && (
                                  <div>
                                    <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Company Info</h4>
                                    <p style={{ margin: 0 }}>{stock.company}</p>
                                  </div>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            {/* Source Status Footer */}
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: 'var(--background)', 
              borderRadius: '8px',
              display: 'flex',
              gap: '2rem',
              fontSize: '0.85rem',
              color: 'var(--text-secondary)'
            }}>
              <div>
                <strong>StockTwits:</strong> {screenerData.sources.stocktwits ? '🟢 Active' : '🔴 Unavailable'}
              </div>
              <div>
                <strong>Twitter:</strong> {screenerData.sources.x_sentiment ? '🟢 Active' : '🔴 Unavailable'}
              </div>
              <div>
                <strong>News:</strong> {screenerData.sources.news ? '🟢 Active' : '🔴 Unavailable'}
              </div>
            </div>
          </>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            Click refresh to load screening data
          </div>
        )}
      </section>
      
      {/* ===== DEEP NEWS ANALYSIS SECTION ===== */}
      <section>
        <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>
          📰 Deep News Analysis
        </h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          Analyze sentiment from recent news articles using FinBERT and VADER models
        </p>
        
        <div style={{ 
          display: 'flex', 
          gap: '1rem', 
          marginBottom: '1.5rem',
          flexWrap: 'wrap'
        }}>
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAnalyzeSentiment()}
            placeholder="Enter ticker or topic (e.g., 'gold', 'AAPL')"
            style={{
              flex: 1,
              minWidth: '250px',
              padding: '0.75rem 1rem',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              color: 'var(--text)',
              fontSize: '1rem'
            }}
          />
          
          <select
            value={numArticles}
            onChange={(e) => setNumArticles(Number(e.target.value))}
            style={{
              padding: '0.75rem 1rem',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              color: 'var(--text)',
              fontSize: '1rem',
              cursor: 'pointer'
            }}
          >
            <option value={5}>5 articles</option>
            <option value={10}>10 articles</option>
            <option value={15}>15 articles</option>
            <option value={20}>20 articles</option>
          </select>
          
          <button
            onClick={handleAnalyzeSentiment}
            disabled={newsLoading}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'var(--accent)',
              border: 'none',
              borderRadius: '8px',
              color: 'white',
              fontSize: '1rem',
              cursor: newsLoading ? 'not-allowed' : 'pointer',
              opacity: newsLoading ? 0.6 : 1,
              fontWeight: 'bold'
            }}
          >
            {newsLoading ? '⏳ Analyzing...' : '🔍 Analyze'}
          </button>
        </div>
        
        {error && (
          <div style={{
            padding: '1rem',
            background: 'rgba(255, 107, 107, 0.1)',
            border: '1px solid var(--negative)',
            borderRadius: '8px',
            color: 'var(--negative)',
            marginBottom: '1rem'
          }}>
            ⚠️ {error}
          </div>
        )}
        
        {newsLoading && <LoadingOverlay message="Analyzing news sentiment..." />}
        
        {newsData && (
          <div style={{ 
            background: 'var(--surface)', 
            padding: '1.5rem', 
            borderRadius: '12px',
            animation: 'fadeIn 0.6s ease-out'
          }}>
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
                Overall Sentiment
              </h3>
              
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
                <SentimentGauge value={newsData.ticker ? 50 : 50} size={150} />
              </div>
              
              <BarChart
                labels={['Positive', 'Neutral', 'Negative']}
                data={[
                  newsData.summary.Positive.count,
                  newsData.summary.Neutral.count,
                  newsData.summary.Negative.count,
                ]}
                colors={['var(--positive)', 'var(--neutral)', 'var(--negative)']}
              />
            </div>
            
            <div>
              <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>
                Analyzed Articles ({newsData.total_articles})
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {newsData.articles.map((article, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '1rem',
                      background: 'var(--background)',
                      borderRadius: '8px',
                      border: '1px solid var(--border)'
                    }}
                  >
                    <div style={{ marginBottom: '0.5rem' }}>
                      <Badge variant={sentimentToVariant(article.sentiment)}>
                        {article.sentiment}
                      </Badge>
                      <span style={{ 
                        marginLeft: '0.5rem', 
                        color: article.polarity >= 0 ? 'var(--positive)' : 'var(--negative)',
                        fontWeight: 'bold'
                      }}>
                        {article.polarity.toFixed(3)}
                      </span>
                    </div>
                    
                    <h4 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>
                      {article.title}
                    </h4>
                    
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      fontSize: '0.85rem',
                      color: 'var(--text-secondary)'
                    }}>
                      <span>{article.published}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>
      
      {/* Global Styles for Animations */}
      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
