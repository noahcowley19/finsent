'use client';

import { useState, useEffect } from 'react';
import { LoadingOverlay, Badge, BarChart } from '@/components';
import { analyzeSentiment, getSocialScreening } from '@/lib/api';
import type { SentimentAnalysisResponse, ScreeningResponse, SentimentArticle, ScreeningStock } from '@/lib/types';

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'];

export default function SentimentPage() {
  // Screener state
  const [screenerLoading, setScreenerLoading] = useState(true);
  const [screenerData, setScreenerData] = useState<ScreeningResponse | null>(null);
  const [selectedTickers, setSelectedTickers] = useState<string[]>(DEFAULT_TICKERS);
  
  // News analyzer state
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsData, setNewsData] = useState<SentimentAnalysisResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('gold');
  const [numArticles, setNumArticles] = useState(5);
  const [error, setError] = useState<string | null>(null);
  
  // Explainer toggles
  const [showScreenerExplainer, setShowScreenerExplainer] = useState(false);

  // Map sentiment to badge variant
  const sentimentToVariant = (sentiment: string): 'success' | 'danger' | 'default' => {
    const lower = sentiment.toLowerCase();
    if (lower === 'positive') return 'success';
    if (lower === 'negative') return 'danger';
    return 'default';
  };

  // Load social screener on mount
  useEffect(() => {
    loadScreeningData();
  }, []);

  const loadScreeningData = async () => {
    setScreenerLoading(true);
    try {
      const data = await getSocialScreening(selectedTickers);
      setScreenerData(data);
    } catch (err) {
      console.error('Failed to load screening data:', err);
    } finally {
      setScreenerLoading(false);
    }
  };

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

  const removeTicker = (ticker: string) => {
    setSelectedTickers(selectedTickers.filter(t => t !== ticker));
  };

  const getSentimentColor = (value: number | null): string => {
    if (value === null) return 'var(--neutral)';
    if (value >= 55) return 'var(--positive)';
    if (value <= 45) return 'var(--negative)';
    return 'var(--neutral)';
  };

  const formatSentimentValue = (value: number | null): string => {
    if (value === null) return 'N/A';
    return value.toFixed(1);
  };

  return (
    <div className="container" style={{ maxWidth: '1400px' }}>
      {newsLoading && <LoadingOverlay message="Analyzing market sentiment..." />}
      
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '3rem', fontWeight: 700, marginBottom: '12px' }}>
          Sentiment Analyzer
        </h1>
        <p className="subtitle" style={{ maxWidth: '800px', margin: '0 auto' }}>
          Tracks real-time sentiment from X, StockTwits (an investing social platform), and financial news from yfinance in a watchlist format. The deep-dive option allows for stock-specific, credible news sentiment data only via bidirectional encoder representations from transformers.
        </p>
      </header>

      {/* Social Screener Section */}
      <div className="card" style={{ padding: '32px', marginBottom: '40px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '4px' }}>
              Social Screener
            </div>
            <div style={{ fontSize: '14px', color: 'var(--secondary)' }}>
              Real-time sentiment from multiple sources
            </div>
          </div>
          <button 
            onClick={loadScreeningData} 
            className="btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ width: '16px', height: '16px' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>

        {/* Ticker chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '20px' }}>
          {selectedTickers.map(ticker => (
            <div 
              key={ticker}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                background: 'var(--neutral-light)',
                borderRadius: '20px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              {ticker}
              <button 
                onClick={() => removeTicker(ticker)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--secondary)',
                  fontSize: '14px',
                  padding: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        {/* Screener table */}
        {screenerLoading ? (
          <div style={{ textAlign: 'center', padding: '48px' }}>
            <div className="spinner" style={{ margin: '0 auto 16px' }} />
            <p className="loading-text">Loading sentiment data...</p>
          </div>
        ) : screenerData ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Company</th>
                  <th style={{ textAlign: 'right' }}>Price</th>
                  <th style={{ textAlign: 'right' }}>Change</th>
                  <th style={{ textAlign: 'right' }}>Volume</th>
                  <th style={{ textAlign: 'center' }} title="StockTwits user sentiment (0-100)">StockTwits</th>
                  <th style={{ textAlign: 'center' }} title="X/Twitter social sentiment (0-100)">X Sent.</th>
                  <th style={{ textAlign: 'center' }} title="News sentiment (0-100)">News</th>
                  <th style={{ textAlign: 'center' }} title="Weighted composite score">Composite</th>
                </tr>
              </thead>
              <tbody>
                {screenerData.results.map((stock: ScreeningStock) => (
                  <tr key={stock.ticker}>
                    <td style={{ fontWeight: 700 }}>{stock.ticker}</td>
                    <td style={{ color: 'var(--secondary)' }}>{stock.company}</td>
                    <td style={{ textAlign: 'right', fontWeight: 600 }}>{stock.price_display}</td>
                    <td style={{ 
                      textAlign: 'right', 
                      fontWeight: 600,
                      color: (stock.pct_change ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)'
                    }}>
                      {stock.change_display}
                    </td>
                    <td style={{ textAlign: 'right' }}>{stock.volume_display}</td>
                    <td style={{ textAlign: 'center', fontWeight: 600, color: getSentimentColor(stock.stocktwits) }}>
                      {formatSentimentValue(stock.stocktwits)}
                    </td>
                    <td style={{ textAlign: 'center', fontWeight: 600, color: getSentimentColor(stock.x_sentiment) }}>
                      {formatSentimentValue(stock.x_sentiment)}
                    </td>
                    <td style={{ textAlign: 'center', fontWeight: 600, color: getSentimentColor(stock.news_sentiment) }}>
                      {formatSentimentValue(stock.news_sentiment)}
                    </td>
                    <td style={{ textAlign: 'center', fontWeight: 700, color: getSentimentColor(stock.composite) }}>
                      {formatSentimentValue(stock.composite)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {/* Source status */}
        {screenerData && (
          <div style={{ display: 'flex', gap: '16px', marginTop: '16px', fontSize: '13px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: screenerData.sources.market_data ? 'var(--positive)' : 'var(--negative)' }} />
              <span>Market Data</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: screenerData.sources.stocktwits ? 'var(--positive)' : 'var(--negative)' }} />
              <span>StockTwits</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: screenerData.sources.x_sentiment ? 'var(--positive)' : 'var(--negative)' }} />
              <span>X Sentiment</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: screenerData.sources.news ? 'var(--positive)' : 'var(--negative)' }} />
              <span>News</span>
            </div>
          </div>
        )}
      </div>

      {/* Explainer Section */}
      <div className="card" style={{ padding: 0, marginBottom: '32px', overflow: 'hidden' }}>
        <button 
          onClick={() => setShowScreenerExplainer(!showScreenerExplainer)}
          style={{
            width: '100%',
            padding: '20px 24px',
            background: 'transparent',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            fontSize: '15px',
            fontWeight: 600,
            color: 'var(--primary)',
          }}
        >
          <span>What do these sentiment scores mean?</span>
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor" 
            strokeWidth={2}
            style={{ 
              width: '20px', 
              height: '20px', 
              transform: showScreenerExplainer ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s ease',
              color: 'var(--secondary)'
            }}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showScreenerExplainer && (
          <div style={{ padding: '0 24px 24px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px' }}>
              <div style={{ padding: '16px', background: 'var(--neutral-light)', borderRadius: '10px' }}>
                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>StockTwits Sentiment (0-100)</h4>
                <p style={{ fontSize: '13px', color: 'var(--secondary)', lineHeight: 1.6 }}>
                  Based on user-labeled bullish/bearish posts from StockTwits. Higher scores indicate more bullish sentiment from retail traders.
                </p>
              </div>
              <div style={{ padding: '16px', background: 'var(--neutral-light)', borderRadius: '10px' }}>
                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>X/Twitter Sentiment (0-100)</h4>
                <p style={{ fontSize: '13px', color: 'var(--secondary)', lineHeight: 1.6 }}>
                  Derived from cashtag ($TICKER) discussions and social media coverage. Analyzes recent posts mentioning the stock.
                </p>
              </div>
              <div style={{ padding: '16px', background: 'var(--neutral-light)', borderRadius: '10px' }}>
                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>News Sentiment (0-100)</h4>
                <p style={{ fontSize: '13px', color: 'var(--secondary)', lineHeight: 1.6 }}>
                  Analyzed from recent Google News articles using natural language processing. Scores above 60 indicate positive coverage, below 40 indicates negative.
                </p>
              </div>
              <div style={{ padding: '16px', background: 'var(--neutral-light)', borderRadius: '10px' }}>
                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>Composite Score</h4>
                <p style={{ fontSize: '13px', color: 'var(--secondary)', lineHeight: 1.6 }}>
                  Weighted average: StockTwits (1.5x weight), X Sentiment (1.0x), News (1.0x). Used to rank overall market sentiment.
                </p>
              </div>
            </div>
            <div style={{ marginTop: '20px', padding: '16px', background: 'var(--neutral-light)', borderRadius: '8px', fontSize: '13px', color: 'var(--neutral-dark)' }}>
              Data is cached for 5 minutes to ensure responsive performance. Sentiment scores should supplement, not replace, fundamental analysis.
            </div>
          </div>
        )}
      </div>

      {/* Section Divider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', margin: '48px 0 32px' }}>
        <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
        <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Deep News Analysis
        </span>
        <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
      </div>

      {/* News Sentiment Analyzer Section */}
      <div className="card" style={{ padding: '40px', marginBottom: '40px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>
          News Sentiment Analyzer
        </h2>
        <p style={{ fontSize: '14px', color: 'var(--secondary)', marginBottom: '24px' }}>
          Deep analysis of news articles using bidirectional encoder representations from transformers to interpret articles scraped from Google News.
        </p>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '12px', marginBottom: '24px' }}>
          <div>
            <label className="input-label" style={{ minHeight: '30px' }}>
              Enter a company, cryptocurrency, or commodity, and a query amount less than 10. One query translates to ~5 articles.
            </label>
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
              placeholder="Enter ticker or asset name"
              className="input-field"
            />
          </div>
          <div>
            <label className="input-label" style={{ minHeight: '30px' }}>&nbsp;</label>
            <input
              type="number"
              value={numArticles}
              onChange={(e) => setNumArticles(parseInt(e.target.value) || 5)}
              min={1}
              max={20}
              placeholder="Count"
              className="input-field"
              style={{ width: '120px' }}
            />
          </div>
          <div style={{ alignSelf: 'end' }}>
            <button onClick={handleAnalyzeSentiment} className="btn-primary">
              Analyze
            </button>
          </div>
        </div>

        {error && (
          <div className="error-message" style={{ marginTop: '20px' }}>
            {error}
          </div>
        )}
      </div>

      {/* News Results */}
      {newsData && (
        <div id="results">
          {/* Chart and summary */}
          <div style={{ display: 'flex', gap: '40px', marginBottom: '40px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <div style={{ width: '280px', height: '280px' }}>
              <BarChart
                labels={['Positive', 'Negative', 'Neutral']}
                data={[
                  newsData.summary.Positive.count,
                  newsData.summary.Negative.count,
                  newsData.summary.Neutral.count
                ]}
                colors={['#00e5a0', '#ff6b6b', '#64748b']}
              />
            </div>
            <div style={{ display: 'flex', gap: '20px', flex: 1, flexWrap: 'wrap' }}>
              <div className="card" style={{ 
                flex: '1 1 150px',
                padding: '24px',
                textAlign: 'center',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--positive)' }} />
                <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                  Positive
                </div>
                <div style={{ fontSize: '3rem', fontWeight: 700, color: 'var(--positive)', marginBottom: '4px' }}>
                  {newsData.summary.Positive.count}
                </div>
                <div style={{ fontSize: '14px', color: 'var(--positive)' }}>
                  {newsData.summary.Positive.percentage.toFixed(1)}%
                </div>
              </div>
              <div className="card" style={{ 
                flex: '1 1 150px',
                padding: '24px',
                textAlign: 'center',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--negative)' }} />
                <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                  Negative
                </div>
                <div style={{ fontSize: '3rem', fontWeight: 700, color: 'var(--negative)', marginBottom: '4px' }}>
                  {newsData.summary.Negative.count}
                </div>
                <div style={{ fontSize: '14px', color: 'var(--negative)' }}>
                  {newsData.summary.Negative.percentage.toFixed(1)}%
                </div>
              </div>
              <div className="card" style={{ 
                flex: '1 1 150px',
                padding: '24px',
                textAlign: 'center',
                position: 'relative',
                overflow: 'hidden'
              }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--neutral)' }} />
                <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                  Neutral
                </div>
                <div style={{ fontSize: '3rem', fontWeight: 700, color: 'var(--neutral)', marginBottom: '4px' }}>
                  {newsData.summary.Neutral.count}
                </div>
                <div style={{ fontSize: '14px', color: 'var(--neutral)' }}>
                  {newsData.summary.Neutral.percentage.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Articles */}
          <div className="card" style={{ padding: '40px' }}>
            <div style={{ marginBottom: '32px', paddingBottom: '24px', borderBottom: '1px solid var(--border)' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                Analyzed Articles ({newsData.total_articles})
              </h2>
              <p style={{ fontSize: '13px', color: 'var(--secondary)', marginTop: '4px' }}>
                Model: {newsData.model}
              </p>
            </div>
            
            {newsData.articles.map((article: SentimentArticle, index: number) => (
              <div 
                key={index} 
                style={{ 
                  padding: '24px 0', 
                  borderBottom: index < newsData.articles.length - 1 ? '1px solid var(--border)' : 'none'
                }}
              >
                <div style={{ marginBottom: '12px' }}>
                  <a 
                    href={article.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{
                      fontSize: '1.0625rem',
                      fontWeight: 600,
                      color: 'var(--primary)',
                      textDecoration: 'none',
                      lineHeight: 1.5,
                    }}
                  >
                    {article.title}
                  </a>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                  <Badge variant={sentimentToVariant(article.sentiment)}>
                    {article.sentiment}
                  </Badge>
                  <span style={{ fontSize: '14px', color: 'var(--secondary)', fontWeight: 500 }}>
                    Polarity: {article.polarity.toFixed(3)}
                  </span>
                  <span style={{ fontSize: '14px', color: 'var(--secondary)' }}>
                    {article.published}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
