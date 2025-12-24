'use client';

import { useState } from 'react';
import { LoadingOverlay } from '@/components';
import { analyzeFinancials } from '@/lib/api';
import type { FinancialsResponse, FinancialMetric } from '@/lib/types';

export default function FinancialsPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FinancialsResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('MSFT');
  const [showExplainer, setShowExplainer] = useState(false);

  const handleAnalyze = async () => {
    if (!tickerInput.trim()) return;
    
    setLoading(true);
    setError(null);
    setData(null);
    
    try {
      const result = await analyzeFinancials(tickerInput.trim().toUpperCase());
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze financials');
    } finally {
      setLoading(false);
    }
  };

  const renderMetricItem = (metric: FinancialMetric, index: number) => (
    <div 
      key={index}
      style={{
        background: 'var(--background)',
        borderRadius: '12px',
        padding: '20px',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '3px',
        background: metric.status === 'positive' ? 'var(--positive)' : 
                    metric.status === 'negative' ? 'var(--negative)' : 'var(--neutral)'
      }} />
      <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
        {metric.name}
      </div>
      <div style={{ 
        fontSize: '1.5rem', 
        fontWeight: 700,
        color: metric.status === 'positive' ? 'var(--positive)' : 
               metric.status === 'negative' ? 'var(--negative)' : 'var(--primary)'
      }}>
        {metric.display}
      </div>
    </div>
  );

  return (
    <div className="container" style={{ maxWidth: '1280px' }}>
      {loading && <LoadingOverlay message="Fetching financial data..." />}
      
      <header style={{ textAlign: 'center', marginBottom: '60px' }}>
        <h1 style={{ fontSize: '3rem', fontWeight: 700, marginBottom: '12px' }}>
          Financial Analyzer
        </h1>
        <p className="subtitle" style={{ maxWidth: '700px', margin: '0 auto' }}>
          Real-time company financial metrics and calculations of academic scoring models. Scores are not definitive; metrics function differently from industry to industry.
        </p>
      </header>

      <div className="card" style={{ padding: '40px', marginBottom: '40px' }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <label className="input-label">
              Enter a stock ticker symbol (e.g, MSFT for Microsoft).
            </label>
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
              placeholder="Enter ticker symbol"
              className="input-field"
            />
          </div>
          <button onClick={handleAnalyze} className="btn-primary" disabled={loading}>
            Analyze
          </button>
        </div>

        {error && <div className="error-message" style={{ marginTop: '20px' }}>{error}</div>}
      </div>

      {data && (
        <div id="results">
          <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
            <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '16px' }}>{data.company.name}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', marginBottom: '12px' }}>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Ticker: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.ticker}</span></div>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Sector: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.sector}</span></div>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Price: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.price ? `${data.company.currency}${data.company.price.toFixed(2)}` : 'N/A'}</span></div>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Market Cap: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.market_cap_display}</span></div>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>Data as of: {new Date(data.timestamp).toLocaleString()}</div>
          </div>

          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '16px' }}>Financial Health Scores</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
              {[
                { name: 'Piotroski F-Score', data: data.scores.piotroski },
                { name: 'Altman Z-Score', data: data.scores.altman },
                { name: 'Beneish M-Score', data: data.scores.beneish }
              ].map((score, i) => (
                <div key={i} className="card" style={{ padding: '24px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '12px' }}>{score.name}</div>
                  <div style={{ fontSize: '2.5rem', fontWeight: 700, marginBottom: '8px', color: score.data.status === 'positive' ? 'var(--positive)' : score.data.status === 'negative' ? 'var(--negative)' : 'var(--warning)' }}>{score.data.display}</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: score.data.status === 'positive' ? 'var(--positive-dark)' : score.data.status === 'negative' ? 'var(--negative-dark)' : 'var(--warning-dark)' }}>{score.data.interpretation}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card" style={{ padding: 0, marginBottom: '32px', overflow: 'hidden' }}>
            <button onClick={() => setShowExplainer(!showExplainer)} style={{ width: '100%', padding: '20px 24px', background: 'transparent', border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', fontSize: '15px', fontWeight: 600, color: 'var(--primary)' }}>
              <span>What do these scores mean?</span>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ width: '20px', height: '20px', transform: showExplainer ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease', color: 'var(--secondary)' }}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
            </button>
            {showExplainer && (
              <div style={{ padding: '0 24px 24px' }}>
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>Piotroski F-Score (0-9)</h4>
                  <p style={{ fontSize: '14px', color: 'var(--secondary)', lineHeight: 1.6 }}>Measures financial strength based on 9 criteria including profitability, leverage, and operating efficiency. Scores 7-9 indicate strong fundamentals, 4-6 average, and 0-3 weak.</p>
                </div>
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>Altman Z-Score</h4>
                  <p style={{ fontSize: '14px', color: 'var(--secondary)', lineHeight: 1.6 }}>Predicts bankruptcy probability. Scores above 2.99 indicate safety, 1.81-2.99 is the grey zone, and below 1.81 signals distress.</p>
                </div>
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>Beneish M-Score</h4>
                  <p style={{ fontSize: '14px', color: 'var(--secondary)', lineHeight: 1.6 }}>Detects earnings manipulation. Scores below -2.22 suggest unlikely manipulation, while scores above -1.78 indicate potential manipulation.</p>
                </div>
                <div style={{ padding: '16px', background: 'var(--neutral-light)', borderRadius: '8px', fontSize: '13px', color: 'var(--neutral-dark)', fontStyle: 'italic' }}>
                  These academic models should be used as one of many tools in financial analysis, not as standalone indicators. Always consider industry context and other factors.
                </div>
              </div>
            )}
          </div>

          <div style={{ marginBottom: '32px' }}>
            <div className="card" style={{ padding: '32px', marginBottom: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '20px' }}>Valuation Metrics</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                {data.metrics.valuation.map(renderMetricItem)}
              </div>
            </div>
            <div className="card" style={{ padding: '32px', marginBottom: '24px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '20px' }}>Profitability Metrics</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                {data.metrics.profitability.map(renderMetricItem)}
              </div>
            </div>
            <div className="card" style={{ padding: '32px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '20px' }}>Leverage Metrics</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                {data.metrics.leverage.map(renderMetricItem)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
