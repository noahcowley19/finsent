'use client';

import { useState } from 'react';
import { LoadingOverlay, Badge } from '@/components';
import { analyzeInsider } from '@/lib/api';
import type { InsiderResponse, InsiderTransaction, ClusterAlert } from '@/lib/types';

export default function InsiderPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<InsiderResponse | null>(null);
  const [tickerInput, setTickerInput] = useState('AAPL');
  const [periodMonths, setPeriodMonths] = useState(12);
  const [showExplainer, setShowExplainer] = useState(false);

  const handleAnalyze = async () => {
    if (!tickerInput.trim()) return;
    
    setLoading(true);
    setError(null);
    setData(null);
    
    try {
      const result = await analyzeInsider(tickerInput.trim().toUpperCase(), periodMonths);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze insider activity');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '1400px' }}>
      {loading && <LoadingOverlay message="Fetching insider trading data..." />}
      
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '3rem', fontWeight: 700, marginBottom: '12px' }}>
          Insider Trading
        </h1>
        <p className="subtitle" style={{ maxWidth: '700px', margin: '0 auto' }}>
          Track insider transactions and institutional ownership changes. Insider buying often indicates company strength, and clustered insider selling often indicates company weakness. These indications are not definitive.
        </p>
      </header>

      <div className="card" style={{ padding: '40px', marginBottom: '40px' }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label className="input-label">Enter a stock ticker symbol (e.g., AAPL for Apple).</label>
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
              placeholder="Enter ticker symbol"
              className="input-field"
            />
          </div>
          <div style={{ minWidth: '150px' }}>
            <label className="input-label">Time Period</label>
            <select
              value={periodMonths}
              onChange={(e) => setPeriodMonths(parseInt(e.target.value))}
              className="input-field"
              style={{ appearance: 'auto' }}
            >
              <option value={3}>3 Months</option>
              <option value={6}>6 Months</option>
              <option value={12}>12 Months</option>
              <option value={24}>24 Months</option>
            </select>
          </div>
          <button onClick={handleAnalyze} className="btn-primary" disabled={loading}>
            Analyze
          </button>
        </div>

        {error && <div className="error-message" style={{ marginTop: '20px' }}>{error}</div>}
      </div>

      {data && (
        <div id="results">
          {/* Company Header */}
          <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
            <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '16px' }}>{data.company.name}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', marginBottom: '12px' }}>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Ticker: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.ticker}</span></div>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Sector: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.sector}</span></div>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Price: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.price_display}</span></div>
              <div><span style={{ fontSize: '13px', color: 'var(--secondary)' }}>Market Cap: </span><span style={{ fontSize: '14px', fontWeight: 600 }}>{data.company.market_cap_display}</span></div>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>Data as of: {new Date(data.timestamp).toLocaleString()}</div>
          </div>

          {/* Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '32px' }}>
            <div className="card" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--positive)' }} />
              <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '12px' }}>Total Buy Value</div>
              <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'var(--positive)', marginBottom: '8px' }}>{data.summary.buy_value_display}</div>
              <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--secondary)' }}>{data.summary.total_buys} transactions</div>
            </div>
            <div className="card" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--negative)' }} />
              <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '12px' }}>Total Sell Value</div>
              <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'var(--negative)', marginBottom: '8px' }}>{data.summary.sell_value_display}</div>
              <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--secondary)' }}>{data.summary.total_sells} transactions</div>
            </div>
            <div className="card" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: data.summary.net_positive ? 'var(--positive)' : 'var(--negative)' }} />
              <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '12px' }}>Net Value</div>
              <div style={{ fontSize: '2.5rem', fontWeight: 700, color: data.summary.net_positive ? 'var(--positive)' : 'var(--negative)', marginBottom: '8px' }}>{data.summary.net_value_display}</div>
              <div style={{ display: 'inline-flex', alignItems: 'center', padding: '6px 14px', borderRadius: '8px', fontSize: '14px', fontWeight: 600, marginTop: '8px', background: data.sentiment.status === 'positive' ? 'var(--positive-light)' : data.sentiment.status === 'negative' ? 'var(--negative-light)' : 'var(--warning-light)', color: data.sentiment.status === 'positive' ? 'var(--positive-dark)' : data.sentiment.status === 'negative' ? 'var(--negative-dark)' : 'var(--warning-dark)' }}>
                {data.sentiment.sentiment}
              </div>
            </div>
          </div>

          {/* Signals */}
          {data.signals.length > 0 && (
            <div style={{ marginBottom: '32px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
                {data.signals.map((signal, i) => (
                  <div key={i} className="card" style={{ padding: '20px', borderLeft: `4px solid ${signal.status === 'positive' ? 'var(--positive)' : signal.status === 'negative' ? 'var(--negative)' : 'var(--warning)'}` }}>
                    <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>{signal.title}</div>
                    <div style={{ fontSize: '13px', color: 'var(--secondary)', lineHeight: 1.5 }}>{signal.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Cluster Alerts */}
          {data.cluster_alerts.length > 0 && (
            <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--border)' }}>
                <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)' }}>Cluster Trading Activity</div>
                <div style={{ fontSize: '13px', color: 'var(--secondary)' }}>{data.cluster_alerts.length} cluster(s) detected</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
                {data.cluster_alerts.map((alert: ClusterAlert, i) => (
                  <div key={i} style={{ background: 'var(--background)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)', position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: alert.status === 'positive' ? 'var(--positive)' : 'var(--negative)' }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <Badge variant={alert.status}>{alert.type === 'cluster_buy' ? 'Cluster Buy' : 'Cluster Sell'}</Badge>
                      <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>{alert.week_display}</div>
                    </div>
                    <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>{alert.message}</div>
                    <div style={{ fontSize: '13px', color: 'var(--secondary)', marginBottom: '12px' }}>{alert.description}</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: alert.status === 'positive' ? 'var(--positive)' : 'var(--negative)' }}>{alert.total_value_display}</div>
                    {alert.insiders.length > 0 && (
                      <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid var(--border)' }}>
                        <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px' }}>Insiders involved:</div>
                        {alert.insiders.map((insider, j) => (
                          <div key={j} style={{ fontSize: '13px', color: 'var(--secondary)', marginBottom: '4px' }}>
                            {insider.name} ({insider.title}) - {insider.value}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Monthly Activity Chart */}
          {data.monthly_data.length > 0 && (
            <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '24px' }}>Monthly Activity</div>
              <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-around', height: '200px', gap: '8px' }}>
                {data.monthly_data.map((month, i) => {
                  const maxVal = Math.max(...data.monthly_data.map(m => Math.max(m.buys, m.sells))) || 1;
                  return (
                    <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, maxWidth: '60px' }}>
                      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '150px' }}>
                        <div style={{ width: '20px', background: 'var(--positive)', borderRadius: '4px 4px 0 0', height: `${(month.buys / maxVal) * 100}%`, minHeight: month.buys > 0 ? '4px' : '0' }} title={`${month.buys} buys`} />
                        <div style={{ width: '20px', background: 'var(--negative)', borderRadius: '4px 4px 0 0', height: `${(month.sells / maxVal) * 100}%`, minHeight: month.sells > 0 ? '4px' : '0' }} title={`${month.sells} sells`} />
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--secondary)', marginTop: '8px', transform: 'rotate(-45deg)', whiteSpace: 'nowrap' }}>{month.label}</div>
                    </div>
                  );
                })}
              </div>
              <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginTop: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
                  <div style={{ width: '12px', height: '12px', background: 'var(--positive)', borderRadius: '2px' }} />
                  <span>Buys</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
                  <div style={{ width: '12px', height: '12px', background: 'var(--negative)', borderRadius: '2px' }} />
                  <span>Sells</span>
                </div>
              </div>
            </div>
          )}

          {/* Transactions Table */}
          {data.has_transaction_data && data.transactions.length > 0 && (
            <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '24px' }}>
                Recent Transactions ({data.transactions.length})
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Insider</th>
                      <th>Title</th>
                      <th>Type</th>
                      <th style={{ textAlign: 'right' }}>Shares</th>
                      <th style={{ textAlign: 'right' }}>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.transactions.slice(0, 20).map((tx: InsiderTransaction, i) => (
                      <tr key={i}>
                        <td>{tx.date}</td>
                        <td style={{ fontWeight: 600 }}>{tx.insider}</td>
                        <td style={{ color: 'var(--secondary)', fontSize: '13px' }}>{tx.title}</td>
                        <td><Badge variant={tx.type_status}>{tx.type}</Badge></td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>{tx.shares_display}</td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>{tx.value_display}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Institutional Holders */}
          {data.has_institutional_data && data.institutional.holders.length > 0 && (
            <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '24px' }}>
                Top Institutional Holders
              </div>
              {data.institutional.holders.slice(0, 10).map((holder, i) => (
                <div key={i} style={{ padding: '16px 0', borderBottom: i < Math.min(data.institutional.holders.length, 10) - 1 ? '1px solid var(--border)' : 'none', display: 'grid', gridTemplateColumns: '1fr auto auto auto', gap: '24px', alignItems: 'center' }}>
                  <div style={{ fontSize: '14px', fontWeight: 500 }}>{holder.name}</div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontWeight: 600 }}>{holder.shares_display}</div>
                    <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>shares</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontWeight: 600 }}>{holder.value_display}</div>
                    <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>value</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontWeight: 600 }}>{holder.percent_display}</div>
                    <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>of float</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Explainer */}
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <button onClick={() => setShowExplainer(!showExplainer)} style={{ width: '100%', padding: '20px 24px', background: 'transparent', border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', fontSize: '15px', fontWeight: 600, color: 'var(--primary)' }}>
              <span>How to interpret insider trading data?</span>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ width: '20px', height: '20px', transform: showExplainer ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease', color: 'var(--secondary)' }}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
            </button>
            {showExplainer && (
              <div style={{ padding: '0 24px 24px' }}>
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>Cluster Buying</h4>
                  <p style={{ fontSize: '14px', color: 'var(--secondary)', lineHeight: 1.6 }}>When multiple insiders buy shares around the same time, it may indicate confidence in the company&apos;s future. This is often seen as a positive signal.</p>
                </div>
                <div style={{ marginBottom: '20px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '6px' }}>Cluster Selling</h4>
                  <p style={{ fontSize: '14px', color: 'var(--secondary)', lineHeight: 1.6 }}>Coordinated selling by multiple insiders may indicate concern about the company&apos;s prospects. However, insiders sell for many reasons (diversification, taxes, etc.).</p>
                </div>
                <div style={{ padding: '16px', background: 'var(--neutral-light)', borderRadius: '8px', fontSize: '13px', color: 'var(--neutral-dark)', fontStyle: 'italic' }}>
                  Insider trading data should be used alongside other analysis methods. Insiders may have non-investment reasons for their transactions.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
