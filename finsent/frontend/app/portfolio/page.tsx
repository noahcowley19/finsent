'use client';

import { useState, useEffect, useCallback } from 'react';
import { LoadingOverlay, DoughnutChart } from '@/components';
import { analyzePortfolio } from '@/lib/api';
import type { PortfolioResponse, PortfolioPosition, StockAnalysis } from '@/lib/types';

export default function PortfolioPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [lastUpdate, setLastUpdate] = useState<string>('--');
  
  // Input state
  const [tickerInput, setTickerInput] = useState('');
  const [sharesInput, setSharesInput] = useState('');
  const [costBasisInput, setCostBasisInput] = useState('');
  
  // CAPM inputs
  const [riskFreeRate, setRiskFreeRate] = useState(2.0);
  const [marketReturn, setMarketReturn] = useState(10.0);
  
  // Allocation view
  const [allocationType, setAllocationType] = useState<'sector' | 'industry' | 'ticker'>('sector');

  // Load positions from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('portfolio_positions');
    if (saved) {
      const parsed = JSON.parse(saved);
      setPositions(parsed);
    }
  }, []);

  // Analyze portfolio when positions change
  const runAnalysis = useCallback(async (currentPositions: PortfolioPosition[]) => {
    if (currentPositions.length === 0) {
      setData(null);
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const result = await analyzePortfolio(
        currentPositions, 
        riskFreeRate / 100, 
        marketReturn / 100
      );
      setData(result);
      setLastUpdate(new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze portfolio');
    } finally {
      setLoading(false);
    }
  }, [riskFreeRate, marketReturn]);

  // Run analysis when positions change
  useEffect(() => {
    if (positions.length > 0) {
      runAnalysis(positions);
    }
  }, [positions, runAnalysis]);

  const addPosition = () => {
    const ticker = tickerInput.trim().toUpperCase();
    const shares = parseFloat(sharesInput);
    const costBasis = parseFloat(costBasisInput);
    
    if (!ticker || isNaN(shares) || shares <= 0) {
      setError('Please enter a valid ticker and shares amount');
      return;
    }

    const newPosition: PortfolioPosition = {
      ticker,
      shares,
      total_cost_basis: isNaN(costBasis) ? 0 : costBasis,
    };

    const newPositions = [...positions, newPosition];
    setPositions(newPositions);
    localStorage.setItem('portfolio_positions', JSON.stringify(newPositions));
    
    // Clear inputs
    setTickerInput('');
    setSharesInput('');
    setCostBasisInput('');
    setError(null);
  };

  const removeShares = (index: number, sharesToRemove: number) => {
    const position = positions[index];
    if (!position) return;

    let newPositions: PortfolioPosition[];
    
    if (sharesToRemove >= position.shares) {
      // Remove entire position
      newPositions = positions.filter((_, i) => i !== index);
    } else {
      // Reduce shares
      const ratio = (position.shares - sharesToRemove) / position.shares;
      newPositions = positions.map((p, i) => {
        if (i === index) {
          return {
            ...p,
            shares: p.shares - sharesToRemove,
            total_cost_basis: p.total_cost_basis * ratio,
          };
        }
        return p;
      });
    }

    setPositions(newPositions);
    localStorage.setItem('portfolio_positions', JSON.stringify(newPositions));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      addPosition();
    }
  };

  const getAllocationData = () => {
    if (!data) return [];
    switch (allocationType) {
      case 'sector': return data.allocation.sector;
      case 'industry': return data.allocation.industry;
      case 'ticker': return data.allocation.ticker;
      default: return [];
    }
  };

  const formatCurrency = (value: number) => {
    return '$' + value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return sign + value.toFixed(2) + '%';
  };

  return (
    <div className="container" style={{ maxWidth: '1600px' }}>
      {loading && <LoadingOverlay message="Loading portfolio data..." />}
      
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 700, textAlign: 'center', marginBottom: '8px' }}>
          Portfolio Dashboard
        </h1>
        <p className="subtitle" style={{ textAlign: 'center' }}>
          Real-time portfolio tracking with comprehensive analytics
        </p>
      </header>

      {error && (
        <div className="error-message" style={{ marginBottom: '24px' }}>
          {error}
        </div>
      )}

      {/* Add Position Card */}
      <div className="card" style={{ marginBottom: '32px', padding: '24px' }}>
        <div style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '20px' }}>
          Add Stock Position
        </div>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '2fr 1.5fr 1.5fr auto', 
          gap: '12px', 
          alignItems: 'end' 
        }}>
          <div>
            <label className="input-label">Ticker Symbol</label>
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
              onKeyPress={handleKeyPress}
              placeholder="AAPL"
              className="input-field"
              style={{ textTransform: 'uppercase' }}
            />
          </div>
          <div>
            <label className="input-label">Shares</label>
            <input
              type="number"
              value={sharesInput}
              onChange={(e) => setSharesInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="10"
              step="0.01"
              min="0"
              className="input-field"
            />
          </div>
          <div>
            <label className="input-label">Total Cost Basis</label>
            <input
              type="number"
              value={costBasisInput}
              onChange={(e) => setCostBasisInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="1500.00"
              step="0.01"
              min="0"
              className="input-field"
            />
          </div>
          <button onClick={addPosition} className="btn-primary" disabled={loading}>
            Add
          </button>
        </div>
      </div>

      {/* Dashboard Content */}
      {data && (
        <>
          {/* Overview Stats */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
            gap: '24px', 
            marginBottom: '32px' 
          }}>
            <div className={`card ${data.portfolio_metrics.total_gain_loss >= 0 ? 'positive' : 'negative'}`} style={{ 
              padding: '24px',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--primary)' }} />
              <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                Total Value
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '4px' }}>
                {formatCurrency(data.portfolio_metrics.total_value)}
              </div>
            </div>
            
            <div className="card" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: data.portfolio_metrics.total_gain_loss >= 0 ? 'var(--positive)' : 'var(--negative)' }} />
              <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                Total Gain/Loss
              </div>
              <div style={{ 
                fontSize: '2rem', 
                fontWeight: 700, 
                marginBottom: '4px',
                color: data.portfolio_metrics.total_gain_loss >= 0 ? 'var(--positive)' : 'var(--negative)'
              }}>
                {formatCurrency(data.portfolio_metrics.total_gain_loss)}
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: data.portfolio_metrics.total_gain_loss_percent >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                {formatPercent(data.portfolio_metrics.total_gain_loss_percent)}
              </div>
            </div>

            <div className="card" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--primary)' }} />
              <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                Total Cost
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '4px' }}>
                {formatCurrency(data.portfolio_metrics.total_cost)}
              </div>
            </div>

            <div className="card" style={{ padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', background: 'var(--primary)' }} />
              <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--secondary)', marginBottom: '8px' }}>
                Positions
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '4px' }}>
                {data.portfolio_metrics.positions_count}
              </div>
            </div>
          </div>

          {/* Holdings Table */}
          <div className="card" style={{ marginBottom: '32px', padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>Portfolio Holdings</div>
              <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>
                Last updated: {lastUpdate}
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th style={{ textAlign: 'right' }}>Shares</th>
                    <th style={{ textAlign: 'right' }}>Cost Basis</th>
                    <th style={{ textAlign: 'right' }}>Current Price</th>
                    <th style={{ textAlign: 'right' }}>Current Value</th>
                    <th style={{ textAlign: 'right' }}>Gain/Loss $</th>
                    <th style={{ textAlign: 'right' }}>Gain/Loss %</th>
                    <th style={{ textAlign: 'right' }}>Beta</th>
                    <th style={{ textAlign: 'right' }}>CAPM Return</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {data.stock_analyses.map((stock: StockAnalysis, index: number) => {
                    const gainLoss = stock.gain_loss ?? 0;
                    const gainLossPercent = stock.gain_loss_percent ?? 0;
                    return (
                      <tr key={stock.ticker}>
                        <td style={{ fontWeight: 700, fontSize: '15px' }}>{stock.ticker}</td>
                        <td style={{ color: 'var(--secondary)', fontSize: '13px' }}>{stock.name}</td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>{stock.shares?.toLocaleString()}</td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>
                          {stock.cost_basis ? formatCurrency(stock.cost_basis) : 'N/A'}
                        </td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>{formatCurrency(stock.current_price)}</td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>
                          {stock.current_value ? formatCurrency(stock.current_value) : 'N/A'}
                        </td>
                        <td style={{ 
                          textAlign: 'right', 
                          fontWeight: 600,
                          color: gainLoss >= 0 ? 'var(--positive)' : 'var(--negative)'
                        }}>
                          {gainLoss >= 0 ? '+' : ''}{formatCurrency(gainLoss)}
                        </td>
                        <td style={{ 
                          textAlign: 'right', 
                          fontWeight: 600,
                          color: gainLossPercent >= 0 ? 'var(--positive)' : 'var(--negative)'
                        }}>
                          {formatPercent(gainLossPercent)}
                        </td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>{stock.beta?.toFixed(2) ?? 'N/A'}</td>
                        <td style={{ textAlign: 'right', fontWeight: 600 }}>
                          {stock.capm ? `${(stock.capm.expected_return * 100).toFixed(2)}%` : 'N/A'}
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <input
                              type="number"
                              id={`removeShares_${index}`}
                              placeholder="Shares"
                              step="0.01"
                              min="0"
                              max={stock.shares}
                              style={{ 
                                width: '80px', 
                                padding: '6px 8px', 
                                fontSize: '13px', 
                                border: '1.5px solid var(--border)', 
                                borderRadius: '6px' 
                              }}
                            />
                            <button 
                              onClick={() => {
                                const input = document.getElementById(`removeShares_${index}`) as HTMLInputElement;
                                const sharesToRemove = parseFloat(input?.value || '0');
                                if (sharesToRemove > 0) {
                                  removeShares(index, sharesToRemove);
                                  input.value = '';
                                }
                              }}
                              className="btn-danger"
                              style={{ padding: '6px 12px', fontSize: '12px' }}
                            >
                              Remove
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* CAPM Section */}
          <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '20px' }}>
              Portfolio CAPM Analysis
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <div>
                <label className="input-label">Risk-Free Rate (%)</label>
                <input
                  type="number"
                  value={riskFreeRate}
                  onChange={(e) => setRiskFreeRate(parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0"
                  max="10"
                  className="input-field"
                />
              </div>
              <div>
                <label className="input-label">Market Return (%)</label>
                <input
                  type="number"
                  value={marketReturn}
                  onChange={(e) => setMarketReturn(parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0"
                  max="20"
                  className="input-field"
                />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '20px' }}>
              <div style={{ background: 'var(--background)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '12px', color: 'var(--secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
                  Expected Return
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700 }}>
                  {(data.portfolio_capm.expected_return * 100).toFixed(2)}%
                </div>
              </div>
              <div style={{ background: 'var(--background)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '12px', color: 'var(--secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
                  Portfolio Beta
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700 }}>
                  {data.portfolio_capm.beta.toFixed(2)}
                </div>
              </div>
              <div style={{ background: 'var(--background)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '12px', color: 'var(--secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
                  Risk Premium
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700 }}>
                  {(data.portfolio_capm.risk_premium * 100).toFixed(2)}%
                </div>
              </div>
              <div style={{ background: 'var(--background)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '12px', color: 'var(--secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
                  Risk-Free Rate
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700 }}>
                  {(data.portfolio_capm.risk_free_rate * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </div>

          {/* Allocation Section - FIXED */}
          <div className="card" style={{ padding: '32px', marginBottom: '32px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>Portfolio Allocation</div>
              <div style={{ display: 'flex', gap: '8px' }}>
                {(['sector', 'industry', 'ticker'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setAllocationType(type)}
                    className={allocationType === type ? 'btn-primary' : 'btn-secondary'}
                    style={{ padding: '8px 16px', fontSize: '13px', fontWeight: 600 }}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ height: '350px' }}>
              <DoughnutChart
                data={getAllocationData().map(item => ({
                  label:
                   (item as { name?: string; ticker?: string }).name ||
                   (item as { ticker: string }).ticker ||
                   'Unknown',
                  value: item.percentage,
                }))}
              />
            </div>
          </div>
        </>
      )}

      {/* Empty State */}
      {!data && !loading && positions.length === 0 && (
        <div style={{ textAlign: 'center', padding: '48px 24px', color: 'var(--secondary)' }}>
          <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.3 }}>📊</div>
          <div style={{ fontSize: '1.125rem', fontWeight: 500, marginBottom: '8px' }}>No positions yet</div>
          <div style={{ fontSize: '14px' }}>Add your first stock position above to start tracking your portfolio</div>
        </div>
      )}
    </div>
  );
}
