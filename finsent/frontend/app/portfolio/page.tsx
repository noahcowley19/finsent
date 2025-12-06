"use client";

import React, { useCallback, useEffect, useMemo, useState } from 'react';

// NOTE: This file includes both a React page/component (default export)
// and a small frontend-friendly "analyzePortfolio" implementation below.
// Use the analyzePortfolio() implementation as a reference or drop-in replacement
// for your backend function. The fetchPriceForTickers() function is a stub —
// replace it with your own real data-fetching logic (I left clear TODOs).

/**
 * Types
 */
function analyzePortfolio(currentPositions, riskFreeRate, marketReturn);
      setData(result);
      setLastUpdate(new Date().toLocaleTimeString('en-US'));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze portfolio');
    } finally {
      setLoading(false);
    }
  }, [riskFreeRate, marketReturn]);

  useEffect(() => {
    if (positions.length > 0) runAnalysis(positions);
  }, [positions, runAnalysis]);

  const addPosition = () => {
    const ticker = tickerInput.trim().toUpperCase();
    const shares = parseFloat(sharesInput);
    const cost = parseFloat(costBasisInput);

    if (!ticker || isNaN(shares) || shares <= 0) {
      setError('Please enter a valid ticker and shares');
      return;
    }

    const newPos: PortfolioPosition = {
      ticker,
      shares,
      total_cost_basis: isNaN(cost) ? 0 : cost,
    };

    setPositions((p) => [...p, newPos]);

    setTickerInput('');
    setSharesInput('');
    setCostBasisInput('');
    setError(null);
  };

  const removeShares = (index: number, sharesToRemove: number) => {
    const position = positions[index];
    if (!position) return;

    if (sharesToRemove >= position.shares) {
      setPositions((p) => p.filter((_, i) => i !== index));
    } else {
      setPositions((p) => p.map((pos, i) => {
        if (i !== index) return pos;
        const remaining = pos.shares - sharesToRemove;
        const ratio = remaining / pos.shares;
        return {
          ...pos,
          shares: remaining,
          total_cost_basis: pos.total_cost_basis * ratio,
        };
      }));
    }
  };

  const allocationData = useMemo(() => {
    if (!data) return [] as AllocationItem[];
    switch (allocationType) {
      case 'sector': return data.allocation.sector;
      case 'industry': return data.allocation.industry;
      default: return data.allocation.ticker;
    }
  }, [allocationType, data]);

  return (
    <div className="max-w-[1400px] mx-auto p-6">
      {/* Header */}
      <header className="mb-6 text-center">
        <h1 className="text-3xl font-bold">Portfolio Dashboard</h1>
        <p className="text-sm text-gray-500">Real-time portfolio tracking with CAPM analytics</p>
      </header>

      {/* Top row - Add position + CAPM inputs */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Add Position Card */}
        <div className="bg-white shadow rounded p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Add Position</h2>
            <div className="text-xs text-gray-400">Positions: {positions.length}</div>
          </div>

          <div className="space-y-3">
            <label className="block text-xs font-medium text-gray-600">Ticker</label>
            <input
              className="w-full input-field p-2 border rounded"
              placeholder="AAPL"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
            />

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-medium text-gray-600">Shares</label>
                <input
                  className="w-full input-field p-2 border rounded"
                  value={sharesInput}
                  onChange={(e) => setSharesInput(e.target.value)}
                  inputMode="decimal"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600">Total Cost Basis ($)</label>
                <input
                  className="w-full input-field p-2 border rounded"
                  value={costBasisInput}
                  onChange={(e) => setCostBasisInput(e.target.value)}
                  inputMode="decimal"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <button
                className="bg-blue-600 text-white px-4 py-2 rounded shadow"
                onClick={addPosition}
              >
                Add
              </button>
              <button
                className="bg-gray-100 px-3 py-2 rounded"
                onClick={() => { setTickerInput(''); setSharesInput(''); setCostBasisInput(''); setError(null); }}
              >
                Clear
              </button>
            </div>

            {error && <div className="text-red-600 text-sm">{error}</div>}
          </div>
        </div>

        {/* CAPM Inputs Card */}
        <div className="bg-white shadow rounded p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold">CAPM Inputs</h2>
            <div className="text-xs text-gray-400">Risk-free & market figures</div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div>
              <label className="block text-xs font-medium text-gray-600">Risk-free rate (%)</label>
              <div className="flex items-center gap-2">
                <input
                  className="w-full p-2 border rounded"
                  type="number"
                  step="0.01"
                  inputMode="decimal"
                  value={(riskFreeRate * 100).toString()}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setRiskFreeRate(isNaN(v) ? 0 : v / 100);
                  }}
                />
                <span className="text-sm text-gray-500">%</span>
              </div>
              <div className="text-xs text-gray-400 mt-1">Enter 2 for 2% — decimals allowed (e.g. 2.1)</div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-600">Market return (%)</label>
              <div className="flex items-center gap-2">
                <input
                  className="w-full p-2 border rounded"
                  type="number"
                  step="0.01"
                  inputMode="decimal"
                  value={(marketReturn * 100).toString()}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setMarketReturn(isNaN(v) ? 0 : v / 100);
                  }}
                />
                <span className="text-sm text-gray-500">%</span>
              </div>
              <div className="text-xs text-gray-400 mt-1">Enter 10 for 10% — decimals allowed (e.g. 10.5)</div>
            </div>

            <div className="space-y-2">
              <div className="bg-gray-50 border rounded p-3">
                <div className="text-xs text-gray-500">Portfolio Expected Return</div>
                <div className="text-xl font-semibold">
                  {data ? (data.portfolio_capm.expected_return * 100).toFixed(2) : '--'}%
                </div>
              </div>

              <div className="bg-gray-50 border rounded p-3">
                <div className="text-xs text-gray-500">Portfolio Beta</div>
                <div className="text-xl font-semibold">{data ? data.portfolio_capm.beta.toFixed(2) : '--'}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main content: holdings table, CAPM metrics, allocation */}
      <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Holdings table (wide) */}
        <section className="lg:col-span-2 bg-white rounded shadow p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">Holdings</h3>
            <div className="text-xs text-gray-400">Last update: {lastUpdate}</div>
          </div>

          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 border-b">
                  <th className="py-2">Ticker</th>
                  <th>Company</th>
                  <th className="text-right">Shares</th>
                  <th className="text-right">Cost</th>
                  <th className="text-right">Price</th>
                  <th className="text-right">Value</th>
                  <th className="text-right">P/L $</th>
                  <th className="text-right">P/L %</th>
                  <th className="text-right">Beta</th>
                  <th className="text-right">CAPM</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {data?.stock_analyses.map((s, i) => (
                  <tr key={s.ticker} className="border-b last:border-b-0">
                    <td className="py-3 font-medium">{s.ticker}</td>
                    <td className="text-gray-500">{s.name}</td>
                    <td className="text-right">{s.shares?.toLocaleString()}</td>
                    <td className="text-right">{s.cost_basis ? formatCurrency(s.cost_basis) : '—'}</td>
                    <td className="text-right">{formatCurrency(s.current_price)}</td>
                    <td className="text-right">{s.current_value ? formatCurrency(s.current_value) : '—'}</td>
                    <td className={`text-right font-semibold ${ (s.gain_loss || 0) >= 0 ? 'text-green-600' : 'text-red-600' }`}>{s.gain_loss ? formatCurrency(s.gain_loss) : '—'}</td>
                    <td className={`text-right ${ (s.gain_loss_percent || 0) >= 0 ? 'text-green-600' : 'text-red-600' }`}>{s.gain_loss_percent ? `${s.gain_loss_percent.toFixed(2)}%` : '—'}</td>
                    <td className="text-right">{s.beta?.toFixed(2) ?? '—'}</td>
                    <td className="text-right">{s.capm ? `${(s.capm.expected_return * 100).toFixed(2)}%` : '—'}</td>
                    <td className="text-right">
                      <div className="flex items-center gap-2 justify-end">
                        <input id={`remove_${i}`} placeholder="0" className="w-20 p-1 border rounded text-sm" />
                        <button
                          className="bg-red-500 text-white px-3 py-1 rounded text-xs"
                          onClick={() => {
                            const input = document.getElementById(`remove_${i}`) as HTMLInputElement;
                            const num = parseFloat(input?.value || '0');
                            if (num > 0) removeShares(i, num);
                            if (input) input.value = '';
                          }}
                        >Remove</button>
                      </div>
                    </td>
                  </tr>
                ))}

                {(!data || data.stock_analyses.length === 0) && (
                  <tr>
                    <td colSpan={11} className="py-6 text-center text-gray-400">No holdings yet — add a position to get started</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Right column: metrics + allocation */}
        <aside className="space-y-6">
          {/* Summary */}
          <div className="bg-white rounded shadow p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold">Portfolio Summary</h4>
              <div className="text-xs text-gray-400">CAPM snapshot</div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 border rounded">
                <div className="text-xs text-gray-500">Total Value</div>
                <div className="font-semibold text-lg">{data ? formatCurrency(data.portfolio_metrics.total_value) : '--'}</div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-xs text-gray-500">Total Cost</div>
                <div className="font-semibold text-lg">{data ? formatCurrency(data.portfolio_metrics.total_cost) : '--'}</div>
              </div>

              <div className={`p-3 border rounded ${data && data.portfolio_metrics.total_gain_loss >= 0 ? 'border-green-200' : 'border-red-200'}`}>
                <div className="text-xs text-gray-500">Total P/L</div>
                <div className={`${data && data.portfolio_metrics.total_gain_loss >= 0 ? 'text-green-600' : 'text-red-600'} font-semibold text-lg`}>{data ? formatCurrency(data.portfolio_metrics.total_gain_loss) : '--'}</div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-xs text-gray-500">P/L %</div>
                <div className="font-semibold text-lg">{data ? `${data.portfolio_metrics.total_gain_loss_percent.toFixed(2)}%` : '--'}</div>
              </div>
            </div>
          </div>

          {/* Allocation (doughnut chart spot) */}
          <div className="bg-white rounded shadow p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold">Allocation</h4>
              <div className="text-sm">
                {(['sector','industry','ticker'] as const).map((t) => (
                  <button key={t} onClick={() => setAllocationType(t)} className={`px-2 py-1 text-xs rounded ${allocationType === t ? 'bg-blue-600 text-white' : 'bg-gray-100'}`} style={{marginLeft:8}}>{t}</button>
                ))}
              </div>
            </div>

            <div className="h-48 flex flex-col justify-center items-center text-sm text-gray-500"> 
              {/* Replace with your DoughnutChart component if you have one. This is a graceful fallback. */}
              {allocationData.length > 0 ? (
                <div className="w-full">
                  {allocationData.slice(0,6).map((a) => (
                    <div key={a.name || a.ticker} className="flex justify-between py-1">
                      <div>{a.name || a.ticker}</div>
                      <div>{a.percentage}%</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div>No allocation data</div>
              )}
            </div>
          </div>
        </aside>
      </main>

      <footer className="mt-6 text-center text-xs text-gray-400">Built with care — replace the mock price fetcher with your real data source for production.</footer>
    </div>
  );
}
