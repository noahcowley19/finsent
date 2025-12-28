'use client';

// =============================================================================
// ADVANCED PORTFOLIO COMPONENTS
// =============================================================================
// Components for correlation matrix, Monte Carlo, and What-If analysis
//
// Location: frontend/components/portfolio/AdvancedPortfolio.tsx
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

// =============================================================================
// TYPES
// =============================================================================

interface CorrelationData {
    tickers: string[];
    matrix: number[][];
}

interface MonteCarloData {
    initial_value: number;
    simulations: number;
    time_horizon_days: number;
    percentile_paths: Record<string, number[]>;
    final_value_stats: {
        mean: number;
        median: number;
        std: number;
        min: number;
        max: number;
        p5: number;
        p95: number;
    };
    expected_return: number;
    worst_case_return: number;
    best_case_return: number;
}

interface VaRData {
    portfolio_value: number;
    confidence_level: number;
    time_horizon_days: number;
    var: {
        percentage: number;
        dollar_amount: number;
        interpretation: string;
    };
    cvar: {
        percentage: number;
        dollar_amount: number;
        interpretation: string;
    };
}

interface Recommendation {
    type: string;
    priority: 'high' | 'medium' | 'low';
    ticker?: string;
    sector?: string;
    message: string;
    current_weight?: number;
    suggested_weight?: number;
}

interface OptimizationData {
    recommendations: Recommendation[];
    diversification_score: number;
    positions_count: number;
    sectors_count: number;
}

interface Trade {
    action: 'buy' | 'sell';
    ticker: string;
    shares: number;
    price: number;
}

interface WhatIfResult {
    current: {
        total_value: number;
        total_positions: number;
        portfolio_beta: number;
        diversification_score: number;
        concentration_risk: string;
    };
    simulated: {
        total_value: number;
        total_positions: number;
        portfolio_beta: number;
        diversification_score: number;
        concentration_risk: string;
    };
    changes: {
        value_change: number;
        beta_change: number;
        diversification_change: number;
    };
}

interface Position {
    ticker: string;
    shares: number;
    total_cost_basis: number;
}

// =============================================================================
// API BASE
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

// =============================================================================
// CORRELATION MATRIX COMPONENT
// =============================================================================

export const CorrelationMatrix: React.FC<{
    tickers: string[];
    onLoad?: () => void;
}> = ({ tickers, onLoad }) => {
    const [data, setData] = useState<CorrelationData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchCorrelation = useCallback(async () => {
        if (tickers.length < 2) return;

        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_BASE}/api/portfolio/correlation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers }),
            });

            if (!res.ok) throw new Error('Failed to fetch correlation');
            const result = await res.json();
            setData(result);
            onLoad?.();
        } catch (err) {
            setError('Unable to calculate correlations');
        } finally {
            setLoading(false);
        }
    }, [tickers.join(',')]);

    useEffect(() => {
        fetchCorrelation();
    }, [fetchCorrelation]);

    const getCorrelationColor = (value: number) => {
        if (value >= 0.7) return 'bg-error-500 text-white';
        if (value >= 0.4) return 'bg-warning-400 text-navy-900';
        if (value >= -0.4) return 'bg-cream-100 text-navy-900';
        if (value >= -0.7) return 'bg-success-300 text-navy-900';
        return 'bg-success-500 text-white';
    };

    if (loading) {
        return (
            <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                    Correlation Matrix
                </h3>
                <div className="animate-pulse h-64 bg-cream-100 rounded-lg" />
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                    Correlation Matrix
                </h3>
                <p className="text-neutral-500 text-center py-8">
                    {error || 'Add at least 2 positions to see correlations'}
                </p>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-heading font-semibold text-heading-md text-navy-900">
                    Correlation Matrix
                </h3>
                <div className="flex items-center gap-2 text-caption">
                    <span className="px-2 py-1 rounded bg-error-500 text-white">High +</span>
                    <span className="px-2 py-1 rounded bg-cream-100">Neutral</span>
                    <span className="px-2 py-1 rounded bg-success-500 text-white">High -</span>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead>
                        <tr>
                            <th className="p-2"></th>
                            {data.tickers.map(ticker => (
                                <th key={ticker} className="p-2 text-caption font-semibold text-navy-900 text-center">
                                    {ticker}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {data.matrix.map((row, i) => (
                            <tr key={data.tickers[i]}>
                                <td className="p-2 text-caption font-semibold text-navy-900">{data.tickers[i]}</td>
                                {row.map((value, j) => (
                                    <td key={j} className="p-1">
                                        <div
                                            className={`p-2 rounded text-center text-caption font-medium ${getCorrelationColor(value)}`}
                                        >
                                            {value.toFixed(2)}
                                        </div>
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <p className="mt-4 text-caption text-neutral-500">
                High positive correlation (red) means stocks tend to move together.
                Negative correlation (green) provides diversification benefits.
            </p>
        </div>
    );
};

// =============================================================================
// MONTE CARLO SIMULATION COMPONENT
// =============================================================================

export const MonteCarloSimulation: React.FC<{
    positions: Position[];
}> = ({ positions }) => {
    const [data, setData] = useState<MonteCarloData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [timeHorizon, setTimeHorizon] = useState(252); // 1 year

    const runSimulation = useCallback(async () => {
        if (positions.length === 0) return;

        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_BASE}/api/portfolio/monte-carlo`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    positions: positions.map(p => ({
                        ticker: p.ticker,
                        shares: p.shares,
                        total_cost_basis: p.total_cost_basis,
                    })),
                    simulations: 1000,
                    time_horizon: timeHorizon,
                }),
            });

            if (!res.ok) throw new Error('Failed to run simulation');
            const result = await res.json();
            setData(result);
        } catch (err) {
            setError('Unable to run simulation');
        } finally {
            setLoading(false);
        }
    }, [positions, timeHorizon]);

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="font-heading font-semibold text-heading-md text-navy-900">
                        Monte Carlo Simulation
                    </h3>
                    <p className="text-body-sm text-neutral-500 mt-1">
                        Project future portfolio values using 1,000 simulations
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    <select
                        value={timeHorizon}
                        onChange={(e) => setTimeHorizon(Number(e.target.value))}
                        className="px-3 py-2 border border-border-medium rounded-lg text-body-sm"
                    >
                        <option value={63}>3 Months</option>
                        <option value={126}>6 Months</option>
                        <option value={252}>1 Year</option>
                        <option value={504}>2 Years</option>
                    </select>

                    <button
                        onClick={runSimulation}
                        disabled={loading || positions.length === 0}
                        className="px-4 py-2 bg-terra-500 text-white rounded-lg text-body-sm font-medium hover:bg-terra-600 disabled:opacity-50"
                    >
                        {loading ? 'Running...' : 'Run Simulation'}
                    </button>
                </div>
            </div>

            {error && (
                <p className="text-error-600 text-center py-4">{error}</p>
            )}

            {data && (
                <div className="space-y-6">
                    {/* Results Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-cream-50 rounded-lg text-center">
                            <p className="text-caption text-neutral-500">Starting Value</p>
                            <p className="text-heading-md font-semibold text-navy-900">
                                ${data.initial_value.toLocaleString()}
                            </p>
                        </div>
                        <div className="p-4 bg-success-50 rounded-lg text-center">
                            <p className="text-caption text-neutral-500">Expected (Median)</p>
                            <p className="text-heading-md font-semibold text-success-700">
                                ${data.final_value_stats.median.toLocaleString()}
                            </p>
                            <p className="text-caption text-success-600">
                                +{data.expected_return.toFixed(1)}%
                            </p>
                        </div>
                        <div className="p-4 bg-error-50 rounded-lg text-center">
                            <p className="text-caption text-neutral-500">Worst Case (5th %ile)</p>
                            <p className="text-heading-md font-semibold text-error-700">
                                ${data.final_value_stats.p5.toLocaleString()}
                            </p>
                            <p className="text-caption text-error-600">
                                {data.worst_case_return.toFixed(1)}%
                            </p>
                        </div>
                        <div className="p-4 bg-navy-50 rounded-lg text-center">
                            <p className="text-caption text-neutral-500">Best Case (95th %ile)</p>
                            <p className="text-heading-md font-semibold text-navy-700">
                                ${data.final_value_stats.p95.toLocaleString()}
                            </p>
                            <p className="text-caption text-navy-600">
                                +{data.best_case_return.toFixed(1)}%
                            </p>
                        </div>
                    </div>

                    {/* Simplified Chart (text representation) */}
                    <div className="p-4 bg-cream-50 rounded-lg">
                        <h4 className="font-medium text-navy-900 mb-3">Projected Range</h4>
                        <div className="h-24 relative flex items-end gap-1">
                            {data.percentile_paths.p50?.map((value, i) => {
                                const minVal = data.percentile_paths.p5?.[i] || value * 0.8;
                                const maxVal = data.percentile_paths.p95?.[i] || value * 1.2;
                                const range = data.final_value_stats.max - data.final_value_stats.min;
                                const height = ((maxVal - minVal) / range) * 100;
                                const bottom = ((minVal - data.final_value_stats.min) / range) * 100;

                                return (
                                    <div
                                        key={i}
                                        className="flex-1 bg-gradient-to-t from-terra-400 to-warning-300 rounded-t"
                                        style={{
                                            height: `${Math.max(height, 5)}%`,
                                            marginBottom: `${bottom}%`
                                        }}
                                    />
                                );
                            })}
                        </div>
                        <div className="flex justify-between text-caption text-neutral-500 mt-2">
                            <span>Now</span>
                            <span>{Math.round(timeHorizon / 21)} months</span>
                        </div>
                    </div>
                </div>
            )}

            {!data && !loading && (
                <div className="text-center py-12 bg-cream-50 rounded-lg">
                    <p className="text-neutral-600">
                        Click "Run Simulation" to project future portfolio values
                    </p>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// VAR CARD COMPONENT
// =============================================================================

export const VaRCard: React.FC<{
    positions: Position[];
}> = ({ positions }) => {
    const [data, setData] = useState<VaRData | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchVaR = async () => {
            if (positions.length === 0) return;

            setLoading(true);
            try {
                const res = await fetch(`${API_BASE}/api/portfolio/var`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        positions: positions.map(p => ({
                            ticker: p.ticker,
                            shares: p.shares,
                            total_cost_basis: p.total_cost_basis,
                        })),
                        confidence: 0.95,
                        time_horizon: 1,
                    }),
                });

                if (res.ok) {
                    const result = await res.json();
                    setData(result);
                }
            } catch (err) {
                console.error('VaR error:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchVaR();
    }, [positions]);

    if (loading) {
        return (
            <div className="bg-white rounded-xl border border-border-light p-6 animate-pulse">
                <div className="h-6 bg-cream-100 rounded w-1/3 mb-4" />
                <div className="h-20 bg-cream-100 rounded" />
            </div>
        );
    }

    if (!data) return null;

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                Value at Risk (95% Confidence)
            </h3>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-4 bg-error-50 rounded-lg">
                    <p className="text-caption text-neutral-500">Daily VaR</p>
                    <p className="text-heading-sm font-semibold text-error-700">
                        ${data.var.dollar_amount.toLocaleString()}
                    </p>
                    <p className="text-caption text-error-600">
                        {data.var.percentage.toFixed(2)}%
                    </p>
                </div>
                <div className="p-4 bg-warning-50 rounded-lg">
                    <p className="text-caption text-neutral-500">CVaR (Expected Shortfall)</p>
                    <p className="text-heading-sm font-semibold text-warning-700">
                        ${data.cvar.dollar_amount.toLocaleString()}
                    </p>
                    <p className="text-caption text-warning-600">
                        {data.cvar.percentage.toFixed(2)}%
                    </p>
                </div>
            </div>

            <p className="text-body-sm text-neutral-600">
                {data.var.interpretation}
            </p>
        </div>
    );
};

// =============================================================================
// OPTIMIZATION RECOMMENDATIONS COMPONENT
// =============================================================================

export const OptimizationPanel: React.FC<{
    positions: Position[];
}> = ({ positions }) => {
    const [data, setData] = useState<OptimizationData | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchOptimization = async () => {
            if (positions.length === 0) return;

            setLoading(true);
            try {
                const res = await fetch(`${API_BASE}/api/portfolio/optimize`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        positions: positions.map(p => ({
                            ticker: p.ticker,
                            shares: p.shares,
                            total_cost_basis: p.total_cost_basis,
                        })),
                    }),
                });

                if (res.ok) {
                    const result = await res.json();
                    setData(result);
                }
            } catch (err) {
                console.error('Optimization error:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchOptimization();
    }, [positions]);

    if (loading) {
        return (
            <div className="bg-white rounded-xl border border-border-light p-6 animate-pulse">
                <div className="h-6 bg-cream-100 rounded w-1/3 mb-4" />
                <div className="space-y-2">
                    <div className="h-16 bg-cream-100 rounded" />
                    <div className="h-16 bg-cream-100 rounded" />
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                Optimization Recommendations
            </h3>

            {/* Stats */}
            {data && (
                <div className="grid grid-cols-3 gap-3 mb-6">
                    <div className="text-center p-3 bg-cream-50 rounded-lg">
                        <p className="text-display-xs font-semibold text-navy-900">
                            {data.diversification_score}
                        </p>
                        <p className="text-caption text-neutral-500">Diversification</p>
                    </div>
                    <div className="text-center p-3 bg-cream-50 rounded-lg">
                        <p className="text-display-xs font-semibold text-navy-900">
                            {data.positions_count}
                        </p>
                        <p className="text-caption text-neutral-500">Holdings</p>
                    </div>
                    <div className="text-center p-3 bg-cream-50 rounded-lg">
                        <p className="text-display-xs font-semibold text-navy-900">
                            {data.sectors_count}
                        </p>
                        <p className="text-caption text-neutral-500">Sectors</p>
                    </div>
                </div>
            )}

            {/* Recommendations */}
            {data?.recommendations && data.recommendations.length > 0 ? (
                <div className="space-y-3">
                    {data.recommendations.map((rec, i) => (
                        <div
                            key={i}
                            className={`p-4 rounded-lg border-l-4 ${rec.priority === 'high'
                                    ? 'bg-error-50 border-l-error-500'
                                    : rec.priority === 'medium'
                                        ? 'bg-warning-50 border-l-warning-500'
                                        : 'bg-cream-50 border-l-neutral-300'
                                }`}
                        >
                            <div className="flex items-start justify-between">
                                <div>
                                    <span className={`text-caption font-medium uppercase ${rec.priority === 'high' ? 'text-error-600' :
                                            rec.priority === 'medium' ? 'text-warning-600' :
                                                'text-neutral-600'
                                        }`}>
                                        {rec.priority} priority
                                    </span>
                                    <p className="text-body-sm text-navy-900 mt-1">{rec.message}</p>
                                </div>
                                {rec.ticker && (
                                    <span className="px-2 py-1 bg-white rounded text-caption font-medium">
                                        {rec.ticker}
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="text-center py-8 bg-success-50 rounded-lg">
                    <svg className="w-12 h-12 text-success-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-success-700 font-medium">Portfolio looks well-balanced!</p>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// WHAT-IF SIMULATOR COMPONENT
// =============================================================================

export const WhatIfSimulator: React.FC<{
    positions: Position[];
}> = ({ positions }) => {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [newTrade, setNewTrade] = useState<Partial<Trade>>({
        action: 'buy',
        ticker: '',
        shares: 0,
        price: 0,
    });
    const [result, setResult] = useState<WhatIfResult | null>(null);
    const [loading, setLoading] = useState(false);

    const addTrade = () => {
        if (!newTrade.ticker || !newTrade.shares || !newTrade.price) return;

        setTrades([...trades, newTrade as Trade]);
        setNewTrade({ action: 'buy', ticker: '', shares: 0, price: 0 });
    };

    const removeTrade = (index: number) => {
        setTrades(trades.filter((_, i) => i !== index));
        setResult(null);
    };

    const simulateTrades = async () => {
        if (trades.length === 0 || positions.length === 0) return;

        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/portfolio/what-if`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    positions: positions.map(p => ({
                        ticker: p.ticker,
                        shares: p.shares,
                        total_cost_basis: p.total_cost_basis,
                    })),
                    trades,
                }),
            });

            if (res.ok) {
                const data = await res.json();
                setResult(data);
            }
        } catch (err) {
            console.error('What-if error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-2">
                What-If Simulator
            </h3>
            <p className="text-body-sm text-neutral-500 mb-6">
                Model trades before executing to see how they'll affect your portfolio
            </p>

            {/* Add Trade Form */}
            <div className="p-4 bg-cream-50 rounded-lg mb-4">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <select
                        value={newTrade.action}
                        onChange={(e) => setNewTrade({ ...newTrade, action: e.target.value as 'buy' | 'sell' })}
                        className="px-3 py-2 border border-border-medium rounded-lg text-body-sm"
                    >
                        <option value="buy">Buy</option>
                        <option value="sell">Sell</option>
                    </select>

                    <input
                        type="text"
                        placeholder="Ticker"
                        value={newTrade.ticker}
                        onChange={(e) => setNewTrade({ ...newTrade, ticker: e.target.value.toUpperCase() })}
                        className="px-3 py-2 border border-border-medium rounded-lg text-body-sm"
                    />

                    <input
                        type="number"
                        placeholder="Shares"
                        value={newTrade.shares || ''}
                        onChange={(e) => setNewTrade({ ...newTrade, shares: Number(e.target.value) })}
                        className="px-3 py-2 border border-border-medium rounded-lg text-body-sm"
                    />

                    <input
                        type="number"
                        placeholder="Price"
                        value={newTrade.price || ''}
                        onChange={(e) => setNewTrade({ ...newTrade, price: Number(e.target.value) })}
                        className="px-3 py-2 border border-border-medium rounded-lg text-body-sm"
                    />

                    <button
                        onClick={addTrade}
                        className="px-4 py-2 bg-navy-600 text-white rounded-lg text-body-sm font-medium hover:bg-navy-700"
                    >
                        Add Trade
                    </button>
                </div>
            </div>

            {/* Trade List */}
            {trades.length > 0 && (
                <div className="space-y-2 mb-4">
                    {trades.map((trade, i) => (
                        <div key={i} className="flex items-center justify-between p-3 bg-cream-50 rounded-lg">
                            <div className="flex items-center gap-3">
                                <span className={`px-2 py-1 rounded text-caption font-medium ${trade.action === 'buy' ? 'bg-success-100 text-success-700' : 'bg-error-100 text-error-700'
                                    }`}>
                                    {trade.action.toUpperCase()}
                                </span>
                                <span className="font-medium">{trade.shares} shares of {trade.ticker}</span>
                                <span className="text-neutral-500">@ ${trade.price}</span>
                            </div>
                            <button
                                onClick={() => removeTrade(i)}
                                className="text-error-500 hover:text-error-700"
                            >
                                ✕
                            </button>
                        </div>
                    ))}

                    <button
                        onClick={simulateTrades}
                        disabled={loading}
                        className="w-full py-3 bg-terra-500 text-white rounded-lg font-medium hover:bg-terra-600 disabled:opacity-50"
                    >
                        {loading ? 'Simulating...' : 'Simulate Trades'}
                    </button>
                </div>
            )}

            {/* Results */}
            {result && (
                <div className="space-y-4 pt-4 border-t border-border-light">
                    <h4 className="font-medium text-navy-900">Simulation Results</h4>

                    <div className="grid grid-cols-2 gap-4">
                        {/* Current */}
                        <div className="p-4 bg-cream-50 rounded-lg">
                            <p className="text-caption text-neutral-500 mb-2">Current Portfolio</p>
                            <p className="text-heading-sm font-semibold text-navy-900">
                                ${result.current.total_value.toLocaleString()}
                            </p>
                            <div className="mt-2 space-y-1 text-caption">
                                <p>Beta: {result.current.portfolio_beta}</p>
                                <p>Positions: {result.current.total_positions}</p>
                            </div>
                        </div>

                        {/* Simulated */}
                        <div className="p-4 bg-terra-50 rounded-lg">
                            <p className="text-caption text-neutral-500 mb-2">After Trades</p>
                            <p className="text-heading-sm font-semibold text-terra-700">
                                ${result.simulated.total_value.toLocaleString()}
                            </p>
                            <div className="mt-2 space-y-1 text-caption">
                                <p>Beta: {result.simulated.portfolio_beta}</p>
                                <p>Positions: {result.simulated.total_positions}</p>
                            </div>
                        </div>
                    </div>

                    {/* Changes */}
                    <div className="grid grid-cols-3 gap-3">
                        <div className={`p-3 rounded-lg text-center ${result.changes.value_change >= 0 ? 'bg-success-50' : 'bg-error-50'
                            }`}>
                            <p className="text-caption text-neutral-500">Value Change</p>
                            <p className={`font-semibold ${result.changes.value_change >= 0 ? 'text-success-700' : 'text-error-700'
                                }`}>
                                {result.changes.value_change >= 0 ? '+' : ''}${result.changes.value_change.toLocaleString()}
                            </p>
                        </div>
                        <div className="p-3 bg-cream-50 rounded-lg text-center">
                            <p className="text-caption text-neutral-500">Beta Change</p>
                            <p className="font-semibold text-navy-700">
                                {result.changes.beta_change >= 0 ? '+' : ''}{result.changes.beta_change.toFixed(2)}
                            </p>
                        </div>
                        <div className="p-3 bg-cream-50 rounded-lg text-center">
                            <p className="text-caption text-neutral-500">Diversification</p>
                            <p className={`font-semibold ${result.changes.diversification_change >= 0 ? 'text-success-700' : 'text-error-700'
                                }`}>
                                {result.changes.diversification_change >= 0 ? '+' : ''}{result.changes.diversification_change.toFixed(1)}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {trades.length === 0 && !result && (
                <div className="text-center py-8 text-neutral-500">
                    Add trades above to see how they would affect your portfolio
                </div>
            )}
        </div>
    );
};

export default {
    CorrelationMatrix,
    MonteCarloSimulation,
    VaRCard,
    OptimizationPanel,
    WhatIfSimulator,
};
