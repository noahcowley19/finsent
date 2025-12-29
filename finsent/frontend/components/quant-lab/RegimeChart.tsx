'use client';

// =============================================================================
// REGIME CHART - HMM Market Regime Overlay
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface RegimeData {
    ticker: string;
    states: number[];
    dates: string[];
    labels: string[];
    colors: string[];
    state_stats: {
        label: string;
        color: string;
        mean_return: number;
        volatility: number;
        count: number;
    }[];
    transition_matrix: number[][];
    current_regime: string;
}

export const RegimeChart: React.FC<{ ticker?: string }> = ({ ticker: propTicker }) => {
    const [ticker, setTicker] = useState(propTicker || 'SPY');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<RegimeData | null>(null);
    const [error, setError] = useState<string | null>(null);

    const fetchRegimes = useCallback(async () => {
        if (!ticker) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/quant/regime`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker.toUpperCase(), n_regimes: 3, period: '5y' }),
            });

            if (!response.ok) throw new Error('Failed to detect regimes');

            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    }, [ticker]);

    useEffect(() => {
        if (propTicker) fetchRegimes();
    }, [propTicker, fetchRegimes]);

    // Group consecutive states for visualization
    const getRegimeBlocks = () => {
        if (!data) return [];

        const blocks: { state: number; start: number; end: number }[] = [];
        let currentState = data.states[0];
        let startIdx = 0;

        for (let i = 1; i <= data.states.length; i++) {
            if (i === data.states.length || data.states[i] !== currentState) {
                blocks.push({ state: currentState, start: startIdx, end: i - 1 });
                if (i < data.states.length) {
                    currentState = data.states[i];
                    startIdx = i;
                }
            }
        }

        return blocks;
    };

    return (
        <div className="p-6 bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-lg font-semibold text-obsidian-900">Regime Detector</h3>
                    <p className="text-sm text-obsidian-500">Hidden Markov Model market state classification</p>
                </div>

                <div className="flex items-center gap-3">
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        className="w-24 px-3 py-1.5 border border-cream-200 rounded-lg text-sm"
                        placeholder="SPY"
                    />
                    <button
                        onClick={fetchRegimes}
                        disabled={loading}
                        className="px-4 py-1.5 bg-obsidian-900 text-white text-sm font-medium rounded-lg hover:bg-obsidian-800 disabled:opacity-50"
                    >
                        {loading ? 'Detecting...' : 'Analyze'}
                    </button>
                </div>
            </div>

            {error && (
                <div className="p-4 bg-coral-50 border border-coral-200 rounded-xl text-coral-700 mb-4">
                    {error}
                </div>
            )}

            {data && (
                <>
                    {/* Current Regime */}
                    <div className="mb-6 p-4 bg-cream-50 rounded-xl">
                        <p className="text-sm text-obsidian-500 mb-1">Current Market Regime</p>
                        <p className={`text-2xl font-bold ${data.current_regime === 'Bull' ? 'text-success-600' :
                                data.current_regime === 'Bear' ? 'text-coral-600' : 'text-obsidian-600'
                            }`}>
                            {data.current_regime === 'Bull' ? '🐂' : data.current_regime === 'Bear' ? '🐻' : '↔️'} {data.current_regime}
                        </p>
                    </div>

                    {/* Regime Timeline */}
                    <div className="mb-6">
                        <p className="text-xs uppercase tracking-wide text-obsidian-400 mb-2">Regime Timeline</p>
                        <div className="h-8 flex rounded-lg overflow-hidden">
                            {getRegimeBlocks().map((block, i) => {
                                const width = ((block.end - block.start + 1) / data.states.length) * 100;
                                return (
                                    <div
                                        key={i}
                                        className="transition-all hover:brightness-110"
                                        style={{
                                            width: `${width}%`,
                                            backgroundColor: data.colors[block.state],
                                        }}
                                        title={`${data.labels[block.state]}: ${data.dates[block.start]} to ${data.dates[block.end]}`}
                                    />
                                );
                            })}
                        </div>
                        <div className="flex justify-between mt-1 text-xs text-obsidian-400">
                            <span>{data.dates[0]}</span>
                            <span>{data.dates[data.dates.length - 1]}</span>
                        </div>
                    </div>

                    {/* Regime Stats */}
                    <div className="grid grid-cols-3 gap-4">
                        {data.state_stats.map((stat, i) => (
                            <div
                                key={i}
                                className="p-4 rounded-xl border"
                                style={{ borderColor: stat.color, backgroundColor: `${stat.color}10` }}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: stat.color }} />
                                    <span className="font-semibold text-obsidian-900">{stat.label}</span>
                                </div>
                                <div className="space-y-1 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-obsidian-500">Avg Return</span>
                                        <span className={stat.mean_return > 0 ? 'text-success-600' : 'text-coral-600'}>
                                            {stat.mean_return.toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-obsidian-500">Volatility</span>
                                        <span>{stat.volatility.toFixed(1)}%</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-obsidian-500">Days</span>
                                        <span>{stat.count}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </>
            )}

            {!data && !loading && !error && (
                <div className="py-12 text-center text-obsidian-400">
                    Enter a ticker and click Analyze to detect market regimes
                </div>
            )}
        </div>
    );
};

export default RegimeChart;
