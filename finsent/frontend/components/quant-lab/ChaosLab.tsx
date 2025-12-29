'use client';

// =============================================================================
// CHAOS LAB - Entropy & Cointegration Analysis
// =============================================================================

import React, { useState, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface EntropyData {
    ticker: string;
    sample_entropy: number | null;
    interpretation: string;
}

interface CointPair {
    pair: string[];
    pvalue: number;
    score: number;
    cointegrated: boolean;
    correlation: number;
}

interface PairsData {
    tickers: string[];
    pairs: CointPair[];
    cointegrated_count: number;
    total_pairs: number;
}

// Entropy Gauge Component
const EntropyGauge: React.FC<{ data: EntropyData }> = ({ data }) => {
    const entropy = data.sample_entropy ?? 0;
    const maxEntropy = 2.5;
    const percentage = Math.min((entropy / maxEntropy) * 100, 100);

    const getColor = () => {
        if (entropy < 0.5) return 'text-success-600';
        if (entropy < 1.0) return 'text-electric-600';
        if (entropy < 1.5) return 'text-amber-600';
        return 'text-coral-600';
    };

    return (
        <div className="p-6 bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50">
            <h4 className="font-semibold text-obsidian-900 mb-4">Sample Entropy</h4>

            <div className="flex items-center gap-6">
                {/* Gauge */}
                <div className="relative w-32 h-32">
                    <svg className="w-full h-full -rotate-90">
                        <circle
                            cx="64"
                            cy="64"
                            r="56"
                            fill="none"
                            stroke="#E4E4E7"
                            strokeWidth="12"
                        />
                        <circle
                            cx="64"
                            cy="64"
                            r="56"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="12"
                            strokeDasharray={`${percentage * 3.52} 352`}
                            className={getColor()}
                        />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className={`text-2xl font-bold ${getColor()}`}>
                            {data.sample_entropy?.toFixed(2) ?? '—'}
                        </span>
                        <span className="text-xs text-obsidian-400">SampEn</span>
                    </div>
                </div>

                {/* Interpretation */}
                <div className="flex-1">
                    <p className="text-sm text-obsidian-500 mb-2">{data.ticker}</p>
                    <p className={`font-medium ${getColor()}`}>{data.interpretation}</p>
                    <div className="mt-3 text-xs text-obsidian-400">
                        <p>Lower entropy = more predictable patterns</p>
                        <p>Higher entropy = more random/efficient</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Pairs Table Component
const PairsTable: React.FC<{ data: PairsData }> = ({ data }) => {
    return (
        <div className="p-6 bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50">
            <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-obsidian-900">Cointegrated Pairs</h4>
                <span className="px-2 py-1 bg-success-100 text-success-700 text-xs font-medium rounded">
                    {data.cointegrated_count} / {data.total_pairs} pairs
                </span>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-cream-200">
                            <th className="text-left py-2 text-obsidian-500 font-medium">Pair</th>
                            <th className="text-right py-2 text-obsidian-500 font-medium">P-Value</th>
                            <th className="text-right py-2 text-obsidian-500 font-medium">Correlation</th>
                            <th className="text-center py-2 text-obsidian-500 font-medium">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.pairs.slice(0, 10).map((pair, i) => (
                            <tr key={i} className="border-b border-cream-100">
                                <td className="py-3 font-mono">{pair.pair.join(' / ')}</td>
                                <td className="py-3 text-right font-mono">
                                    <span className={pair.pvalue < 0.05 ? 'text-success-600' : 'text-obsidian-600'}>
                                        {pair.pvalue.toFixed(4)}
                                    </span>
                                </td>
                                <td className="py-3 text-right font-mono">{pair.correlation.toFixed(2)}</td>
                                <td className="py-3 text-center">
                                    {pair.cointegrated ? (
                                        <span className="px-2 py-1 bg-success-100 text-success-700 text-xs rounded">
                                            Cointegrated
                                        </span>
                                    ) : (
                                        <span className="px-2 py-1 bg-cream-100 text-obsidian-500 text-xs rounded">
                                            Not Coint.
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {data.cointegrated_count > 0 && (
                <p className="mt-4 text-xs text-obsidian-400">
                    💡 Cointegrated pairs may be suitable for pairs trading strategies
                </p>
            )}
        </div>
    );
};

// Main Chaos Lab Component
export const ChaosLab: React.FC = () => {
    const [entropyTicker, setEntropyTicker] = useState('');
    const [pairsTickers, setPairsTickers] = useState('SPY, QQQ, IWM, DIA, GLD');
    const [entropyData, setEntropyData] = useState<EntropyData | null>(null);
    const [pairsData, setPairsData] = useState<PairsData | null>(null);
    const [entropyLoading, setEntropyLoading] = useState(false);
    const [pairsLoading, setPairsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchEntropy = useCallback(async () => {
        if (!entropyTicker) return;

        setEntropyLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/quant/entropy/${entropyTicker.toUpperCase()}`);
            if (!response.ok) throw new Error('Failed to calculate entropy');
            const data = await response.json();
            setEntropyData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setEntropyLoading(false);
        }
    }, [entropyTicker]);

    const fetchPairs = useCallback(async () => {
        const tickers = pairsTickers.split(',').map(t => t.trim().toUpperCase()).filter(Boolean);
        if (tickers.length < 2) return;

        setPairsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/quant/pairs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers }),
            });
            if (!response.ok) throw new Error('Failed to find pairs');
            const data = await response.json();
            setPairsData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setPairsLoading(false);
        }
    }, [pairsTickers]);

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h2 className="text-2xl font-bold text-obsidian-900">Chaos Lab</h2>
                <p className="text-obsidian-500">Measure predictability and find trading opportunities</p>
            </div>

            {error && (
                <div className="p-4 bg-coral-50 border border-coral-200 rounded-xl text-coral-700">
                    {error}
                </div>
            )}

            {/* Entropy Section */}
            <div className="p-6 bg-cream-50 rounded-2xl">
                <h3 className="font-semibold text-obsidian-900 mb-4">🔮 Sample Entropy (Predictability)</h3>

                <div className="flex items-center gap-3 mb-6">
                    <input
                        type="text"
                        value={entropyTicker}
                        onChange={(e) => setEntropyTicker(e.target.value.toUpperCase())}
                        className="flex-1 px-4 py-2 border border-cream-200 rounded-lg"
                        placeholder="Enter ticker (e.g., AAPL)"
                    />
                    <button
                        onClick={fetchEntropy}
                        disabled={entropyLoading || !entropyTicker}
                        className="px-6 py-2 bg-obsidian-900 text-white font-medium rounded-lg hover:bg-obsidian-800 disabled:opacity-50"
                    >
                        {entropyLoading ? 'Calculating...' : 'Calculate'}
                    </button>
                </div>

                {entropyData && <EntropyGauge data={entropyData} />}
            </div>

            {/* Pairs Section */}
            <div className="p-6 bg-cream-50 rounded-2xl">
                <h3 className="font-semibold text-obsidian-900 mb-4">🔗 Cointegration (Pairs Trading)</h3>

                <div className="flex items-center gap-3 mb-6">
                    <input
                        type="text"
                        value={pairsTickers}
                        onChange={(e) => setPairsTickers(e.target.value)}
                        className="flex-1 px-4 py-2 border border-cream-200 rounded-lg"
                        placeholder="Enter tickers separated by commas"
                    />
                    <button
                        onClick={fetchPairs}
                        disabled={pairsLoading}
                        className="px-6 py-2 bg-obsidian-900 text-white font-medium rounded-lg hover:bg-obsidian-800 disabled:opacity-50"
                    >
                        {pairsLoading ? 'Searching...' : 'Find Pairs'}
                    </button>
                </div>

                {pairsData && <PairsTable data={pairsData} />}
            </div>
        </div>
    );
};

export default ChaosLab;
