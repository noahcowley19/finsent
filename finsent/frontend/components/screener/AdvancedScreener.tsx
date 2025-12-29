'use client';

// =============================================================================
// ADVANCED SCREENER V2 - ML-Powered Stock Discovery
// =============================================================================

import React, { useState, useCallback, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Stock {
    ticker: string;
    name: string;
    sector: string;
    price: number;
    change_pct: number;
    market_cap: number;
    pe_ratio: number;
    rsi: number;
    volume_zscore: number;
    return_zscore: number;
    piotroski_score: number;
    altman_zscore: number;
    cluster_id?: number;
    cluster_name?: string;
    is_anomaly?: boolean;
}

interface Cluster {
    id: number;
    name: string;
    color: string;
    count: number;
    avg_return_5d: number;
    avg_volatility: number;
}

// Z-Score Slider Component
const ZScoreSlider: React.FC<{
    label: string;
    value: [number, number];
    onChange: (value: [number, number]) => void;
}> = ({ label, value, onChange }) => {
    return (
        <div className="space-y-2">
            <div className="flex justify-between text-sm">
                <span className="text-obsidian-600">{label}</span>
                <span className="font-mono text-obsidian-400">
                    {value[0].toFixed(1)} to {value[1].toFixed(1)}
                </span>
            </div>
            <div className="flex items-center gap-2">
                <input
                    type="range"
                    min="-3"
                    max="3"
                    step="0.5"
                    value={value[0]}
                    onChange={(e) => onChange([parseFloat(e.target.value), value[1]])}
                    className="flex-1 h-2 bg-cream-200 rounded-lg appearance-none cursor-pointer"
                />
                <input
                    type="range"
                    min="-3"
                    max="3"
                    step="0.5"
                    value={value[1]}
                    onChange={(e) => onChange([value[0], parseFloat(e.target.value)])}
                    className="flex-1 h-2 bg-cream-200 rounded-lg appearance-none cursor-pointer"
                />
            </div>
            <div className="flex justify-between text-xs text-obsidian-400">
                <span>-3σ</span>
                <span>0</span>
                <span>+3σ</span>
            </div>
        </div>
    );
};

// Cluster Dropdown Component
const ClusterDropdown: React.FC<{
    clusters: Cluster[];
    selected: number | null;
    onChange: (id: number | null) => void;
}> = ({ clusters, selected, onChange }) => {
    return (
        <div className="space-y-2">
            <label className="block text-sm text-obsidian-600">Behavioral Cluster</label>
            <select
                value={selected ?? ''}
                onChange={(e) => onChange(e.target.value ? parseInt(e.target.value) : null)}
                className="w-full px-3 py-2 bg-white border border-cream-200 rounded-lg text-sm"
            >
                <option value="">All Clusters</option>
                {clusters.map((cluster) => (
                    <option key={cluster.id} value={cluster.id}>
                        {cluster.name} ({cluster.count} stocks)
                    </option>
                ))}
            </select>
        </div>
    );
};

// Results Table Component
const ResultsTable: React.FC<{
    stocks: Stock[];
    sortBy: string;
    sortAsc: boolean;
    onSort: (key: string) => void;
}> = ({ stocks, sortBy, sortAsc, onSort }) => {
    const formatMarketCap = (value: number) => {
        if (!value) return '—';
        if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
        if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
        if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
        return `$${value.toLocaleString()}`;
    };

    const SortHeader: React.FC<{ label: string; field: string }> = ({ label, field }) => (
        <th
            onClick={() => onSort(field)}
            className="px-3 py-2 text-left text-xs font-medium text-obsidian-500 cursor-pointer hover:text-obsidian-900"
        >
            {label} {sortBy === field && (sortAsc ? '↑' : '↓')}
        </th>
    );

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead className="bg-cream-50 border-b border-cream-200">
                    <tr>
                        <SortHeader label="Ticker" field="ticker" />
                        <SortHeader label="Price" field="price" />
                        <SortHeader label="Chg %" field="change_pct" />
                        <SortHeader label="Mkt Cap" field="market_cap" />
                        <SortHeader label="P/E" field="pe_ratio" />
                        <SortHeader label="RSI" field="rsi" />
                        <SortHeader label="Vol Z" field="volume_zscore" />
                        <SortHeader label="F-Score" field="piotroski_score" />
                        <SortHeader label="Altman Z" field="altman_zscore" />
                        <th className="px-3 py-2 text-left text-xs font-medium text-obsidian-500">Cluster</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-cream-100">
                    {stocks.map((stock) => (
                        <tr key={stock.ticker} className="hover:bg-cream-50 transition-colors">
                            <td className="px-3 py-3">
                                <div className="flex items-center gap-2">
                                    {stock.is_anomaly && (
                                        <span className="w-2 h-2 bg-amber-400 rounded-full" title="Anomaly" />
                                    )}
                                    <span className="font-medium text-obsidian-900">{stock.ticker}</span>
                                </div>
                            </td>
                            <td className="px-3 py-3 font-mono">${stock.price?.toFixed(2) ?? '—'}</td>
                            <td className={`px-3 py-3 font-mono ${(stock.change_pct ?? 0) >= 0 ? 'text-success-600' : 'text-coral-600'
                                }`}>
                                {stock.change_pct !== null ? `${stock.change_pct > 0 ? '+' : ''}${stock.change_pct.toFixed(2)}%` : '—'}
                            </td>
                            <td className="px-3 py-3">{formatMarketCap(stock.market_cap)}</td>
                            <td className="px-3 py-3 font-mono">{stock.pe_ratio?.toFixed(1) ?? '—'}</td>
                            <td className="px-3 py-3">
                                <span className={`px-2 py-0.5 text-xs rounded ${(stock.rsi ?? 50) < 30 ? 'bg-coral-100 text-coral-700' :
                                        (stock.rsi ?? 50) > 70 ? 'bg-success-100 text-success-700' :
                                            'bg-cream-100 text-obsidian-600'
                                    }`}>
                                    {stock.rsi?.toFixed(0) ?? '—'}
                                </span>
                            </td>
                            <td className="px-3 py-3">
                                <span className={`px-2 py-0.5 text-xs font-mono rounded ${Math.abs(stock.volume_zscore ?? 0) > 2 ? 'bg-electric-100 text-electric-700' : 'bg-cream-100'
                                    }`}>
                                    {stock.volume_zscore?.toFixed(1) ?? '—'}
                                </span>
                            </td>
                            <td className="px-3 py-3">
                                <span className={`px-2 py-0.5 text-xs rounded ${(stock.piotroski_score ?? 0) >= 7 ? 'bg-success-100 text-success-700' :
                                        (stock.piotroski_score ?? 0) >= 4 ? 'bg-amber-100 text-amber-700' :
                                            'bg-coral-100 text-coral-700'
                                    }`}>
                                    {stock.piotroski_score ?? '—'}
                                </span>
                            </td>
                            <td className="px-3 py-3">
                                <span className={`px-2 py-0.5 text-xs rounded ${(stock.altman_zscore ?? 0) > 2.99 ? 'bg-success-100 text-success-700' :
                                        (stock.altman_zscore ?? 0) > 1.81 ? 'bg-amber-100 text-amber-700' :
                                            'bg-coral-100 text-coral-700'
                                    }`}>
                                    {stock.altman_zscore?.toFixed(1) ?? '—'}
                                </span>
                            </td>
                            <td className="px-3 py-3">
                                {stock.cluster_name && (
                                    <span className="px-2 py-0.5 text-xs bg-electric-100 text-electric-700 rounded">
                                        {stock.cluster_name}
                                    </span>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

// Find Alike Modal
const FindAlikeModal: React.FC<{
    isOpen: boolean;
    onClose: () => void;
}> = ({ isOpen, onClose }) => {
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<Stock[]>([]);
    const [target, setTarget] = useState<Stock | null>(null);

    const findSimilar = async () => {
        if (!ticker) return;

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE}/api/screener/v2/similar/${ticker}`);
            if (!response.ok) throw new Error('Failed');
            const data = await response.json();
            setTarget(data.target);
            setResults(data.similar || []);
        } catch {
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-obsidian-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden">
                <div className="p-6 border-b border-cream-200">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-obsidian-900">Find Similar Stocks</h3>
                        <button onClick={onClose} className="text-obsidian-400 hover:text-obsidian-600">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    <div className="flex items-center gap-3 mt-4">
                        <input
                            type="text"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="Enter ticker (e.g., TSLA)"
                            className="flex-1 px-4 py-2 border border-cream-200 rounded-lg"
                        />
                        <button
                            onClick={findSimilar}
                            disabled={loading || !ticker}
                            className="px-6 py-2 bg-obsidian-900 text-white font-medium rounded-lg hover:bg-obsidian-800 disabled:opacity-50"
                        >
                            {loading ? 'Searching...' : 'Find'}
                        </button>
                    </div>
                </div>

                <div className="p-6 overflow-y-auto max-h-96">
                    {target && (
                        <div className="mb-4 p-4 bg-electric-50 rounded-xl">
                            <p className="text-sm text-electric-600 mb-1">Target Stock</p>
                            <p className="text-lg font-semibold text-obsidian-900">
                                {target.ticker} - {target.name}
                            </p>
                        </div>
                    )}

                    {results.length > 0 ? (
                        <div className="space-y-2">
                            {results.map((stock, i) => (
                                <div key={stock.ticker} className="flex items-center justify-between p-3 bg-cream-50 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <span className="w-6 h-6 bg-obsidian-900 text-white text-xs flex items-center justify-center rounded">
                                            {i + 1}
                                        </span>
                                        <div>
                                            <p className="font-medium text-obsidian-900">{stock.ticker}</p>
                                            <p className="text-xs text-obsidian-500">{stock.name}</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="font-mono">${stock.price?.toFixed(2)}</p>
                                        <p className={`text-xs ${(stock.change_pct ?? 0) >= 0 ? 'text-success-600' : 'text-coral-600'}`}>
                                            {stock.change_pct?.toFixed(2)}%
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : !loading && ticker && (
                        <p className="text-center text-obsidian-400 py-8">No similar stocks found</p>
                    )}
                </div>
            </div>
        </div>
    );
};

// Main Advanced Screener Component
export const AdvancedScreener: React.FC = () => {
    const [loading, setLoading] = useState(false);
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [clusters, setClusters] = useState<Cluster[]>([]);
    const [sortBy, setSortBy] = useState('market_cap');
    const [sortAsc, setSortAsc] = useState(false);
    const [findAlikeOpen, setFindAlikeOpen] = useState(false);

    // Filters
    const [volumeZScore, setVolumeZScore] = useState<[number, number]>([-3, 3]);
    const [returnZScore, setReturnZScore] = useState<[number, number]>([-3, 3]);
    const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
    const [showAnomalies, setShowAnomalies] = useState(false);
    const [minPiotroski, setMinPiotroski] = useState(0);
    const [minAltman, setMinAltman] = useState(0);

    // Fetch clusters on mount
    useEffect(() => {
        fetch(`${API_BASE}/api/screener/v2/clusters`)
            .then(res => res.json())
            .then(data => setClusters(data.clusters || []))
            .catch(() => { });
    }, []);

    const runScan = useCallback(async () => {
        setLoading(true);

        const filters: Record<string, any> = {};

        if (volumeZScore[0] > -3) filters.volume_zscore = { min: volumeZScore[0] };
        if (volumeZScore[1] < 3) filters.volume_zscore = { ...filters.volume_zscore, max: volumeZScore[1] };
        if (returnZScore[0] > -3) filters.return_zscore = { min: returnZScore[0] };
        if (returnZScore[1] < 3) filters.return_zscore = { ...filters.return_zscore, max: returnZScore[1] };
        if (minPiotroski > 0) filters.piotroski_score = { min: minPiotroski };
        if (minAltman > 0) filters.altman_zscore = { min: minAltman };
        if (selectedCluster !== null) filters.cluster_id = { equals: selectedCluster };
        if (showAnomalies) filters.is_anomaly = { equals: true };

        try {
            const response = await fetch(`${API_BASE}/api/screener/v2/scan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filters,
                    sort_by: sortBy,
                    ascending: sortAsc,
                    limit: 100,
                    include_clustering: true,
                    include_anomalies: true,
                }),
            });

            const data = await response.json();
            setStocks(data.results || []);
        } catch {
            setStocks([]);
        } finally {
            setLoading(false);
        }
    }, [volumeZScore, returnZScore, selectedCluster, showAnomalies, minPiotroski, minAltman, sortBy, sortAsc]);

    const handleSort = (field: string) => {
        if (sortBy === field) {
            setSortAsc(!sortAsc);
        } else {
            setSortBy(field);
            setSortAsc(false);
        }
    };

    return (
        <div className="min-h-screen bg-cream-50">
            {/* Header */}
            <div className="bg-white border-b border-cream-200 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-obsidian-900">Advanced Stock Screener</h1>
                        <p className="text-obsidian-500">ML-powered discovery with statistical filters</p>
                    </div>
                    <button
                        onClick={() => setFindAlikeOpen(true)}
                        className="px-4 py-2 bg-electric-100 text-electric-700 font-medium rounded-lg hover:bg-electric-200 transition-colors"
                    >
                        🔍 Find Alike
                    </button>
                </div>
            </div>

            <div className="flex">
                {/* Left Sidebar - Filters */}
                <div className="w-80 bg-white border-r border-cream-200 p-6 space-y-6 min-h-screen">
                    <div>
                        <h3 className="text-sm font-semibold text-obsidian-900 uppercase tracking-wide mb-4">
                            📊 Z-Score Filters
                        </h3>
                        <div className="space-y-4">
                            <ZScoreSlider label="Volume Z-Score" value={volumeZScore} onChange={setVolumeZScore} />
                            <ZScoreSlider label="Return Z-Score" value={returnZScore} onChange={setReturnZScore} />
                        </div>
                    </div>

                    <div className="border-t border-cream-200 pt-6">
                        <h3 className="text-sm font-semibold text-obsidian-900 uppercase tracking-wide mb-4">
                            🤖 ML Filters
                        </h3>
                        <div className="space-y-4">
                            <ClusterDropdown clusters={clusters} selected={selectedCluster} onChange={setSelectedCluster} />

                            <label className="flex items-center gap-3 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={showAnomalies}
                                    onChange={(e) => setShowAnomalies(e.target.checked)}
                                    className="w-4 h-4 rounded border-cream-300"
                                />
                                <span className="text-sm text-obsidian-600">Show Anomalies Only</span>
                            </label>
                        </div>
                    </div>

                    <div className="border-t border-cream-200 pt-6">
                        <h3 className="text-sm font-semibold text-obsidian-900 uppercase tracking-wide mb-4">
                            🎯 Fundamental Filters
                        </h3>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-obsidian-600 mb-1">Min Piotroski F-Score</label>
                                <select
                                    value={minPiotroski}
                                    onChange={(e) => setMinPiotroski(parseInt(e.target.value))}
                                    className="w-full px-3 py-2 bg-white border border-cream-200 rounded-lg text-sm"
                                >
                                    <option value={0}>Any</option>
                                    <option value={4}>4+ (Moderate)</option>
                                    <option value={7}>7+ (Strong)</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm text-obsidian-600 mb-1">Min Altman Z-Score</label>
                                <select
                                    value={minAltman}
                                    onChange={(e) => setMinAltman(parseFloat(e.target.value))}
                                    className="w-full px-3 py-2 bg-white border border-cream-200 rounded-lg text-sm"
                                >
                                    <option value={0}>Any</option>
                                    <option value={1.81}>1.81+ (Grey Zone)</option>
                                    <option value={2.99}>2.99+ (Safe Zone)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={runScan}
                        disabled={loading}
                        className="w-full py-3 bg-obsidian-900 text-white font-semibold rounded-xl hover:bg-obsidian-800 disabled:opacity-50 transition-all"
                    >
                        {loading ? 'Scanning...' : 'Run Scan'}
                    </button>
                </div>

                {/* Right - Results */}
                <div className="flex-1 p-6">
                    <div className="bg-white rounded-2xl border border-cream-200 overflow-hidden">
                        <div className="p-4 border-b border-cream-200 flex items-center justify-between">
                            <p className="text-sm text-obsidian-500">
                                {stocks.length} stocks found
                            </p>
                        </div>

                        {stocks.length > 0 ? (
                            <ResultsTable
                                stocks={stocks}
                                sortBy={sortBy}
                                sortAsc={sortAsc}
                                onSort={handleSort}
                            />
                        ) : (
                            <div className="py-16 text-center text-obsidian-400">
                                {loading ? (
                                    <div className="flex items-center justify-center gap-2">
                                        <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                        Scanning universe...
                                    </div>
                                ) : (
                                    <>
                                        <p className="text-lg">No stocks to display</p>
                                        <p className="text-sm mt-1">Adjust filters and click "Run Scan"</p>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <FindAlikeModal isOpen={findAlikeOpen} onClose={() => setFindAlikeOpen(false)} />
        </div>
    );
};

export default AdvancedScreener;
