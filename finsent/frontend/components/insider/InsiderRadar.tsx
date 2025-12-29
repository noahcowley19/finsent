'use client';

// =============================================================================
// INSIDER RADAR V2 - Atmospheric Glass Theme (Cream/Obsidian)
// =============================================================================
// Reskinned from Brutalist Dark to match the Caveray design system
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

// Types
interface Trade {
    ticker: string;
    insider_name: string;
    insider_title: string;
    transaction_type: 'Buy' | 'Sell';
    shares: number;
    value: number;
    date: string;
    conviction_score: number;
    conviction_color: string;
    insider_rank: number;
}

interface CongressTrade {
    politician: string;
    ticker: string;
    transaction_type: string;
    amount: string;
    date: string;
    chamber: string;
    party: string;
    conflict: {
        is_conflict: boolean;
        committee: string;
        severity: string;
        label: string;
        color: string;
    };
}

interface Cluster {
    cluster_detected: boolean;
    buy_count: number;
    sell_count: number;
    signal: string;
    color: string;
}

// Conviction Gauge Component - Cream Theme
const ConvictionGauge: React.FC<{ score: number; color: string }> = ({ score, color }) => {
    const radius = 28;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    return (
        <div className="relative w-16 h-16">
            <svg className="w-16 h-16 -rotate-90">
                <circle
                    cx="32"
                    cy="32"
                    r={radius}
                    fill="none"
                    stroke="#E7E5E4"
                    strokeWidth="6"
                />
                <circle
                    cx="32"
                    cy="32"
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="6"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    className="transition-all duration-500"
                />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-obsidian-900">{score}</span>
            </div>
        </div>
    );
};

// Trade Card Component - Cream Theme
const TradeCard: React.FC<{ trade: Trade }> = ({ trade }) => {
    const isBuy = trade.transaction_type === 'Buy';

    return (
        <div className="bg-white/50 backdrop-blur-md border border-cream-200 rounded-2xl p-4 flex items-center gap-4 hover:shadow-md transition-shadow">
            <ConvictionGauge score={trade.conviction_score} color={trade.conviction_color} />

            <div className="flex-1">
                <div className="flex items-center gap-2">
                    <span className={`text-lg font-bold ${isBuy ? 'text-success-600' : 'text-coral-600'}`}>
                        {trade.ticker}
                    </span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${isBuy ? 'bg-success-100 text-success-700' : 'bg-coral-100 text-coral-700'}`}>
                        {trade.transaction_type.toUpperCase()}
                    </span>
                </div>

                <div className="text-sm text-obsidian-600 mt-1">
                    {trade.insider_name} • {trade.insider_title}
                </div>

                <div className="flex items-center gap-4 mt-2 text-xs text-obsidian-500">
                    <span>${(trade.value / 1e6).toFixed(1)}M</span>
                    <span>{trade.shares.toLocaleString()} shares</span>
                    <span>{trade.date}</span>
                </div>
            </div>

            <div className="text-right">
                <div className="text-[10px] text-obsidian-500 uppercase">Rank</div>
                <div className="text-lg font-bold text-obsidian-900">{trade.insider_rank}/5</div>
            </div>
        </div>
    );
};

// Congress Trade Card Component - Cream Theme
const CongressCard: React.FC<{ trade: CongressTrade }> = ({ trade }) => {
    const isBuy = trade.transaction_type === 'Buy';
    const hasConflict = trade.conflict?.is_conflict;

    return (
        <div className={`bg-white/50 backdrop-blur-md border rounded-2xl p-4 hover:shadow-md transition-shadow ${hasConflict ? 'border-amber-400' : 'border-cream-200'}`}>
            {hasConflict && (
                <div
                    className="text-[10px] font-medium mb-2 px-2 py-1 rounded-full inline-block"
                    style={{ backgroundColor: trade.conflict.color + '20', color: trade.conflict.color }}
                >
                    {trade.conflict.label} • {trade.conflict.committee}
                </div>
            )}

            <div className="flex items-center justify-between">
                <div>
                    <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${trade.party === 'D' ? 'bg-blue-500' : 'bg-red-500'}`} />
                        <span className="text-obsidian-900 font-semibold">{trade.politician}</span>
                        <span className="text-[10px] text-obsidian-500">{trade.chamber}</span>
                    </div>

                    <div className="flex items-center gap-2 mt-2">
                        <span className={`text-lg font-bold ${isBuy ? 'text-success-600' : 'text-coral-600'}`}>
                            {trade.ticker}
                        </span>
                        <span className="text-sm text-obsidian-600">{trade.amount}</span>
                    </div>
                </div>

                <div className="text-right">
                    <div className={`text-sm font-semibold ${isBuy ? 'text-success-600' : 'text-coral-600'}`}>
                        {trade.transaction_type}
                    </div>
                    <div className="text-xs text-obsidian-500 mt-1">{trade.date}</div>
                </div>
            </div>
        </div>
    );
};

// Cluster Badge Component - Cream Theme
const ClusterBadge: React.FC<{ cluster: Cluster }> = ({ cluster }) => {
    if (!cluster) return null;

    const getBgColor = () => {
        if (cluster.signal.includes('STRONG BUY')) return 'bg-success-100 text-success-700 border-success-200';
        if (cluster.signal.includes('BUY')) return 'bg-success-50 text-success-600 border-success-100';
        if (cluster.signal.includes('SELL')) return 'bg-coral-100 text-coral-700 border-coral-200';
        return 'bg-cream-100 text-obsidian-600 border-cream-200';
    };

    return (
        <div className={`px-4 py-3 rounded-2xl border ${getBgColor()}`}>
            <div className="flex items-center gap-3">
                <span className="text-lg">
                    {cluster.cluster_detected ? '🔥' : '📊'}
                </span>
                <div>
                    <div className="font-semibold">{cluster.signal}</div>
                    <div className="text-[10px] opacity-70">
                        {cluster.buy_count} buyers • {cluster.sell_count} sellers (7d)
                    </div>
                </div>
            </div>
        </div>
    );
};

// Main Insider Radar Component
export const InsiderRadar: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'corporate' | 'congress'>('corporate');
    const [ticker, setTicker] = useState('AAPL');
    const [loading, setLoading] = useState(false);
    const [trades, setTrades] = useState<Trade[]>([]);
    const [cluster, setCluster] = useState<Cluster | null>(null);
    const [congressTrades, setCongressTrades] = useState<CongressTrade[]>([]);
    const [conflictsCount, setConflictsCount] = useState(0);

    const fetchCorporateTrades = useCallback(async () => {
        if (!ticker) return;

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE}/api/insider/v2/trades/${ticker}`);
            const data = await response.json();
            setTrades(data.trades || []);
            setCluster(data.cluster || null);
        } catch (error) {
            console.error('Failed to fetch trades:', error);
            setTrades([]);
        } finally {
            setLoading(false);
        }
    }, [ticker]);

    const fetchCongressTrades = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_BASE}/api/insider/v2/congress?days=30`);
            const data = await response.json();
            setCongressTrades(data.trades || []);
            setConflictsCount(data.conflicts_count || 0);
        } catch (error) {
            console.error('Failed to fetch congress trades:', error);
            setCongressTrades([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (activeTab === 'corporate') {
            fetchCorporateTrades();
        } else {
            fetchCongressTrades();
        }
    }, [activeTab, fetchCorporateTrades, fetchCongressTrades]);

    return (
        <div className="min-h-screen bg-cream-50">
            {/* Header */}
            <div className="border-b border-cream-200 bg-white/50 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div>
                            <h1 className="text-2xl sm:text-3xl font-bold text-obsidian-900 tracking-tight">
                                Insider Radar
                            </h1>
                            <p className="text-sm text-obsidian-500 mt-1">
                                Smart Money Tracking • ML-Powered Signals
                            </p>
                        </div>

                        {/* Tab Switcher */}
                        <div className="flex items-center gap-2 p-1 bg-cream-100 rounded-xl">
                            <button
                                onClick={() => setActiveTab('corporate')}
                                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${activeTab === 'corporate'
                                    ? 'bg-electric-500 text-white shadow-sm'
                                    : 'text-obsidian-600 hover:text-obsidian-900'
                                    }`}
                            >
                                Corporate
                            </button>
                            <button
                                onClick={() => setActiveTab('congress')}
                                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${activeTab === 'congress'
                                    ? 'bg-amber-500 text-white shadow-sm'
                                    : 'text-obsidian-600 hover:text-obsidian-900'
                                    }`}
                            >
                                Congress {conflictsCount > 0 && <span className="ml-1">({conflictsCount} ⚠️)</span>}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {activeTab === 'corporate' ? (
                    <div className="space-y-6">
                        {/* Search Bar */}
                        <div className="flex items-center gap-4">
                            <input
                                type="text"
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                                placeholder="Enter ticker..."
                                className="flex-1 px-4 py-3 bg-white/50 backdrop-blur-md border border-cream-200 rounded-xl text-obsidian-900 placeholder-obsidian-400 focus:border-electric-500 focus:ring-2 focus:ring-electric-200 outline-none transition-all"
                            />
                            <button
                                onClick={fetchCorporateTrades}
                                disabled={loading}
                                className="px-6 py-3 bg-electric-500 text-white font-semibold rounded-xl hover:bg-electric-600 disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Scanning...' : 'Scan'}
                            </button>
                        </div>

                        {/* Cluster Signal */}
                        {cluster && <ClusterBadge cluster={cluster} />}

                        {/* Trades List */}
                        <div className="space-y-3">
                            {trades.length > 0 ? (
                                trades.map((trade, i) => <TradeCard key={i} trade={trade} />)
                            ) : !loading && (
                                <div className="text-center py-12 bg-white/50 backdrop-blur-md border border-cream-200 rounded-2xl">
                                    <p className="text-lg text-obsidian-600">No smart money trades found</p>
                                    <p className="text-sm text-obsidian-500 mt-1">Try a different ticker</p>
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* Conflict Counter */}
                        {conflictsCount > 0 && (
                            <div className="bg-amber-50 border border-amber-200 p-4 rounded-2xl">
                                <div className="text-amber-700 font-semibold">
                                    ⚠️ {conflictsCount} potential committee conflicts detected
                                </div>
                                <div className="text-xs text-amber-600 mt-1">
                                    Politicians trading stocks within their committee jurisdiction
                                </div>
                            </div>
                        )}

                        {/* Congress Trades List */}
                        <div className="space-y-3">
                            {congressTrades.length > 0 ? (
                                congressTrades.map((trade, i) => <CongressCard key={i} trade={trade} />)
                            ) : !loading && (
                                <div className="text-center py-12 bg-white/50 backdrop-blur-md border border-cream-200 rounded-2xl">
                                    <p className="text-lg text-obsidian-600">Loading congressional trades...</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="border-t border-cream-200 bg-white/30 backdrop-blur-sm">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 text-center">
                    <span className="text-xs text-obsidian-400">
                        Data Sources: SEC Form 4 • House Stock Watcher • Senate Stock Watcher
                    </span>
                </div>
            </div>
        </div>
    );
};

export default InsiderRadar;
