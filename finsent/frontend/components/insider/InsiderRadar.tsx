'use client';

// =============================================================================
// INSIDER RADAR V2 - Smart Money Tracking Dashboard
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

// Conviction Gauge Component
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
                    stroke="#2a2a2a"
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
                <span className="text-sm font-mono font-bold text-white">{score}</span>
            </div>
        </div>
    );
};

// Trade Card Component
const TradeCard: React.FC<{ trade: Trade }> = ({ trade }) => {
    const isBuy = trade.transaction_type === 'Buy';

    return (
        <div className="bg-[#121212] border border-[#2a2a2a] p-4 flex items-center gap-4">
            <ConvictionGauge score={trade.conviction_score} color={trade.conviction_color} />

            <div className="flex-1">
                <div className="flex items-center gap-2">
                    <span className={`text-lg font-mono font-bold ${isBuy ? 'text-[#00ff88]' : 'text-[#ff0066]'}`}>
                        {trade.ticker}
                    </span>
                    <span className={`text-[10px] px-2 py-0.5 rounded ${isBuy ? 'bg-[#00ff8820] text-[#00ff88]' : 'bg-[#ff006620] text-[#ff0066]'}`}>
                        {trade.transaction_type.toUpperCase()}
                    </span>
                </div>

                <div className="text-sm text-[#888] mt-1">
                    {trade.insider_name} • {trade.insider_title}
                </div>

                <div className="flex items-center gap-4 mt-2 text-xs text-[#666]">
                    <span>${(trade.value / 1e6).toFixed(1)}M</span>
                    <span>{trade.shares.toLocaleString()} shares</span>
                    <span>{trade.date}</span>
                </div>
            </div>

            <div className="text-right">
                <div className="text-[10px] text-[#666] uppercase">Rank</div>
                <div className="text-lg font-mono text-white">{trade.insider_rank}/5</div>
            </div>
        </div>
    );
};

// Congress Trade Card Component
const CongressCard: React.FC<{ trade: CongressTrade }> = ({ trade }) => {
    const isBuy = trade.transaction_type === 'Buy';
    const hasConflict = trade.conflict?.is_conflict;

    return (
        <div className={`bg-[#121212] border p-4 ${hasConflict ? 'border-[#ff6600]' : 'border-[#2a2a2a]'}`}>
            {hasConflict && (
                <div
                    className="text-[10px] font-mono mb-2 px-2 py-1 rounded inline-block"
                    style={{ backgroundColor: trade.conflict.color + '20', color: trade.conflict.color }}
                >
                    {trade.conflict.label} • {trade.conflict.committee}
                </div>
            )}

            <div className="flex items-center justify-between">
                <div>
                    <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${trade.party === 'D' ? 'bg-blue-500' : 'bg-red-500'}`} />
                        <span className="text-white font-medium">{trade.politician}</span>
                        <span className="text-[10px] text-[#666]">{trade.chamber}</span>
                    </div>

                    <div className="flex items-center gap-2 mt-2">
                        <span className={`text-lg font-mono ${isBuy ? 'text-[#00ff88]' : 'text-[#ff0066]'}`}>
                            {trade.ticker}
                        </span>
                        <span className="text-sm text-[#888]">{trade.amount}</span>
                    </div>
                </div>

                <div className="text-right">
                    <div className={`text-sm font-mono ${isBuy ? 'text-[#00ff88]' : 'text-[#ff0066]'}`}>
                        {trade.transaction_type}
                    </div>
                    <div className="text-xs text-[#666] mt-1">{trade.date}</div>
                </div>
            </div>
        </div>
    );
};

// Cluster Badge Component
const ClusterBadge: React.FC<{ cluster: Cluster }> = ({ cluster }) => {
    if (!cluster) return null;

    return (
        <div
            className="px-4 py-2 rounded text-sm font-mono"
            style={{ backgroundColor: cluster.color + '20', color: cluster.color }}
        >
            <div className="flex items-center gap-3">
                <span className="text-lg">
                    {cluster.cluster_detected ? '🔥' : '📊'}
                </span>
                <div>
                    <div className="font-bold">{cluster.signal}</div>
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
        <div className="min-h-screen bg-[#0a0a0a] text-white">
            {/* Header */}
            <div className="border-b border-[#2a2a2a] px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-mono font-bold tracking-tight">
                            INSIDER RADAR V2
                        </h1>
                        <p className="text-xs text-[#666] mt-1 font-mono">
                            SMART MONEY TRACKING • ML-POWERED SIGNALS
                        </p>
                    </div>

                    {/* Tab Switcher */}
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setActiveTab('corporate')}
                            className={`px-4 py-2 text-xs font-mono transition-colors ${activeTab === 'corporate'
                                    ? 'bg-[#00ff88] text-black'
                                    : 'bg-[#121212] text-[#888] hover:text-white'
                                }`}
                        >
                            CORPORATE
                        </button>
                        <button
                            onClick={() => setActiveTab('congress')}
                            className={`px-4 py-2 text-xs font-mono transition-colors ${activeTab === 'congress'
                                    ? 'bg-[#ff6600] text-black'
                                    : 'bg-[#121212] text-[#888] hover:text-white'
                                }`}
                        >
                            CONGRESS {conflictsCount > 0 && `(${conflictsCount} ⚠️)`}
                        </button>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="p-4">
                {activeTab === 'corporate' ? (
                    <div className="space-y-4">
                        {/* Search Bar */}
                        <div className="flex items-center gap-4">
                            <input
                                type="text"
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                                placeholder="Enter ticker..."
                                className="flex-1 px-4 py-3 bg-[#121212] border border-[#2a2a2a] text-white font-mono focus:border-[#00ff88] outline-none"
                            />
                            <button
                                onClick={fetchCorporateTrades}
                                disabled={loading}
                                className="px-6 py-3 bg-[#00ff88] text-black font-mono font-bold hover:bg-[#00cc6a] disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'SCANNING...' : 'SCAN'}
                            </button>
                        </div>

                        {/* Cluster Signal */}
                        {cluster && <ClusterBadge cluster={cluster} />}

                        {/* Trades List */}
                        <div className="space-y-2">
                            {trades.length > 0 ? (
                                trades.map((trade, i) => <TradeCard key={i} trade={trade} />)
                            ) : !loading && (
                                <div className="text-center py-12 text-[#666]">
                                    <p className="text-lg">No smart money trades found</p>
                                    <p className="text-sm mt-1">Try a different ticker</p>
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {/* Conflict Counter */}
                        {conflictsCount > 0 && (
                            <div className="bg-[#ff660020] border border-[#ff6600] p-4 rounded">
                                <div className="text-[#ff6600] font-mono">
                                    ⚠️ {conflictsCount} potential committee conflicts detected
                                </div>
                                <div className="text-xs text-[#888] mt-1">
                                    Politicians trading stocks within their committee jurisdiction
                                </div>
                            </div>
                        )}

                        {/* Congress Trades List */}
                        <div className="space-y-2">
                            {congressTrades.length > 0 ? (
                                congressTrades.map((trade, i) => <CongressCard key={i} trade={trade} />)
                            ) : !loading && (
                                <div className="text-center py-12 text-[#666]">
                                    <p className="text-lg">Loading congressional trades...</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="border-t border-[#2a2a2a] px-6 py-3 text-center">
                <span className="text-[10px] text-[#444] font-mono">
                    DATA SOURCES: SEC FORM 4 • HOUSE STOCK WATCHER • SENATE STOCK WATCHER
                </span>
            </div>
        </div>
    );
};

export default InsiderRadar;
