'use client';

import React, { useState, useEffect } from 'react';
import { Section } from '@/components/layout';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Mover {
    ticker: string;
    name: string;
    sector: string;
    price: number;
    change: number;
    change_pct: number;
    volume: number;
    volume_ratio?: number;
}

interface Summary {
    top_gainers: Mover[];
    top_losers: Mover[];
    most_active: Mover[];
    unusual_volume: Mover[];
    market_breadth: {
        advances: number;
        declines: number;
        unchanged: number;
        advance_decline_ratio: number;
    };
}

export default function MoversPage() {
    const [summary, setSummary] = useState<Summary | null>(null);
    const [activeTab, setActiveTab] = useState<'gainers' | 'losers' | 'active' | 'volume'>('gainers');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_BASE}/api/movers/summary`)
            .then(res => res.json())
            .then(data => setSummary(data))
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    const formatVolume = (vol: number) => {
        if (!vol) return 'N/A';
        if (vol >= 1e9) return `${(vol / 1e9).toFixed(1)}B`;
        if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`;
        if (vol >= 1e3) return `${(vol / 1e3).toFixed(0)}K`;
        return vol.toString();
    };

    const getActiveData = () => {
        if (!summary) return [];
        switch (activeTab) {
            case 'gainers': return summary.top_gainers;
            case 'losers': return summary.top_losers;
            case 'active': return summary.most_active;
            case 'volume': return summary.unusual_volume;
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="w-10 h-10 border-4 border-navy-200 border-t-navy-600 rounded-full animate-spin" />
            </div>
        );
    }

    return (
        <>
            <Section spacing="md" background="gradient">
                <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                    Market Movers
                </h1>
                <p className="text-body-md text-neutral-600">
                    Today's biggest gainers, losers, and most active stocks.
                </p>
            </Section>

            {/* Market Breadth */}
            {summary?.market_breadth && (
                <Section spacing="sm" background="secondary">
                    <div className="flex flex-wrap justify-center gap-6">
                        <div className="text-center">
                            <p className="text-display-xs font-semibold text-success-600">{summary.market_breadth.advances}</p>
                            <p className="text-caption text-neutral-500">Advancing</p>
                        </div>
                        <div className="text-center">
                            <p className="text-display-xs font-semibold text-error-600">{summary.market_breadth.declines}</p>
                            <p className="text-caption text-neutral-500">Declining</p>
                        </div>
                        <div className="text-center">
                            <p className="text-display-xs font-semibold text-neutral-600">{summary.market_breadth.unchanged}</p>
                            <p className="text-caption text-neutral-500">Unchanged</p>
                        </div>
                        <div className="text-center">
                            <p className={`text-display-xs font-semibold ${(summary.market_breadth.advance_decline_ratio || 0) >= 1 ? 'text-success-600' : 'text-error-600'}`}>
                                {summary.market_breadth.advance_decline_ratio?.toFixed(2)}
                            </p>
                            <p className="text-caption text-neutral-500">A/D Ratio</p>
                        </div>
                    </div>
                </Section>
            )}

            <Section spacing="md" background="default">
                {/* Tabs */}
                <div className="flex border-b border-border-light mb-6">
                    {[
                        { id: 'gainers', label: 'Top Gainers', icon: '📈' },
                        { id: 'losers', label: 'Top Losers', icon: '📉' },
                        { id: 'active', label: 'Most Active', icon: '🔥' },
                        { id: 'volume', label: 'Unusual Volume', icon: '📊' },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`px-4 py-3 text-body-sm font-medium border-b-2 -mb-px transition-colors ${activeTab === tab.id
                                    ? 'border-terra-500 text-terra-600'
                                    : 'border-transparent text-neutral-500 hover:text-neutral-700'
                                }`}
                        >
                            {tab.icon} {tab.label}
                        </button>
                    ))}
                </div>

                {/* Data Table */}
                <div className="bg-white rounded-xl border border-border-light overflow-hidden">
                    <table className="w-full">
                        <thead className="bg-cream-50 border-b border-border-light">
                            <tr>
                                <th className="px-4 py-3 text-left text-caption font-semibold text-navy-900">Stock</th>
                                <th className="px-4 py-3 text-left text-caption font-semibold text-navy-900">Sector</th>
                                <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Price</th>
                                <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Change</th>
                                <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Volume</th>
                                {activeTab === 'volume' && (
                                    <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Vol Ratio</th>
                                )}
                            </tr>
                        </thead>
                        <tbody>
                            {getActiveData().map((stock, i) => (
                                <tr key={stock.ticker} className={`border-b border-border-light ${i % 2 === 0 ? 'bg-white' : 'bg-cream-50/50'} hover:bg-terra-50/30`}>
                                    <td className="px-4 py-3">
                                        <div className="font-semibold text-navy-900">{stock.ticker}</div>
                                        <div className="text-caption text-neutral-500 truncate max-w-[150px]">{stock.name}</div>
                                    </td>
                                    <td className="px-4 py-3 text-body-sm text-neutral-600">{stock.sector}</td>
                                    <td className="px-4 py-3 text-right font-medium">${stock.price?.toFixed(2)}</td>
                                    <td className="px-4 py-3 text-right">
                                        <span className={`font-medium ${(stock.change_pct || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                            {(stock.change_pct || 0) >= 0 ? '+' : ''}{stock.change_pct?.toFixed(2)}%
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-right text-body-sm">{formatVolume(stock.volume)}</td>
                                    {activeTab === 'volume' && (
                                        <td className="px-4 py-3 text-right">
                                            <span className="px-2 py-1 bg-warning-100 text-warning-700 rounded text-caption font-medium">
                                                {stock.volume_ratio?.toFixed(1)}x
                                            </span>
                                        </td>
                                    )}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </Section>
        </>
    );
}
