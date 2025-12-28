'use client';

import React, { useState, useEffect } from 'react';
import { Section } from '@/components/layout';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Sector {
    sector: string;
    etf: string;
    price: number;
    change_1d: number;
    change_1w?: number;
    change_1m?: number;
    color: string;
}

interface RotationData {
    sectors: Array<{
        sector: string;
        return_1w: number;
        return_1m: number;
        return_3m: number;
        momentum_score: number;
    }>;
    leaders: string[];
    laggards: string[];
    market_phase: string;
    phase_description: string;
}

export default function SectorsPage() {
    const [heatmapData, setHeatmapData] = useState<Sector[]>([]);
    const [rotationData, setRotationData] = useState<RotationData | null>(null);
    const [loading, setLoading] = useState(true);
    const [period, setPeriod] = useState<'1d' | '1w' | '1m'>('1d');

    useEffect(() => {
        Promise.all([
            fetch(`${API_BASE}/api/sectors/heatmap`).then(r => r.json()),
            fetch(`${API_BASE}/api/sectors/rotation`).then(r => r.json()),
        ])
            .then(([heatmap, rotation]) => {
                setHeatmapData(heatmap.sectors || []);
                setRotationData(rotation);
            })
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    const getChangeForPeriod = (sector: Sector) => {
        switch (period) {
            case '1d': return sector.change_1d;
            case '1w': return sector.change_1w;
            case '1m': return sector.change_1m;
            default: return sector.change_1d;
        }
    };

    const getHeatmapColor = (change: number) => {
        if (change >= 2) return 'bg-success-600 text-white';
        if (change >= 1) return 'bg-success-500 text-white';
        if (change >= 0.5) return 'bg-success-400 text-white';
        if (change >= 0) return 'bg-success-200 text-success-900';
        if (change >= -0.5) return 'bg-error-200 text-error-900';
        if (change >= -1) return 'bg-error-400 text-white';
        if (change >= -2) return 'bg-error-500 text-white';
        return 'bg-error-600 text-white';
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
                    Sector Heatmap
                </h1>
                <p className="text-body-md text-neutral-600">
                    Visualize sector performance and identify rotation patterns.
                </p>
            </Section>

            {/* Market Phase */}
            {rotationData && (
                <Section spacing="sm" background="secondary">
                    <div className="flex flex-col md:flex-row items-center justify-center gap-6">
                        <div className="text-center">
                            <span className={`inline-block px-4 py-2 rounded-full font-semibold ${rotationData.market_phase === 'Risk-On' ? 'bg-success-100 text-success-700' :
                                    rotationData.market_phase === 'Risk-Off' ? 'bg-error-100 text-error-700' :
                                        'bg-neutral-100 text-neutral-700'
                                }`}>
                                {rotationData.market_phase}
                            </span>
                            <p className="text-body-sm text-neutral-600 mt-2">{rotationData.phase_description}</p>
                        </div>
                        <div className="flex gap-4">
                            <div>
                                <p className="text-caption text-neutral-500">Leading</p>
                                <div className="flex gap-1">
                                    {rotationData.leaders.map(s => (
                                        <span key={s} className="px-2 py-1 bg-success-100 text-success-700 rounded text-caption font-medium">
                                            {s.split(' ')[0]}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <p className="text-caption text-neutral-500">Lagging</p>
                                <div className="flex gap-1">
                                    {rotationData.laggards.map(s => (
                                        <span key={s} className="px-2 py-1 bg-error-100 text-error-700 rounded text-caption font-medium">
                                            {s.split(' ')[0]}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </Section>
            )}

            <Section spacing="md" background="default">
                {/* Period Selector */}
                <div className="flex justify-center mb-6">
                    <div className="inline-flex bg-cream-100 rounded-lg p-1">
                        {(['1d', '1w', '1m'] as const).map(p => (
                            <button
                                key={p}
                                onClick={() => setPeriod(p)}
                                className={`px-4 py-2 rounded-md text-body-sm font-medium transition-colors ${period === p ? 'bg-white text-navy-900 shadow-sm' : 'text-neutral-600 hover:text-navy-900'
                                    }`}
                            >
                                {p === '1d' ? 'Today' : p === '1w' ? '1 Week' : '1 Month'}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Heatmap Grid */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {heatmapData.map(sector => {
                        const change = getChangeForPeriod(sector) || 0;
                        return (
                            <div
                                key={sector.sector}
                                className={`p-6 rounded-xl ${getHeatmapColor(change)} transition-all hover:scale-105 cursor-pointer`}
                            >
                                <p className="font-semibold truncate">{sector.sector}</p>
                                <p className="text-display-xs font-bold mt-2">
                                    {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                                </p>
                                <p className="text-caption opacity-80 mt-1">{sector.etf}</p>
                            </div>
                        );
                    })}
                </div>

                {/* Rotation Table */}
                {rotationData && (
                    <div className="mt-8">
                        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                            Sector Momentum
                        </h2>
                        <div className="bg-white rounded-xl border border-border-light overflow-hidden">
                            <table className="w-full">
                                <thead className="bg-cream-50 border-b border-border-light">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-caption font-semibold text-navy-900">Sector</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">1 Week</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">1 Month</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">3 Month</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Momentum</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rotationData.sectors.map((s, i) => (
                                        <tr key={s.sector} className={`border-b border-border-light ${i % 2 === 0 ? 'bg-white' : 'bg-cream-50/50'}`}>
                                            <td className="px-4 py-3 font-medium text-navy-900">{s.sector}</td>
                                            <td className={`px-4 py-3 text-right font-medium ${(s.return_1w || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                                {(s.return_1w || 0) >= 0 ? '+' : ''}{s.return_1w?.toFixed(2)}%
                                            </td>
                                            <td className={`px-4 py-3 text-right font-medium ${(s.return_1m || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                                {(s.return_1m || 0) >= 0 ? '+' : ''}{s.return_1m?.toFixed(2)}%
                                            </td>
                                            <td className={`px-4 py-3 text-right font-medium ${(s.return_3m || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                                {(s.return_3m || 0) >= 0 ? '+' : ''}{s.return_3m?.toFixed(2)}%
                                            </td>
                                            <td className="px-4 py-3 text-right">
                                                <span className={`px-2 py-1 rounded text-caption font-medium ${s.momentum_score >= 5 ? 'bg-success-100 text-success-700' :
                                                        s.momentum_score >= 0 ? 'bg-warning-100 text-warning-700' :
                                                            'bg-error-100 text-error-700'
                                                    }`}>
                                                    {s.momentum_score?.toFixed(1)}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </Section>
        </>
    );
}
