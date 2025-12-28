'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Section } from '@/components/layout';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Stock {
    ticker: string;
    name: string;
    sector: string;
    price: number;
    change_pct: number;
    market_cap: number;
    pe_ratio: number;
    dividend_yield: number;
    rsi: number;
    [key: string]: any;
}

interface Preset {
    id: string;
    name: string;
    description: string;
}

export default function ScreenerPage() {
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [presets, setPresets] = useState<Preset[]>([]);
    const [selectedPreset, setSelectedPreset] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [sortBy, setSortBy] = useState('market_cap');
    const [sortAsc, setSortAsc] = useState(false);

    useEffect(() => {
        fetch(`${API_BASE}/api/screener/presets`)
            .then(res => res.json())
            .then(data => setPresets(data.presets || []))
            .catch(console.error);
    }, []);

    const runScreen = async (preset?: string) => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/screener/scan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    preset: preset || selectedPreset,
                    sort_by: sortBy,
                    ascending: sortAsc,
                    limit: 50
                }),
            });
            const data = await res.json();
            setStocks(data.results || []);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const formatMarketCap = (cap: number) => {
        if (!cap) return 'N/A';
        if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
        if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
        if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
        return `$${cap.toLocaleString()}`;
    };

    return (
        <>
            <Section spacing="md" background="gradient">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div>
                        <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                            Stock Screener
                        </h1>
                        <p className="text-body-md text-neutral-600 max-w-2xl">
                            Find stocks matching specific criteria using preset screens or custom filters.
                        </p>
                    </div>
                </div>
            </Section>

            <Section spacing="md" background="default">
                {/* Preset Buttons */}
                <div className="mb-6">
                    <h2 className="font-heading font-semibold text-heading-sm text-navy-900 mb-3">
                        Quick Screens
                    </h2>
                    <div className="flex flex-wrap gap-2">
                        {presets.map(preset => (
                            <button
                                key={preset.id}
                                onClick={() => {
                                    setSelectedPreset(preset.id);
                                    runScreen(preset.id);
                                }}
                                className={`px-4 py-2 rounded-lg text-body-sm font-medium transition-colors ${selectedPreset === preset.id
                                        ? 'bg-terra-500 text-white'
                                        : 'bg-cream-100 text-navy-700 hover:bg-cream-200'
                                    }`}
                            >
                                {preset.name}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Results Table */}
                {loading ? (
                    <div className="text-center py-12">
                        <div className="w-10 h-10 border-4 border-navy-200 border-t-navy-600 rounded-full animate-spin mx-auto mb-4" />
                        <p className="text-neutral-600">Scanning stocks...</p>
                    </div>
                ) : stocks.length > 0 ? (
                    <div className="bg-white rounded-xl border border-border-light overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead className="bg-cream-50 border-b border-border-light">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-caption font-semibold text-navy-900">Ticker</th>
                                        <th className="px-4 py-3 text-left text-caption font-semibold text-navy-900">Sector</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Price</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Change</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Market Cap</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">P/E</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">Div Yield</th>
                                        <th className="px-4 py-3 text-right text-caption font-semibold text-navy-900">RSI</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stocks.map((stock, i) => (
                                        <tr key={stock.ticker} className={`border-b border-border-light ${i % 2 === 0 ? 'bg-white' : 'bg-cream-50/50'}`}>
                                            <td className="px-4 py-3">
                                                <div className="font-semibold text-navy-900">{stock.ticker}</div>
                                                <div className="text-caption text-neutral-500 truncate max-w-[150px]">{stock.name}</div>
                                            </td>
                                            <td className="px-4 py-3 text-body-sm text-neutral-600">{stock.sector}</td>
                                            <td className="px-4 py-3 text-right font-medium">${stock.price?.toFixed(2)}</td>
                                            <td className={`px-4 py-3 text-right font-medium ${(stock.change_pct || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                                {(stock.change_pct || 0) >= 0 ? '+' : ''}{stock.change_pct?.toFixed(2)}%
                                            </td>
                                            <td className="px-4 py-3 text-right text-body-sm">{formatMarketCap(stock.market_cap)}</td>
                                            <td className="px-4 py-3 text-right text-body-sm">{stock.pe_ratio?.toFixed(1) || 'N/A'}</td>
                                            <td className="px-4 py-3 text-right text-body-sm">{stock.dividend_yield?.toFixed(2) || '0'}%</td>
                                            <td className={`px-4 py-3 text-right font-medium ${(stock.rsi || 50) < 30 ? 'text-success-600' :
                                                    (stock.rsi || 50) > 70 ? 'text-error-600' : 'text-neutral-600'
                                                }`}>
                                                {stock.rsi?.toFixed(0) || 'N/A'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ) : (
                    <div className="text-center py-16 bg-cream-50 rounded-xl">
                        <svg className="w-16 h-16 text-neutral-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                        <p className="text-neutral-600 font-medium mb-2">Select a Screen</p>
                        <p className="text-neutral-500 text-body-sm">Click one of the preset screens above to find matching stocks</p>
                    </div>
                )}
            </Section>
        </>
    );
}
