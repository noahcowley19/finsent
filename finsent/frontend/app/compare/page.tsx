'use client';

import React, { useState } from 'react';
import { Section } from '@/components/layout';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Stock {
    ticker: string;
    name: string;
    sector: string;
    price: number;
    market_cap: number;
    pe_ratio: number;
    forward_pe: number;
    pb_ratio: number;
    profit_margin: number;
    roe: number;
    revenue_growth: number;
    dividend_yield: number;
    beta: number;
    return_1m: number;
    return_3m: number;
    return_1y: number;
    volatility: number;
    sharpe_ratio: number;
    [key: string]: any;
}

interface CompareResult {
    stocks: Stock[];
    winners: Record<string, Record<string, string>>;
    win_counts: Record<string, number>;
    overall_winner: string;
}

export default function ComparePage() {
    const [tickers, setTickers] = useState<string[]>(['', '']);
    const [result, setResult] = useState<CompareResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const updateTicker = (index: number, value: string) => {
        const newTickers = [...tickers];
        newTickers[index] = value.toUpperCase();
        setTickers(newTickers);
    };

    const addTicker = () => {
        if (tickers.length < 5) {
            setTickers([...tickers, '']);
        }
    };

    const removeTicker = (index: number) => {
        if (tickers.length > 2) {
            setTickers(tickers.filter((_, i) => i !== index));
        }
    };

    const compare = async () => {
        const validTickers = tickers.filter(t => t.trim());
        if (validTickers.length < 2) {
            setError('Enter at least 2 tickers');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const res = await fetch(`${API_BASE}/api/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers: validTickers }),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            setResult(data);
        } catch (err: any) {
            setError(err.message || 'Comparison failed');
        } finally {
            setLoading(false);
        }
    };

    const formatValue = (key: string, value: any): string => {
        if (value === null || value === undefined) return 'N/A';
        if (key === 'market_cap') {
            if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
            if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
            return `$${(value / 1e6).toFixed(0)}M`;
        }
        if (['profit_margin', 'roe', 'revenue_growth', 'dividend_yield', 'return_1m', 'return_3m', 'return_1y', 'volatility'].includes(key)) {
            return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
        }
        if (typeof value === 'number') return value.toFixed(2);
        return String(value);
    };

    const isWinner = (ticker: string, category: string, metric: string): boolean => {
        return result?.winners?.[category]?.[metric] === ticker;
    };

    const metrics = [
        { key: 'pe_ratio', label: 'P/E Ratio', category: 'valuation' },
        { key: 'forward_pe', label: 'Forward P/E', category: 'valuation' },
        { key: 'pb_ratio', label: 'P/B Ratio', category: 'valuation' },
        { key: 'profit_margin', label: 'Profit Margin', category: 'profitability' },
        { key: 'roe', label: 'ROE', category: 'profitability' },
        { key: 'revenue_growth', label: 'Revenue Growth', category: 'growth' },
        { key: 'dividend_yield', label: 'Dividend Yield', category: 'dividends' },
        { key: 'return_1m', label: '1 Month Return', category: 'performance' },
        { key: 'return_3m', label: '3 Month Return', category: 'performance' },
        { key: 'return_1y', label: '1 Year Return', category: 'performance' },
        { key: 'volatility', label: 'Volatility', category: 'risk' },
        { key: 'beta', label: 'Beta', category: 'risk' },
    ];

    return (
        <>
            <Section spacing="md" background="gradient">
                <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                    Stock Comparison
                </h1>
                <p className="text-body-md text-neutral-600 mb-6">
                    Compare up to 5 stocks side by side across valuation, performance, and risk metrics.
                </p>

                <div className="flex flex-wrap gap-3 items-end">
                    {tickers.map((ticker, i) => (
                        <div key={i} className="relative">
                            <input
                                type="text"
                                value={ticker}
                                onChange={(e) => updateTicker(i, e.target.value)}
                                placeholder={`Stock ${i + 1}`}
                                className="w-28 px-3 py-3 rounded-lg border border-border-medium bg-white focus:outline-none focus:ring-2 focus:ring-terra-500"
                            />
                            {tickers.length > 2 && (
                                <button
                                    onClick={() => removeTicker(i)}
                                    className="absolute -top-2 -right-2 w-5 h-5 bg-error-500 text-white rounded-full text-caption"
                                >
                                    ×
                                </button>
                            )}
                        </div>
                    ))}
                    {tickers.length < 5 && (
                        <button onClick={addTicker} className="px-3 py-3 border-2 border-dashed border-neutral-300 rounded-lg text-neutral-500 hover:border-terra-500 hover:text-terra-500">
                            + Add
                        </button>
                    )}
                    <button
                        onClick={compare}
                        disabled={loading}
                        className="px-6 py-3 bg-terra-500 text-white rounded-lg font-medium hover:bg-terra-600 disabled:opacity-50"
                    >
                        {loading ? 'Comparing...' : 'Compare'}
                    </button>
                </div>

                {error && <p className="text-error-600 mt-3">{error}</p>}
            </Section>

            {result && (
                <Section spacing="md" background="default">
                    {/* Winner Banner */}
                    <div className="bg-gradient-to-r from-terra-500 to-warning-500 text-white p-6 rounded-xl mb-6">
                        <p className="text-caption uppercase opacity-80">Overall Winner</p>
                        <p className="text-display-xs font-bold">{result.overall_winner}</p>
                        <p className="text-body-sm mt-2 opacity-90">
                            Wins in {result.win_counts[result.overall_winner]} of {Object.values(result.winners).flatMap(c => Object.values(c)).length} categories
                        </p>
                    </div>

                    {/* Comparison Table */}
                    <div className="bg-white rounded-xl border border-border-light overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead className="bg-cream-50 border-b border-border-light">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-caption font-semibold text-navy-900 sticky left-0 bg-cream-50">Metric</th>
                                        {result.stocks.map(stock => (
                                            <th key={stock.ticker} className="px-4 py-3 text-center">
                                                <div className="font-semibold text-navy-900">{stock.ticker}</div>
                                                <div className="text-caption text-neutral-500 truncate max-w-[120px]">{stock.name}</div>
                                                {stock.ticker === result.overall_winner && (
                                                    <span className="inline-block mt-1 px-2 py-0.5 bg-terra-100 text-terra-700 rounded text-caption">
                                                        🏆 Winner
                                                    </span>
                                                )}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr className="border-b border-border-light bg-cream-50/50">
                                        <td className="px-4 py-2 font-semibold text-navy-700 sticky left-0 bg-cream-50/50">Price</td>
                                        {result.stocks.map(stock => (
                                            <td key={stock.ticker} className="px-4 py-2 text-center font-medium">
                                                ${stock.price?.toFixed(2)}
                                            </td>
                                        ))}
                                    </tr>
                                    <tr className="border-b border-border-light">
                                        <td className="px-4 py-2 font-semibold text-navy-700 sticky left-0 bg-white">Market Cap</td>
                                        {result.stocks.map(stock => (
                                            <td key={stock.ticker} className="px-4 py-2 text-center">
                                                {formatValue('market_cap', stock.market_cap)}
                                            </td>
                                        ))}
                                    </tr>

                                    {metrics.map((metric, i) => (
                                        <tr key={metric.key} className={`border-b border-border-light ${i % 2 === 0 ? 'bg-white' : 'bg-cream-50/30'}`}>
                                            <td className={`px-4 py-2 font-medium text-navy-700 sticky left-0 ${i % 2 === 0 ? 'bg-white' : 'bg-cream-50/30'}`}>
                                                {metric.label}
                                            </td>
                                            {result.stocks.map(stock => {
                                                const winner = isWinner(stock.ticker, metric.category, metric.key);
                                                return (
                                                    <td key={stock.ticker} className={`px-4 py-2 text-center ${winner ? 'bg-success-50' : ''}`}>
                                                        <span className={winner ? 'font-bold text-success-700' : ''}>
                                                            {formatValue(metric.key, stock[metric.key])}
                                                            {winner && ' ✓'}
                                                        </span>
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </Section>
            )}
        </>
    );
}
