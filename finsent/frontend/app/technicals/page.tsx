'use client';

import React, { useState } from 'react';
import { Section } from '@/components/layout';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface TechnicalData {
    ticker: string;
    name: string;
    current_price: number;
    change_pct: number;
    rsi: number;
    rsi_interpretation: string;
    macd: number;
    macd_signal: number;
    macd_histogram: number;
    macd_interpretation: string;
    bb_upper: number;
    bb_middle: number;
    bb_lower: number;
    bb_position: number;
    stochastic_k: number;
    stochastic_d: number;
    stochastic_interpretation: string;
    sma_20: number;
    sma_50: number;
    sma_200: number;
    above_sma_20: boolean;
    above_sma_50: boolean;
    above_sma_200: boolean;
    atr: number;
    atr_percent: number;
    trend: string;
    trend_sentiment: string;
    overall_sentiment: string;
    signals: Array<{
        type: string;
        indicator: string;
        reason: string;
    }>;
    signal_summary: {
        buy_count: number;
        sell_count: number;
        hold_count: number;
    };
}

export default function TechnicalsPage() {
    const [ticker, setTicker] = useState('');
    const [data, setData] = useState<TechnicalData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const analyze = async () => {
        if (!ticker.trim()) return;
        setLoading(true);
        setError('');

        try {
            const res = await fetch(`${API_BASE}/api/technicals/${ticker.toUpperCase()}`);
            if (!res.ok) throw new Error('Failed to fetch');
            const result = await res.json();
            if (result.error) throw new Error(result.error);
            setData(result);
        } catch (err: any) {
            setError(err.message || 'Unable to analyze');
            setData(null);
        } finally {
            setLoading(false);
        }
    };

    const IndicatorCard = ({ title, value, interpretation, children }: { title: string; value: React.ReactNode; interpretation?: string; children?: React.ReactNode }) => (
        <div className="bg-white rounded-xl border border-border-light p-5">
            <h3 className="text-caption font-semibold text-neutral-500 uppercase tracking-wide mb-2">{title}</h3>
            <div className="text-heading-md font-bold text-navy-900">{value}</div>
            {interpretation && (
                <span className={`inline-block mt-2 px-2 py-1 rounded text-caption font-medium ${interpretation === 'Oversold' || interpretation === 'Bullish' ? 'bg-success-100 text-success-700' :
                        interpretation === 'Overbought' || interpretation === 'Bearish' ? 'bg-error-100 text-error-700' :
                            'bg-neutral-100 text-neutral-700'
                    }`}>
                    {interpretation}
                </span>
            )}
            {children}
        </div>
    );

    return (
        <>
            <Section spacing="md" background="gradient">
                <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                    Technical Analysis
                </h1>
                <p className="text-body-md text-neutral-600 mb-6">
                    Comprehensive technical indicators and trading signals.
                </p>

                <div className="flex gap-3 max-w-md">
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        onKeyDown={(e) => e.key === 'Enter' && analyze()}
                        placeholder="Enter ticker (e.g., AAPL)"
                        className="flex-1 px-4 py-3 rounded-lg border border-border-medium bg-white focus:outline-none focus:ring-2 focus:ring-terra-500"
                    />
                    <button
                        onClick={analyze}
                        disabled={loading || !ticker.trim()}
                        className="px-6 py-3 bg-terra-500 text-white rounded-lg font-medium hover:bg-terra-600 disabled:opacity-50"
                    >
                        {loading ? 'Analyzing...' : 'Analyze'}
                    </button>
                </div>

                {error && <p className="text-error-600 mt-3">{error}</p>}
            </Section>

            {data && (
                <Section spacing="md" background="default">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h2 className="text-heading-lg font-bold text-navy-900">{data.ticker}</h2>
                            <p className="text-neutral-600">{data.name}</p>
                        </div>
                        <div className="text-right">
                            <p className="text-heading-md font-bold">${data.current_price?.toFixed(2)}</p>
                            <p className={`font-medium ${(data.change_pct || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                {(data.change_pct || 0) >= 0 ? '+' : ''}{data.change_pct?.toFixed(2)}%
                            </p>
                        </div>
                    </div>

                    {/* Overall Sentiment */}
                    <div className={`p-6 rounded-xl mb-6 ${data.overall_sentiment === 'Bullish' ? 'bg-success-50 border border-success-200' :
                            data.overall_sentiment === 'Bearish' ? 'bg-error-50 border border-error-200' :
                                'bg-cream-50 border border-neutral-200'
                        }`}>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-caption font-semibold text-neutral-500">OVERALL SIGNAL</p>
                                <p className={`text-display-xs font-bold ${data.overall_sentiment === 'Bullish' ? 'text-success-700' :
                                        data.overall_sentiment === 'Bearish' ? 'text-error-700' :
                                            'text-neutral-700'
                                    }`}>
                                    {data.overall_sentiment}
                                </p>
                            </div>
                            <div className="flex gap-4 text-center">
                                <div>
                                    <p className="text-display-xs font-bold text-success-600">{data.signal_summary.buy_count}</p>
                                    <p className="text-caption text-neutral-500">Buy</p>
                                </div>
                                <div>
                                    <p className="text-display-xs font-bold text-error-600">{data.signal_summary.sell_count}</p>
                                    <p className="text-caption text-neutral-500">Sell</p>
                                </div>
                                <div>
                                    <p className="text-display-xs font-bold text-neutral-600">{data.signal_summary.hold_count}</p>
                                    <p className="text-caption text-neutral-500">Hold</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Indicators Grid */}
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                        <IndicatorCard title="RSI (14)" value={data.rsi?.toFixed(1)} interpretation={data.rsi_interpretation} />

                        <IndicatorCard title="MACD" value={data.macd?.toFixed(3)} interpretation={data.macd_interpretation}>
                            <div className="mt-2 text-caption text-neutral-500">
                                Signal: {data.macd_signal?.toFixed(3)} | Hist: {data.macd_histogram?.toFixed(3)}
                            </div>
                        </IndicatorCard>

                        <IndicatorCard title="Stochastic" value={`${data.stochastic_k?.toFixed(1)} / ${data.stochastic_d?.toFixed(1)}`} interpretation={data.stochastic_interpretation} />

                        <IndicatorCard title="Bollinger Position" value={`${data.bb_position?.toFixed(1)}%`}>
                            <div className="mt-2 text-caption text-neutral-500">
                                Upper: ${data.bb_upper?.toFixed(2)} | Lower: ${data.bb_lower?.toFixed(2)}
                            </div>
                        </IndicatorCard>

                        <IndicatorCard title="Trend" value={data.trend} interpretation={data.trend_sentiment === 'bullish' ? 'Bullish' : data.trend_sentiment === 'bearish' ? 'Bearish' : 'Neutral'} />

                        <IndicatorCard title="ATR (Volatility)" value={`$${data.atr?.toFixed(2)}`}>
                            <div className="mt-2 text-caption text-neutral-500">{data.atr_percent?.toFixed(2)}% of price</div>
                        </IndicatorCard>
                    </div>

                    {/* Moving Averages */}
                    <div className="bg-white rounded-xl border border-border-light p-5 mb-6">
                        <h3 className="font-semibold text-navy-900 mb-4">Moving Averages</h3>
                        <div className="grid grid-cols-3 gap-4">
                            {[
                                { label: 'SMA 20', value: data.sma_20, above: data.above_sma_20 },
                                { label: 'SMA 50', value: data.sma_50, above: data.above_sma_50 },
                                { label: 'SMA 200', value: data.sma_200, above: data.above_sma_200 },
                            ].map(ma => (
                                <div key={ma.label} className="text-center">
                                    <p className="text-caption text-neutral-500">{ma.label}</p>
                                    <p className="font-semibold">${ma.value?.toFixed(2) || 'N/A'}</p>
                                    {ma.value && (
                                        <span className={`inline-block px-2 py-0.5 rounded text-caption font-medium mt-1 ${ma.above ? 'bg-success-100 text-success-700' : 'bg-error-100 text-error-700'
                                            }`}>
                                            {ma.above ? 'Above' : 'Below'}
                                        </span>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Signals */}
                    {data.signals.length > 0 && (
                        <div className="bg-white rounded-xl border border-border-light p-5">
                            <h3 className="font-semibold text-navy-900 mb-4">Active Signals</h3>
                            <div className="space-y-3">
                                {data.signals.map((signal, i) => (
                                    <div key={i} className={`p-3 rounded-lg flex items-center gap-3 ${signal.type === 'buy' ? 'bg-success-50' :
                                            signal.type === 'sell' ? 'bg-error-50' : 'bg-cream-50'
                                        }`}>
                                        <span className={`w-2 h-2 rounded-full ${signal.type === 'buy' ? 'bg-success-500' :
                                                signal.type === 'sell' ? 'bg-error-500' : 'bg-neutral-400'
                                            }`} />
                                        <span className="font-medium text-caption uppercase">{signal.indicator}</span>
                                        <span className="text-body-sm text-neutral-600">{signal.reason}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </Section>
            )}
        </>
    );
}
