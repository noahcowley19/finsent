'use client';

import React, { useState, useEffect } from 'react';
import { Section } from '@/components/layout';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface DividendStock {
    ticker: string;
    name: string;
    price: number;
    dividend_yield: number;
    dividend_rate: number;
    annual_dividend: number;
    payout_ratio: number;
    ex_dividend_date: string;
    days_until_ex_div: number;
    dividend_growth_5y: number;
    dividend_safety: string;
    safety_score: number;
}

interface DRIPProjection {
    year: number;
    portfolio_value: number;
    annual_dividend: number;
    total_dividends_received: number;
}

export default function DividendsPage() {
    const [ticker, setTicker] = useState('');
    const [stockData, setStockData] = useState<DividendStock | null>(null);
    const [loading, setLoading] = useState(false);

    // DRIP Calculator
    const [dripConfig, setDripConfig] = useState({
        initial_investment: 10000,
        dividend_yield: 3,
        years: 20,
        price_growth: 5,
    });
    const [dripResult, setDripResult] = useState<{ projections: DRIPProjection[]; summary: any } | null>(null);
    const [dripLoading, setDripLoading] = useState(false);

    const analyzeDividend = async () => {
        if (!ticker.trim()) return;
        setLoading(true);

        try {
            const res = await fetch(`${API_BASE}/api/dividends/analysis/${ticker.toUpperCase()}`);
            const data = await res.json();
            if (!data.error) {
                setStockData(data);
                if (data.dividend_yield) {
                    setDripConfig(prev => ({ ...prev, dividend_yield: data.dividend_yield }));
                }
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const calculateDRIP = async () => {
        setDripLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/dividends/drip-calculator`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dripConfig),
            });
            const data = await res.json();
            setDripResult(data);
        } catch (err) {
            console.error(err);
        } finally {
            setDripLoading(false);
        }
    };

    const getSafetyColor = (safety: string) => {
        switch (safety) {
            case 'Safe': return 'bg-success-100 text-success-700';
            case 'Moderate': return 'bg-warning-100 text-warning-700';
            case 'Caution': return 'bg-orange-100 text-orange-700';
            case 'At Risk': return 'bg-error-100 text-error-700';
            default: return 'bg-neutral-100 text-neutral-700';
        }
    };

    return (
        <>
            <Section spacing="md" background="gradient">
                <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                    Dividend Tracker
                </h1>
                <p className="text-body-md text-neutral-600 mb-6">
                    Analyze dividend stocks and project future income with DRIP calculator.
                </p>

                <div className="flex gap-3 max-w-md">
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        onKeyDown={(e) => e.key === 'Enter' && analyzeDividend()}
                        placeholder="Enter ticker (e.g., JNJ)"
                        className="flex-1 px-4 py-3 rounded-lg border border-border-medium bg-white focus:outline-none focus:ring-2 focus:ring-terra-500"
                    />
                    <button
                        onClick={analyzeDividend}
                        disabled={loading}
                        className="px-6 py-3 bg-terra-500 text-white rounded-lg font-medium hover:bg-terra-600 disabled:opacity-50"
                    >
                        {loading ? 'Loading...' : 'Analyze'}
                    </button>
                </div>
            </Section>

            <Section spacing="md" background="default">
                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Stock Analysis */}
                    <div>
                        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                            Dividend Analysis
                        </h2>

                        {stockData ? (
                            <div className="bg-white rounded-xl border border-border-light p-6">
                                <div className="flex justify-between items-start mb-6">
                                    <div>
                                        <h3 className="text-heading-sm font-bold text-navy-900">{stockData.ticker}</h3>
                                        <p className="text-neutral-600">{stockData.name}</p>
                                    </div>
                                    <span className={`px-3 py-1 rounded-full text-caption font-medium ${getSafetyColor(stockData.dividend_safety || '')}`}>
                                        {stockData.dividend_safety || 'N/A'}
                                    </span>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-cream-50 rounded-lg text-center">
                                        <p className="text-caption text-neutral-500">Dividend Yield</p>
                                        <p className="text-heading-md font-bold text-success-600">{stockData.dividend_yield?.toFixed(2)}%</p>
                                    </div>
                                    <div className="p-4 bg-cream-50 rounded-lg text-center">
                                        <p className="text-caption text-neutral-500">Annual Dividend</p>
                                        <p className="text-heading-md font-bold text-navy-900">${stockData.annual_dividend?.toFixed(2)}</p>
                                    </div>
                                    <div className="p-4 bg-cream-50 rounded-lg text-center">
                                        <p className="text-caption text-neutral-500">Payout Ratio</p>
                                        <p className={`text-heading-md font-bold ${(stockData.payout_ratio || 0) > 80 ? 'text-error-600' : 'text-navy-900'}`}>
                                            {stockData.payout_ratio?.toFixed(0)}%
                                        </p>
                                    </div>
                                    <div className="p-4 bg-cream-50 rounded-lg text-center">
                                        <p className="text-caption text-neutral-500">5Y Growth</p>
                                        <p className={`text-heading-md font-bold ${(stockData.dividend_growth_5y || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                            {stockData.dividend_growth_5y?.toFixed(1) || 'N/A'}%
                                        </p>
                                    </div>
                                </div>

                                {stockData.ex_dividend_date && (
                                    <div className="mt-4 p-4 bg-terra-50 rounded-lg">
                                        <p className="text-caption text-terra-600">Next Ex-Dividend Date</p>
                                        <p className="font-semibold text-terra-700">
                                            {stockData.ex_dividend_date}
                                            {stockData.days_until_ex_div !== undefined && (
                                                <span className="ml-2 text-caption">
                                                    ({stockData.days_until_ex_div > 0 ? `in ${stockData.days_until_ex_div} days` : 'passed'})
                                                </span>
                                            )}
                                        </p>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="bg-cream-50 rounded-xl p-12 text-center">
                                <p className="text-neutral-500">Enter a ticker above to analyze dividends</p>
                            </div>
                        )}
                    </div>

                    {/* DRIP Calculator */}
                    <div>
                        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                            DRIP Calculator
                        </h2>

                        <div className="bg-white rounded-xl border border-border-light p-6">
                            <div className="grid grid-cols-2 gap-4 mb-6">
                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Initial Investment</label>
                                    <input
                                        type="number"
                                        value={dripConfig.initial_investment}
                                        onChange={(e) => setDripConfig({ ...dripConfig, initial_investment: Number(e.target.value) })}
                                        className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                    />
                                </div>
                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Dividend Yield %</label>
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={dripConfig.dividend_yield}
                                        onChange={(e) => setDripConfig({ ...dripConfig, dividend_yield: Number(e.target.value) })}
                                        className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                    />
                                </div>
                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Years</label>
                                    <input
                                        type="number"
                                        value={dripConfig.years}
                                        onChange={(e) => setDripConfig({ ...dripConfig, years: Number(e.target.value) })}
                                        className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                    />
                                </div>
                                <div>
                                    <label className="block text-caption text-neutral-500 mb-1">Price Growth %</label>
                                    <input
                                        type="number"
                                        step="0.5"
                                        value={dripConfig.price_growth}
                                        onChange={(e) => setDripConfig({ ...dripConfig, price_growth: Number(e.target.value) })}
                                        className="w-full px-3 py-2 border border-border-medium rounded-lg"
                                    />
                                </div>
                            </div>

                            <button
                                onClick={calculateDRIP}
                                disabled={dripLoading}
                                className="w-full py-3 bg-navy-600 text-white rounded-lg font-medium hover:bg-navy-700 disabled:opacity-50"
                            >
                                {dripLoading ? 'Calculating...' : 'Calculate DRIP'}
                            </button>

                            {dripResult && (
                                <div className="mt-6">
                                    <div className="grid grid-cols-2 gap-4 mb-4">
                                        <div className="p-4 bg-success-50 rounded-lg text-center">
                                            <p className="text-caption text-neutral-500">Final Value</p>
                                            <p className="text-heading-sm font-bold text-success-700">
                                                ${dripResult.summary.final_value?.toLocaleString()}
                                            </p>
                                        </div>
                                        <div className="p-4 bg-terra-50 rounded-lg text-center">
                                            <p className="text-caption text-neutral-500">Total Return</p>
                                            <p className="text-heading-sm font-bold text-terra-700">
                                                +{dripResult.summary.total_return?.toFixed(0)}%
                                            </p>
                                        </div>
                                        <div className="p-4 bg-cream-50 rounded-lg text-center">
                                            <p className="text-caption text-neutral-500">Total Dividends</p>
                                            <p className="text-heading-sm font-bold text-navy-900">
                                                ${dripResult.summary.total_dividends?.toLocaleString()}
                                            </p>
                                        </div>
                                        <div className="p-4 bg-cream-50 rounded-lg text-center">
                                            <p className="text-caption text-neutral-500">Annual Income (Y{dripConfig.years})</p>
                                            <p className="text-heading-sm font-bold text-navy-900">
                                                ${dripResult.summary.final_annual_dividend?.toLocaleString()}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </Section>
        </>
    );
}
