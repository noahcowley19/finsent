'use client';

// =============================================================================
// ECONOMIC INDICATORS DASHBOARD
// =============================================================================
// Dashboard for tracking key economic data affecting markets
//
// Location: frontend/app/economic/page.tsx
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Section, Grid } from '@/components/layout';

// =============================================================================
// TYPES
// =============================================================================

interface StaticIndicator {
    value: number;
    unit: string;
    name: string;
    description: string;
    simple_explanation: string;
    source: string;
    last_updated: string;
    trend: 'up' | 'down' | 'stable';
    impact: string;
}

interface YieldPoint {
    maturity: string;
    yield: number;
}

interface IndicatorData {
    static: Record<string, StaticIndicator>;
    treasury: {
        yields: Record<string, number>;
        timestamp: string;
    };
}

interface YieldCurveData {
    current_curve: YieldPoint[];
    spreads: Record<string, number>;
    shape: string;
    shape_explanation: string;
    inverted: boolean;
}

interface Explainer {
    indicator: string;
    what_it_is: string;
    why_it_matters: string;
    market_impact: string;
    current_context: string;
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

async function fetchIndicators(): Promise<IndicatorData> {
    const res = await fetch(`${API_BASE}/api/economic/indicators`);
    if (!res.ok) throw new Error('Failed to fetch indicators');
    return res.json();
}

async function fetchYieldCurve(): Promise<YieldCurveData> {
    const res = await fetch(`${API_BASE}/api/economic/yield-curve`);
    if (!res.ok) throw new Error('Failed to fetch yield curve');
    return res.json();
}

async function fetchExplainers(): Promise<{ explainers: Explainer[] }> {
    const res = await fetch(`${API_BASE}/api/economic/explainers`);
    if (!res.ok) throw new Error('Failed to fetch explainers');
    return res.json();
}

// =============================================================================
// COMPONENTS
// =============================================================================

const TrendArrow: React.FC<{ trend: 'up' | 'down' | 'stable' }> = ({ trend }) => {
    if (trend === 'up') {
        return (
            <svg className="w-5 h-5 text-error-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
        );
    }
    if (trend === 'down') {
        return (
            <svg className="w-5 h-5 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
        );
    }
    return (
        <svg className="w-5 h-5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
        </svg>
    );
};

const IndicatorCard: React.FC<{
    indicator: StaticIndicator;
    label: string;
}> = ({ indicator, label }) => {
    const [showExplanation, setShowExplanation] = useState(false);

    const getIconForLabel = (label: string) => {
        switch (label) {
            case 'fed_funds_rate':
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                );
            case 'inflation_rate':
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                );
            case 'unemployment_rate':
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                );
            case 'gdp_growth':
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                );
            default:
                return null;
        }
    };

    return (
        <div className="bg-white rounded-xl border border-border-light p-6 hover:shadow-lg transition-shadow relative group">
            <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-navy-100 to-navy-50 flex items-center justify-center text-navy-600">
                    {getIconForLabel(label)}
                </div>
                <TrendArrow trend={indicator.trend} />
            </div>

            <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-2">
                {indicator.name}
            </h3>

            <div className="flex items-baseline gap-1 mb-3">
                <span className="font-display text-display-md text-navy-900">
                    {indicator.value}
                </span>
                <span className="text-body-lg text-neutral-500">{indicator.unit}</span>
            </div>

            <p className="text-body-sm text-neutral-600 mb-3">
                {indicator.simple_explanation}
            </p>

            <div className="flex items-center gap-2 text-caption text-neutral-400">
                <span>{indicator.source}</span>
                <span>•</span>
                <span>{indicator.last_updated}</span>
            </div>

            {/* Hover tooltip for impact */}
            <div className="absolute bottom-full left-0 right-0 mb-2 p-3 bg-navy-900 text-white text-body-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                <strong>Market Impact:</strong> {indicator.impact}
            </div>
        </div>
    );
};

const YieldCurveChart: React.FC<{ data: YieldCurveData }> = ({ data }) => {
    if (!data.current_curve || data.current_curve.length === 0) {
        return (
            <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                    Yield Curve
                </h3>
                <p className="text-neutral-500">Loading yield curve data...</p>
            </div>
        );
    }

    const maxYield = Math.max(...data.current_curve.map(p => p.yield)) + 0.5;
    const minYield = Math.min(...data.current_curve.map(p => p.yield)) - 0.5;
    const range = maxYield - minYield;

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="font-heading font-semibold text-heading-md text-navy-900">
                        Treasury Yield Curve
                    </h3>
                    <p className="text-body-sm text-neutral-500 mt-1">
                        Current shape: <span className={`font-medium ${data.inverted ? 'text-error-500' : 'text-success-500'}`}>{data.shape}</span>
                    </p>
                </div>
                {data.inverted && (
                    <span className="px-3 py-1 bg-error-100 text-error-700 text-body-sm font-medium rounded-full">
                        ⚠️ Inverted
                    </span>
                )}
            </div>

            {/* Chart */}
            <div className="h-64 flex items-end gap-4 border-b border-l border-border-light p-4 relative">
                {data.current_curve.map((point, index) => {
                    const height = ((point.yield - minYield) / range) * 100;
                    return (
                        <div key={point.maturity} className="flex-1 flex flex-col items-center">
                            <div
                                className="w-full bg-gradient-to-t from-navy-600 to-navy-400 rounded-t-lg transition-all hover:from-terra-600 hover:to-terra-400"
                                style={{ height: `${height}%` }}
                            >
                                <div className="relative">
                                    <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-body-sm font-medium text-navy-900 whitespace-nowrap">
                                        {point.yield.toFixed(2)}%
                                    </span>
                                </div>
                            </div>
                            <span className="text-caption text-neutral-600 mt-2">{point.maturity}</span>
                        </div>
                    );
                })}
            </div>

            {/* Explanation */}
            <div className="mt-4 p-3 bg-cream-50 rounded-lg">
                <p className="text-body-sm text-neutral-700">
                    <strong>What this means:</strong> {data.shape_explanation}
                </p>
            </div>

            {/* Spreads */}
            <div className="grid grid-cols-2 gap-4 mt-4">
                {data.spreads['10Y_3M'] !== undefined && (
                    <div className="p-3 bg-cream-50 rounded-lg">
                        <p className="text-caption text-neutral-500">10Y-3M Spread</p>
                        <p className={`text-heading-sm font-semibold ${data.spreads['10Y_3M'] < 0 ? 'text-error-600' : 'text-success-600'}`}>
                            {data.spreads['10Y_3M'] > 0 ? '+' : ''}{data.spreads['10Y_3M']?.toFixed(2)}%
                        </p>
                    </div>
                )}
                {data.spreads['10Y_2Y'] !== undefined && (
                    <div className="p-3 bg-cream-50 rounded-lg">
                        <p className="text-caption text-neutral-500">10Y-2Y Spread</p>
                        <p className={`text-heading-sm font-semibold ${data.spreads['10Y_2Y'] < 0 ? 'text-error-600' : 'text-success-600'}`}>
                            {data.spreads['10Y_2Y'] > 0 ? '+' : ''}{data.spreads['10Y_2Y']?.toFixed(2)}%
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

const ExplainerCard: React.FC<{ explainer: Explainer }> = ({ explainer }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="bg-white rounded-xl border border-border-light overflow-hidden">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-cream-50 transition-colors"
            >
                <span className="font-heading font-semibold text-heading-sm text-navy-900">
                    {explainer.indicator}
                </span>
                <svg
                    className={`w-5 h-5 text-neutral-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isExpanded && (
                <div className="p-4 pt-0 space-y-3 border-t border-border-light">
                    <div>
                        <h4 className="text-caption font-medium text-terra-600 uppercase tracking-wide mb-1">What is it?</h4>
                        <p className="text-body-sm text-neutral-700">{explainer.what_it_is}</p>
                    </div>
                    <div>
                        <h4 className="text-caption font-medium text-terra-600 uppercase tracking-wide mb-1">Why it matters</h4>
                        <p className="text-body-sm text-neutral-700">{explainer.why_it_matters}</p>
                    </div>
                    <div>
                        <h4 className="text-caption font-medium text-terra-600 uppercase tracking-wide mb-1">Market Impact</h4>
                        <p className="text-body-sm text-neutral-700">{explainer.market_impact}</p>
                    </div>
                    <div className="p-3 bg-navy-50 rounded-lg">
                        <h4 className="text-caption font-medium text-navy-700 mb-1">Current Context</h4>
                        <p className="text-body-sm text-navy-900">{explainer.current_context}</p>
                    </div>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// MAIN PAGE
// =============================================================================

export default function EconomicDashboard() {
    const [indicators, setIndicators] = useState<IndicatorData | null>(null);
    const [yieldCurve, setYieldCurve] = useState<YieldCurveData | null>(null);
    const [explainers, setExplainers] = useState<Explainer[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const [indicatorsData, yieldData, explainersData] = await Promise.all([
                fetchIndicators(),
                fetchYieldCurve(),
                fetchExplainers(),
            ]);

            setIndicators(indicatorsData);
            setYieldCurve(yieldData);
            setExplainers(explainersData.explainers);
        } catch (err) {
            setError('Failed to load economic data. Please try again.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();
    }, [loadData]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-navy-200 border-t-navy-600 rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-neutral-600">Loading economic indicators...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <p className="text-error-600 mb-4">{error}</p>
                    <button
                        onClick={loadData}
                        className="px-4 py-2 bg-terra-500 text-white rounded-lg hover:bg-terra-600"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <>
            {/* Header */}
            <Section spacing="md" background="gradient">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div>
                        <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                            Economic Indicators
                        </h1>
                        <p className="text-body-md text-neutral-600 max-w-2xl">
                            Track key economic data points that affect market performance.
                            Understanding these indicators helps you make more informed investment decisions.
                        </p>
                    </div>
                    <button
                        onClick={loadData}
                        className="px-4 py-2.5 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors flex items-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Refresh
                    </button>
                </div>
            </Section>

            {/* Key Indicators */}
            <Section spacing="md" background="default">
                <h2 className="font-heading font-semibold text-heading-lg text-navy-900 mb-6">
                    Key Indicators
                </h2>
                <Grid cols={1} colsMd={2} colsLg={4} gap="md">
                    {indicators?.static && Object.entries(indicators.static).map(([key, indicator]) => (
                        <IndicatorCard key={key} label={key} indicator={indicator} />
                    ))}
                </Grid>
            </Section>

            {/* Yield Curve */}
            <Section spacing="md" background="secondary">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                        {yieldCurve && <YieldCurveChart data={yieldCurve} />}
                    </div>

                    {/* Treasury Yields */}
                    <div className="bg-white rounded-xl border border-border-light p-6">
                        <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-6">
                            Current Treasury Yields
                        </h3>
                        <div className="space-y-4">
                            {indicators?.treasury?.yields && Object.entries(indicators.treasury.yields)
                                .filter(([key]) => !key.includes('spread') && !key.includes('inverted'))
                                .map(([maturity, value]) => (
                                    <div key={maturity} className="flex items-center justify-between p-3 bg-cream-50 rounded-lg">
                                        <span className="font-medium text-navy-900">{maturity} Treasury</span>
                                        <span className="font-display text-heading-sm text-navy-700">{value?.toFixed(2)}%</span>
                                    </div>
                                ))
                            }
                        </div>
                    </div>
                </div>
            </Section>

            {/* Explainers */}
            <Section spacing="lg" background="default">
                <h2 className="font-heading font-semibold text-heading-lg text-navy-900 mb-6">
                    Understanding the Indicators
                </h2>
                <p className="text-body-md text-neutral-600 mb-6">
                    Click on each indicator to learn more about what it measures and how it affects markets.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {explainers.map((explainer) => (
                        <ExplainerCard key={explainer.indicator} explainer={explainer} />
                    ))}
                </div>
            </Section>
        </>
    );
}
