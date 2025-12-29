'use client';

// =============================================================================
// MACRO INTELLIGENCE HUB - Atmospheric Glass Theme (Cream/Obsidian)
// =============================================================================
// Reskinned from Brutalist Dark to match the Caveray design system
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

// Types
interface LiquidityData {
    total: { value: number; formatted: string };
    breakdown: {
        fed: { value: number; formatted: string };
        ecb: { value: number; formatted: string };
        boj: { value: number; formatted: string };
    };
}

interface YieldCurveData {
    curve: { maturity: string; years: number; yield: number }[];
    spreads: { '10Y_2Y': number; '10Y_3M': number };
    inverted: boolean;
}

interface RecessionData {
    probability: number;
    level: string;
    color: string;
}

interface FedSpeakData {
    statement: { date: string; excerpt: string };
    sentiment: { score: number; label: string; color: string };
}

interface InflationData {
    nowcast: number;
    official_cpi: number;
    trend: string;
    color: string;
}

interface StressData {
    tickers: string[];
    labels: string[];
    matrix: number[][];
    stress_flags: { condition: string; severity: string; color: string }[];
    overall_stress: boolean;
}

interface VitalData {
    name: string;
    value: number;
    unit: string;
    trend: string;
    sparkline: number[];
}

// Sparkline Component - Cream Theme
const Sparkline: React.FC<{ data: number[]; color?: string }> = ({ data, color = '#22C55E' }) => {
    if (!data || data.length === 0) return null;

    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;

    const points = data.map((v, i) => {
        const x = (i / (data.length - 1)) * 80;
        const y = 20 - ((v - min) / range) * 16;
        return `${x},${y}`;
    }).join(' ');

    return (
        <svg width="80" height="24" className="inline-block ml-2">
            <polyline
                points={points}
                fill="none"
                stroke={color}
                strokeWidth="1.5"
            />
        </svg>
    );
};

// Module Card Wrapper - Glass Morphism
const ModuleCard: React.FC<{
    title: string;
    children: React.ReactNode;
    span?: 1 | 2;
}> = ({ title, children, span = 1 }) => (
    <div className={`bg-white/50 backdrop-blur-md border border-cream-200 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow ${span === 2 ? 'lg:col-span-2' : ''}`}>
        <h3 className="text-xs uppercase tracking-widest text-obsidian-500 mb-4 font-medium">{title}</h3>
        {children}
    </div>
);

// Module A: Global Liquidity
const LiquidityModule: React.FC<{ data: LiquidityData | null }> = ({ data }) => {
    if (!data) return <div className="text-obsidian-400 text-sm">Loading...</div>;

    return (
        <div className="space-y-4">
            <div className="text-center">
                <div className="text-4xl font-bold text-obsidian-900">
                    {data.total.formatted}
                </div>
                <div className="text-xs text-obsidian-500 mt-1">Global Central Bank Liquidity</div>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-cream-100/80 p-3 rounded-xl">
                    <div className="text-sm font-semibold text-electric-600">{data.breakdown.fed.formatted}</div>
                    <div className="text-[10px] text-obsidian-500">FED</div>
                </div>
                <div className="bg-cream-100/80 p-3 rounded-xl">
                    <div className="text-sm font-semibold text-electric-600">{data.breakdown.ecb.formatted}</div>
                    <div className="text-[10px] text-obsidian-500">ECB</div>
                </div>
                <div className="bg-cream-100/80 p-3 rounded-xl">
                    <div className="text-sm font-semibold text-electric-600">{data.breakdown.boj.formatted}</div>
                    <div className="text-[10px] text-obsidian-500">BOJ</div>
                </div>
            </div>
        </div>
    );
};

// Module B: Yield Curve & Recession
const YieldCurveModule: React.FC<{
    curve: YieldCurveData | null;
    recession: RecessionData | null
}> = ({ curve, recession }) => {
    if (!curve || !recession) return <div className="text-obsidian-400 text-sm">Loading...</div>;

    const maxYield = Math.max(...curve.curve.map(c => c.yield));
    const minYield = Math.min(...curve.curve.map(c => c.yield));
    const range = maxYield - minYield || 1;

    return (
        <div className="space-y-4">
            {/* Recession Probability */}
            <div className="flex items-center justify-between">
                <span className="text-xs text-obsidian-500 uppercase">Recession Probability</span>
                <span
                    className="text-2xl font-bold"
                    style={{ color: recession.probability > 50 ? '#EF4444' : recession.probability > 30 ? '#F59E0B' : '#22C55E' }}
                >
                    {recession.probability}%
                </span>
            </div>

            {/* Probability Bar */}
            <div className="h-2 bg-cream-200 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all"
                    style={{
                        width: `${Math.min(recession.probability, 100)}%`,
                        backgroundColor: recession.probability > 50 ? '#EF4444' : recession.probability > 30 ? '#F59E0B' : '#22C55E'
                    }}
                />
            </div>

            {/* Yield Curve Chart */}
            <div className="flex items-end justify-between h-20 gap-1">
                {curve.curve.map((point) => {
                    const height = ((point.yield - minYield) / range) * 100;
                    return (
                        <div key={point.maturity} className="flex-1 flex flex-col items-center">
                            <div
                                className={`w-full rounded-t ${curve.inverted ? 'bg-coral-500' : 'bg-electric-500'}`}
                                style={{ height: `${Math.max(height, 5)}%` }}
                            />
                            <span className="text-[8px] text-obsidian-500 mt-1">{point.maturity}</span>
                        </div>
                    );
                })}
            </div>

            {/* Spreads */}
            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-cream-100/80 p-2 rounded-lg">
                    <span className="text-obsidian-500">10Y-2Y:</span>
                    <span className={`ml-2 font-semibold ${curve.spreads['10Y_2Y'] < 0 ? 'text-coral-600' : 'text-success-600'}`}>
                        {curve.spreads['10Y_2Y']}%
                    </span>
                </div>
                <div className="bg-cream-100/80 p-2 rounded-lg">
                    <span className="text-obsidian-500">10Y-3M:</span>
                    <span className={`ml-2 font-semibold ${curve.spreads['10Y_3M'] < 0 ? 'text-coral-600' : 'text-success-600'}`}>
                        {curve.spreads['10Y_3M']}%
                    </span>
                </div>
            </div>
        </div>
    );
};

// Module C: Fed Speak Decoder
const FedSpeakModule: React.FC<{ data: FedSpeakData | null }> = ({ data }) => {
    if (!data) return <div className="text-obsidian-400 text-sm">Loading...</div>;

    const gaugePosition = ((data.sentiment.score + 1) / 2) * 100;

    return (
        <div className="space-y-4">
            {/* Hawk/Dove Gauge */}
            <div className="relative">
                <div className="flex justify-between text-[10px] text-obsidian-500 mb-1">
                    <span>DOVE</span>
                    <span>NEUTRAL</span>
                    <span>HAWK</span>
                </div>
                <div className="h-3 bg-gradient-to-r from-success-400 via-amber-400 to-coral-500 rounded-full relative">
                    <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full border-2 border-obsidian-300 shadow-md transition-all"
                        style={{ left: `calc(${gaugePosition}% - 8px)` }}
                    />
                </div>
            </div>

            {/* Score */}
            <div className="text-center">
                <span
                    className="text-3xl font-bold"
                    style={{ color: data.sentiment.score > 0 ? '#EF4444' : data.sentiment.score < 0 ? '#22C55E' : '#F59E0B' }}
                >
                    {data.sentiment.score > 0 ? '+' : ''}{data.sentiment.score}
                </span>
                <span
                    className="ml-2 text-sm uppercase font-medium"
                    style={{ color: data.sentiment.score > 0 ? '#EF4444' : data.sentiment.score < 0 ? '#22C55E' : '#F59E0B' }}
                >
                    {data.sentiment.label}
                </span>
            </div>

            {/* Statement Excerpt */}
            <div className="text-xs text-obsidian-600 italic leading-relaxed border-l-2 border-cream-300 pl-3">
                "{data.statement.excerpt.slice(0, 150)}..."
            </div>

            <div className="text-[10px] text-obsidian-500">
                Last Statement: {data.statement.date}
            </div>
        </div>
    );
};

// Module D: Inflation Nowcast
const InflationModule: React.FC<{ data: InflationData | null }> = ({ data }) => {
    if (!data) return <div className="text-obsidian-400 text-sm">Loading...</div>;

    const inflationColor = data.nowcast > 3 ? '#EF4444' : data.nowcast > 2 ? '#F59E0B' : '#22C55E';

    return (
        <div className="space-y-4">
            <div className="flex items-end justify-between">
                <div>
                    <div className="text-[10px] text-obsidian-500 uppercase">Inflation Nowcast</div>
                    <div
                        className="text-4xl font-bold"
                        style={{ color: inflationColor }}
                    >
                        {data.nowcast}%
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-[10px] text-obsidian-500 uppercase">Official CPI</div>
                    <div className="text-xl font-semibold text-obsidian-600">{data.official_cpi}%</div>
                </div>
            </div>

            {/* Comparison Bar */}
            <div className="space-y-2">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-obsidian-500 w-16">NOWCAST</span>
                    <div className="flex-1 h-3 bg-cream-200 rounded-full overflow-hidden">
                        <div
                            className="h-full rounded-full"
                            style={{
                                width: `${(data.nowcast / 5) * 100}%`,
                                backgroundColor: inflationColor
                            }}
                        />
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-obsidian-500 w-16">OFFICIAL</span>
                    <div className="flex-1 h-3 bg-cream-200 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-obsidian-400 rounded-full"
                            style={{ width: `${(data.official_cpi / 5) * 100}%` }}
                        />
                    </div>
                </div>
            </div>

            <div
                className="text-xs uppercase text-center py-2 rounded-lg font-medium"
                style={{ backgroundColor: inflationColor + '15', color: inflationColor }}
            >
                Trend: {data.trend}
            </div>
        </div>
    );
};

// Module E: Stress Heatmap
const StressModule: React.FC<{ data: StressData | null }> = ({ data }) => {
    if (!data) return <div className="text-obsidian-400 text-sm">Loading...</div>;

    const getColor = (val: number) => {
        if (val >= 0.7) return '#22C55E';
        if (val >= 0.3) return '#86EFAC';
        if (val >= -0.3) return '#D4D4D8';
        if (val >= -0.7) return '#FDA4AF';
        return '#EF4444';
    };

    const getTextColor = (val: number) => {
        if (val >= 0.3 || val <= -0.3) return '#FFFFFF';
        return '#71717A';
    };

    return (
        <div className="space-y-3">
            {/* Correlation Matrix */}
            <div className="overflow-x-auto">
                <table className="w-full text-[10px]">
                    <thead>
                        <tr>
                            <th></th>
                            {data.labels.map((l) => (
                                <th key={l} className="text-obsidian-500 font-normal px-1">{l}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {data.matrix.map((row, i) => (
                            <tr key={i}>
                                <td className="text-obsidian-500 pr-2">{data.labels[i]}</td>
                                {row.map((val, j) => (
                                    <td
                                        key={j}
                                        className="text-center p-1 rounded"
                                        style={{ backgroundColor: getColor(val) }}
                                    >
                                        <span className="text-[8px] font-medium" style={{ color: getTextColor(val) }}>
                                            {val.toFixed(1)}
                                        </span>
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Stress Flags */}
            {data.stress_flags.length > 0 && (
                <div className="space-y-1">
                    {data.stress_flags.map((flag, i) => (
                        <div
                            key={i}
                            className="text-[10px] px-2 py-1 rounded-lg font-medium"
                            style={{ backgroundColor: flag.color + '15', color: flag.color }}
                        >
                            ⚠ {flag.condition}
                        </div>
                    ))}
                </div>
            )}

            {/* Overall Status */}
            <div className={`text-center text-xs py-2 rounded-lg font-medium ${data.overall_stress
                ? 'bg-coral-100 text-coral-700'
                : 'bg-success-100 text-success-700'
                }`}>
                {data.overall_stress ? '⚠ ELEVATED STRESS' : '✓ NORMAL CONDITIONS'}
            </div>
        </div>
    );
};

// Module F: Economic Vitals
const VitalsModule: React.FC<{ data: Record<string, VitalData> | null }> = ({ data }) => {
    if (!data) return <div className="text-obsidian-400 text-sm">Loading...</div>;

    const getTrendColor = (trend: string) => {
        if (trend === 'up') return '#22C55E';
        if (trend === 'down') return '#EF4444';
        return '#F59E0B';
    };

    const vitals = Object.values(data);

    return (
        <div className="space-y-2">
            {vitals.map((vital) => (
                <div
                    key={vital.name}
                    className="flex items-center justify-between bg-cream-100/80 p-3 rounded-xl"
                >
                    <span className="text-[10px] text-obsidian-500 uppercase flex-1">{vital.name}</span>
                    <div className="flex items-center gap-2">
                        <Sparkline data={vital.sparkline} color={getTrendColor(vital.trend)} />
                        <span className="text-sm font-semibold text-obsidian-900">
                            {vital.value}{vital.unit}
                        </span>
                    </div>
                </div>
            ))}
        </div>
    );
};

// Main Dashboard Component
export const MacroDashboard: React.FC = () => {
    const [loading, setLoading] = useState(true);
    const [liquidity, setLiquidity] = useState<LiquidityData | null>(null);
    const [yieldCurve, setYieldCurve] = useState<YieldCurveData | null>(null);
    const [recession, setRecession] = useState<RecessionData | null>(null);
    const [fedSpeak, setFedSpeak] = useState<FedSpeakData | null>(null);
    const [inflation, setInflation] = useState<InflationData | null>(null);
    const [stress, setStress] = useState<StressData | null>(null);
    const [vitals, setVitals] = useState<Record<string, VitalData> | null>(null);
    const [lastUpdate, setLastUpdate] = useState<string>('');

    const fetchData = useCallback(async () => {
        setLoading(true);

        try {
            const response = await fetch(`${API_BASE}/api/macro/dashboard`);
            const data = await response.json();

            setLiquidity(data.liquidity);
            setYieldCurve(data.yield_curve);
            setRecession(data.recession);
            setFedSpeak(data.fed_speak);
            setInflation(data.inflation);
            setStress(data.stress);
            setVitals(data.vitals?.vitals);
            setLastUpdate(new Date().toLocaleTimeString());
        } catch (error) {
            console.error('Failed to fetch macro data:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
        // Refresh every 5 minutes
        const interval = setInterval(fetchData, 5 * 60 * 1000);
        return () => clearInterval(interval);
    }, [fetchData]);

    return (
        <div className="min-h-screen bg-cream-50">
            {/* Header */}
            <div className="border-b border-cream-200 bg-white/50 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-2xl sm:text-3xl font-bold text-obsidian-900 tracking-tight">
                                Macro Intelligence Hub
                            </h1>
                            <p className="text-sm text-obsidian-500 mt-1">
                                Global Economic Radar • Real-Time Analysis
                            </p>
                        </div>
                        <div className="text-right">
                            <button
                                onClick={fetchData}
                                disabled={loading}
                                className="px-5 py-2.5 bg-electric-500 text-white text-sm font-medium rounded-xl hover:bg-electric-600 transition-colors disabled:opacity-50"
                            >
                                {loading ? 'Syncing...' : 'Refresh'}
                            </button>
                            <div className="text-xs text-obsidian-500 mt-2">
                                Last: {lastUpdate}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Grid */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <ModuleCard title="Global Liquidity Impulse">
                        <LiquidityModule data={liquidity} />
                    </ModuleCard>

                    <ModuleCard title="Yield Curve & Recession">
                        <YieldCurveModule curve={yieldCurve} recession={recession} />
                    </ModuleCard>

                    <ModuleCard title="Fed Speak Decoder">
                        <FedSpeakModule data={fedSpeak} />
                    </ModuleCard>

                    <ModuleCard title="Inflation Nowcast">
                        <InflationModule data={inflation} />
                    </ModuleCard>

                    <ModuleCard title="Stress Heatmap">
                        <StressModule data={stress} />
                    </ModuleCard>

                    <ModuleCard title="Economic Vitals">
                        <VitalsModule data={vitals} />
                    </ModuleCard>
                </div>
            </div>

            {/* Footer */}
            <div className="border-t border-cream-200 bg-white/30 backdrop-blur-sm">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 text-center">
                    <span className="text-xs text-obsidian-400">
                        Data Sources: Federal Reserve • YFinance • BLS • Atlanta Fed
                    </span>
                </div>
            </div>
        </div>
    );
};

export default MacroDashboard;
