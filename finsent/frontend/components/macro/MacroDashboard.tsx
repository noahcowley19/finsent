'use client';

// =============================================================================
// MACRO INTELLIGENCE HUB - Brutalist Dark Mode Dashboard
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

// Sparkline Component
const Sparkline: React.FC<{ data: number[]; color?: string }> = ({ data, color = '#00ff88' }) => {
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

// Module Card Wrapper
const ModuleCard: React.FC<{
    title: string;
    children: React.ReactNode;
    span?: 1 | 2;
}> = ({ title, children, span = 1 }) => (
    <div className={`bg-[#121212] border border-[#2a2a2a] p-4 ${span === 2 ? 'col-span-2' : ''}`}>
        <h3 className="text-xs uppercase tracking-widest text-[#666] mb-3 font-mono">{title}</h3>
        {children}
    </div>
);

// Module A: Global Liquidity
const LiquidityModule: React.FC<{ data: LiquidityData | null }> = ({ data }) => {
    if (!data) return <div className="text-[#666]">Loading...</div>;

    return (
        <div className="space-y-4">
            <div className="text-center">
                <div className="text-4xl font-mono text-white font-bold">
                    {data.total.formatted}
                </div>
                <div className="text-xs text-[#666] mt-1">GLOBAL CENTRAL BANK LIQUIDITY</div>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-[#0a0a0a] p-3 border border-[#2a2a2a]">
                    <div className="text-sm font-mono text-[#00d4ff]">{data.breakdown.fed.formatted}</div>
                    <div className="text-[10px] text-[#666]">FED</div>
                </div>
                <div className="bg-[#0a0a0a] p-3 border border-[#2a2a2a]">
                    <div className="text-sm font-mono text-[#00d4ff]">{data.breakdown.ecb.formatted}</div>
                    <div className="text-[10px] text-[#666]">ECB</div>
                </div>
                <div className="bg-[#0a0a0a] p-3 border border-[#2a2a2a]">
                    <div className="text-sm font-mono text-[#00d4ff]">{data.breakdown.boj.formatted}</div>
                    <div className="text-[10px] text-[#666]">BOJ</div>
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
    if (!curve || !recession) return <div className="text-[#666]">Loading...</div>;

    // Simple yield curve visualization
    const maxYield = Math.max(...curve.curve.map(c => c.yield));
    const minYield = Math.min(...curve.curve.map(c => c.yield));
    const range = maxYield - minYield || 1;

    return (
        <div className="space-y-4">
            {/* Recession Probability */}
            <div className="flex items-center justify-between">
                <span className="text-xs text-[#666] uppercase">Recession Probability</span>
                <span
                    className="text-2xl font-mono font-bold"
                    style={{ color: recession.color }}
                >
                    {recession.probability}%
                </span>
            </div>

            {/* Probability Bar */}
            <div className="h-2 bg-[#0a0a0a] rounded">
                <div
                    className="h-full rounded transition-all"
                    style={{
                        width: `${Math.min(recession.probability, 100)}%`,
                        backgroundColor: recession.color
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
                                className={`w-full rounded-t ${curve.inverted ? 'bg-[#ff0066]' : 'bg-[#00ff88]'}`}
                                style={{ height: `${Math.max(height, 5)}%` }}
                            />
                            <span className="text-[8px] text-[#666] mt-1">{point.maturity}</span>
                        </div>
                    );
                })}
            </div>

            {/* Spreads */}
            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-[#0a0a0a] p-2 border border-[#2a2a2a]">
                    <span className="text-[#666]">10Y-2Y:</span>
                    <span className={`ml-2 font-mono ${curve.spreads['10Y_2Y'] < 0 ? 'text-[#ff0066]' : 'text-[#00ff88]'}`}>
                        {curve.spreads['10Y_2Y']}%
                    </span>
                </div>
                <div className="bg-[#0a0a0a] p-2 border border-[#2a2a2a]">
                    <span className="text-[#666]">10Y-3M:</span>
                    <span className={`ml-2 font-mono ${curve.spreads['10Y_3M'] < 0 ? 'text-[#ff0066]' : 'text-[#00ff88]'}`}>
                        {curve.spreads['10Y_3M']}%
                    </span>
                </div>
            </div>
        </div>
    );
};

// Module C: Fed Speak Decoder
const FedSpeakModule: React.FC<{ data: FedSpeakData | null }> = ({ data }) => {
    if (!data) return <div className="text-[#666]">Loading...</div>;

    const gaugePosition = ((data.sentiment.score + 1) / 2) * 100;

    return (
        <div className="space-y-4">
            {/* Hawk/Dove Gauge */}
            <div className="relative">
                <div className="flex justify-between text-[10px] text-[#666] mb-1">
                    <span>DOVE</span>
                    <span>NEUTRAL</span>
                    <span>HAWK</span>
                </div>
                <div className="h-3 bg-gradient-to-r from-[#00ff88] via-[#ffaa00] to-[#ff0066] rounded relative">
                    <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full border-2 border-[#0a0a0a] shadow-lg transition-all"
                        style={{ left: `calc(${gaugePosition}% - 8px)` }}
                    />
                </div>
            </div>

            {/* Score */}
            <div className="text-center">
                <span
                    className="text-3xl font-mono font-bold"
                    style={{ color: data.sentiment.color }}
                >
                    {data.sentiment.score > 0 ? '+' : ''}{data.sentiment.score}
                </span>
                <span
                    className="ml-2 text-sm uppercase"
                    style={{ color: data.sentiment.color }}
                >
                    {data.sentiment.label}
                </span>
            </div>

            {/* Statement Excerpt */}
            <div className="text-xs text-[#888] italic leading-relaxed border-l-2 border-[#2a2a2a] pl-3">
                "{data.statement.excerpt.slice(0, 150)}..."
            </div>

            <div className="text-[10px] text-[#666]">
                Last Statement: {data.statement.date}
            </div>
        </div>
    );
};

// Module D: Inflation Nowcast
const InflationModule: React.FC<{ data: InflationData | null }> = ({ data }) => {
    if (!data) return <div className="text-[#666]">Loading...</div>;

    return (
        <div className="space-y-4">
            <div className="flex items-end justify-between">
                <div>
                    <div className="text-[10px] text-[#666] uppercase">Inflation Nowcast</div>
                    <div
                        className="text-4xl font-mono font-bold"
                        style={{ color: data.color }}
                    >
                        {data.nowcast}%
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-[10px] text-[#666] uppercase">Official CPI</div>
                    <div className="text-xl font-mono text-[#888]">{data.official_cpi}%</div>
                </div>
            </div>

            {/* Comparison Bar */}
            <div className="space-y-1">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[#666] w-16">NOWCAST</span>
                    <div className="flex-1 h-4 bg-[#0a0a0a] rounded">
                        <div
                            className="h-full rounded"
                            style={{
                                width: `${(data.nowcast / 5) * 100}%`,
                                backgroundColor: data.color
                            }}
                        />
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[#666] w-16">OFFICIAL</span>
                    <div className="flex-1 h-4 bg-[#0a0a0a] rounded">
                        <div
                            className="h-full bg-[#666] rounded"
                            style={{ width: `${(data.official_cpi / 5) * 100}%` }}
                        />
                    </div>
                </div>
            </div>

            <div
                className="text-xs uppercase text-center py-1 rounded"
                style={{ backgroundColor: data.color + '20', color: data.color }}
            >
                Trend: {data.trend}
            </div>
        </div>
    );
};

// Module E: Stress Heatmap
const StressModule: React.FC<{ data: StressData | null }> = ({ data }) => {
    if (!data) return <div className="text-[#666]">Loading...</div>;

    const getColor = (val: number) => {
        if (val >= 0.7) return '#00ff88';
        if (val >= 0.3) return '#00ff8866';
        if (val >= -0.3) return '#444';
        if (val >= -0.7) return '#ff006666';
        return '#ff0066';
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
                                <th key={l} className="text-[#666] font-normal px-1">{l}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {data.matrix.map((row, i) => (
                            <tr key={i}>
                                <td className="text-[#666] pr-2">{data.labels[i]}</td>
                                {row.map((val, j) => (
                                    <td
                                        key={j}
                                        className="text-center p-1"
                                        style={{ backgroundColor: getColor(val) }}
                                    >
                                        <span className="text-[8px] text-white font-mono">
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
                            className="text-[10px] px-2 py-1 rounded"
                            style={{ backgroundColor: flag.color + '20', color: flag.color }}
                        >
                            ⚠ {flag.condition}
                        </div>
                    ))}
                </div>
            )}

            {/* Overall Status */}
            <div className={`text-center text-xs py-2 rounded ${data.overall_stress
                    ? 'bg-[#ff006620] text-[#ff0066]'
                    : 'bg-[#00ff8820] text-[#00ff88]'
                }`}>
                {data.overall_stress ? '⚠ ELEVATED STRESS' : '✓ NORMAL CONDITIONS'}
            </div>
        </div>
    );
};

// Module F: Economic Vitals
const VitalsModule: React.FC<{ data: Record<string, VitalData> | null }> = ({ data }) => {
    if (!data) return <div className="text-[#666]">Loading...</div>;

    const getTrendColor = (trend: string) => {
        if (trend === 'up') return '#00ff88';
        if (trend === 'down') return '#ff0066';
        return '#ffaa00';
    };

    const vitals = Object.values(data);

    return (
        <div className="space-y-2">
            {vitals.map((vital) => (
                <div
                    key={vital.name}
                    className="flex items-center justify-between bg-[#0a0a0a] p-2 border border-[#2a2a2a]"
                >
                    <span className="text-[10px] text-[#666] uppercase flex-1">{vital.name}</span>
                    <div className="flex items-center gap-2">
                        <Sparkline data={vital.sparkline} color={getTrendColor(vital.trend)} />
                        <span className="text-sm font-mono text-white">
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
        <div className="min-h-screen bg-[#0a0a0a] text-white">
            {/* Header */}
            <div className="border-b border-[#2a2a2a] px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-mono font-bold tracking-tight">
                            MACRO INTELLIGENCE HUB
                        </h1>
                        <p className="text-xs text-[#666] mt-1 font-mono">
                            GLOBAL ECONOMIC RADAR • REAL-TIME ANALYSIS
                        </p>
                    </div>
                    <div className="text-right">
                        <button
                            onClick={fetchData}
                            disabled={loading}
                            className="px-4 py-2 bg-[#121212] border border-[#2a2a2a] text-xs font-mono hover:bg-[#1a1a1a] transition-colors disabled:opacity-50"
                        >
                            {loading ? 'SYNCING...' : 'REFRESH'}
                        </button>
                        <div className="text-[10px] text-[#666] mt-1">
                            Last: {lastUpdate}
                        </div>
                    </div>
                </div>
            </div>

            {/* Grid */}
            <div className="p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <ModuleCard title="Module A: Global Liquidity Impulse">
                        <LiquidityModule data={liquidity} />
                    </ModuleCard>

                    <ModuleCard title="Module B: Yield Curve & Recession">
                        <YieldCurveModule curve={yieldCurve} recession={recession} />
                    </ModuleCard>

                    <ModuleCard title="Module C: Fed Speak Decoder">
                        <FedSpeakModule data={fedSpeak} />
                    </ModuleCard>

                    <ModuleCard title="Module D: Inflation Nowcast">
                        <InflationModule data={inflation} />
                    </ModuleCard>

                    <ModuleCard title="Module E: Stress Heatmap">
                        <StressModule data={stress} />
                    </ModuleCard>

                    <ModuleCard title="Module F: Economic Vitals">
                        <VitalsModule data={vitals} />
                    </ModuleCard>
                </div>
            </div>

            {/* Footer */}
            <div className="border-t border-[#2a2a2a] px-6 py-3 text-center">
                <span className="text-[10px] text-[#444] font-mono">
                    DATA SOURCES: FEDERAL RESERVE • YFINANCE • BLS • ATLANTA FED
                </span>
            </div>
        </div>
    );
};

export default MacroDashboard;
