'use client';

// =============================================================================
// RISK DASHBOARD - Advanced Portfolio Risk Metrics
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

// Types
interface RiskReturnMetrics {
    sharpe_ratio: number | null;
    sortino_ratio: number | null;
    treynor_ratio: number | null;
    information_ratio: number | null;
    jensens_alpha: number | null;
}

interface VolatilityMetrics {
    annualized_volatility: number | null;
    portfolio_beta: number | null;
    max_drawdown: number | null;
}

interface VaRMetrics {
    var_95_historical: number | null;
    var_99_historical: number | null;
    var_95_parametric: number | null;
    var_99_parametric: number | null;
    var_95_dollar: number | null;
    var_99_dollar: number | null;
}

interface ConcentrationMetrics {
    hhi_index: number | null;
    hhi_interpretation: string;
    effective_n: number | null;
    top_holding_weight: number | null;
}

interface RiskAdjustedSummary {
    risk_grade: string;
    diversification_grade: string;
}

interface AdvancedMetrics {
    portfolio_value: number;
    positions_count: number;
    risk_return_metrics: RiskReturnMetrics;
    volatility_metrics: VolatilityMetrics;
    var_metrics: VaRMetrics;
    concentration_metrics: ConcentrationMetrics;
    risk_adjusted_summary: RiskAdjustedSummary;
    capm_analysis: {
        expected_return: number;
        beta: number;
        risk_free_rate: number;
    };
}

interface Position {
    ticker: string;
    shares: number;
    total_cost_basis: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

// Metric Card Component
interface MetricCardProps {
    label: string;
    value: string | number | null;
    tooltip: string;
    color?: 'default' | 'positive' | 'negative' | 'warning';
    suffix?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, tooltip, color = 'default', suffix = '' }) => {
    const [showTooltip, setShowTooltip] = useState(false);

    const colorClasses = {
        default: 'text-obsidian-900',
        positive: 'text-success-600',
        negative: 'text-coral-600',
        warning: 'text-amber-600',
    };

    return (
        <div
            className="relative p-4 bg-white/80 backdrop-blur-lg rounded-xl border border-cream-200/50 hover:shadow-lg transition-shadow cursor-help"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
        >
            <p className="text-sm text-obsidian-500 mb-1">{label}</p>
            <p className={`text-2xl font-bold ${colorClasses[color]}`}>
                {value !== null && value !== undefined ? `${value}${suffix}` : '—'}
            </p>

            {showTooltip && (
                <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-obsidian-900 text-white text-sm rounded-lg shadow-xl">
                    {tooltip}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-obsidian-900" />
                </div>
            )}
        </div>
    );
};

// Grade Badge Component
const GradeBadge: React.FC<{ grade: string; label: string }> = ({ grade, label }) => {
    const gradeColors: Record<string, string> = {
        A: 'bg-success-100 text-success-700 border-success-200',
        B: 'bg-electric-100 text-electric-700 border-electric-200',
        C: 'bg-amber-100 text-amber-700 border-amber-200',
        D: 'bg-coral-100 text-coral-700 border-coral-200',
        F: 'bg-red-100 text-red-700 border-red-200',
    };

    return (
        <div className="flex items-center gap-3 p-4 bg-white/80 backdrop-blur-lg rounded-xl border border-cream-200/50">
            <div
                className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold border ${gradeColors[grade] || gradeColors.C
                    }`}
            >
                {grade}
            </div>
            <div>
                <p className="text-sm text-obsidian-500">{label}</p>
                <p className="text-lg font-semibold text-obsidian-900">
                    {grade === 'A' ? 'Excellent' : grade === 'B' ? 'Good' : grade === 'C' ? 'Average' : grade === 'D' ? 'Below Average' : 'Poor'}
                </p>
            </div>
        </div>
    );
};

// Main RiskDashboard Component
export const RiskDashboard: React.FC<{ positions: Position[] }> = ({ positions }) => {
    const [metrics, setMetrics] = useState<AdvancedMetrics | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchAdvancedMetrics = useCallback(async () => {
        if (!positions || positions.length === 0) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/portfolio/advanced-metrics`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ positions }),
            });

            if (!response.ok) throw new Error('Failed to fetch metrics');

            const data = await response.json();
            setMetrics(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    }, [positions]);

    useEffect(() => {
        fetchAdvancedMetrics();
    }, [fetchAdvancedMetrics]);

    if (loading) {
        return (
            <div className="p-8 bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 animate-pulse">
                <div className="h-8 bg-cream-200 rounded w-48 mb-6" />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[...Array(8)].map((_, i) => (
                        <div key={i} className="h-24 bg-cream-100 rounded-xl" />
                    ))}
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-6 bg-coral-50 border border-coral-200 rounded-xl text-coral-700">
                <p className="font-medium">Error loading risk metrics</p>
                <p className="text-sm opacity-75">{error}</p>
            </div>
        );
    }

    if (!metrics) return null;

    const getValueColor = (value: number | null, threshold: { positive: number; warning: number }): 'default' | 'positive' | 'negative' | 'warning' => {
        if (value === null) return 'default';
        if (value > threshold.positive) return 'positive';
        if (value > threshold.warning) return 'warning';
        return 'negative';
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-obsidian-900">Risk Analytics</h2>
                <button
                    onClick={fetchAdvancedMetrics}
                    className="px-4 py-2 text-sm font-medium text-obsidian-600 hover:text-obsidian-900 hover:bg-cream-100 rounded-lg transition-colors"
                >
                    Refresh
                </button>
            </div>

            {/* Grades */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <GradeBadge grade={metrics.risk_adjusted_summary.risk_grade} label="Risk Performance" />
                <GradeBadge grade={metrics.risk_adjusted_summary.diversification_grade} label="Diversification" />
            </div>

            {/* Risk/Return Metrics */}
            <div className="p-6 bg-white/60 backdrop-blur-lg rounded-2xl border border-cream-200/50">
                <h3 className="text-lg font-semibold text-obsidian-900 mb-4">Risk-Adjusted Returns</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <MetricCard
                        label="Sharpe Ratio"
                        value={metrics.risk_return_metrics.sharpe_ratio}
                        tooltip="Excess return per unit of total risk. Above 1.0 is good, above 2.0 is excellent."
                        color={getValueColor(metrics.risk_return_metrics.sharpe_ratio, { positive: 1, warning: 0.5 })}
                    />
                    <MetricCard
                        label="Sortino Ratio"
                        value={metrics.risk_return_metrics.sortino_ratio}
                        tooltip="Excess return per unit of downside risk. Better than Sharpe for asymmetric returns."
                        color={getValueColor(metrics.risk_return_metrics.sortino_ratio, { positive: 1, warning: 0.5 })}
                    />
                    <MetricCard
                        label="Treynor Ratio"
                        value={metrics.risk_return_metrics.treynor_ratio}
                        tooltip="Excess return per unit of systematic (market) risk."
                        suffix="%"
                    />
                    <MetricCard
                        label="Information Ratio"
                        value={metrics.risk_return_metrics.information_ratio}
                        tooltip="Active return per unit of tracking error vs S&P 500."
                    />
                    <MetricCard
                        label="Jensen's Alpha"
                        value={metrics.risk_return_metrics.jensens_alpha}
                        tooltip="Abnormal return above CAPM prediction. Positive alpha means outperformance."
                        suffix="%"
                        color={getValueColor(metrics.risk_return_metrics.jensens_alpha, { positive: 0, warning: -2 })}
                    />
                </div>
            </div>

            {/* Volatility Metrics */}
            <div className="p-6 bg-white/60 backdrop-blur-lg rounded-2xl border border-cream-200/50">
                <h3 className="text-lg font-semibold text-obsidian-900 mb-4">Volatility & Risk</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <MetricCard
                        label="Annual Volatility"
                        value={metrics.volatility_metrics.annualized_volatility}
                        tooltip="Annualized standard deviation of returns. Lower is less risky."
                        suffix="%"
                    />
                    <MetricCard
                        label="Portfolio Beta"
                        value={metrics.volatility_metrics.portfolio_beta}
                        tooltip="Sensitivity to market movements. Beta of 1.0 moves with the market."
                        color={metrics.volatility_metrics.portfolio_beta && metrics.volatility_metrics.portfolio_beta > 1.2 ? 'warning' : 'default'}
                    />
                    <MetricCard
                        label="Max Drawdown"
                        value={metrics.volatility_metrics.max_drawdown}
                        tooltip="Largest peak-to-trough decline. Measures worst-case historical loss."
                        suffix="%"
                        color="negative"
                    />
                </div>
            </div>

            {/* VaR Metrics */}
            <div className="p-6 bg-white/60 backdrop-blur-lg rounded-2xl border border-cream-200/50">
                <h3 className="text-lg font-semibold text-obsidian-900 mb-4">Value at Risk (VaR)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                        label="VaR 95% (Daily)"
                        value={metrics.var_metrics.var_95_historical}
                        tooltip="5% chance of losing more than this on any given day."
                        suffix="%"
                        color="negative"
                    />
                    <MetricCard
                        label="VaR 99% (Daily)"
                        value={metrics.var_metrics.var_99_historical}
                        tooltip="1% chance of losing more than this on any given day."
                        suffix="%"
                        color="negative"
                    />
                    <MetricCard
                        label="VaR 95% ($)"
                        value={metrics.var_metrics.var_95_dollar?.toLocaleString() ?? null}
                        tooltip="Dollar amount at risk at 95% confidence level."
                        color="negative"
                    />
                    <MetricCard
                        label="VaR 99% ($)"
                        value={metrics.var_metrics.var_99_dollar?.toLocaleString() ?? null}
                        tooltip="Dollar amount at risk at 99% confidence level."
                        color="negative"
                    />
                </div>
            </div>

            {/* Concentration Metrics */}
            <div className="p-6 bg-white/60 backdrop-blur-lg rounded-2xl border border-cream-200/50">
                <h3 className="text-lg font-semibold text-obsidian-900 mb-4">Portfolio Concentration</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                        label="HHI Index"
                        value={metrics.concentration_metrics.hhi_index}
                        tooltip="Herfindahl-Hirschman Index. Below 0.15 is diversified, above 0.25 is concentrated."
                        color={metrics.concentration_metrics.hhi_index && metrics.concentration_metrics.hhi_index > 0.25 ? 'warning' : 'positive'}
                    />
                    <MetricCard
                        label="Concentration"
                        value={metrics.concentration_metrics.hhi_interpretation}
                        tooltip="Overall concentration classification based on HHI."
                    />
                    <MetricCard
                        label="Effective Holdings"
                        value={metrics.concentration_metrics.effective_n}
                        tooltip="Equivalent number of equally-weighted holdings. Higher is more diversified."
                    />
                    <MetricCard
                        label="Top Holding"
                        value={metrics.concentration_metrics.top_holding_weight}
                        tooltip="Weight of the largest single position."
                        suffix="%"
                        color={metrics.concentration_metrics.top_holding_weight && metrics.concentration_metrics.top_holding_weight > 25 ? 'warning' : 'default'}
                    />
                </div>
            </div>
        </div>
    );
};

export default RiskDashboard;
