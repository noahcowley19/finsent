'use client';

// =============================================================================
// EARNINGS CALENDAR PAGE
// =============================================================================
// Earnings calendar with analysis features
//
// Location: frontend/app/earnings/page.tsx
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Section, Grid } from '@/components/layout';
import { useWatchlist } from '@/lib/hooks';

// =============================================================================
// TYPES
// =============================================================================

interface EarningsEvent {
    ticker: string;
    company_name: string;
    next_earnings_date: string;
    days_until_earnings: number;
    eps_estimate?: number;
    revenue_estimate?: number;
    current_price?: number;
    month_change_pct?: number;
    sector?: string;
}

interface EarningsHistory {
    quarter: string;
    date: string;
    eps_actual?: number;
    eps_estimate?: number;
    eps_surprise?: number;
    surprise_pct?: number;
    result?: 'beat' | 'miss' | 'met';
}

interface EarningsHistoryData {
    ticker: string;
    company_name: string;
    history: EarningsHistory[];
    stats: {
        total_quarters: number;
        beats: number;
        misses: number;
        beat_rate?: number;
        avg_surprise_pct?: number;
    };
}

interface EarningsAnalysis {
    ticker: string;
    company_name: string;
    upcoming_earnings?: {
        date: string;
        days_until: number;
        eps_estimate?: number;
        revenue_estimate?: number;
    };
    market_context?: {
        current_price?: number;
        month_change_pct?: number;
        momentum_20d?: number;
        recent_volume_ratio?: number;
    };
    volatility?: {
        historical_volatility?: number;
        implied_volatility?: number;
        iv_hv_ratio?: number;
        iv_status?: string;
        iv_interpretation?: string;
    };
    historical_performance?: {
        total_quarters: number;
        beats: number;
        misses: number;
        beat_rate?: number;
        avg_surprise_pct?: number;
    };
    recent_quarters?: EarningsHistory[];
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

async function fetchEarningsCalendar(tickers: string[]): Promise<{ earnings: EarningsEvent[] }> {
    const res = await fetch(`${API_BASE}/api/earnings/calendar?tickers=${tickers.join(',')}`);
    if (!res.ok) throw new Error('Failed to fetch earnings calendar');
    return res.json();
}

async function fetchEarningsHistory(ticker: string): Promise<EarningsHistoryData> {
    const res = await fetch(`${API_BASE}/api/earnings/history/${ticker}`);
    if (!res.ok) throw new Error('Failed to fetch earnings history');
    return res.json();
}

async function fetchEarningsAnalysis(ticker: string): Promise<EarningsAnalysis> {
    const res = await fetch(`${API_BASE}/api/earnings/analysis/${ticker}`);
    if (!res.ok) throw new Error('Failed to fetch earnings analysis');
    return res.json();
}

// =============================================================================
// COMPONENTS
// =============================================================================

const EarningsCard: React.FC<{
    event: EarningsEvent;
    onSelect: (ticker: string) => void;
    isSelected: boolean;
}> = ({ event, onSelect, isSelected }) => {
    const daysUntil = event.days_until_earnings;
    const urgencyClass = daysUntil <= 7 ? 'border-l-error-500' :
        daysUntil <= 14 ? 'border-l-warning-500' :
            'border-l-success-500';

    return (
        <div
            className={`bg-white rounded-xl border border-border-light border-l-4 ${urgencyClass} p-4 cursor-pointer transition-all hover:shadow-md ${isSelected ? 'ring-2 ring-terra-500' : ''}`}
            onClick={() => onSelect(event.ticker)}
        >
            <div className="flex items-start justify-between mb-3">
                <div>
                    <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
                        {event.ticker}
                    </h3>
                    <p className="text-body-sm text-neutral-600 line-clamp-1">{event.company_name}</p>
                </div>
                <span className={`px-2 py-1 rounded-full text-caption font-medium ${daysUntil <= 7 ? 'bg-error-100 text-error-700' :
                    daysUntil <= 14 ? 'bg-warning-100 text-warning-700' :
                        'bg-success-100 text-success-700'
                    }`}>
                    {daysUntil} days
                </span>
            </div>

            <div className="text-body-sm text-neutral-700 mb-2">
                <span className="font-medium">Earnings:</span> {event.next_earnings_date}
            </div>

            {event.eps_estimate && (
                <div className="text-body-sm text-neutral-600">
                    <span className="font-medium">EPS Est:</span> ${event.eps_estimate.toFixed(2)}
                </div>
            )}

            {event.current_price && (
                <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border-light">
                    <span className="text-body-sm font-medium">${event.current_price.toFixed(2)}</span>
                    {event.month_change_pct !== undefined && (
                        <span className={`text-caption ${event.month_change_pct >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                            {event.month_change_pct >= 0 ? '+' : ''}{event.month_change_pct.toFixed(1)}% (1M)
                        </span>
                    )}
                </div>
            )}
        </div>
    );
};

const EarningsHistoryChart: React.FC<{ history: EarningsHistory[] }> = ({ history }) => {
    if (!history || history.length === 0) {
        return (
            <div className="p-4 text-center text-neutral-500">
                No earnings history available
            </div>
        );
    }

    const maxSurprise = Math.max(...history.map(h => Math.abs(h.surprise_pct || 0)), 10);

    return (
        <div className="space-y-2">
            {history.slice(0, 8).map((quarter, index) => (
                <div key={quarter.quarter || index} className="flex items-center gap-4">
                    <div className="w-20 text-caption text-neutral-600 text-right">
                        {quarter.quarter || quarter.date}
                    </div>
                    <div className="flex-1 h-8 bg-cream-100 rounded relative overflow-hidden">
                        {quarter.surprise_pct !== undefined && quarter.surprise_pct !== 0 && (
                            <div
                                className={`absolute top-0 bottom-0 ${quarter.surprise_pct >= 0 ? 'bg-success-400 left-1/2' : 'bg-error-400 right-1/2'}`}
                                style={{
                                    width: `${Math.min(Math.abs(quarter.surprise_pct) / maxSurprise * 50, 50)}%`,
                                    [quarter.surprise_pct >= 0 ? 'left' : 'right']: '50%'
                                }}
                            />
                        )}
                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className={`text-caption font-medium ${quarter.result === 'beat' ? 'text-success-700' :
                                quarter.result === 'miss' ? 'text-error-700' :
                                    'text-neutral-700'
                                }`}>
                                {quarter.result === 'beat' ? '✓ Beat' :
                                    quarter.result === 'miss' ? '✗ Miss' :
                                        'Met'}
                                {quarter.surprise_pct !== undefined && ` (${quarter.surprise_pct > 0 ? '+' : ''}${quarter.surprise_pct.toFixed(1)}%)`}
                            </span>
                        </div>
                    </div>
                    <div className="w-16 text-right">
                        <span className="text-body-sm font-medium">${quarter.eps_actual?.toFixed(2) || 'N/A'}</span>
                    </div>
                </div>
            ))}
        </div>
    );
};

const VolatilityIndicator: React.FC<{ volatility?: EarningsAnalysis['volatility'] }> = ({ volatility }) => {
    if (!volatility) return null;

    const iv = volatility.implied_volatility;
    const hv = volatility.historical_volatility;

    return (
        <div className="bg-cream-50 rounded-xl p-4">
            <h4 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                Volatility Analysis
            </h4>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center p-3 bg-white rounded-lg">
                    <p className="text-caption text-neutral-500">Historical Vol</p>
                    <p className="text-heading-sm font-semibold text-navy-900">
                        {hv?.toFixed(1) || 'N/A'}%
                    </p>
                </div>
                <div className="text-center p-3 bg-white rounded-lg">
                    <p className="text-caption text-neutral-500">Implied Vol</p>
                    <p className={`text-heading-sm font-semibold ${volatility.iv_status === 'elevated' ? 'text-warning-600' : 'text-navy-900'
                        }`}>
                        {iv?.toFixed(1) || 'N/A'}%
                    </p>
                </div>
            </div>

            {volatility.iv_interpretation && (
                <p className="text-body-sm text-neutral-700">
                    <strong>What this means:</strong> {volatility.iv_interpretation}
                </p>
            )}
        </div>
    );
};

// =============================================================================
// MAIN PAGE
// =============================================================================

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM'];

export default function EarningsCalendarPage() {
    return <EarningsCalendarContent />;
}

function EarningsCalendarContent() {
    const { items = [] } = useWatchlist();
    const [events, setEvents] = useState<EarningsEvent[]>([]);
    const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<EarningsAnalysis | null>(null);
    const [loading, setLoading] = useState(true);
    const [analysisLoading, setAnalysisLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [searchTicker, setSearchTicker] = useState('');

    // Get tickers from watchlist or use defaults
    const tickers = (items && items.length > 0)
        ? items.map((item: any) => {
            if (typeof item === 'string') return item;
            return item.ticker || item.symbol || '';
        }).filter(Boolean).slice(0, 20)
        : DEFAULT_TICKERS;

    const loadCalendar = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const data = await fetchEarningsCalendar(tickers);
            setEvents(data.earnings || []);
        } catch (err) {
            setError('Failed to load earnings calendar');
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [tickers.join(',')]);

    const loadAnalysis = useCallback(async (ticker: string) => {
        setAnalysisLoading(true);

        try {
            const data = await fetchEarningsAnalysis(ticker);
            setAnalysis(data);
        } catch (err) {
            console.error('Failed to load analysis:', err);
        } finally {
            setAnalysisLoading(false);
        }
    }, []);

    useEffect(() => {
        loadCalendar();
    }, [loadCalendar]);

    useEffect(() => {
        if (selectedTicker) {
            loadAnalysis(selectedTicker);
        }
    }, [selectedTicker, loadAnalysis]);

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchTicker.trim()) {
            setSelectedTicker(searchTicker.trim().toUpperCase());
        }
    };

    // Separate upcoming from future
    const upcomingEvents = events.filter(e => e.days_until_earnings <= 14);
    const futureEvents = events.filter(e => e.days_until_earnings > 14);

    return (
        <>
            {/* Header */}
            <Section spacing="md" background="gradient">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div>
                        <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
                            Earnings Calendar
                        </h1>
                        <p className="text-body-md text-neutral-600 max-w-2xl">
                            Track upcoming earnings for your watchlist stocks. Click any stock to see
                            detailed analysis including historical performance and volatility.
                        </p>
                    </div>

                    {/* Search */}
                    <form onSubmit={handleSearch} className="flex gap-2">
                        <input
                            type="text"
                            value={searchTicker}
                            onChange={(e) => setSearchTicker(e.target.value.toUpperCase())}
                            placeholder="Analyze ticker..."
                            className="px-4 py-2.5 border border-border-medium rounded-lg text-body-sm w-32 focus:outline-none focus:ring-2 focus:ring-terra-500"
                        />
                        <button
                            type="submit"
                            className="px-4 py-2.5 bg-terra-500 rounded-lg text-body-sm font-medium text-white hover:bg-terra-600 transition-colors"
                        >
                            Analyze
                        </button>
                    </form>
                </div>
            </Section>

            {/* Main Content */}
            <Section spacing="md" background="default">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Calendar Column */}
                    <div className="lg:col-span-2 space-y-6">
                        {loading ? (
                            <div className="text-center py-12">
                                <div className="w-10 h-10 border-4 border-navy-200 border-t-navy-600 rounded-full animate-spin mx-auto mb-4" />
                                <p className="text-neutral-600">Loading earnings calendar...</p>
                            </div>
                        ) : error ? (
                            <div className="text-center py-12">
                                <p className="text-error-600 mb-4">{error}</p>
                                <button onClick={loadCalendar} className="px-4 py-2 bg-terra-500 text-white rounded-lg">
                                    Retry
                                </button>
                            </div>
                        ) : (
                            <>
                                {/* Upcoming (within 14 days) */}
                                {upcomingEvents.length > 0 && (
                                    <div>
                                        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4 flex items-center gap-2">
                                            <span className="w-3 h-3 bg-warning-500 rounded-full animate-pulse" />
                                            Upcoming (Next 2 Weeks)
                                        </h2>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                            {upcomingEvents.map(event => (
                                                <EarningsCard
                                                    key={event.ticker}
                                                    event={event}
                                                    onSelect={setSelectedTicker}
                                                    isSelected={selectedTicker === event.ticker}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Future */}
                                {futureEvents.length > 0 && (
                                    <div>
                                        <h2 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                                            Scheduled Later
                                        </h2>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                            {futureEvents.map(event => (
                                                <EarningsCard
                                                    key={event.ticker}
                                                    event={event}
                                                    onSelect={setSelectedTicker}
                                                    isSelected={selectedTicker === event.ticker}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {events.length === 0 && (
                                    <div className="text-center py-12 bg-cream-50 rounded-xl">
                                        <p className="text-neutral-600">
                                            No upcoming earnings found for your watchlist stocks.
                                        </p>
                                        <p className="text-neutral-500 text-body-sm mt-2">
                                            Add stocks to your watchlist or search for a specific ticker.
                                        </p>
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                    {/* Analysis Panel */}
                    <div className="space-y-4">
                        {selectedTicker ? (
                            analysisLoading ? (
                                <div className="bg-white rounded-xl border border-border-light p-6 text-center">
                                    <div className="w-8 h-8 border-3 border-navy-200 border-t-navy-600 rounded-full animate-spin mx-auto mb-4" />
                                    <p className="text-neutral-600">Loading analysis...</p>
                                </div>
                            ) : analysis ? (
                                <>
                                    {/* Company Info */}
                                    <div className="bg-white rounded-xl border border-border-light p-6">
                                        <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-2">
                                            {analysis.ticker}
                                        </h3>
                                        <p className="text-body-md text-neutral-600 mb-4">{analysis.company_name}</p>

                                        {analysis.upcoming_earnings && (
                                            <div className="p-3 bg-warning-50 rounded-lg mb-4">
                                                <p className="text-body-sm text-warning-800">
                                                    <strong>Earnings in {analysis.upcoming_earnings.days_until} days</strong>
                                                    <br />
                                                    {analysis.upcoming_earnings.date}
                                                </p>
                                            </div>
                                        )}

                                        {analysis.market_context && (
                                            <div className="grid grid-cols-2 gap-3 text-body-sm">
                                                <div>
                                                    <p className="text-neutral-500">Price</p>
                                                    <p className="font-semibold">${analysis.market_context.current_price?.toFixed(2)}</p>
                                                </div>
                                                <div>
                                                    <p className="text-neutral-500">20D Momentum</p>
                                                    <p className={`font-semibold ${(analysis.market_context.momentum_20d || 0) >= 0 ? 'text-success-600' : 'text-error-600'}`}>
                                                        {(analysis.market_context.momentum_20d || 0) >= 0 ? '+' : ''}
                                                        {analysis.market_context.momentum_20d?.toFixed(1)}%
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Historical Performance */}
                                    {analysis.historical_performance && (
                                        <div className="bg-white rounded-xl border border-border-light p-6">
                                            <h4 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                                                Historical Performance
                                            </h4>
                                            <div className="grid grid-cols-3 gap-3 text-center mb-4">
                                                <div className="p-2 bg-cream-50 rounded-lg">
                                                    <p className="text-display-xs font-semibold text-navy-900">
                                                        {analysis.historical_performance.beats}
                                                    </p>
                                                    <p className="text-caption text-success-600">Beats</p>
                                                </div>
                                                <div className="p-2 bg-cream-50 rounded-lg">
                                                    <p className="text-display-xs font-semibold text-navy-900">
                                                        {analysis.historical_performance.misses}
                                                    </p>
                                                    <p className="text-caption text-error-600">Misses</p>
                                                </div>
                                                <div className="p-2 bg-cream-50 rounded-lg">
                                                    <p className="text-display-xs font-semibold text-terra-600">
                                                        {analysis.historical_performance.beat_rate?.toFixed(0)}%
                                                    </p>
                                                    <p className="text-caption text-neutral-600">Beat Rate</p>
                                                </div>
                                            </div>

                                            {analysis.recent_quarters && analysis.recent_quarters.length > 0 && (
                                                <EarningsHistoryChart history={analysis.recent_quarters} />
                                            )}
                                        </div>
                                    )}

                                    {/* Volatility */}
                                    <VolatilityIndicator volatility={analysis.volatility} />
                                </>
                            ) : (
                                <div className="bg-white rounded-xl border border-border-light p-6 text-center">
                                    <p className="text-neutral-600">Unable to load analysis</p>
                                </div>
                            )
                        ) : (
                            <div className="bg-cream-50 rounded-xl p-6 text-center">
                                <svg className="w-12 h-12 text-neutral-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                <p className="text-neutral-600 font-medium mb-2">Select a Stock</p>
                                <p className="text-body-sm text-neutral-500">
                                    Click on any earnings card or search for a ticker to see detailed analysis
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </Section>
        </>
    );
}
