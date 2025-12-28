'use client';

// =============================================================================
// ALERTS COMPONENTS
// =============================================================================
// Components for displaying portfolio alerts
//
// Location: frontend/components/alerts/AlertsPanel.tsx
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

// =============================================================================
// TYPES
// =============================================================================

interface Alert {
    type: string;
    ticker: string;
    severity: 'high' | 'medium' | 'low';
    title: string;
    description: string;
    data?: Record<string, any>;
    timestamp: string;
}

interface AlertType {
    type: string;
    name: string;
    description: string;
    severity_levels: string[];
}

// =============================================================================
// API BASE
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

// =============================================================================
// ALERT CARD COMPONENT
// =============================================================================

export const AlertCard: React.FC<{
    alert: Alert;
    compact?: boolean;
    onDismiss?: () => void;
}> = ({ alert, compact = false, onDismiss }) => {
    const getSeverityStyles = (severity: string) => {
        switch (severity) {
            case 'high':
                return {
                    container: 'bg-error-50 border-l-error-500',
                    badge: 'bg-error-100 text-error-700',
                    icon: 'text-error-500',
                };
            case 'medium':
                return {
                    container: 'bg-warning-50 border-l-warning-500',
                    badge: 'bg-warning-100 text-warning-700',
                    icon: 'text-warning-500',
                };
            default:
                return {
                    container: 'bg-cream-50 border-l-neutral-400',
                    badge: 'bg-neutral-100 text-neutral-700',
                    icon: 'text-neutral-500',
                };
        }
    };

    const getAlertIcon = (type: string) => {
        switch (type) {
            case 'volume_spike':
                return (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                );
            case 'price_breakout_high':
            case 'price_breakout_low':
                return (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                );
            case 'earnings_approaching':
                return (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                );
            case 'high_put_call_ratio':
                return (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                );
            case 'beta_divergence':
                return (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                    </svg>
                );
            default:
                return (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                );
        }
    };

    const styles = getSeverityStyles(alert.severity);

    if (compact) {
        return (
            <div className={`p-3 rounded-lg border-l-4 ${styles.container} flex items-center gap-3`}>
                <span className={styles.icon}>{getAlertIcon(alert.type)}</span>
                <div className="flex-1 min-w-0">
                    <p className="text-body-sm font-medium text-navy-900 truncate">{alert.title}</p>
                </div>
                <span className="text-caption text-neutral-500 whitespace-nowrap">{alert.ticker}</span>
            </div>
        );
    }

    return (
        <div className={`p-4 rounded-lg border-l-4 ${styles.container}`}>
            <div className="flex items-start gap-3">
                <span className={styles.icon}>{getAlertIcon(alert.type)}</span>

                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium text-navy-900">{alert.title}</h4>
                        <span className={`px-2 py-0.5 rounded text-caption font-medium ${styles.badge}`}>
                            {alert.severity}
                        </span>
                    </div>

                    <p className="text-body-sm text-neutral-600 mb-2">{alert.description}</p>

                    {/* Alert-specific data */}
                    {alert.data && (
                        <div className="flex flex-wrap gap-2">
                            {Object.entries(alert.data).map(([key, value]) => (
                                <span key={key} className="px-2 py-1 bg-white rounded text-caption">
                                    <span className="text-neutral-500">{key.replace(/_/g, ' ')}: </span>
                                    <span className="font-medium">{typeof value === 'number' ? value.toLocaleString() : value}</span>
                                </span>
                            ))}
                        </div>
                    )}
                </div>

                {onDismiss && (
                    <button
                        onClick={onDismiss}
                        className="text-neutral-400 hover:text-neutral-600"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                )}
            </div>
        </div>
    );
};

// =============================================================================
// ALERTS PANEL COMPONENT
// =============================================================================

export const AlertsPanel: React.FC<{
    tickers: string[];
    maxAlerts?: number;
    compact?: boolean;
    showRefresh?: boolean;
}> = ({ tickers, maxAlerts = 10, compact = false, showRefresh = true }) => {
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dismissed, setDismissed] = useState<Set<string>>(new Set());

    const fetchAlerts = useCallback(async () => {
        if (tickers.length === 0) return;

        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_BASE}/api/alerts/check`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers }),
            });

            if (!res.ok) throw new Error('Failed to fetch alerts');
            const data = await res.json();
            setAlerts(data.alerts || []);
        } catch (err) {
            setError('Unable to load alerts');
        } finally {
            setLoading(false);
        }
    }, [tickers.join(',')]);

    useEffect(() => {
        fetchAlerts();
    }, [fetchAlerts]);

    const dismissAlert = (alert: Alert) => {
        const key = `${alert.ticker}-${alert.type}`;
        setDismissed(prev => new Set(prev).add(key));
    };

    const visibleAlerts = alerts
        .filter(a => !dismissed.has(`${a.ticker}-${a.type}`))
        .slice(0, maxAlerts);

    if (tickers.length === 0) {
        return (
            <div className="bg-white rounded-xl border border-border-light p-6">
                <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-4">
                    Portfolio Alerts
                </h3>
                <p className="text-neutral-500 text-center py-6">
                    Add stocks to your watchlist to receive alerts
                </p>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-xl border border-border-light p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-heading font-semibold text-heading-md text-navy-900">
                    Portfolio Alerts
                    {visibleAlerts.length > 0 && (
                        <span className="ml-2 px-2 py-0.5 bg-error-100 text-error-700 rounded-full text-caption">
                            {visibleAlerts.length}
                        </span>
                    )}
                </h3>

                {showRefresh && (
                    <button
                        onClick={fetchAlerts}
                        disabled={loading}
                        className="p-2 text-neutral-400 hover:text-neutral-600 hover:bg-cream-50 rounded-lg transition-colors"
                    >
                        <svg className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                    </button>
                )}
            </div>

            {error && (
                <p className="text-error-600 text-center py-4">{error}</p>
            )}

            {loading && alerts.length === 0 ? (
                <div className="space-y-3">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="animate-pulse h-16 bg-cream-100 rounded-lg" />
                    ))}
                </div>
            ) : visibleAlerts.length > 0 ? (
                <div className="space-y-3">
                    {visibleAlerts.map((alert, i) => (
                        <AlertCard
                            key={`${alert.ticker}-${alert.type}-${i}`}
                            alert={alert}
                            compact={compact}
                            onDismiss={compact ? undefined : () => dismissAlert(alert)}
                        />
                    ))}
                </div>
            ) : (
                <div className="text-center py-8 bg-success-50 rounded-lg">
                    <svg className="w-12 h-12 text-success-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-success-700 font-medium">No alerts</p>
                    <p className="text-success-600 text-body-sm">Your watchlist stocks look stable</p>
                </div>
            )}
        </div>
    );
};

// =============================================================================
// ALERT BADGE (for navbar)
// =============================================================================

export const AlertBadge: React.FC<{
    tickers: string[];
}> = ({ tickers }) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        const fetchCount = async () => {
            if (tickers.length === 0) return;

            try {
                const res = await fetch(`${API_BASE}/api/alerts/check`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tickers }),
                });

                if (res.ok) {
                    const data = await res.json();
                    setCount(data.count || 0);
                }
            } catch {
                // Silent fail
            }
        };

        fetchCount();
        const interval = setInterval(fetchCount, 5 * 60 * 1000); // Refresh every 5 min

        return () => clearInterval(interval);
    }, [tickers.join(',')]);

    if (count === 0) return null;

    return (
        <span className="absolute -top-1 -right-1 w-5 h-5 bg-error-500 text-white text-caption font-bold rounded-full flex items-center justify-center">
            {count > 9 ? '9+' : count}
        </span>
    );
};

export default AlertsPanel;
