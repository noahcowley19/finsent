'use client';

// =============================================================================
// ACTIVITY TRACKER
// =============================================================================
// Utility for tracking and storing user activities
// Used for recent activity display on dashboard and analytics
//
// Location: frontend/components/utils/activity-tracker.ts
// =============================================================================

export type ActivityType = 'search' | 'analysis' | 'watchlist' | 'portfolio' | 'quant' | 'login';

export interface Activity {
    id: string;
    type: ActivityType;
    title: string;
    description: string;
    timestamp: Date;
    link?: string;
    metadata?: Record<string, unknown>;
}

interface StoredActivity extends Omit<Activity, 'timestamp'> {
    timestamp: string;
}

const STORAGE_KEY = 'caveray_user_activities';
const MAX_ACTIVITIES = 50;
const ACTIVITY_RETENTION_DAYS = 30;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

const generateId = (): string => {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

const cleanOldActivities = (activities: StoredActivity[]): StoredActivity[] => {
    const cutoff = Date.now() - (ACTIVITY_RETENTION_DAYS * 24 * 60 * 60 * 1000);
    return activities.filter(a => new Date(a.timestamp).getTime() > cutoff);
};

// =============================================================================
// STORAGE FUNCTIONS
// =============================================================================

export const getActivities = (): Activity[] => {
    if (typeof window === 'undefined') return [];

    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (!stored) return [];

        const activities = JSON.parse(stored) as StoredActivity[];
        const cleaned = cleanOldActivities(activities);

        // Convert timestamps back to Date objects
        return cleaned.map(a => ({
            ...a,
            timestamp: new Date(a.timestamp)
        }));
    } catch (err) {
        console.error('Failed to load activities:', err);
        return [];
    }
};

export const saveActivity = (activity: Omit<Activity, 'id' | 'timestamp'>): Activity => {
    const newActivity: Activity = {
        ...activity,
        id: generateId(),
        timestamp: new Date(),
    };

    if (typeof window === 'undefined') return newActivity;

    try {
        const existing = getActivities();
        const updated = [newActivity, ...existing].slice(0, MAX_ACTIVITIES);

        // Convert to storable format
        const storable: StoredActivity[] = updated.map(a => ({
            ...a,
            timestamp: a.timestamp.toISOString(),
        }));

        localStorage.setItem(STORAGE_KEY, JSON.stringify(storable));
        return newActivity;
    } catch (err) {
        console.error('Failed to save activity:', err);
        return newActivity;
    }
};

export const clearActivities = (): void => {
    if (typeof window === 'undefined') return;

    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
        console.error('Failed to clear activities:', err);
    }
};

// =============================================================================
// CONVENIENCE FUNCTIONS FOR COMMON ACTIVITY TYPES
// =============================================================================

export const trackSearch = (ticker: string, companyName?: string): Activity => {
    return saveActivity({
        type: 'search',
        title: `Searched for ${ticker}`,
        description: companyName || ticker,
        link: `/stock/${ticker}`,
    });
};

export const trackAnalysis = (ticker: string, analysisType: string, summary?: string): Activity => {
    return saveActivity({
        type: 'analysis',
        title: `${analysisType} Analysis`,
        description: summary || `${ticker} - ${analysisType} analysis completed`,
        link: `/sentiment/${ticker}`,
        metadata: { analysisType },
    });
};

export const trackWatchlistAdd = (ticker: string, companyName?: string): Activity => {
    return saveActivity({
        type: 'watchlist',
        title: 'Added to Watchlist',
        description: `${ticker} - ${companyName || 'Stock added'}`,
        link: '/watchlist',
    });
};

export const trackWatchlistRemove = (ticker: string): Activity => {
    return saveActivity({
        type: 'watchlist',
        title: 'Removed from Watchlist',
        description: `${ticker} removed from your watchlist`,
        link: '/watchlist',
    });
};

export const trackPortfolioUpdate = (action: string, ticker?: string, shares?: number): Activity => {
    const description = ticker && shares
        ? `${action} ${shares} shares of ${ticker}`
        : action;

    return saveActivity({
        type: 'portfolio',
        title: 'Portfolio Updated',
        description,
        link: '/portfolio',
        metadata: { ticker, shares },
    });
};

export const trackQuantLabRun = (strategyName: string, symbol: string): Activity => {
    return saveActivity({
        type: 'quant',
        title: 'Strategy Backtest',
        description: `Ran ${strategyName} on ${symbol}`,
        link: '/quant-lab',
        metadata: { strategyName, symbol },
    });
};

export const trackLogin = (): Activity => {
    return saveActivity({
        type: 'login',
        title: 'Signed In',
        description: 'Successfully signed into your account',
    });
};

// =============================================================================
// REACT HOOK
// =============================================================================

import { useState, useEffect, useCallback } from 'react';

export const useActivities = (limit: number = 10) => {
    const [activities, setActivities] = useState<Activity[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    const refresh = useCallback(() => {
        const loaded = getActivities().slice(0, limit);
        setActivities(loaded);
        setIsLoading(false);
    }, [limit]);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const addActivity = useCallback((activity: Omit<Activity, 'id' | 'timestamp'>) => {
        const newActivity = saveActivity(activity);
        setActivities(prev => [newActivity, ...prev].slice(0, limit));
        return newActivity;
    }, [limit]);

    const clear = useCallback(() => {
        clearActivities();
        setActivities([]);
    }, []);

    return {
        activities,
        isLoading,
        refresh,
        addActivity,
        clear,
        trackSearch,
        trackAnalysis,
        trackWatchlistAdd,
        trackWatchlistRemove,
        trackPortfolioUpdate,
        trackQuantLabRun,
        trackLogin,
    };
};

export default useActivities;
