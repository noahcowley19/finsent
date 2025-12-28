'use client';

// =============================================================================
// RECENT SEARCHES COMPONENT
// =============================================================================
// Shows user's recent stock searches with quick-access buttons
//
// Location: frontend/components/search/RecentSearches.tsx
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';

interface RecentSearch {
    ticker: string;
    companyName?: string;
    timestamp: number;
}

const STORAGE_KEY = 'caveray_recent_searches';
const MAX_SEARCHES = 10;
const RETENTION_DAYS = 30;

// =============================================================================
// STORAGE FUNCTIONS
// =============================================================================

export const getRecentSearches = (): RecentSearch[] => {
    if (typeof window === 'undefined') return [];

    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (!stored) return [];

        const searches = JSON.parse(stored) as RecentSearch[];
        const cutoff = Date.now() - (RETENTION_DAYS * 24 * 60 * 60 * 1000);

        return searches.filter(s => s.timestamp > cutoff);
    } catch (err) {
        console.error('Failed to load recent searches:', err);
        return [];
    }
};

export const addRecentSearch = (ticker: string, companyName?: string): void => {
    if (typeof window === 'undefined') return;

    try {
        const existing = getRecentSearches();

        // Remove if already exists (will be re-added at top)
        const filtered = existing.filter(s => s.ticker !== ticker.toUpperCase());

        const newSearch: RecentSearch = {
            ticker: ticker.toUpperCase(),
            companyName,
            timestamp: Date.now(),
        };

        const updated = [newSearch, ...filtered].slice(0, MAX_SEARCHES);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (err) {
        console.error('Failed to add recent search:', err);
    }
};

export const clearRecentSearches = (): void => {
    if (typeof window === 'undefined') return;

    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
        console.error('Failed to clear recent searches:', err);
    }
};

// =============================================================================
// COMPONENT
// =============================================================================

interface RecentSearchesProps {
    /** Maximum number of searches to display */
    limit?: number;
    /** Show as compact pills or list */
    variant?: 'pills' | 'list';
    /** On search click callback */
    onSearchClick?: (ticker: string) => void;
    /** Additional className */
    className?: string;
}

export const RecentSearches: React.FC<RecentSearchesProps> = ({
    limit = 5,
    variant = 'pills',
    onSearchClick,
    className = '',
}) => {
    const [searches, setSearches] = useState<RecentSearch[]>([]);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);
        setSearches(getRecentSearches().slice(0, limit));
    }, [limit]);

    const handleClear = useCallback(() => {
        clearRecentSearches();
        setSearches([]);
    }, []);

    const handleClick = useCallback((ticker: string) => {
        if (onSearchClick) {
            onSearchClick(ticker);
        }
    }, [onSearchClick]);

    // Wait for client-side render
    if (!isClient || searches.length === 0) {
        return null;
    }

    if (variant === 'pills') {
        return (
            <div className={`${className}`}>
                <div className="flex items-center justify-between mb-3">
                    <h4 className="text-body-sm font-medium text-neutral-600">Recent Searches</h4>
                    <button
                        onClick={handleClear}
                        className="text-caption text-neutral-400 hover:text-neutral-600 transition-colors"
                    >
                        Clear
                    </button>
                </div>
                <div className="flex flex-wrap gap-2">
                    {searches.map((search) => (
                        <Link
                            key={search.ticker}
                            href={`/stock/${search.ticker}`}
                            onClick={() => handleClick(search.ticker)}
                            className="
                inline-flex items-center gap-1.5
                px-3 py-1.5
                bg-cream-100 hover:bg-cream-200
                text-navy-700
                text-body-sm font-medium
                rounded-full
                transition-all duration-fast
                hover:-translate-y-0.5 hover:shadow-sm
                group
              "
                        >
                            <span className="text-terra-500 font-semibold">{search.ticker}</span>
                            {search.companyName && (
                                <span className="text-neutral-500 text-caption hidden sm:inline">
                                    {search.companyName.slice(0, 15)}
                                </span>
                            )}
                            <svg
                                className="w-3 h-3 text-neutral-400 group-hover:text-navy-500 transition-colors"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                        </Link>
                    ))}
                </div>
            </div>
        );
    }

    // List variant
    return (
        <div className={`bg-white rounded-xl border border-border-light p-4 ${className}`}>
            <div className="flex items-center justify-between mb-4">
                <h4 className="font-heading font-semibold text-heading-sm text-navy-900">
                    Recent Searches
                </h4>
                <button
                    onClick={handleClear}
                    className="text-body-sm text-neutral-500 hover:text-error-500 transition-colors"
                >
                    Clear All
                </button>
            </div>
            <div className="space-y-2">
                {searches.map((search, index) => (
                    <Link
                        key={search.ticker}
                        href={`/stock/${search.ticker}`}
                        onClick={() => handleClick(search.ticker)}
                        className="
              flex items-center justify-between
              p-3
              rounded-lg
              hover:bg-cream-50
              transition-colors
              group
            "
                        style={{ animationDelay: `${index * 50}ms` }}
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-navy-100 flex items-center justify-center">
                                <span className="font-semibold text-body-sm text-navy-700">
                                    {search.ticker.slice(0, 2)}
                                </span>
                            </div>
                            <div>
                                <p className="font-medium text-body-md text-navy-900">{search.ticker}</p>
                                {search.companyName && (
                                    <p className="text-caption text-neutral-500">{search.companyName}</p>
                                )}
                            </div>
                        </div>
                        <svg
                            className="w-5 h-5 text-neutral-400 group-hover:text-terra-500 transition-colors"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </Link>
                ))}
            </div>
        </div>
    );
};

// =============================================================================
// HOOK
// =============================================================================

export const useRecentSearches = (limit: number = 10) => {
    const [searches, setSearches] = useState<RecentSearch[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    const refresh = useCallback(() => {
        setSearches(getRecentSearches().slice(0, limit));
        setIsLoading(false);
    }, [limit]);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const addSearch = useCallback((ticker: string, companyName?: string) => {
        addRecentSearch(ticker, companyName);
        refresh();
    }, [refresh]);

    const clear = useCallback(() => {
        clearRecentSearches();
        setSearches([]);
    }, []);

    return {
        searches,
        isLoading,
        addSearch,
        clear,
        refresh,
    };
};

export default RecentSearches;
