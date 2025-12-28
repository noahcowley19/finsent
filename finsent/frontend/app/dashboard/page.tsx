'use client';

// =============================================================================
// DASHBOARD PAGE - Modern SaaS Aesthetic
// =============================================================================

import React, { useEffect } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';
import { Section, Grid } from '@/components/layout';
import { StatsCard, RecentActivity, WatchlistPreview } from '@/components/dashboard';
import type { Activity as DashboardActivity, WatchlistStock } from '@/components/dashboard';
import { useWatchlist, usePortfolio } from '@/lib/hooks';
import { useActivities, type Activity } from '@/components/utils/activity-tracker';

// Convert activity tracker format to dashboard Activity format
const convertActivity = (activity: Activity): DashboardActivity => ({
  id: activity.id,
  type: activity.type === 'quant' ? 'analysis' : activity.type as DashboardActivity['type'],
  title: activity.title,
  description: activity.description,
  timestamp: activity.timestamp,
  link: activity.link,
});


function DashboardContent() {
  const { user } = useAuth();
  const firstName = user?.name?.split(' ')[0] || 'there';

  // Use real hooks
  const { enrichedItems, refresh: refreshWatchlist } = useWatchlist();
  const { positions, analysis, analyze } = usePortfolio();
  const { activities: rawActivities, isLoading: activitiesLoading } = useActivities(10);

  // Convert activity tracker format to dashboard format
  const userActivities: DashboardActivity[] = rawActivities.map(convertActivity);

  // Refresh data on mount
  useEffect(() => {
    if (enrichedItems.length > 0) {
      refreshWatchlist();
    }
    if (positions.length > 0) {
      analyze();
    }
  }, []);

  // Convert enriched watchlist items to dashboard format
  const watchlistStocks: WatchlistStock[] = enrichedItems.slice(0, 5).map(item => ({
    symbol: item.ticker,
    name: item.company || item.ticker,
    price: item.price || 0,
    change: item.change || 0,
    changePercent: item.pct_change || 0,
  }));

  // Calculate stats
  const searchesUsed = 3;
  const searchesLimit = user?.tier === 'pro' ? '∞' : 10;
  const analysesUsed = 1;
  const analysesLimit = user?.tier === 'pro' ? '∞' : 3;

  // Portfolio value from analysis
  const portfolioValue = analysis?.portfolio_metrics?.total_value || 0;
  const portfolioChange = analysis?.portfolio_metrics?.total_gain_loss_percent || 0;

  return (
    <div className="min-h-screen bg-ink-50">
      {/* Header Section */}
      <Section spacing="md" background="white">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="text-display-sm lg:text-display-md text-ink-900 tracking-tight mb-2">
              Welcome back, {firstName}
            </h1>
            <p className="text-body-md text-ink-500">
              Here&apos;s what&apos;s happening with your investments today.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/search"
              className="inline-flex items-center gap-2 h-10 px-4 bg-white border border-ink-200 rounded-lg text-body-sm font-medium text-ink-700 hover:bg-ink-50 hover:border-ink-300 transition-all"
            >
              <svg className="w-4 h-4 text-ink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Search
            </Link>
            <Link
              href="/sentiment"
              className="inline-flex items-center gap-2 h-10 px-4 bg-ink-900 rounded-lg text-body-sm font-medium text-white hover:bg-ink-800 transition-all"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Analysis
            </Link>
          </div>
        </div>
      </Section>

      {/* Stats Grid */}
      <Section spacing="sm" background="default">
        <Grid cols={1} colsMd={2} colsLg={4} gap="md">
          <StatsCard
            title="Searches Today"
            value={`${searchesUsed}/${searchesLimit}`}
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            }
            iconColor="default"
          />
          <StatsCard
            title="Analyses Today"
            value={`${analysesUsed}/${analysesLimit}`}
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            }
            iconColor="accent"
          />
          <StatsCard
            title="Portfolio Value"
            value={portfolioValue > 0 ? `$${portfolioValue.toLocaleString()}` : '$0'}
            change={portfolioValue > 0 ? { value: portfolioChange, label: 'total' } : undefined}
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
            iconColor="success"
          />
          <StatsCard
            title="Watchlist"
            value={enrichedItems.length}
            subtitle={user?.tier === 'free' ? `${10 - enrichedItems.length} slots left` : 'Unlimited'}
            icon={
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            }
            iconColor="warning"
          />
        </Grid>
      </Section>

      {/* Main Content */}
      <Section spacing="md" background="default">
        <Grid cols={1} colsLg={3} gap="lg">
          {/* Activity & Quick Actions - 2 columns */}
          <div className="lg:col-span-2 space-y-5">
            <RecentActivity activities={userActivities} />

            {/* Quick Actions */}
            <div className="bg-white rounded-xl border border-ink-200/50 p-5">
              <h3 className="font-semibold text-heading-sm text-ink-900 tracking-tight mb-4">
                Quick Actions
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Link
                  href="/sentiment"
                  className="flex flex-col items-center gap-2 p-4 rounded-xl bg-ink-50 hover:bg-ink-100 transition-colors group"
                >
                  <div className="w-10 h-10 rounded-lg bg-accent/10 text-accent flex items-center justify-center group-hover:bg-accent/20 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-ink-700">Sentiment</span>
                </Link>
                <Link
                  href="/financials"
                  className="flex flex-col items-center gap-2 p-4 rounded-xl bg-ink-50 hover:bg-ink-100 transition-colors group"
                >
                  <div className="w-10 h-10 rounded-lg bg-ink-200 text-ink-600 flex items-center justify-center group-hover:bg-ink-300 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-ink-700">Financials</span>
                </Link>
                <Link
                  href="/insider"
                  className="flex flex-col items-center gap-2 p-4 rounded-xl bg-ink-50 hover:bg-ink-100 transition-colors group"
                >
                  <div className="w-10 h-10 rounded-lg bg-success-50 text-success-600 flex items-center justify-center group-hover:bg-success-100 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-ink-700">Insider</span>
                </Link>
                <Link
                  href="/portfolio"
                  className="flex flex-col items-center gap-2 p-4 rounded-xl bg-ink-50 hover:bg-ink-100 transition-colors group"
                >
                  <div className="w-10 h-10 rounded-lg bg-warning-50 text-warning-600 flex items-center justify-center group-hover:bg-warning-100 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-ink-700">Portfolio</span>
                </Link>
              </div>
            </div>

            {/* Upgrade banner for free users */}
            {user?.tier === 'free' && (
              <div className="bg-gradient-to-r from-ink-900 to-ink-800 rounded-xl p-6">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h3 className="font-semibold text-heading-sm text-white mb-1">
                      Upgrade to Pro
                    </h3>
                    <p className="text-body-sm text-ink-300">
                      Unlimited analyses, Quant Lab access, and more.
                    </p>
                  </div>
                  <Link
                    href="/pricing"
                    className="px-4 py-2.5 bg-white text-ink-900 font-medium text-body-sm rounded-lg hover:bg-ink-50 transition-colors flex-shrink-0"
                  >
                    View Plans
                  </Link>
                </div>
              </div>
            )}
          </div>

          {/* Watchlist - 1 column */}
          <div>
            <WatchlistPreview stocks={watchlistStocks} />
          </div>
        </Grid>
      </Section>
    </div>
  );
}

export default function DashboardPage() {
  return <DashboardContent />;
}
