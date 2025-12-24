'use client';

// =============================================================================
// DASHBOARD PAGE
// =============================================================================
// Main dashboard for authenticated users
//
// Location: frontend/app/dashboard/page.tsx
//
// =============================================================================

import React, { useEffect } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';
import { PageHeader, Container, Section, Grid, AuthGuard } from '@/components/layout';
import { StatsCard, RecentActivity, WatchlistPreview } from '@/components/dashboard';
import type { Activity, WatchlistStock } from '@/components/dashboard';
import { useWatchlist, usePortfolio } from '@/lib/hooks';

// Mock activities - ideally this would come from a user activity API
const mockActivities: Activity[] = [
  {
    id: '1',
    type: 'search',
    title: 'Searched for AAPL',
    description: 'Apple Inc.',
    timestamp: new Date(Date.now() - 1000 * 60 * 5),
    link: '/stock/AAPL',
  },
  {
    id: '2',
    type: 'analysis',
    title: 'Sentiment Analysis',
    description: 'TSLA - Bullish sentiment detected',
    timestamp: new Date(Date.now() - 1000 * 60 * 30),
    link: '/sentiment/TSLA',
  },
  {
    id: '3',
    type: 'watchlist',
    title: 'Added to Watchlist',
    description: 'NVDA - NVIDIA Corporation',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2),
  },
  {
    id: '4',
    type: 'portfolio',
    title: 'Portfolio Updated',
    description: 'Added 10 shares of MSFT',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24),
  },
];

function DashboardContent() {
  const { user } = useAuth();
  const firstName = user?.name?.split(' ')[0] || 'there';
  
  // Use real hooks
  const { enrichedItems, refresh: refreshWatchlist } = useWatchlist();
  const { positions, analysis, analyze } = usePortfolio();

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
    name: item.name || item.ticker,
    price: item.current_price || 0,
    change: item.change_dollar || 0,
    changePercent: item.change_percent || 0,
  }));

  // Calculate stats (mock for now - ideally from user API)
  const searchesUsed = 3;
  const searchesLimit = user?.tier === 'pro' ? '∞' : 10;
  const analysesUsed = 1;
  const analysesLimit = user?.tier === 'pro' ? '∞' : 3;
  
  // Portfolio value from analysis
  const portfolioValue = analysis?.portfolio_metrics?.total_value || 0;
  const portfolioChange = analysis?.portfolio_metrics?.total_gain_loss_percent || 0;

  return (
    <>
      {/* Header */}
      <Section spacing="md" background="gradient">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="font-display text-display-sm lg:text-display-md text-navy-900 mb-2">
              Welcome back, {firstName}
            </h1>
            <p className="text-body-md text-neutral-600">
              Here&apos;s what&apos;s happening with your investments today.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/search"
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Search Stocks
            </Link>
            <Link
              href="/sentiment"
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-terra-500 rounded-lg text-body-sm font-medium text-white hover:bg-terra-600 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              New Analysis
            </Link>
          </div>
        </div>
      </Section>

      {/* Stats */}
      <Section spacing="md" background="default">
        <Grid cols={1} colsMd={2} colsLg={4} gap="md">
          <StatsCard
            title="Searches Today"
            value={`${searchesUsed}/${searchesLimit}`}
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            }
            iconColor="navy"
          />
          <StatsCard
            title="Analyses Today"
            value={`${analysesUsed}/${analysesLimit}`}
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            }
            iconColor="terra"
          />
          <StatsCard
            title="Portfolio Value"
            value={portfolioValue > 0 ? `$${portfolioValue.toLocaleString()}` : '$0'}
            change={portfolioValue > 0 ? { value: portfolioChange, label: 'total return' } : undefined}
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
            iconColor="success"
          />
          <StatsCard
            title="Watchlist Stocks"
            value={enrichedItems.length}
            subtitle={user?.tier === 'free' ? `${10 - enrichedItems.length} slots remaining` : 'Unlimited'}
            icon={
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            }
            iconColor="warning"
          />
        </Grid>
      </Section>

      {/* Main content */}
      <Section spacing="lg" background="default">
        <Grid cols={1} colsLg={3} gap="lg">
          {/* Activity & Quick Actions - 2 columns */}
          <div className="lg:col-span-2 space-y-6">
            <RecentActivity activities={mockActivities} />

            {/* Quick Actions */}
            <div className="bg-white rounded-xl border border-border-light p-6">
              <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-4">
                Quick Actions
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Link
                  href="/sentiment"
                  className="flex flex-col items-center gap-2 p-4 rounded-lg bg-cream-50 hover:bg-cream-100 transition-colors text-center"
                >
                  <div className="w-10 h-10 rounded-lg bg-terra-100 text-terra-600 flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-navy-700">Sentiment</span>
                </Link>
                <Link
                  href="/financials"
                  className="flex flex-col items-center gap-2 p-4 rounded-lg bg-cream-50 hover:bg-cream-100 transition-colors text-center"
                >
                  <div className="w-10 h-10 rounded-lg bg-navy-100 text-navy-600 flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-navy-700">Financials</span>
                </Link>
                <Link
                  href="/insider"
                  className="flex flex-col items-center gap-2 p-4 rounded-lg bg-cream-50 hover:bg-cream-100 transition-colors text-center"
                >
                  <div className="w-10 h-10 rounded-lg bg-success-100 text-success-600 flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-navy-700">Insider</span>
                </Link>
                <Link
                  href="/portfolio"
                  className="flex flex-col items-center gap-2 p-4 rounded-lg bg-cream-50 hover:bg-cream-100 transition-colors text-center"
                >
                  <div className="w-10 h-10 rounded-lg bg-warning-100 text-warning-600 flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <span className="text-body-sm font-medium text-navy-700">Portfolio</span>
                </Link>
              </div>
            </div>

            {/* Upgrade banner for free users */}
            {user?.tier === 'free' && (
              <div className="bg-gradient-to-r from-terra-500 to-terra-600 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-heading font-semibold text-lg mb-1">
                      Upgrade to Pro
                    </h3>
                    <p className="text-white/80 text-body-sm">
                      Get unlimited analyses, Quant Lab access, and more.
                    </p>
                  </div>
                  <Link
                    href="/pricing"
                    className="px-4 py-2 bg-white text-terra-600 font-medium rounded-lg hover:bg-cream-50 transition-colors"
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
    </>
  );
}

export default function DashboardPage() {
  return (
    <AuthGuard>
      <DashboardContent />
    </AuthGuard>
  );
}
