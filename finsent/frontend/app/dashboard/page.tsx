'use client';

// =============================================================================
// DASHBOARD PAGE - Enhanced with More Widgets
// =============================================================================

import React, { useEffect } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';
import { Section, Grid } from '@/components/layout';
import { StatsCard, RecentActivity, WatchlistPreview } from '@/components/dashboard';
import type { Activity as DashboardActivity, WatchlistStock } from '@/components/dashboard';
import { useWatchlist, usePortfolio, useMarketMovers } from '@/lib/hooks';
import { useActivities, type Activity } from '@/components/utils/activity-tracker';
import { ScrollReveal } from '@/components/ui';

// Convert activity tracker format to dashboard Activity format
const convertActivity = (activity: Activity): DashboardActivity => ({
  id: activity.id,
  type: activity.type === 'quant' ? 'analysis' : activity.type as DashboardActivity['type'],
  title: activity.title,
  description: activity.description,
  timestamp: activity.timestamp,
  link: activity.link,
});

// =============================================================================
// MARKET OVERVIEW WIDGET
// =============================================================================

const MarketOverviewWidget: React.FC = () => {
  const indices = [
    { name: 'S&P 500', value: '4,567.89', change: '+1.2%', up: true },
    { name: 'NASDAQ', value: '14,234.56', change: '+1.8%', up: true },
    { name: 'DOW', value: '35,678.90', change: '+0.8%', up: true },
    { name: 'VIX', value: '15.23', change: '-3.2%', up: false },
  ];

  return (
    <div className="bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-5">
      <h3 className="font-semibold text-lg text-obsidian-900 mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-electric-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
        Market Overview
      </h3>
      <div className="grid grid-cols-2 gap-3">
        {indices.map((index) => (
          <div
            key={index.name}
            className="p-3 rounded-xl bg-cream-50/80 border border-cream-200/30"
          >
            <p className="text-xs text-obsidian-500 mb-1">{index.name}</p>
            <p className="text-lg font-semibold text-obsidian-900">{index.value}</p>
            <p className={`text-sm font-medium ${index.up ? 'text-success-600' : 'text-coral-600'}`}>
              {index.change}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

// =============================================================================
// SENTIMENT HEATMAP WIDGET
// =============================================================================

const SentimentHeatmapWidget: React.FC = () => {
  const sectors = [
    { name: 'Technology', sentiment: 78, color: 'bg-success-500' },
    { name: 'Finance', sentiment: 65, color: 'bg-success-400' },
    { name: 'Healthcare', sentiment: 55, color: 'bg-amber-400' },
    { name: 'Energy', sentiment: 42, color: 'bg-amber-500' },
    { name: 'Consumer', sentiment: 70, color: 'bg-success-400' },
    { name: 'Industrial', sentiment: 48, color: 'bg-amber-400' },
  ];

  return (
    <div className="bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-5">
      <h3 className="font-semibold text-lg text-obsidian-900 mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-coral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Sector Sentiment
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {sectors.map((sector) => (
          <div
            key={sector.name}
            className="p-3 rounded-xl bg-cream-50/80 border border-cream-200/30"
          >
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-obsidian-500">{sector.name}</p>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${sector.sentiment >= 60 ? 'bg-success-100 text-success-700' : 'bg-amber-100 text-amber-700'
                }`}>
                {sector.sentiment}
              </span>
            </div>
            <div className="h-1.5 bg-cream-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${sector.color}`}
                style={{ width: `${sector.sentiment}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// =============================================================================
// TRENDING NEWS WIDGET
// =============================================================================

const TrendingNewsWidget: React.FC = () => {
  const news = [
    { title: 'Fed signals potential rate cuts in 2024', time: '2h ago', source: 'Reuters' },
    { title: 'Apple announces new AI features for iPhone', time: '4h ago', source: 'Bloomberg' },
    { title: 'Oil prices surge on supply concerns', time: '5h ago', source: 'WSJ' },
  ];

  return (
    <div className="bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-5">
      <h3 className="font-semibold text-lg text-obsidian-900 mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9.5a2.5 2.5 0 00-2.5-2.5H15" />
        </svg>
        Trending News
      </h3>
      <div className="space-y-3">
        {news.map((item, index) => (
          <div
            key={index}
            className="p-3 rounded-xl bg-cream-50/80 border border-cream-200/30 hover:bg-cream-100 transition-colors cursor-pointer"
          >
            <p className="text-sm text-obsidian-800 font-medium mb-1 line-clamp-2">{item.title}</p>
            <div className="flex items-center gap-2 text-xs text-obsidian-400">
              <span>{item.source}</span>
              <span>•</span>
              <span>{item.time}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// =============================================================================
// TOP MOVERS WIDGET
// =============================================================================

const TopMoversWidget: React.FC = () => {
  const { data: moversData } = useMarketMovers();

  const gainers = (moversData?.gainers || []).slice(0, 3);
  const losers = (moversData?.losers || []).slice(0, 3);

  return (
    <div className="bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-5">
      <h3 className="font-semibold text-lg text-obsidian-900 mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-electric-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
        Top Movers
      </h3>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-xs text-success-600 font-medium mb-2">🚀 Gainers</p>
          <div className="space-y-2">
            {gainers.map((stock) => (
              <Link
                key={stock.ticker}
                href={`/stock/${stock.ticker}`}
                className="flex items-center justify-between p-2 rounded-lg bg-success-50/50 hover:bg-success-100/50 transition-colors"
              >
                <span className="text-sm font-medium text-obsidian-900">{stock.ticker}</span>
                <span className="text-xs font-semibold text-success-600">+{stock.change_percent?.toFixed(1)}%</span>
              </Link>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs text-coral-600 font-medium mb-2">📉 Losers</p>
          <div className="space-y-2">
            {losers.map((stock) => (
              <Link
                key={stock.ticker}
                href={`/stock/${stock.ticker}`}
                className="flex items-center justify-between p-2 rounded-lg bg-coral-50/50 hover:bg-coral-100/50 transition-colors"
              >
                <span className="text-sm font-medium text-obsidian-900">{stock.ticker}</span>
                <span className="text-xs font-semibold text-coral-600">{stock.change_percent?.toFixed(1)}%</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// MAIN DASHBOARD CONTENT
// =============================================================================

function DashboardContent() {
  const { user } = useAuth();
  const firstName = user?.name?.split(' ')[0] || 'there';

  const { enrichedItems, refresh: refreshWatchlist } = useWatchlist();
  const { positions, analysis, analyze } = usePortfolio();
  const { activities: rawActivities, isLoading: activitiesLoading } = useActivities(10);

  const userActivities: DashboardActivity[] = rawActivities.map(convertActivity);

  useEffect(() => {
    if (enrichedItems.length > 0) {
      refreshWatchlist();
    }
    if (positions.length > 0) {
      analyze();
    }
  }, []);

  const watchlistStocks: WatchlistStock[] = enrichedItems.slice(0, 5).map(item => ({
    symbol: item.ticker,
    name: item.company || item.ticker,
    price: item.price || 0,
    change: item.change || 0,
    changePercent: item.pct_change || 0,
  }));

  const searchesUsed = 3;
  const searchesLimit = user?.tier === 'pro' ? '∞' : 10;
  const analysesUsed = 1;
  const analysesLimit = user?.tier === 'pro' ? '∞' : 3;
  const portfolioValue = analysis?.portfolio_metrics?.total_value || 0;
  const portfolioChange = analysis?.portfolio_metrics?.total_gain_loss_percent || 0;

  return (
    <div className="min-h-screen bg-cream-50">
      {/* Header Section */}
      <Section spacing="md" background="white">
        <ScrollReveal>
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-3xl lg:text-4xl font-bold text-obsidian-900 tracking-tight mb-2">
                Welcome back, {firstName}
              </h1>
              <p className="text-lg text-obsidian-500">
                Here&apos;s what&apos;s happening with your investments today.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/search"
                className="inline-flex items-center gap-2 h-10 px-4 bg-white border border-cream-300 rounded-xl text-sm font-medium text-obsidian-700 hover:bg-cream-100 transition-all"
              >
                <svg className="w-4 h-4 text-obsidian-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Search
              </Link>
              <Link
                href="/sentiment"
                className="inline-flex items-center gap-2 h-10 px-4 bg-obsidian-900 rounded-xl text-sm font-medium text-white hover:bg-obsidian-850 transition-all hover:-translate-y-0.5 hover:shadow-lg"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Analysis
              </Link>
            </div>
          </div>
        </ScrollReveal>
      </Section>

      {/* Stats Grid */}
      <Section spacing="sm" background="default">
        <Grid cols={1} colsMd={2} colsLg={4} gap="md">
          <ScrollReveal delay={0}>
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
          </ScrollReveal>
          <ScrollReveal delay={50}>
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
          </ScrollReveal>
          <ScrollReveal delay={100}>
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
          </ScrollReveal>
          <ScrollReveal delay={150}>
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
          </ScrollReveal>
        </Grid>
      </Section>

      {/* Market Overview Row */}
      <Section spacing="sm" background="default">
        <Grid cols={1} colsMd={2} colsLg={4} gap="md">
          <ScrollReveal delay={0}>
            <MarketOverviewWidget />
          </ScrollReveal>
          <ScrollReveal delay={50}>
            <SentimentHeatmapWidget />
          </ScrollReveal>
          <ScrollReveal delay={100}>
            <TopMoversWidget />
          </ScrollReveal>
          <ScrollReveal delay={150}>
            <TrendingNewsWidget />
          </ScrollReveal>
        </Grid>
      </Section>

      {/* Main Content */}
      <Section spacing="md" background="default">
        <Grid cols={1} colsLg={3} gap="lg">
          {/* Activity & Quick Actions - 2 columns */}
          <div className="lg:col-span-2 space-y-5">
            <ScrollReveal>
              <RecentActivity activities={userActivities} />
            </ScrollReveal>

            {/* Quick Actions */}
            <ScrollReveal delay={100}>
              <div className="bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50 shadow-glass p-5">
                <h3 className="font-semibold text-lg text-obsidian-900 mb-4">
                  Quick Actions
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <Link
                    href="/sentiment"
                    className="flex flex-col items-center gap-2 p-4 rounded-xl bg-cream-50/80 hover:bg-cream-100 transition-all hover:-translate-y-0.5 group"
                  >
                    <div className="w-10 h-10 rounded-lg bg-electric-100 text-electric-600 flex items-center justify-center group-hover:bg-electric-200 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <span className="text-sm font-medium text-obsidian-700">Sentiment</span>
                  </Link>
                  <Link
                    href="/screener"
                    className="flex flex-col items-center gap-2 p-4 rounded-xl bg-cream-50/80 hover:bg-cream-100 transition-all hover:-translate-y-0.5 group"
                  >
                    <div className="w-10 h-10 rounded-lg bg-coral-100 text-coral-600 flex items-center justify-center group-hover:bg-coral-200 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                      </svg>
                    </div>
                    <span className="text-sm font-medium text-obsidian-700">Screener</span>
                  </Link>
                  <Link
                    href="/insider"
                    className="flex flex-col items-center gap-2 p-4 rounded-xl bg-cream-50/80 hover:bg-cream-100 transition-all hover:-translate-y-0.5 group"
                  >
                    <div className="w-10 h-10 rounded-lg bg-success-100 text-success-600 flex items-center justify-center group-hover:bg-success-200 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                    <span className="text-sm font-medium text-obsidian-700">Insider</span>
                  </Link>
                  <Link
                    href="/portfolio"
                    className="flex flex-col items-center gap-2 p-4 rounded-xl bg-cream-50/80 hover:bg-cream-100 transition-all hover:-translate-y-0.5 group"
                  >
                    <div className="w-10 h-10 rounded-lg bg-amber-100 text-amber-600 flex items-center justify-center group-hover:bg-amber-200 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                    <span className="text-sm font-medium text-obsidian-700">Portfolio</span>
                  </Link>
                </div>
              </div>
            </ScrollReveal>

            {/* Upgrade banner for free users */}
            {user?.tier === 'free' && (
              <ScrollReveal delay={150}>
                <div className="bg-gradient-to-r from-obsidian-900 to-obsidian-800 rounded-2xl p-6 shadow-diffuse">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <h3 className="font-semibold text-lg text-white mb-1">
                        Upgrade to Pro
                      </h3>
                      <p className="text-sm text-obsidian-300">
                        Unlimited analyses, Quant Lab access, and more.
                      </p>
                    </div>
                    <Link
                      href="/pricing"
                      className="px-5 py-2.5 bg-white text-obsidian-900 font-medium text-sm rounded-xl hover:bg-cream-100 transition-colors flex-shrink-0"
                    >
                      View Plans
                    </Link>
                  </div>
                </div>
              </ScrollReveal>
            )}
          </div>

          {/* Watchlist - 1 column */}
          <div>
            <ScrollReveal delay={100}>
              <WatchlistPreview stocks={watchlistStocks} />
            </ScrollReveal>
          </div>
        </Grid>
      </Section>
    </div>
  );
}

export default function DashboardPage() {
  return <DashboardContent />;
}
