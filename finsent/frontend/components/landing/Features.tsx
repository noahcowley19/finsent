'use client';

// =============================================================================
// FEATURES SECTION - FIXED VERSION
// =============================================================================

import React from 'react';

interface Feature {
  icon: React.ReactNode;
  title: string;
  description: string;
  accent: 'orange' | 'purple' | 'blue';
  badge?: string;
}

const features: Feature[] = [
  {
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    title: 'Sentiment Analysis',
    description: 'AI-powered analysis of social media, news, and forums to gauge market sentiment in real-time.',
    accent: 'orange',
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    title: 'Financial Analysis',
    description: 'Deep-dive into company financials with automated ratio analysis, trend detection, and peer comparison.',
    accent: 'purple',
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
    title: 'Insider Trading',
    description: 'Track what executives and insiders are buying and selling with real-time SEC filing analysis.',
    accent: 'blue',
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    title: 'Portfolio Tracking',
    description: 'Monitor your investments with real-time valuations, performance metrics, and allocation insights.',
    accent: 'orange',
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    title: 'Quant Lab',
    description: 'Build and backtest custom trading strategies with our powerful quantitative analysis tools.',
    accent: 'purple',
    badge: 'Pro',
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
      </svg>
    ),
    title: 'Smart Alerts',
    description: 'Get notified when sentiment shifts, insiders trade, or your stocks hit key price levels.',
    accent: 'blue',
    badge: 'Pro',
  },
];

const accentClasses = {
  orange: {
    bg: 'bg-gradient-to-br from-orange-100 to-orange-50',
    icon: 'text-orange-500',
    hover: 'group-hover:from-orange-200 group-hover:to-orange-100',
  },
  purple: {
    bg: 'bg-gradient-to-br from-purple-100 to-purple-50',
    icon: 'text-purple-500',
    hover: 'group-hover:from-purple-200 group-hover:to-purple-100',
  },
  blue: {
    bg: 'bg-gradient-to-br from-blue-100 to-blue-50',
    icon: 'text-blue-500',
    hover: 'group-hover:from-blue-200 group-hover:to-blue-100',
  },
};

export const Features: React.FC = () => {
  return (
    <section className="py-24 lg:py-32 bg-cream-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16 lg:mb-20">
          <p className="text-body-sm font-semibold text-terra-500 uppercase tracking-widest mb-4">
            Features
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-6">
            Everything you need to invest smarter
          </h2>
          <p className="text-body-lg text-navy-600/70 leading-relaxed">
            Powerful tools that combine AI, data science, and financial expertise
            to give you an edge in the market.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
          {features.map((feature, index) => {
            const colors = accentClasses[feature.accent];

            return (
              <div
                key={feature.title}
                className="group relative bg-white rounded-2xl p-8 border border-navy-100/50 transition-all duration-300 ease-out hover:shadow-xl hover:shadow-navy-900/5 hover:-translate-y-1 hover:border-navy-200/50"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className={`w-14 h-14 rounded-2xl mb-6 flex items-center justify-center transition-all duration-300 ${colors.bg} ${colors.hover}`}>
                  <div className={`${colors.icon} transition-transform duration-300 group-hover:scale-110`}>
                    {feature.icon}
                  </div>
                </div>

                {feature.badge && (
                  <span className="absolute top-6 right-6 px-3 py-1 text-[11px] font-bold uppercase tracking-wider bg-gradient-to-r from-terra-500 to-pink-500 text-white rounded-full shadow-sm">
                    {feature.badge}
                  </span>
                )}

                <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-body-md text-navy-600/70 leading-relaxed">
                  {feature.description}
                </p>

                <div className="absolute bottom-0 left-8 right-8 h-0.5 bg-gradient-to-r from-transparent via-terra-500/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-full" />
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Features;
