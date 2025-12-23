'use client';

// =============================================================================
// FEATURES SECTION
// =============================================================================
// Feature showcase with icons and descriptions
//
// Location: frontend/components/landing/Features.tsx
//
// =============================================================================

import React from 'react';

const features = [
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    title: 'Sentiment Analysis',
    description: 'AI-powered analysis of social media, news, and forums to gauge market sentiment in real-time.',
    color: 'terra',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    title: 'Financial Analysis',
    description: 'Deep-dive into company financials with automated ratio analysis, trend detection, and peer comparison.',
    color: 'navy',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
    title: 'Insider Trading',
    description: 'Track what executives and insiders are buying and selling with real-time SEC filing analysis.',
    color: 'terra',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    title: 'Portfolio Tracking',
    description: 'Monitor your investments with real-time valuations, performance metrics, and allocation insights.',
    color: 'navy',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    title: 'Quant Lab',
    description: 'Build and backtest custom trading strategies with our powerful quantitative analysis tools.',
    color: 'terra',
    badge: 'Pro',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
      </svg>
    ),
    title: 'Smart Alerts',
    description: 'Get notified when sentiment shifts, insiders trade, or your stocks hit key price levels.',
    color: 'navy',
    badge: 'Pro',
  },
];

export const Features: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-cream-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <p className="text-overline text-terra-500 uppercase tracking-widest mb-4">
            Features
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-4">
            Everything you need to invest smarter
          </h2>
          <p className="text-body-lg text-neutral-600">
            Powerful tools that combine AI, data science, and financial expertise 
            to give you an edge in the market.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="
                group relative
                bg-white rounded-2xl p-8
                border border-border-light
                transition-all duration-normal
                hover:border-border-medium hover:shadow-lg hover:-translate-y-1
              "
            >
              {/* Icon */}
              <div
                className={`
                  w-12 h-12 rounded-xl mb-6
                  flex items-center justify-center
                  transition-transform duration-normal group-hover:scale-110
                  ${feature.color === 'terra' 
                    ? 'bg-terra-100 text-terra-600' 
                    : 'bg-navy-100 text-navy-600'
                  }
                `}
              >
                {feature.icon}
              </div>

              {/* Badge */}
              {feature.badge && (
                <span className="absolute top-6 right-6 px-2 py-1 text-[10px] font-bold uppercase tracking-wider bg-terra-500 text-white rounded">
                  {feature.badge}
                </span>
              )}

              {/* Content */}
              <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-3">
                {feature.title}
              </h3>
              <p className="text-body-md text-neutral-600">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
