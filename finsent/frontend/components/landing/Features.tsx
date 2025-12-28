'use client';

import React from 'react';

// =============================================================================
// FEATURES SECTION - Bento Grid Layout
// =============================================================================

const features = [
  {
    title: 'Sentiment Analysis',
    description: 'AI-powered analysis of news, social media, and market data to gauge investor sentiment.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    size: 'large',
  },
  {
    title: 'Insider Trading',
    description: 'Track executive and institutional trading patterns in real-time.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    size: 'normal',
  },
  {
    title: 'Quant Lab',
    description: 'Build and backtest custom trading strategies with our advanced tools.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    size: 'normal',
  },
  {
    title: 'Stock Screener',
    description: 'Filter stocks by sentiment, financials, technicals, and more with advanced screening tools.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
      </svg>
    ),
    size: 'large',
  },
  {
    title: 'Portfolio Analytics',
    description: 'Track your positions and analyze portfolio risk and performance.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    size: 'normal',
  },
  {
    title: 'Market Movers',
    description: 'Stay ahead with real-time alerts on the biggest market movements.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    size: 'normal',
  },
];

export const Features: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <p className="text-body-sm font-medium text-accent mb-3">Features</p>
          <h2 className="text-display-sm lg:text-display-md text-ink-900 mb-4">
            Everything you need to invest smarter
          </h2>
          <p className="text-body-lg text-ink-500 max-w-2xl mx-auto">
            Powerful tools to analyze the market, track your portfolio, and make data-driven decisions.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className={`
                group relative p-6 lg:p-8 bg-ink-50 rounded-2xl 
                border border-transparent hover:border-ink-200/50 
                transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg
                ${feature.size === 'large' ? 'lg:col-span-2' : ''}
              `}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-white flex items-center justify-center text-ink-600 shadow-sm group-hover:shadow-md group-hover:text-accent transition-all">
                  {feature.icon}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-heading-md text-ink-900 mb-2 tracking-tight">
                    {feature.title}
                  </h3>
                  <p className="text-body-sm text-ink-500 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
