'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// FEATURES SECTION - Bento Grid with Glass Cards
// =============================================================================
// Features vibrant glass cards with varying widths and functional accent colors.
// Uses scroll-triggered reveal animations.
// =============================================================================

const features = [
  {
    title: 'Sentiment Analysis',
    description: 'AI-powered analysis of news, social media, and market data to gauge investor sentiment in real-time.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    accent: 'coral',
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
    accent: 'electric',
    size: 'normal',
  },
  {
    title: 'Quant Lab',
    description: 'Build and backtest custom trading strategies with advanced tools.',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    accent: 'amber',
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
    accent: 'electric',
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
    accent: 'coral',
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
    accent: 'amber',
    size: 'normal',
  },
];

const accentStyles = {
  coral: {
    bg: 'bg-coral-50',
    iconBg: 'bg-coral-100',
    iconColor: 'text-coral-500',
    border: 'border-coral-200/50',
  },
  electric: {
    bg: 'bg-electric-50',
    iconBg: 'bg-electric-100',
    iconColor: 'text-electric-500',
    border: 'border-electric-200/50',
  },
  amber: {
    bg: 'bg-amber-50',
    iconBg: 'bg-amber-100',
    iconColor: 'text-amber-500',
    border: 'border-amber-200/50',
  },
};

export const Features: React.FC = () => {
  return (
    <section className="py-24 lg:py-32 bg-cream-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal>
          <div className="text-center mb-16">
            <p className="text-sm font-medium text-electric-500 mb-3 tracking-wide uppercase">
              Features
            </p>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-obsidian-900 mb-4 tracking-tightest">
              Everything you need to invest smarter
            </h2>
            <p className="text-lg text-obsidian-500 max-w-2xl mx-auto">
              Powerful tools to analyze the market, track your portfolio, and make data-driven decisions.
            </p>
          </div>
        </ScrollReveal>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-5">
          {features.map((feature, index) => {
            const accent = accentStyles[feature.accent as keyof typeof accentStyles];
            return (
              <ScrollReveal key={feature.title} delay={index * 75}>
                <div
                  className={`
                    group relative p-6 lg:p-8 rounded-2xl 
                    bg-white/80 backdrop-blur-lg
                    border border-white/50
                    shadow-glass
                    transition-all duration-300 
                    hover:-translate-y-1 hover:shadow-glass-lg
                    ${feature.size === 'large' ? 'lg:col-span-2' : ''}
                  `}
                >
                  <div className="flex items-start gap-4">
                    {/* Icon */}
                    <div
                      className={`
                        w-12 h-12 rounded-xl flex items-center justify-center 
                        ${accent.iconBg} ${accent.iconColor}
                        transition-all duration-300 group-hover:scale-110
                      `}
                    >
                      {feature.icon}
                    </div>

                    {/* Content */}
                    <div className="flex-1">
                      <h3 className="font-semibold text-lg text-obsidian-900 mb-2 tracking-tight">
                        {feature.title}
                      </h3>
                      <p className="text-sm text-obsidian-500 leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>

                  {/* Subtle hover accent line */}
                  <div
                    className={`
                      absolute bottom-0 left-6 right-6 h-0.5 rounded-full
                      ${accent.iconBg}
                      opacity-0 group-hover:opacity-100
                      transition-opacity duration-300
                    `}
                  />
                </div>
              </ScrollReveal>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Features;
