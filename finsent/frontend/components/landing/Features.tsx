'use client';

// =============================================================================
// FEATURES SECTION - REDESIGNED
// =============================================================================
// Premium feature showcase with react-icons and hover animations
//
// Location: frontend/components/landing/Features.tsx
// =============================================================================

import React from 'react';
import {
  HiChartBar,
  HiTrendingUp,
  HiUserGroup,
  HiCollection,
  HiLightningBolt,
  HiBell
} from 'react-icons/hi';
import { IconType } from 'react-icons';

interface Feature {
  icon: IconType;
  title: string;
  description: string;
  accent: 'orange' | 'purple' | 'blue';
  badge?: string;
}

const features: Feature[] = [
  {
    icon: HiChartBar,
    title: 'Sentiment Analysis',
    description: 'AI-powered analysis of social media, news, and forums to gauge market sentiment in real-time.',
    accent: 'orange',
  },
  {
    icon: HiTrendingUp,
    title: 'Financial Analysis',
    description: 'Deep-dive into company financials with automated ratio analysis, trend detection, and peer comparison.',
    accent: 'purple',
  },
  {
    icon: HiUserGroup,
    title: 'Insider Trading',
    description: 'Track what executives and insiders are buying and selling with real-time SEC filing analysis.',
    accent: 'blue',
  },
  {
    icon: HiCollection,
    title: 'Portfolio Tracking',
    description: 'Monitor your investments with real-time valuations, performance metrics, and allocation insights.',
    accent: 'orange',
  },
  {
    icon: HiLightningBolt,
    title: 'Quant Lab',
    description: 'Build and backtest custom trading strategies with our powerful quantitative analysis tools.',
    accent: 'purple',
    badge: 'Pro',
  },
  {
    icon: HiBell,
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
        {/* Section header */}
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

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            const colors = accentClasses[feature.accent];

            return (
              <div
                key={feature.title}
                className="
                  group relative
                  bg-white rounded-2xl p-8
                  border border-navy-100/50
                  transition-all duration-300 ease-out
                  hover:shadow-xl hover:shadow-navy-900/5
                  hover:-translate-y-1
                  hover:border-navy-200/50
                "
                style={{
                  animationDelay: `${index * 100}ms`,
                }}
              >
                {/* Icon container */}
                <div
                  className={`
                    w-14 h-14 rounded-2xl mb-6
                    flex items-center justify-center
                    transition-all duration-300
                    ${colors.bg} ${colors.hover}
                  `}
                >
                  <Icon className={`w-7 h-7 ${colors.icon} transition-transform duration-300 group-hover:scale-110`} />
                </div>

                {/* Badge */}
                {feature.badge && (
                  <span className="
                    absolute top-6 right-6 
                    px-3 py-1 
                    text-[11px] font-bold uppercase tracking-wider 
                    bg-gradient-to-r from-terra-500 to-pink-500 
                    text-white rounded-full
                    shadow-sm
                  ">
                    {feature.badge}
                  </span>
                )}

                {/* Content */}
                <h3 className="font-heading font-semibold text-heading-md text-navy-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-body-md text-navy-600/70 leading-relaxed">
                  {feature.description}
                </p>

                {/* Hover indicator */}
                <div className="
                  absolute bottom-0 left-8 right-8 h-0.5 
                  bg-gradient-to-r from-transparent via-terra-500/50 to-transparent
                  opacity-0 group-hover:opacity-100
                  transition-opacity duration-300
                  rounded-full
                " />
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Features;
