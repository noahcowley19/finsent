'use client';

import React from 'react';
import Link from 'next/link';
import { AtmosphericBackground } from './AtmosphericBackground';
import { FloatingDashboard } from './FloatingDashboard';
import { ScrollReveal, MagneticButton, ArrowIcon } from '@/components/ui';

// =============================================================================
// HERO SECTION - Properly Centered Modern Layout
// =============================================================================

export const Hero: React.FC = () => {
  return (
    <AtmosphericBackground className="min-h-screen flex items-center">
      <div className="w-full">
        {/* Centered Container */}
        <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-16 lg:py-24">
          {/* Two-Column Grid with Equal Spacing */}
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">

            {/* Left Column - Content (Centered within column) */}
            <div className="flex flex-col items-start lg:items-start justify-center">
              {/* Announcement Badge */}
              <ScrollReveal delay={0}>
                <div className="inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-white/90 backdrop-blur-sm border border-cream-200 shadow-sm">
                  <span className="w-2 h-2 rounded-full bg-success-500 animate-pulse" />
                  <span className="text-sm font-medium text-obsidian-700">
                    Now with AI-Powered Insights
                  </span>
                </div>
              </ScrollReveal>

              {/* Main Headline */}
              <ScrollReveal delay={100}>
                <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-obsidian-900 mb-6 tracking-tight leading-[1.1]">
                  Invest with
                  <br />
                  <span className="text-gradient">Confidence</span>
                </h1>
              </ScrollReveal>

              {/* Subheadline */}
              <ScrollReveal delay={200}>
                <p className="text-lg lg:text-xl text-obsidian-500 mb-10 leading-relaxed max-w-lg">
                  Real-time sentiment analysis, insider trading alerts, and
                  predictive analytics—all in one powerful platform.
                </p>
              </ScrollReveal>

              {/* CTA Buttons */}
              <ScrollReveal delay={300}>
                <div className="flex flex-col sm:flex-row items-start gap-4 mb-12">
                  <MagneticButton
                    href="/signup"
                    variant="primary"
                    size="lg"
                    icon={<ArrowIcon className="w-4 h-4" />}
                  >
                    Start Free Trial
                  </MagneticButton>

                  <MagneticButton
                    href="/search"
                    variant="secondary"
                    size="lg"
                    icon={
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                    }
                    iconPosition="left"
                  >
                    Search Stocks
                  </MagneticButton>
                </div>
              </ScrollReveal>

              {/* Trust Indicators */}
              <ScrollReveal delay={400}>
                <div className="flex flex-wrap items-center gap-6 text-sm text-obsidian-500">
                  {[
                    'Free 14-day trial',
                    'No credit card',
                    'Cancel anytime',
                  ].map((item) => (
                    <div key={item} className="flex items-center gap-2">
                      <svg className="w-4 h-4 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </ScrollReveal>
            </div>

            {/* Right Column - Dashboard (Centered within column) */}
            <div className="hidden lg:flex items-center justify-center">
              <ScrollReveal delay={200}>
                <FloatingDashboard className="w-full max-w-xl" />
              </ScrollReveal>
            </div>
          </div>
        </div>

        {/* Stats Section - Full Width Centered */}
        <div className="border-t border-cream-200/50 bg-white/30 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-12 lg:py-16">
            <ScrollReveal delay={500}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-8 lg:gap-12">
                {[
                  { value: '50K+', label: 'Stocks Analyzed' },
                  { value: '2.5M+', label: 'Data Points Daily' },
                  { value: '99.9%', label: 'Uptime SLA' },
                  { value: '4.9/5', label: 'User Rating' },
                ].map((stat) => (
                  <div key={stat.label} className="text-center">
                    <p className="text-3xl lg:text-4xl font-bold text-obsidian-900 mb-1 tracking-tight">
                      {stat.value}
                    </p>
                    <p className="text-sm text-obsidian-400">{stat.label}</p>
                  </div>
                ))}
              </div>
            </ScrollReveal>
          </div>
        </div>
      </div>
    </AtmosphericBackground>
  );
};

export default Hero;
