'use client';

import React from 'react';
import Link from 'next/link';
import { AtmosphericBackground } from './AtmosphericBackground';
import { FloatingDashboard } from './FloatingDashboard';
import { ScrollReveal, MagneticButton, ArrowIcon } from '@/components/ui';

// =============================================================================
// HERO SECTION - Linear-Modernist "Atmospheric Glass" Aesthetic
// =============================================================================
// Asymmetric layout with headline on left and floating dashboard on right.
// Features scroll-triggered animations and magnetic button effects.
// =============================================================================

export const Hero: React.FC = () => {
  return (
    <AtmosphericBackground className="min-h-screen flex items-center">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
        {/* Asymmetric Grid: Text Left, Dashboard Right */}
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">

          {/* Left Column - Content */}
          <div className="max-w-xl">
            {/* Announcement Badge */}
            <ScrollReveal delay={0}>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 mb-8 rounded-full bg-white/80 backdrop-blur-sm border border-cream-300/50 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-electric-500 animate-pulse" />
                <span className="text-sm text-obsidian-600">
                  AI-Powered Financial Intelligence
                </span>
              </div>
            </ScrollReveal>

            {/* Main Headline - Bold Grotesque with -2% tracking */}
            <ScrollReveal delay={100}>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-obsidian-900 mb-6 tracking-tightest leading-[1.08]">
                Make Smarter{' '}
                <span className="text-gradient">Investment</span>{' '}
                Decisions
              </h1>
            </ScrollReveal>

            {/* Subheadline */}
            <ScrollReveal delay={200}>
              <p className="text-lg lg:text-xl text-obsidian-500 mb-10 leading-relaxed">
                Analyze market sentiment, track insider trading, and discover
                opportunities before the crowd. All in one powerful platform.
              </p>
            </ScrollReveal>

            {/* CTA Buttons with Magnetic Effect */}
            <ScrollReveal delay={300}>
              <div className="flex flex-col sm:flex-row items-start gap-3 mb-12">
                <MagneticButton
                  href="/signup"
                  variant="primary"
                  size="lg"
                  icon={<ArrowIcon className="w-4 h-4" />}
                >
                  Get started free
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
                  Search any stock
                </MagneticButton>
              </div>
            </ScrollReveal>

            {/* Trust Indicators */}
            <ScrollReveal delay={400}>
              <div className="flex flex-wrap items-center gap-6 text-sm text-obsidian-400">
                {[
                  'Free to start',
                  'No credit card required',
                  'Cancel anytime',
                ].map((item, index) => (
                  <div key={item} className="flex items-center gap-1.5">
                    <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </ScrollReveal>
          </div>

          {/* Right Column - Floating Dashboard */}
          <div className="hidden lg:block">
            <ScrollReveal delay={200}>
              <FloatingDashboard className="w-full max-w-lg ml-auto" />
            </ScrollReveal>
          </div>
        </div>

        {/* Stats Section */}
        <ScrollReveal delay={500}>
          <div className="mt-20 pt-12 border-t border-cream-300/50">
            <div className="grid grid-cols-3 gap-8 max-w-3xl mx-auto">
              {[
                { value: '50K+', label: 'Stocks Analyzed' },
                { value: '2M+', label: 'Data Points' },
                { value: '98%', label: 'Accuracy' },
              ].map((stat) => (
                <div key={stat.label} className="text-center">
                  <p className="text-2xl sm:text-3xl lg:text-4xl font-bold text-obsidian-900 mb-1 tracking-tight">
                    {stat.value}
                  </p>
                  <p className="text-sm text-obsidian-400">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
        </ScrollReveal>
      </div>
    </AtmosphericBackground>
  );
};

export default Hero;
