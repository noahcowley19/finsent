'use client';

import React from 'react';
import Image from 'next/image';
import { AtmosphericBackground } from './AtmosphericBackground';
import { ScrollReveal, MagneticButton, ArrowIcon } from '@/components/ui';

// =============================================================================
// HERO SECTION - Centered Layout with Product Screenshot
// =============================================================================

export const Hero: React.FC = () => {
  return (
    <AtmosphericBackground className="min-h-screen">
      {/* Main Content - Truly Centered */}
      <div className="flex flex-col items-center justify-center min-h-screen px-6 py-20">

        {/* Text Content - Centered */}
        <div className="max-w-4xl mx-auto text-center mb-12">
          {/* Badge */}
          <ScrollReveal delay={0}>
            <div className="inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-white/90 backdrop-blur-sm border border-cream-200 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-success-500 animate-pulse" />
              <span className="text-sm font-medium text-obsidian-700">
                AI-Powered Financial Intelligence
              </span>
            </div>
          </ScrollReveal>

          {/* Headline */}
          <ScrollReveal delay={100}>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-obsidian-900 mb-6 tracking-tight leading-[1.08]">
              Invest with{' '}
              <span className="text-gradient">Confidence</span>
            </h1>
          </ScrollReveal>

          {/* Subheadline */}
          <ScrollReveal delay={200}>
            <p className="text-lg lg:text-xl text-obsidian-500 mb-10 max-w-2xl mx-auto leading-relaxed">
              Real-time sentiment analysis, insider trading alerts, and
              predictive analytics—all in one powerful platform.
            </p>
          </ScrollReveal>

          {/* CTAs */}
          <ScrollReveal delay={300}>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
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
            <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-obsidian-500">
              {['Free 14-day trial', 'No credit card', 'Cancel anytime'].map((item) => (
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

        {/* Product Screenshot - Centered Below Text */}
        <ScrollReveal delay={500}>
          <div className="relative w-full max-w-5xl mx-auto">
            {/* Glow Effect */}
            <div
              className="absolute inset-0 opacity-30 blur-3xl -z-10"
              style={{
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.15) 100%)',
                transform: 'scale(1.1) translateY(10%)',
              }}
            />

            {/* Screenshot Container */}
            <div
              className="relative bg-white rounded-2xl lg:rounded-3xl overflow-hidden border border-cream-200/50"
              style={{
                boxShadow: '0 50px 100px -20px rgba(0, 0, 0, 0.15), 0 30px 60px -15px rgba(0, 0, 0, 0.1)',
              }}
            >
              {/* Browser Chrome */}
              <div className="flex items-center gap-2 px-4 py-3 border-b border-cream-100 bg-cream-50/50">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-coral-400" />
                  <div className="w-3 h-3 rounded-full bg-amber-400" />
                  <div className="w-3 h-3 rounded-full bg-success-400" />
                </div>
                <div className="flex-1 flex justify-center">
                  <div className="px-4 py-1 rounded-lg bg-cream-100 text-obsidian-400 text-xs font-medium">
                    app.caveray.com
                  </div>
                </div>
                <div className="w-12" />
              </div>

              {/* Screenshot Image */}
              <div className="relative aspect-[16/10] bg-cream-50">
                <Image
                  src="/images/dashboard-screenshot.png"
                  alt="Caveray Dashboard - Social Sentiment Screener"
                  fill
                  className="object-cover object-top"
                  priority
                />
              </div>
            </div>
          </div>
        </ScrollReveal>
      </div>

      {/* Stats Bar - Full Width at Bottom */}
      <div className="border-t border-cream-200/50 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <ScrollReveal delay={600}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
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
    </AtmosphericBackground>
  );
};

export default Hero;
