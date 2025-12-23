'use client';

// =============================================================================
// HERO SECTION
// =============================================================================
// Landing page hero with headline, subheadline, CTAs, and animated background
//
// Location: frontend/components/landing/Hero.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export const Hero: React.FC = () => {
  return (
    <section className="relative min-h-[90vh] flex items-center overflow-hidden bg-gradient-to-br from-navy-900 via-navy-800 to-navy-900">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Grid pattern */}
        <div className="absolute inset-0 opacity-[0.03]">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
                <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
          </svg>
        </div>

        {/* Gradient orbs */}
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-terra-500/20 rounded-full blur-[128px] animate-pulse" />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-navy-400/20 rounded-full blur-[128px] animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-terra-500/10 rounded-full blur-[200px]" />

        {/* Floating chart lines */}
        <svg className="absolute bottom-0 left-0 right-0 h-64 opacity-10" preserveAspectRatio="none" viewBox="0 0 1440 320">
          <path
            fill="none"
            stroke="url(#chartGradient)"
            strokeWidth="2"
            d="M0,160 C180,100 360,200 540,140 C720,80 900,220 1080,160 C1260,100 1350,180 1440,120"
            className="animate-pulse"
          />
          <path
            fill="none"
            stroke="url(#chartGradient)"
            strokeWidth="2"
            d="M0,200 C180,260 360,180 540,240 C720,300 900,160 1080,220 C1260,280 1350,200 1440,260"
            className="animate-pulse"
            style={{ animationDelay: '0.5s' }}
          />
          <defs>
            <linearGradient id="chartGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#954C2E" />
              <stop offset="50%" stopColor="#B86B4A" />
              <stop offset="100%" stopColor="#954C2E" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <div className="max-w-4xl">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/10 mb-8 animate-fade-in">
            <span className="w-2 h-2 rounded-full bg-success-400 animate-pulse" />
            <span className="text-body-sm text-white/80">
              AI-Powered Financial Intelligence
            </span>
          </div>

          {/* Headline */}
          <h1 className="font-display text-display-lg lg:text-display-xl text-white mb-6 animate-fade-in-up">
            Make Smarter Investment Decisions with{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-terra-400 to-terra-300">
              AI-Powered Insights
            </span>
          </h1>

          {/* Subheadline */}
          <p className="text-body-lg lg:text-body-xl text-white/70 mb-10 max-w-2xl animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            Analyze market sentiment, track insider trading, and discover opportunities 
            before the crowd. All in one powerful platform built for modern investors.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row gap-4 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
            <Link
              href="/signup"
              className="
                inline-flex items-center justify-center
                px-8 py-4
                bg-terra-500 text-white
                font-heading font-semibold text-body-lg
                rounded-xl
                transition-all duration-fast
                hover:bg-terra-600 hover:-translate-y-1 hover:shadow-terra-lg
                focus:outline-none focus:ring-2 focus:ring-terra-500 focus:ring-offset-2 focus:ring-offset-navy-900
              "
            >
              Start Free Trial
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link
              href="/search"
              className="
                inline-flex items-center justify-center
                px-8 py-4
                bg-white/10 text-white
                font-heading font-semibold text-body-lg
                rounded-xl
                border border-white/20
                backdrop-blur-sm
                transition-all duration-fast
                hover:bg-white/20 hover:-translate-y-1
                focus:outline-none focus:ring-2 focus:ring-white/50
              "
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Try Stock Search
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 mt-16 pt-10 border-t border-white/10 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
            <div>
              <p className="font-display text-display-sm text-white mb-1">50K+</p>
              <p className="text-body-sm text-white/50">Stocks Analyzed</p>
            </div>
            <div>
              <p className="font-display text-display-sm text-white mb-1">2M+</p>
              <p className="text-body-sm text-white/50">Sentiment Data Points</p>
            </div>
            <div>
              <p className="font-display text-display-sm text-white mb-1">98%</p>
              <p className="text-body-sm text-white/50">Accuracy Rate</p>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-cream-50 to-transparent" />
    </section>
  );
};

export default Hero;
