'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';

// =============================================================================
// HERO SECTION - Modern SaaS Aesthetic
// =============================================================================

export const Hero: React.FC = () => {
  return (
    <section className="relative min-h-[100vh] flex items-center overflow-hidden bg-ink-50">
      {/* Subtle gradient background */}
      <div className="absolute inset-0 bg-gradient-to-b from-white via-ink-50 to-ink-100" />

      {/* Subtle grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.015]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'%3E%3Cg fill='none' stroke='%23000' stroke-width='1'%3E%3Cpath d='M0 0h60v60H0z'/%3E%3C/g%3E%3C/svg%3E")`,
        }}
      />

      {/* Ambient glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-radial from-accent/5 via-transparent to-transparent blur-3xl pointer-events-none" />

      {/* Content */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <div className="flex flex-col items-center text-center max-w-4xl mx-auto">
          {/* Announcement Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 mb-8 bg-white rounded-full border border-ink-200/50 shadow-sm animate-fade-in">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span className="text-body-sm text-ink-600">
              AI-Powered Financial Intelligence
            </span>
          </div>

          {/* Main Headline */}
          <h1
            className="text-display-lg lg:text-display-xl xl:text-display-2xl text-ink-900 mb-6 animate-fade-in-up"
            style={{ animationDelay: '100ms' }}
          >
            Make Smarter{' '}
            <span className="text-gradient">Investment</span>{' '}
            Decisions
          </h1>

          {/* Subheadline */}
          <p
            className="text-body-lg lg:text-xl text-ink-500 mb-10 max-w-2xl leading-relaxed animate-fade-in-up"
            style={{ animationDelay: '200ms' }}
          >
            Analyze market sentiment, track insider trading, and discover opportunities
            before the crowd. All in one powerful platform.
          </p>

          {/* CTA Buttons */}
          <div
            className="flex flex-col sm:flex-row items-center gap-3 mb-12 animate-fade-in-up"
            style={{ animationDelay: '300ms' }}
          >
            <Link
              href="/signup"
              className="inline-flex items-center justify-center gap-2 h-12 px-6 text-body-md font-medium text-white bg-ink-900 hover:bg-ink-800 rounded-xl transition-all duration-150 hover:-translate-y-0.5 hover:shadow-lg"
            >
              Get started free
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link
              href="/search"
              className="inline-flex items-center justify-center gap-2 h-12 px-6 text-body-md font-medium text-ink-700 bg-white hover:bg-ink-50 border border-ink-200 rounded-xl transition-all duration-150"
            >
              <svg className="w-4 h-4 text-ink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Search any stock
            </Link>
          </div>

          {/* Trust Indicators */}
          <div
            className="flex flex-wrap items-center justify-center gap-6 text-body-sm text-ink-400 animate-fade-in-up"
            style={{ animationDelay: '400ms' }}
          >
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span>Free to start</span>
            </div>
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span>No credit card required</span>
            </div>
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span>Cancel anytime</span>
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div
          className="mt-20 pt-12 border-t border-ink-200/50 animate-fade-in-up"
          style={{ animationDelay: '500ms' }}
        >
          <div className="grid grid-cols-3 gap-8 max-w-3xl mx-auto">
            <div className="text-center">
              <p className="text-display-sm lg:text-display-md text-ink-900 mb-1">50K+</p>
              <p className="text-body-sm text-ink-400">Stocks Analyzed</p>
            </div>
            <div className="text-center">
              <p className="text-display-sm lg:text-display-md text-ink-900 mb-1">2M+</p>
              <p className="text-body-sm text-ink-400">Data Points</p>
            </div>
            <div className="text-center">
              <p className="text-display-sm lg:text-display-md text-ink-900 mb-1">98%</p>
              <p className="text-body-sm text-ink-400">Accuracy</p>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-ink-50 to-transparent pointer-events-none" />
    </section>
  );
};

export default Hero;
