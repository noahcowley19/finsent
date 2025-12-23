'use client';

// =============================================================================
// FINAL CTA SECTION
// =============================================================================
// Bottom call-to-action banner
//
// Location: frontend/components/landing/FinalCTA.tsx
//
// =============================================================================

import React from 'react';
import Link from 'next/link';

export const FinalCTA: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-gradient-to-br from-navy-900 via-navy-800 to-navy-900 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0">
        {/* Gradient orbs */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-terra-500/10 rounded-full blur-[150px]" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-navy-400/10 rounded-full blur-[150px]" />
        
        {/* Grid pattern */}
        <div className="absolute inset-0 opacity-[0.02]">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="cta-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="white" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#cta-grid)" />
          </svg>
        </div>
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Icon */}
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-terra-500/20 mb-8">
          <svg className="w-8 h-8 text-terra-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>

        {/* Headline */}
        <h2 className="font-display text-display-md lg:text-display-lg text-white mb-6">
          Ready to invest smarter?
        </h2>

        {/* Subheadline */}
        <p className="text-body-lg lg:text-body-xl text-white/70 mb-10 max-w-2xl mx-auto">
          Join thousands of investors using Caveray to discover opportunities, 
          analyze sentiment, and make data-driven decisions.
        </p>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/signup"
            className="
              inline-flex items-center justify-center
              px-8 py-4 min-w-[200px]
              bg-terra-500 text-white
              font-heading font-semibold text-body-lg
              rounded-xl
              transition-all duration-fast
              hover:bg-terra-600 hover:-translate-y-1 hover:shadow-terra-lg
              focus:outline-none focus:ring-2 focus:ring-terra-500 focus:ring-offset-2 focus:ring-offset-navy-900
            "
          >
            Get Started Free
            <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
          <Link
            href="/pricing"
            className="
              inline-flex items-center justify-center
              px-8 py-4 min-w-[200px]
              bg-transparent text-white
              font-heading font-semibold text-body-lg
              rounded-xl
              border border-white/20
              transition-all duration-fast
              hover:bg-white/10 hover:-translate-y-1
              focus:outline-none focus:ring-2 focus:ring-white/50
            "
          >
            View Pricing
          </Link>
        </div>

        {/* Trust badges */}
        <div className="flex flex-wrap items-center justify-center gap-6 mt-12 text-white/50 text-body-sm">
          <span className="flex items-center gap-2">
            <svg className="w-5 h-5 text-success-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            No credit card required
          </span>
          <span className="flex items-center gap-2">
            <svg className="w-5 h-5 text-success-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            Free forever plan
          </span>
          <span className="flex items-center gap-2">
            <svg className="w-5 h-5 text-success-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            Cancel anytime
          </span>
        </div>
      </div>
    </section>
  );
};

export default FinalCTA;
