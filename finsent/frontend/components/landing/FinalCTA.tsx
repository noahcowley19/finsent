'use client';

// =============================================================================
// FINAL CTA SECTION - FIXED VERSION
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { MeshGradient } from '@/components/ui/MeshGradient';

export const FinalCTA: React.FC = () => {
  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
      <MeshGradient className="z-0" intensity={0.8} />

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/70 backdrop-blur-md rounded-full border border-white/50 shadow-sm mb-8">
          <svg className="w-4 h-4 text-terra-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
          </svg>
          <span className="text-body-sm font-medium text-navy-700">Ready to get started?</span>
        </div>

        <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-6">
          Start making smarter investment decisions today
        </h2>

        <p className="text-body-lg text-navy-600/80 mb-10 max-w-2xl mx-auto">
          Join thousands of investors who use Caveray to analyze markets,
          track sentiment, and discover opportunities.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/signup"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-navy-900 text-white font-heading font-semibold text-body-lg rounded-xl transition-all duration-200 hover:bg-navy-800 hover:scale-[1.02] shadow-lg shadow-navy-900/20"
          >
            Get Started Free
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
          <Link
            href="/search"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-white/80 backdrop-blur-md text-navy-700 font-heading font-medium text-body-lg rounded-xl border border-white/50 transition-all duration-200 hover:bg-white hover:shadow-lg"
          >
            Try Stock Search
          </Link>
        </div>

        <p className="text-body-sm text-navy-500/70 mt-8">
          No credit card required • Free plan available forever
        </p>
      </div>
    </section>
  );
};

export default FinalCTA;
