'use client';

import React from 'react';
import Link from 'next/link';

// =============================================================================
// FINAL CTA - Bottom call to action
// =============================================================================

export const FinalCTA: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-ink-900">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 className="text-display-sm lg:text-display-md text-white mb-4">
          Ready to invest smarter?
        </h2>
        <p className="text-body-lg text-ink-400 mb-8 max-w-xl mx-auto">
          Join thousands of investors using AI-powered insights to make better decisions.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <Link
            href="/signup"
            className="inline-flex items-center justify-center gap-2 h-12 px-6 text-body-md font-medium text-ink-900 bg-white hover:bg-ink-100 rounded-xl transition-all duration-150"
          >
            Get started for free
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
          <Link
            href="/pricing"
            className="inline-flex items-center justify-center h-12 px-6 text-body-md font-medium text-ink-300 hover:text-white transition-colors"
          >
            View pricing
          </Link>
        </div>
      </div>
    </section>
  );
};

export default FinalCTA;
