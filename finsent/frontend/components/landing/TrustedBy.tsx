'use client';

import React from 'react';

// =============================================================================
// TRUSTED BY - Social proof logos
// =============================================================================

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-12 bg-white border-y border-ink-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <p className="text-body-sm text-ink-400 text-center mb-8">
          Trusted by investors at leading companies
        </p>
        <div className="flex flex-wrap items-center justify-center gap-x-12 gap-y-6">
          {/* Placeholder logos - would be replaced with actual company logos */}
          {['Company 1', 'Company 2', 'Company 3', 'Company 4', 'Company 5'].map((company) => (
            <div
              key={company}
              className="text-body-lg font-semibold text-ink-300 tracking-tight"
            >
              {company}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TrustedBy;
