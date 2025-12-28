'use client';

// =============================================================================
// TRUSTED BY SECTION - REDESIGNED
// =============================================================================
// Clean social proof section with animated text
//
// Location: frontend/components/landing/TrustedBy.tsx
// =============================================================================

import React from 'react';

const sources = [
  'Bloomberg',
  'Reuters',
  'Yahoo Finance',
  'MarketWatch',
  'CNBC',
  'WSJ',
];

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-12 lg:py-16 bg-cream-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Label */}
        <p className="text-center text-body-sm text-navy-500/60 mb-8">
          Powered by real-time data from leading financial sources
        </p>

        {/* Logo grid with subtle animation */}
        <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-6">
          {sources.map((source, index) => (
            <div
              key={source}
              className="
                text-navy-900/30 
                font-heading font-bold text-lg lg:text-xl 
                tracking-tight
                transition-all duration-300
                hover:text-navy-900/60
              "
              style={{
                animationDelay: `${index * 100}ms`,
              }}
            >
              {source}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TrustedBy;
