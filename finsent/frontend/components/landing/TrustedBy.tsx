'use client';

// =============================================================================
// TRUSTED BY SECTION
// =============================================================================
// Social proof section with company logos
//
// Location: frontend/components/landing/TrustedBy.tsx
//
// =============================================================================

import React from 'react';

// Placeholder logos - replace with actual client/partner logos
const logos = [
  { name: 'Bloomberg', width: 140 },
  { name: 'Reuters', width: 120 },
  { name: 'Yahoo Finance', width: 150 },
  { name: 'MarketWatch', width: 140 },
  { name: 'CNBC', width: 100 },
  { name: 'WSJ', width: 80 },
];

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-12 lg:py-16 bg-cream-50 border-b border-border-light">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Label */}
        <p className="text-center text-body-sm text-neutral-500 uppercase tracking-widest mb-8">
          Trusted by investors who rely on data from
        </p>

        {/* Logo grid */}
        <div className="flex flex-wrap items-center justify-center gap-x-12 gap-y-8">
          {logos.map((logo) => (
            <div
              key={logo.name}
              className="flex items-center justify-center opacity-40 hover:opacity-70 transition-opacity duration-normal"
              style={{ width: logo.width }}
            >
              {/* Placeholder - replace with actual logo images */}
              <div className="text-navy-900 font-heading font-bold text-xl tracking-tight">
                {logo.name}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TrustedBy;
