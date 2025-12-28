'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// TRUSTED BY - Logo Section
// =============================================================================
// Displays partner/trust logos with subtle styling matching the new aesthetic.
// =============================================================================

const logos = [
  { name: 'TechCrunch', color: '#00D084' },
  { name: 'Bloomberg', color: '#333' },
  { name: 'Forbes', color: '#B8001C' },
  { name: 'The Verge', color: '#FF002F' },
  { name: 'WSJ', color: '#1A1A1A' },
];

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-16 lg:py-20 bg-cream-100/50 border-y border-cream-200/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollReveal>
          <p className="text-center text-sm text-obsidian-400 mb-8 font-medium uppercase tracking-wider">
            Trusted by investors worldwide
          </p>

          <div className="flex flex-wrap items-center justify-center gap-8 lg:gap-12">
            {logos.map((logo) => (
              <div
                key={logo.name}
                className="text-lg font-semibold text-obsidian-300 hover:text-obsidian-500 transition-colors duration-200 cursor-default"
              >
                {logo.name}
              </div>
            ))}
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
};

export default TrustedBy;
