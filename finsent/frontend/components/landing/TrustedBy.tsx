'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// TRUSTED BY - Infinite Scrolling Logo Marquee
// =============================================================================
// Displays data provider logos with smooth infinite scroll animation.
// =============================================================================

const dataProviders = [
  { name: 'Bloomberg', logo: '📊' },
  { name: 'Reuters', logo: '📰' },
  { name: 'Refinitiv', logo: '📈' },
  { name: 'S&P Global', logo: '🏛️' },
  { name: 'Morningstar', logo: '⭐' },
  { name: 'FactSet', logo: '📋' },
  { name: 'Yahoo Finance', logo: '💜' },
  { name: 'Alpha Vantage', logo: '🔷' },
  { name: 'Polygon.io', logo: '🔶' },
  { name: 'IEX Cloud', logo: '☁️' },
];

// Duplicate array for seamless infinite scroll
const allProviders = [...dataProviders, ...dataProviders];

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-12 lg:py-16 bg-cream-100/50 border-y border-cream-200/50 overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollReveal>
          <p className="text-center text-sm text-obsidian-400 mb-8 font-medium tracking-wide">
            Trusted by users who rely on data from:
          </p>
        </ScrollReveal>
      </div>

      {/* Infinite Scrolling Marquee */}
      <div className="relative">
        {/* Gradient fade edges */}
        <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-cream-100/80 to-transparent z-10 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-cream-100/80 to-transparent z-10 pointer-events-none" />

        {/* Scrolling container */}
        <div className="flex animate-marquee hover:pause">
          {allProviders.map((provider, index) => (
            <div
              key={`${provider.name}-${index}`}
              className="flex-shrink-0 flex items-center gap-3 mx-8 group"
            >
              <span className="text-2xl opacity-60 group-hover:opacity-100 transition-opacity duration-300">
                {provider.logo}
              </span>
              <span className="text-base font-medium text-obsidian-400 group-hover:text-obsidian-600 transition-colors duration-300 whitespace-nowrap">
                {provider.name}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Add marquee animation via style tag */}
      <style jsx>{`
        @keyframes marquee {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-50%);
          }
        }
        .animate-marquee {
          animation: marquee 30s linear infinite;
        }
        .animate-marquee:hover {
          animation-play-state: paused;
        }
      `}</style>
    </section>
  );
};

export default TrustedBy;
