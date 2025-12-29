'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// TRUSTED BY - Infinite Scrolling Brand Marquee with SVG Logos
// =============================================================================

// SVG logo components for data providers
const BloombergLogo = () => (
  <svg viewBox="0 0 120 24" className="h-5 w-auto" fill="currentColor">
    <path d="M10.5 4h-6v16h6c4.4 0 8-3.6 8-8s-3.6-8-8-8zm0 12.8h-2.8V7.2h2.8c2.6 0 4.8 2.1 4.8 4.8s-2.2 4.8-4.8 4.8zM25 4h3.2v16H25V4zm10.5 0c-4.4 0-8 3.6-8 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 12.8c-2.6 0-4.8-2.1-4.8-4.8s2.1-4.8 4.8-4.8 4.8 2.1 4.8 4.8-2.2 4.8-4.8 4.8zM53 4c-4.4 0-8 3.6-8 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 12.8c-2.6 0-4.8-2.1-4.8-4.8s2.1-4.8 4.8-4.8 4.8 2.1 4.8 4.8-2.2 4.8-4.8 4.8zM69 4h3.5l4 8.5 4-8.5h3.5v16h-3v-10l-3.2 7h-2.6l-3.2-7v10H69V4zm25.5 0c-4.4 0-8 3.6-8 8s3.6 8 8 8h6v-3.2h-6c-2.6 0-4.8-2.1-4.8-4.8s2.1-4.8 4.8-4.8h6V4h-6z" />
  </svg>
);

const ReutersLogo = () => (
  <svg viewBox="0 0 100 24" className="h-5 w-auto" fill="currentColor">
    <path d="M8 4C3.6 4 0 7.6 0 12s3.6 8 8 8h2v-3.2H8c-2.6 0-4.8-2.2-4.8-4.8S5.4 7.2 8 7.2h12v9.6h-6V20h9.2V4H8zm26 0h-6.8v16H30V4h4zm10.5 0c-4.4 0-8 3.6-8 8s3.6 8 8 8 8-3.6 8-8-3.6-8-8-8zm0 12.8c-2.6 0-4.8-2.2-4.8-4.8s2.2-4.8 4.8-4.8 4.8 2.2 4.8 4.8-2.2 4.8-4.8 4.8z" />
  </svg>
);

const MorningstarLogo = () => (
  <svg viewBox="0 0 24 24" className="h-6 w-auto" fill="currentColor">
    <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 16.8l-6.2 4.5 2.4-7.4L2 9.4h7.6z" />
  </svg>
);

const FactSetLogo = () => (
  <svg viewBox="0 0 80 24" className="h-5 w-auto" fill="currentColor">
    <path d="M4 4v16h3.2v-6h8v-3.2h-8V7.2h10V4H4zm18 0v16h3.2V4H22zm10 0v16h12v-3.2h-8.8v-3.6h6v-3.2h-6V7.2H44V4H32z" />
  </svg>
);

const PolygonLogo = () => (
  <svg viewBox="0 0 24 24" className="h-6 w-auto" fill="currentColor">
    <polygon points="12,2 22,8.5 22,15.5 12,22 2,15.5 2,8.5" />
  </svg>
);

const AlphaVantageLogo = () => (
  <svg viewBox="0 0 24 24" className="h-6 w-auto" fill="currentColor">
    <path d="M12 2L2 22h6l4-8 4 8h6L12 2zm0 6l2.5 5h-5L12 8z" />
  </svg>
);

const IEXLogo = () => (
  <svg viewBox="0 0 60 24" className="h-5 w-auto" fill="currentColor">
    <path d="M4 4v16h3.2V4H4zm10 0v16h3.2V4H14zm10 0v16h12v-3.2h-8.8v-3.6h6v-3.2h-6V7.2H36V4H24zm18 0l6 8-6 8h4l4-5.3 4 5.3h4l-6-8 6-8h-4l-4 5.3L50 4h-4z" />
  </svg>
);

const SPGlobalLogo = () => (
  <svg viewBox="0 0 80 24" className="h-5 w-auto" fill="currentColor">
    <path d="M8 4C3.6 4 0 6.4 0 10s2.4 4.8 6 5.6c2 .4 4.8 1.2 4.8 2.8 0 1.2-1.2 2.4-3.2 2.4-2.4 0-4-1.6-4-3.2H0c0 3.2 3.2 6.4 7.6 6.4 4 0 6.8-2.4 6.8-5.6 0-4-3.2-5.2-6.8-6-2-.4-4-.8-4-2.4 0-1.2 1.2-2.4 2.8-2.4 1.6 0 3.2 1.2 3.2 2.8h3.6C13.2 6.4 10.8 4 8 4zm16.8.4c-2 0-3.6.8-4.4 2V4.8h-3.2V24h3.2v-8.8c0-2 1.2-4 3.6-4 2.8 0 3.6 2 3.6 4.4V24h3.2v-9.2c0-4-2-6.4-6-6.4zM42 4c-4.4 0-7.6 3.2-7.6 8.4 0 4.4 2.4 8 7.2 8 3.6 0 6-2 6.8-5.2h-3.2c-.4 1.6-1.6 2.4-3.6 2.4-2.4 0-4-1.6-4-4h11.2v-1.2c0-5.2-2.8-8.4-6.8-8.4zm-4.4 6.4c.4-2 2-3.6 4.4-3.6 2 0 3.6 1.6 4 3.6h-8.4z" />
  </svg>
);

const dataProviders = [
  { name: 'Bloomberg', Logo: BloombergLogo },
  { name: 'Reuters', Logo: ReutersLogo },
  { name: 'S&P Global', Logo: SPGlobalLogo },
  { name: 'Morningstar', Logo: MorningstarLogo },
  { name: 'FactSet', Logo: FactSetLogo },
  { name: 'Polygon', Logo: PolygonLogo },
  { name: 'Alpha Vantage', Logo: AlphaVantageLogo },
  { name: 'IEX Cloud', Logo: IEXLogo },
];

// Duplicate for seamless infinite scroll
const allProviders = [...dataProviders, ...dataProviders];

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-16 lg:py-20 bg-cream-50/80 border-y border-cream-200/50 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12">
        <ScrollReveal>
          <p className="text-center text-sm text-obsidian-400 mb-10 font-medium tracking-wide uppercase">
            Powered by industry-leading data providers
          </p>
        </ScrollReveal>
      </div>

      {/* Infinite Scrolling Marquee */}
      <div className="relative">
        {/* Gradient fade edges */}
        <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-cream-50/90 to-transparent z-10 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-cream-50/90 to-transparent z-10 pointer-events-none" />

        {/* Scrolling container */}
        <div className="flex animate-marquee">
          {allProviders.map((provider, index) => (
            <div
              key={`${provider.name}-${index}`}
              className="flex-shrink-0 flex items-center gap-3 mx-10 group"
            >
              <div className="text-obsidian-300 group-hover:text-obsidian-500 transition-colors duration-300">
                <provider.Logo />
              </div>
              <span className="text-sm font-medium text-obsidian-400 group-hover:text-obsidian-600 transition-colors duration-300 whitespace-nowrap">
                {provider.name}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Marquee animation */}
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
          animation: marquee 40s linear infinite;
        }
        .animate-marquee:hover {
          animation-play-state: paused;
        }
      `}</style>
    </section>
  );
};

export default TrustedBy;
