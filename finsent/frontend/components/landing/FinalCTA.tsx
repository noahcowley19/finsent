'use client';

// =============================================================================
// FINAL CTA SECTION - REDESIGNED
// =============================================================================
// Bottom call-to-action with mesh gradient background
//
// Location: frontend/components/landing/FinalCTA.tsx
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { HiArrowRight, HiSparkles } from 'react-icons/hi';
import { MeshGradient } from '@/components/ui/MeshGradient';

export const FinalCTA: React.FC = () => {
  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
      {/* Mesh Gradient Background */}
      <MeshGradient className="z-0" intensity={0.8} />

      {/* Content */}
      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/70 backdrop-blur-md rounded-full border border-white/50 shadow-sm mb-8">
          <HiSparkles className="w-4 h-4 text-terra-500" />
          <span className="text-body-sm font-medium text-navy-700">Ready to get started?</span>
        </div>

        {/* Headline */}
        <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-6">
          Start making smarter investment decisions today
        </h2>

        {/* Description */}
        <p className="text-body-lg text-navy-600/80 mb-10 max-w-2xl mx-auto">
          Join thousands of investors who use Caveray to analyze markets,
          track sentiment, and discover opportunities.
        </p>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/signup"
            className="
              inline-flex items-center justify-center gap-2
              px-8 py-4
              bg-navy-900 text-white
              font-heading font-semibold text-body-lg
              rounded-xl
              transition-all duration-200
              hover:bg-navy-800 hover:scale-[1.02]
              shadow-lg shadow-navy-900/20
            "
          >
            Get Started Free
            <HiArrowRight className="w-5 h-5" />
          </Link>
          <Link
            href="/search"
            className="
              inline-flex items-center justify-center gap-2
              px-8 py-4
              bg-white/80 backdrop-blur-md
              text-navy-700
              font-heading font-medium text-body-lg
              rounded-xl
              border border-white/50
              transition-all duration-200
              hover:bg-white hover:shadow-lg
            "
          >
            Try Stock Search
          </Link>
        </div>

        {/* Trust indicators */}
        <p className="text-body-sm text-navy-500/70 mt-8">
          No credit card required • Free plan available forever
        </p>
      </div>
    </section>
  );
};

export default FinalCTA;
