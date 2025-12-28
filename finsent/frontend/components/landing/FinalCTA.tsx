'use client';

import React from 'react';
import { ScrollReveal, MagneticButton, ArrowIcon } from '@/components/ui';
import { AtmosphericBackground } from './AtmosphericBackground';

// =============================================================================
// FINAL CTA SECTION - Atmospheric with Magnetic Button
// =============================================================================

export const FinalCTA: React.FC = () => {
  return (
    <AtmosphericBackground className="py-24 lg:py-32" variant="intense">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <ScrollReveal>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-obsidian-900 mb-6 tracking-tightest">
            Ready to invest smarter?
          </h2>
        </ScrollReveal>

        <ScrollReveal delay={100}>
          <p className="text-lg lg:text-xl text-obsidian-500 mb-10 max-w-2xl mx-auto">
            Join thousands of investors using Caveray to make data-driven decisions.
            Start your free trial today.
          </p>
        </ScrollReveal>

        <ScrollReveal delay={200}>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <MagneticButton
              href="/signup"
              variant="primary"
              size="lg"
              icon={<ArrowIcon className="w-4 h-4" />}
            >
              Get started free
            </MagneticButton>

            <MagneticButton
              href="/pricing"
              variant="secondary"
              size="lg"
            >
              View pricing
            </MagneticButton>
          </div>
        </ScrollReveal>

        <ScrollReveal delay={300}>
          <p className="text-sm text-obsidian-400 mt-8">
            No credit card required • 14-day free trial • Cancel anytime
          </p>
        </ScrollReveal>
      </div>
    </AtmosphericBackground>
  );
};

export default FinalCTA;
