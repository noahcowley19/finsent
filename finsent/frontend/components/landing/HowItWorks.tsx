'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// HOW IT WORKS - Step-by-step process with Glass Cards
// =============================================================================
// Three-step process displayed with glass cards and scroll reveal animations.
// =============================================================================

const steps = [
  {
    number: '01',
    title: 'Search Any Stock',
    description: 'Enter a ticker symbol or company name to start your analysis.',
    accent: 'electric',
  },
  {
    number: '02',
    title: 'Get AI Insights',
    description: 'Our AI analyzes sentiment from news, social media, and SEC filings.',
    accent: 'coral',
  },
  {
    number: '03',
    title: 'Make Better Decisions',
    description: 'Use data-driven insights to inform your investment strategy.',
    accent: 'amber',
  },
];

const accentClasses = {
  electric: 'bg-electric-500',
  coral: 'bg-coral-500',
  amber: 'bg-amber-500',
};

export const HowItWorks: React.FC = () => {
  return (
    <section className="py-24 lg:py-32 bg-white">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal>
          <div className="text-center mb-16">
            <p className="text-sm font-medium text-electric-500 mb-3 tracking-wide uppercase">
              How it works
            </p>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-obsidian-900 mb-4 tracking-tightest">
              Three simple steps
            </h2>
            <p className="text-lg text-obsidian-500 max-w-xl mx-auto">
              Start making smarter investment decisions in minutes, not hours.
            </p>
          </div>
        </ScrollReveal>

        {/* Steps Grid */}
        <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
          {steps.map((step, index) => (
            <ScrollReveal key={step.number} delay={index * 100}>
              <div className="relative text-center p-8 rounded-2xl bg-white/80 backdrop-blur-lg border border-cream-200/50 shadow-glass hover:shadow-glass-lg transition-all duration-300 hover:-translate-y-1">
                {/* Step Number */}
                <div
                  className={`
                    inline-flex items-center justify-center w-14 h-14 rounded-2xl 
                    ${accentClasses[step.accent as keyof typeof accentClasses]}
                    text-white text-lg font-bold mb-5
                  `}
                >
                  {step.number}
                </div>

                {/* Content */}
                <h3 className="font-semibold text-xl text-obsidian-900 mb-3 tracking-tight">
                  {step.title}
                </h3>
                <p className="text-sm text-obsidian-500 leading-relaxed">
                  {step.description}
                </p>

                {/* Connector line (hidden on mobile) */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 lg:-right-6 w-8 lg:w-12 h-px bg-cream-300" />
                )}
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
