'use client';

import React from 'react';

// =============================================================================
// HOW IT WORKS - Step-by-step process
// =============================================================================

const steps = [
  {
    number: '01',
    title: 'Search Any Stock',
    description: 'Enter a ticker symbol or company name to start your analysis.',
  },
  {
    number: '02',
    title: 'Get AI Insights',
    description: 'Our AI analyzes sentiment from news, social media, and SEC filings.',
  },
  {
    number: '03',
    title: 'Make Better Decisions',
    description: 'Use data-driven insights to inform your investment strategy.',
  },
];

export const HowItWorks: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-white">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <p className="text-body-sm font-medium text-accent mb-3">How it works</p>
          <h2 className="text-display-sm lg:text-display-md text-ink-900 mb-4">
            Three simple steps
          </h2>
          <p className="text-body-lg text-ink-500 max-w-xl mx-auto">
            Start making smarter investment decisions in minutes, not hours.
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-3 gap-8 lg:gap-12">
          {steps.map((step, index) => (
            <div key={step.number} className="text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-ink-900 text-white text-body-sm font-semibold mb-4">
                {step.number}
              </div>
              <h3 className="font-semibold text-heading-md text-ink-900 mb-2 tracking-tight">
                {step.title}
              </h3>
              <p className="text-body-sm text-ink-500 leading-relaxed">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
