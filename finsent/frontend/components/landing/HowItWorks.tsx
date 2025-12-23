'use client';

// =============================================================================
// HOW IT WORKS SECTION
// =============================================================================
// Step-by-step explanation of the platform
//
// Location: frontend/components/landing/HowItWorks.tsx
//
// =============================================================================

import React from 'react';

const steps = [
  {
    number: '01',
    title: 'Search any stock',
    description: 'Enter a ticker symbol or company name to instantly access comprehensive analysis.',
    image: '/images/how-it-works-1.png', // Placeholder
  },
  {
    number: '02',
    title: 'Analyze the data',
    description: 'Review AI-powered sentiment scores, financial metrics, and insider trading activity.',
    image: '/images/how-it-works-2.png', // Placeholder
  },
  {
    number: '03',
    title: 'Make informed decisions',
    description: 'Use our insights to build your portfolio with confidence and track performance over time.',
    image: '/images/how-it-works-3.png', // Placeholder
  },
];

export const HowItWorks: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-cream-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <p className="text-overline text-terra-500 uppercase tracking-widest mb-4">
            How It Works
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-4">
            From search to insight in seconds
          </h2>
          <p className="text-body-lg text-neutral-600">
            Our platform makes complex financial analysis simple and accessible.
          </p>
        </div>

        {/* Steps */}
        <div className="space-y-16 lg:space-y-24">
          {steps.map((step, index) => (
            <div
              key={step.number}
              className={`
                flex flex-col lg:flex-row items-center gap-12 lg:gap-20
                ${index % 2 === 1 ? 'lg:flex-row-reverse' : ''}
              `}
            >
              {/* Content */}
              <div className="flex-1 max-w-lg">
                {/* Step number */}
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-terra-500 text-white font-display font-bold text-xl mb-6">
                  {step.number}
                </div>

                <h3 className="font-heading font-semibold text-heading-xl text-navy-900 mb-4">
                  {step.title}
                </h3>
                <p className="text-body-lg text-neutral-600">
                  {step.description}
                </p>
              </div>

              {/* Image placeholder */}
              <div className="flex-1 w-full max-w-xl">
                <div className="relative aspect-[4/3] rounded-2xl bg-white border border-border-light shadow-lg overflow-hidden">
                  {/* Placeholder content - replace with actual screenshots */}
                  <div className="absolute inset-0 bg-gradient-to-br from-cream-50 to-cream-100 flex items-center justify-center">
                    <div className="text-center p-8">
                      <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-navy-100 flex items-center justify-center">
                        <svg className="w-8 h-8 text-navy-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <p className="text-body-sm text-neutral-400">
                        Screenshot placeholder
                      </p>
                    </div>
                  </div>

                  {/* Decorative elements */}
                  <div className="absolute top-4 left-4 flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-error-400" />
                    <div className="w-3 h-3 rounded-full bg-warning-400" />
                    <div className="w-3 h-3 rounded-full bg-success-400" />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
