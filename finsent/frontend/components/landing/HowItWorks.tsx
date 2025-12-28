'use client';

// =============================================================================
// HOW IT WORKS SECTION - FIXED VERSION
// =============================================================================

import React from 'react';

interface Step {
  number: string;
  icon: React.ReactNode;
  title: string;
  description: string;
}

const steps: Step[] = [
  {
    number: '01',
    icon: (
      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    title: 'Search any stock',
    description: 'Enter a ticker symbol or company name to instantly access comprehensive analysis.',
  },
  {
    number: '02',
    icon: (
      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    title: 'Analyze the data',
    description: 'Review AI-powered sentiment scores, financial metrics, and insider trading activity.',
  },
  {
    number: '03',
    icon: (
      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    title: 'Make informed decisions',
    description: 'Use our insights to build your portfolio with confidence and track performance over time.',
  },
];

export const HowItWorks: React.FC = () => {
  return (
    <section className="py-24 lg:py-32 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16 lg:mb-20">
          <p className="text-body-sm font-semibold text-terra-500 uppercase tracking-widest mb-4">
            How It Works
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-6">
            From search to insight in seconds
          </h2>
          <p className="text-body-lg text-navy-600/70">
            Our platform makes complex financial analysis simple and accessible.
          </p>
        </div>

        <div className="relative">
          <div className="hidden lg:block absolute top-20 left-[16%] right-[16%] h-0.5 bg-gradient-to-r from-terra-200 via-terra-300 to-terra-200" />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 lg:gap-8">
            {steps.map((step) => (
              <div key={step.number} className="relative flex flex-col items-center text-center">
                <div className="relative z-10 w-16 h-16 rounded-2xl bg-gradient-to-br from-terra-500 to-pink-500 shadow-lg shadow-terra-500/30 flex items-center justify-center mb-6">
                  {step.icon}
                </div>

                <span className="absolute -top-2 -right-2 z-20 w-8 h-8 rounded-full bg-navy-900 text-white text-caption font-bold flex items-center justify-center shadow-sm lg:hidden">
                  {step.number}
                </span>

                <h3 className="font-heading font-semibold text-heading-lg text-navy-900 mb-3">
                  {step.title}
                </h3>
                <p className="text-body-md text-navy-600/70 max-w-xs">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
