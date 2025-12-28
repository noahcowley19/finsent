'use client';

// =============================================================================
// PRICING SECTION - FIXED VERSION
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';

const plans = [
  {
    name: 'Free',
    description: 'Perfect for getting started with market analysis.',
    priceMonthly: 0,
    priceAnnual: 0,
    features: [
      'Stock search & basic data',
      '10 searches per day',
      '3 sentiment analyses per day',
      '5 portfolio positions',
      '10 watchlist stocks',
      '7-day analysis history',
    ],
    limitations: [
      'No Quant Lab access',
      'No export features',
      'No email alerts',
    ],
    cta: 'Get Started Free',
    href: '/signup',
    highlighted: false,
  },
  {
    name: 'Pro',
    description: 'For serious investors who want the full picture.',
    priceMonthly: 9.99,
    priceAnnual: 7.99,
    features: [
      'Everything in Free, plus:',
      'Unlimited searches',
      'Unlimited sentiment analyses',
      'Unlimited portfolio positions',
      'Unlimited watchlist stocks',
      'Quant Lab access',
      'Export to CSV/PDF',
      'Email alerts',
      '90-day analysis history',
      'Priority support',
    ],
    limitations: [],
    cta: 'Start Pro Trial',
    href: '/signup?plan=pro',
    highlighted: true,
  },
];

export const Pricing: React.FC = () => {
  const [isAnnual, setIsAnnual] = useState(true);

  return (
    <section className="py-24 lg:py-32 bg-cream-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <p className="text-body-sm font-semibold text-terra-500 uppercase tracking-widest mb-4">
            Pricing
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-6">
            Simple, transparent pricing
          </h2>
          <p className="text-body-lg text-navy-600/70">
            Start free and upgrade when you&apos;re ready for more power.
          </p>
        </div>

        <div className="flex items-center justify-center gap-4 mb-16">
          <span className={`text-body-md transition-colors duration-200 ${!isAnnual ? 'text-navy-900 font-semibold' : 'text-navy-400'}`}>
            Monthly
          </span>
          <button
            onClick={() => setIsAnnual(!isAnnual)}
            className={`relative w-16 h-9 rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-terra-500 focus:ring-offset-2 ${isAnnual ? 'bg-gradient-to-r from-terra-500 to-pink-500' : 'bg-navy-200'}`}
            aria-label="Toggle annual billing"
          >
            <span className={`absolute top-1.5 w-6 h-6 rounded-full bg-white shadow-md transition-all duration-300 ease-out ${isAnnual ? 'left-8' : 'left-1.5'}`} />
          </button>
          <span className={`text-body-md transition-colors duration-200 ${isAnnual ? 'text-navy-900 font-semibold' : 'text-navy-400'}`}>
            Annual
          </span>
          {isAnnual && (
            <span className="ml-2 px-3 py-1.5 text-caption font-semibold text-white bg-gradient-to-r from-green-500 to-emerald-500 rounded-full shadow-sm">
              Save 20%
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`relative rounded-3xl overflow-hidden ${plan.highlighted ? 'bg-navy-900 text-white shadow-2xl shadow-navy-900/20' : 'bg-white border border-navy-100/50 shadow-lg shadow-navy-900/5'}`}
            >
              {plan.highlighted && (
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-terra-500 via-pink-500 to-purple-500" />
              )}

              <div className="p-8 lg:p-10">
                {plan.highlighted && (
                  <div className="flex items-center gap-2 mb-6">
                    <svg className="w-5 h-5 text-terra-400" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span className="text-body-sm font-semibold text-terra-400 uppercase tracking-wider">
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="mb-6">
                  <h3 className={`font-heading font-bold text-2xl mb-2 ${plan.highlighted ? 'text-white' : 'text-navy-900'}`}>
                    {plan.name}
                  </h3>
                  <p className={`text-body-md ${plan.highlighted ? 'text-white/60' : 'text-navy-500'}`}>
                    {plan.description}
                  </p>
                </div>

                <div className="mb-8">
                  <div className="flex items-baseline gap-2">
                    <span className={`font-display text-5xl ${plan.highlighted ? 'text-white' : 'text-navy-900'}`}>
                      ${isAnnual ? plan.priceAnnual : plan.priceMonthly}
                    </span>
                    {plan.priceMonthly > 0 && (
                      <span className={`text-body-lg ${plan.highlighted ? 'text-white/50' : 'text-navy-400'}`}>
                        /month
                      </span>
                    )}
                  </div>
                  {isAnnual && plan.priceMonthly > 0 && (
                    <p className={`text-body-sm mt-2 ${plan.highlighted ? 'text-white/50' : 'text-navy-400'}`}>
                      Billed annually (${(plan.priceAnnual * 12).toFixed(0)}/year)
                    </p>
                  )}
                </div>

                <Link
                  href={plan.href}
                  className={`flex items-center justify-center gap-2 w-full py-4 rounded-xl font-heading font-semibold text-body-md transition-all duration-200 ${plan.highlighted ? 'bg-white text-navy-900 hover:bg-cream-50 shadow-lg' : 'bg-navy-900 text-white hover:bg-navy-800'} hover:scale-[1.02]`}
                >
                  {plan.cta}
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </Link>

                <div className="mt-8 pt-8 border-t border-white/10">
                  <ul className="space-y-4">
                    {plan.features.map((feature, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <div className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center ${plan.highlighted ? 'bg-green-500/20' : 'bg-green-100'}`}>
                          <svg className={`w-3 h-3 ${plan.highlighted ? 'text-green-400' : 'text-green-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                        <span className={`text-body-sm ${plan.highlighted ? 'text-white/80' : 'text-navy-600'}`}>
                          {feature}
                        </span>
                      </li>
                    ))}
                    {plan.limitations.map((limitation, index) => (
                      <li key={`limit-${index}`} className="flex items-start gap-3">
                        <div className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center ${plan.highlighted ? 'bg-white/5' : 'bg-navy-50'}`}>
                          <svg className={`w-3 h-3 ${plan.highlighted ? 'text-white/30' : 'text-navy-300'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </div>
                        <span className={`text-body-sm ${plan.highlighted ? 'text-white/40' : 'text-navy-400'}`}>
                          {limitation}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-center gap-6 mt-12 text-body-sm text-navy-500">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span>No credit card required for free plan</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span>Cancel anytime</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span>7-day money-back guarantee</span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Pricing;
