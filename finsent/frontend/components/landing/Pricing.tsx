'use client';

// =============================================================================
// PRICING SECTION
// =============================================================================
// Pricing cards with monthly/annual toggle
//
// Location: frontend/components/landing/Pricing.tsx
//
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
    <section className="py-20 lg:py-28 bg-cream-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <p className="text-overline text-terra-500 uppercase tracking-widest mb-4">
            Pricing
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-4">
            Simple, transparent pricing
          </h2>
          <p className="text-body-lg text-neutral-600">
            Start free and upgrade when you&apos;re ready for more power.
          </p>
        </div>

        {/* Billing toggle */}
        <div className="flex items-center justify-center gap-4 mb-12">
          <span className={`text-body-md ${!isAnnual ? 'text-navy-900 font-medium' : 'text-neutral-500'}`}>
            Monthly
          </span>
          <button
            onClick={() => setIsAnnual(!isAnnual)}
            className={`
              relative w-14 h-8 rounded-full
              transition-colors duration-fast
              focus:outline-none focus:ring-2 focus:ring-terra-500 focus:ring-offset-2
              ${isAnnual ? 'bg-terra-500' : 'bg-neutral-300'}
            `}
            aria-label="Toggle annual billing"
          >
            <span
              className={`
                absolute top-1 w-6 h-6 rounded-full bg-white shadow-md
                transition-transform duration-fast
                ${isAnnual ? 'left-7' : 'left-1'}
              `}
            />
          </button>
          <span className={`text-body-md ${isAnnual ? 'text-navy-900 font-medium' : 'text-neutral-500'}`}>
            Annual
          </span>
          {isAnnual && (
            <span className="ml-2 px-2 py-1 text-caption font-medium text-success-700 bg-success-100 rounded-full">
              Save 20%
            </span>
          )}
        </div>

        {/* Pricing cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`
                relative rounded-2xl p-8 lg:p-10
                ${plan.highlighted
                  ? 'bg-navy-900 text-white ring-4 ring-terra-500/50'
                  : 'bg-white border border-border-light'
                }
              `}
            >
              {/* Popular badge */}
              {plan.highlighted && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                  <span className="px-4 py-1.5 bg-terra-500 text-white text-caption font-semibold uppercase tracking-wider rounded-full">
                    Most Popular
                  </span>
                </div>
              )}

              {/* Plan header */}
              <div className="mb-8">
                <h3 className={`font-heading font-semibold text-heading-lg mb-2 ${plan.highlighted ? 'text-white' : 'text-navy-900'}`}>
                  {plan.name}
                </h3>
                <p className={`text-body-md ${plan.highlighted ? 'text-white/70' : 'text-neutral-600'}`}>
                  {plan.description}
                </p>
              </div>

              {/* Price */}
              <div className="mb-8">
                <div className="flex items-baseline gap-2">
                  <span className={`font-display text-display-md ${plan.highlighted ? 'text-white' : 'text-navy-900'}`}>
                    ${isAnnual ? plan.priceAnnual : plan.priceMonthly}
                  </span>
                  {plan.priceMonthly > 0 && (
                    <span className={`text-body-md ${plan.highlighted ? 'text-white/60' : 'text-neutral-500'}`}>
                      /month
                    </span>
                  )}
                </div>
                {isAnnual && plan.priceMonthly > 0 && (
                  <p className={`text-body-sm mt-1 ${plan.highlighted ? 'text-white/60' : 'text-neutral-500'}`}>
                    Billed annually (${(plan.priceAnnual * 12).toFixed(0)}/year)
                  </p>
                )}
              </div>

              {/* CTA */}
              <Link
                href={plan.href}
                className={`
                  block w-full py-4 rounded-xl
                  font-heading font-semibold text-body-md text-center
                  transition-all duration-fast
                  ${plan.highlighted
                    ? 'bg-terra-500 text-white hover:bg-terra-600 hover:-translate-y-0.5 hover:shadow-terra'
                    : 'bg-navy-900 text-white hover:bg-navy-800 hover:-translate-y-0.5'
                  }
                `}
              >
                {plan.cta}
              </Link>

              {/* Features */}
              <div className="mt-8 pt-8 border-t border-white/10">
                <ul className="space-y-4">
                  {plan.features.map((feature, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <svg
                        className={`w-5 h-5 flex-shrink-0 mt-0.5 ${plan.highlighted ? 'text-success-400' : 'text-success-500'}`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      <span className={`text-body-sm ${plan.highlighted ? 'text-white/90' : 'text-neutral-700'}`}>
                        {feature}
                      </span>
                    </li>
                  ))}
                  {plan.limitations.map((limitation, index) => (
                    <li key={`limit-${index}`} className="flex items-start gap-3">
                      <svg
                        className={`w-5 h-5 flex-shrink-0 mt-0.5 ${plan.highlighted ? 'text-white/30' : 'text-neutral-300'}`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                      <span className={`text-body-sm ${plan.highlighted ? 'text-white/50' : 'text-neutral-400'}`}>
                        {limitation}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        {/* Guarantee */}
        <p className="text-center text-body-sm text-neutral-500 mt-10">
          ✓ No credit card required for free plan &nbsp;•&nbsp; ✓ Cancel anytime &nbsp;•&nbsp; ✓ 7-day money-back guarantee
        </p>
      </div>
    </section>
  );
};

export default Pricing;
