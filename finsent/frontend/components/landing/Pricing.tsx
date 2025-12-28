'use client';

import React, { useState } from 'react';
import Link from 'next/link';

// =============================================================================
// PRICING SECTION - Modern SaaS Aesthetic
// =============================================================================

type BillingPeriod = 'monthly' | 'yearly';

const plans = [
  {
    name: 'Free',
    description: 'Perfect for getting started',
    monthlyPrice: 0,
    yearlyPrice: 0,
    features: [
      '10 stock searches per day',
      '3 sentiment analyses per day',
      'Basic watchlist (10 stocks)',
      'Market movers dashboard',
      'Email support',
    ],
    cta: 'Get started',
    ctaHref: '/signup',
    popular: false,
  },
  {
    name: 'Pro',
    description: 'For serious investors',
    monthlyPrice: 29,
    yearlyPrice: 290,
    features: [
      'Unlimited stock searches',
      'Unlimited sentiment analyses',
      'Unlimited watchlist',
      'Portfolio analytics',
      'Quant Lab access',
      'Insider trading tracker',
      'API access',
      'Priority support',
    ],
    cta: 'Start free trial',
    ctaHref: '/signup?plan=pro',
    popular: true,
  },
];

export const Pricing: React.FC = () => {
  const [billing, setBilling] = useState<BillingPeriod>('monthly');

  return (
    <section className="py-20 lg:py-28 bg-ink-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <p className="text-body-sm font-medium text-accent mb-3">Pricing</p>
          <h2 className="text-display-sm lg:text-display-md text-ink-900 mb-4">
            Simple, transparent pricing
          </h2>
          <p className="text-body-lg text-ink-500 max-w-xl mx-auto">
            Start free and upgrade when you need more. No hidden fees.
          </p>
        </div>

        {/* Billing Toggle */}
        <div className="flex items-center justify-center gap-3 mb-12">
          <button
            onClick={() => setBilling('monthly')}
            className={`px-4 py-2 text-body-sm font-medium rounded-lg transition-all ${billing === 'monthly'
                ? 'bg-ink-900 text-white'
                : 'text-ink-600 hover:text-ink-900'
              }`}
          >
            Monthly
          </button>
          <button
            onClick={() => setBilling('yearly')}
            className={`px-4 py-2 text-body-sm font-medium rounded-lg transition-all ${billing === 'yearly'
                ? 'bg-ink-900 text-white'
                : 'text-ink-600 hover:text-ink-900'
              }`}
          >
            Yearly
            <span className="ml-1.5 text-body-xs text-success-600">Save 17%</span>
          </button>
        </div>

        {/* Pricing Cards */}
        <div className="grid md:grid-cols-2 gap-6 lg:gap-8">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`
                relative p-8 rounded-2xl bg-white border transition-all duration-200
                ${plan.popular
                  ? 'border-accent shadow-lg shadow-accent/10'
                  : 'border-ink-200/50 hover:border-ink-300/50 hover:shadow-lg'
                }
              `}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                  <span className="px-3 py-1 text-body-xs font-medium text-white bg-accent rounded-full">
                    Most popular
                  </span>
                </div>
              )}

              <div className="mb-6">
                <h3 className="text-heading-lg font-semibold text-ink-900 tracking-tight mb-1">
                  {plan.name}
                </h3>
                <p className="text-body-sm text-ink-500">{plan.description}</p>
              </div>

              <div className="mb-6">
                <div className="flex items-baseline gap-1">
                  <span className="text-display-md text-ink-900">
                    ${billing === 'monthly' ? plan.monthlyPrice : plan.yearlyPrice}
                  </span>
                  <span className="text-body-sm text-ink-400">
                    /{billing === 'monthly' ? 'mo' : 'yr'}
                  </span>
                </div>
              </div>

              <ul className="space-y-3 mb-8">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-2.5">
                    <svg className="w-5 h-5 text-success-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    <span className="text-body-sm text-ink-600">{feature}</span>
                  </li>
                ))}
              </ul>

              <Link
                href={plan.ctaHref}
                className={`
                  block w-full py-3 text-center text-body-sm font-medium rounded-xl transition-all duration-150
                  ${plan.popular
                    ? 'bg-ink-900 text-white hover:bg-ink-800 hover:-translate-y-0.5 hover:shadow-lg'
                    : 'bg-ink-100 text-ink-700 hover:bg-ink-200'
                  }
                `}
              >
                {plan.cta}
              </Link>
            </div>
          ))}
        </div>

        {/* Footer Note */}
        <p className="text-center text-body-sm text-ink-400 mt-8">
          All plans include a 14-day free trial. Cancel anytime.
        </p>
      </div>
    </section>
  );
};

export default Pricing;
