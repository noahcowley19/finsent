'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ScrollReveal, MagneticButton, ArrowIcon } from '@/components/ui';

// =============================================================================
// PRICING SECTION - Glass Morphism Cards
// =============================================================================
// Pricing cards with glassmorphism effect and obsidian primary buttons.
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
    <section className="py-24 lg:py-32 bg-cream-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal>
          <div className="text-center mb-12">
            <p className="text-sm font-medium text-electric-500 mb-3 tracking-wide uppercase">
              Pricing
            </p>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-obsidian-900 mb-4 tracking-tightest">
              Simple, transparent pricing
            </h2>
            <p className="text-lg text-obsidian-500 max-w-xl mx-auto">
              Start free and upgrade when you need more. No hidden fees.
            </p>
          </div>
        </ScrollReveal>

        {/* Billing Toggle */}
        <ScrollReveal delay={100}>
          <div className="flex items-center justify-center gap-1 mb-12 p-1 rounded-xl bg-cream-100 max-w-xs mx-auto">
            <button
              onClick={() => setBilling('monthly')}
              className={`
                flex-1 px-4 py-2.5 text-sm font-medium rounded-lg transition-all duration-200
                ${billing === 'monthly'
                  ? 'bg-white text-obsidian-900 shadow-sm'
                  : 'text-obsidian-500 hover:text-obsidian-700'
                }
              `}
            >
              Monthly
            </button>
            <button
              onClick={() => setBilling('yearly')}
              className={`
                flex-1 px-4 py-2.5 text-sm font-medium rounded-lg transition-all duration-200
                ${billing === 'yearly'
                  ? 'bg-white text-obsidian-900 shadow-sm'
                  : 'text-obsidian-500 hover:text-obsidian-700'
                }
              `}
            >
              Yearly
              <span className="ml-1.5 text-xs text-success-600 font-semibold">-17%</span>
            </button>
          </div>
        </ScrollReveal>

        {/* Pricing Cards */}
        <div className="grid md:grid-cols-2 gap-6 lg:gap-8">
          {plans.map((plan, index) => (
            <ScrollReveal key={plan.name} delay={150 + index * 75}>
              <div
                className={`
                  relative p-8 rounded-3xl transition-all duration-300 hover:-translate-y-1
                  ${plan.popular
                    ? 'bg-white shadow-diffuse border-2 border-electric-500/20'
                    : 'bg-white/80 backdrop-blur-lg border border-cream-200/50 shadow-glass hover:shadow-glass-lg'
                  }
                `}
              >
                {/* Popular Badge */}
                {plan.popular && (
                  <div className="absolute -top-3.5 left-1/2 -translate-x-1/2">
                    <span className="px-4 py-1.5 text-xs font-semibold text-white bg-electric-500 rounded-full shadow-lg">
                      Most popular
                    </span>
                  </div>
                )}

                {/* Plan Header */}
                <div className="mb-6">
                  <h3 className="text-xl font-bold text-obsidian-900 tracking-tight mb-1">
                    {plan.name}
                  </h3>
                  <p className="text-sm text-obsidian-500">{plan.description}</p>
                </div>

                {/* Price */}
                <div className="mb-8">
                  <div className="flex items-baseline gap-1">
                    <span className="text-4xl lg:text-5xl font-bold text-obsidian-900 tracking-tight">
                      ${billing === 'monthly' ? plan.monthlyPrice : plan.yearlyPrice}
                    </span>
                    <span className="text-sm text-obsidian-400">
                      /{billing === 'monthly' ? 'mo' : 'yr'}
                    </span>
                  </div>
                </div>

                {/* Features */}
                <ul className="space-y-3.5 mb-8">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-3">
                      <svg className="w-5 h-5 text-success-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm text-obsidian-600">{feature}</span>
                    </li>
                  ))}
                </ul>

                {/* CTA Button */}
                <MagneticButton
                  href={plan.ctaHref}
                  variant={plan.popular ? 'primary' : 'secondary'}
                  size="lg"
                  className="w-full justify-center"
                  icon={plan.popular ? <ArrowIcon className="w-4 h-4" /> : undefined}
                >
                  {plan.cta}
                </MagneticButton>
              </div>
            </ScrollReveal>
          ))}
        </div>

        {/* Footer Note */}
        <ScrollReveal delay={300}>
          <p className="text-center text-sm text-obsidian-400 mt-10">
            All plans include a 14-day free trial. Cancel anytime.
          </p>
        </ScrollReveal>
      </div>
    </section>
  );
};

export default Pricing;
