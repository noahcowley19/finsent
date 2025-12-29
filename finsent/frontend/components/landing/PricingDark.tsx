'use client';

// =============================================================================
// PRICING DARK - Dark Theme Pricing Section
// =============================================================================
// Pricing section for the dark-themed Financial OS landing page
//
// Location: frontend/components/landing/PricingDark.tsx
// =============================================================================

import React, { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

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
        cta: 'Get Started',
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
        cta: 'Start Free Trial',
        ctaHref: '/signup?plan=pro',
        popular: true,
    },
];

export const PricingDark: React.FC = () => {
    const [billing, setBilling] = useState<BillingPeriod>('monthly');

    return (
        <section className="py-24 lg:py-32 bg-obsidian-950">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-12"
                >
                    <p className="text-sm font-medium text-electric-400 mb-3 tracking-wide uppercase">
                        Pricing
                    </p>
                    <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-4 tracking-tight">
                        Simple, transparent pricing
                    </h2>
                    <p className="text-lg text-obsidian-400 max-w-xl mx-auto">
                        Start free and upgrade when you need more. No hidden fees.
                    </p>
                </motion.div>

                {/* Billing Toggle */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 }}
                    className="flex items-center justify-center gap-1 mb-12 p-1 rounded-xl bg-obsidian-900 border border-obsidian-800 max-w-xs mx-auto"
                >
                    <button
                        onClick={() => setBilling('monthly')}
                        className={`
              flex-1 px-4 py-2.5 text-sm font-medium rounded-lg transition-all duration-200
              ${billing === 'monthly'
                                ? 'bg-electric-500 text-white shadow-glow-sm'
                                : 'text-obsidian-400 hover:text-white'
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
                                ? 'bg-electric-500 text-white shadow-glow-sm'
                                : 'text-obsidian-400 hover:text-white'
                            }
            `}
                    >
                        Yearly
                        <span className="ml-1.5 text-xs text-success-400 font-semibold">-17%</span>
                    </button>
                </motion.div>

                {/* Pricing Cards */}
                <div className="grid md:grid-cols-2 gap-6 lg:gap-8">
                    {plans.map((plan, index) => (
                        <motion.div
                            key={plan.name}
                            initial={{ opacity: 0, y: 40 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 + index * 0.1 }}
                            whileHover={{ y: -4 }}
                            className={`
                relative p-8 rounded-3xl transition-all duration-300
                ${plan.popular
                                    ? 'bg-obsidian-900/80 border-2 border-electric-500/40 shadow-glow-lg'
                                    : 'bg-obsidian-900/50 backdrop-blur-lg border border-obsidian-800 hover:border-obsidian-700'
                                }
              `}
                        >
                            {/* Popular Badge */}
                            {plan.popular && (
                                <div className="absolute -top-3.5 left-1/2 -translate-x-1/2">
                                    <span className="px-4 py-1.5 text-xs font-semibold text-white bg-electric-500 rounded-full shadow-glow-md">
                                        Most popular
                                    </span>
                                </div>
                            )}

                            {/* Plan Header */}
                            <div className="mb-6">
                                <h3 className="text-xl font-bold text-white tracking-tight mb-1">
                                    {plan.name}
                                </h3>
                                <p className="text-sm text-obsidian-400">{plan.description}</p>
                            </div>

                            {/* Price */}
                            <div className="mb-8">
                                <div className="flex items-baseline gap-1">
                                    <span className="text-4xl lg:text-5xl font-bold text-white tracking-tight">
                                        ${billing === 'monthly' ? plan.monthlyPrice : plan.yearlyPrice}
                                    </span>
                                    <span className="text-sm text-obsidian-500">
                                        /{billing === 'monthly' ? 'mo' : 'yr'}
                                    </span>
                                </div>
                            </div>

                            {/* Features */}
                            <ul className="space-y-3.5 mb-8">
                                {plan.features.map((feature) => (
                                    <li key={feature} className="flex items-start gap-3">
                                        <svg className="w-5 h-5 text-success-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        <span className="text-sm text-obsidian-300">{feature}</span>
                                    </li>
                                ))}
                            </ul>

                            {/* CTA Button */}
                            <Link
                                href={plan.ctaHref}
                                className={`
                  flex items-center justify-center gap-2 w-full py-4 rounded-xl font-semibold transition-all duration-200
                  ${plan.popular
                                        ? 'bg-electric-500 text-white hover:bg-electric-600 hover:-translate-y-0.5 shadow-glow-sm hover:shadow-glow-md'
                                        : 'bg-obsidian-800 text-white border border-obsidian-700 hover:border-obsidian-600 hover:bg-obsidian-700'
                                    }
                `}
                            >
                                {plan.cta}
                                {plan.popular && (
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                    </svg>
                                )}
                            </Link>
                        </motion.div>
                    ))}
                </div>

                {/* Footer Note */}
                <motion.p
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.3 }}
                    className="text-center text-sm text-obsidian-500 mt-10"
                >
                    All plans include a 14-day free trial. Cancel anytime.
                </motion.p>
            </div>
        </section>
    );
};

export default PricingDark;
