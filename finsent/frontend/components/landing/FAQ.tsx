'use client';

import React, { useState } from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// FAQ SECTION - Glass Accordion
// =============================================================================

const faqs = [
  {
    question: 'How accurate is the sentiment analysis?',
    answer: 'Our AI-powered sentiment analysis achieves 98% accuracy by analyzing millions of data points from news articles, social media, SEC filings, and market data. The model is continuously trained and updated.',
  },
  {
    question: 'Can I cancel my subscription anytime?',
    answer: 'Yes, you can cancel your subscription at any time. If you cancel, you\'ll continue to have access until the end of your current billing period. No questions asked.',
  },
  {
    question: 'What payment methods do you accept?',
    answer: 'We accept all major credit cards (Visa, Mastercard, American Express), as well as PayPal. Enterprise customers can also pay via invoice.',
  },
  {
    question: 'Is my data secure?',
    answer: 'Absolutely. We use bank-level encryption (AES-256) for all data transmission and storage. Your financial data is never shared with third parties and is protected by SOC 2 Type II compliant infrastructure.',
  },
  {
    question: 'Do you offer an API?',
    answer: 'Yes, Pro subscribers get full access to our API, which allows you to integrate Caveray\'s sentiment analysis and market intelligence into your own applications and trading systems.',
  },
];

export const FAQ: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  return (
    <section className="py-24 lg:py-32 bg-cream-50">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal>
          <div className="text-center mb-12">
            <p className="text-sm font-medium text-electric-500 mb-3 tracking-wide uppercase">
              FAQ
            </p>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-obsidian-900 mb-4 tracking-tightest">
              Frequently asked questions
            </h2>
            <p className="text-lg text-obsidian-500">
              Everything you need to know about Caveray.
            </p>
          </div>
        </ScrollReveal>

        {/* FAQ Accordion */}
        <div className="space-y-3">
          {faqs.map((faq, index) => (
            <ScrollReveal key={index} delay={index * 50}>
              <div
                className={`
                  rounded-2xl overflow-hidden transition-all duration-300
                  ${openIndex === index
                    ? 'bg-white shadow-glass-lg'
                    : 'bg-white/80 backdrop-blur-lg border border-cream-200/50 shadow-glass'
                  }
                `}
              >
                <button
                  onClick={() => setOpenIndex(openIndex === index ? null : index)}
                  className="w-full flex items-center justify-between px-6 py-5 text-left"
                >
                  <span className="font-semibold text-obsidian-900 pr-4">
                    {faq.question}
                  </span>
                  <div
                    className={`
                      flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center
                      ${openIndex === index ? 'bg-electric-100 text-electric-600' : 'bg-cream-100 text-obsidian-400'}
                      transition-all duration-200
                    `}
                  >
                    <svg
                      className={`w-4 h-4 transition-transform duration-300 ${openIndex === index ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </button>

                <div
                  className={`
                    overflow-hidden transition-all duration-300 ease-out
                    ${openIndex === index ? 'max-h-96' : 'max-h-0'}
                  `}
                >
                  <div className="px-6 pb-5 text-obsidian-600 leading-relaxed">
                    {faq.answer}
                  </div>
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQ;
