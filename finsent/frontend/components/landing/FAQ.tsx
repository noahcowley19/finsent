'use client';

import React, { useState } from 'react';

// =============================================================================
// FAQ - Frequently asked questions
// =============================================================================

const faqs = [
  {
    question: 'What data sources does Caveray use?',
    answer: 'We analyze news articles, social media sentiment, SEC filings, earnings transcripts, and more to provide comprehensive market intelligence.',
  },
  {
    question: 'How accurate is the sentiment analysis?',
    answer: 'Our AI models achieve 98% accuracy on sentiment classification, validated against expert analyst ratings and historical price movements.',
  },
  {
    question: 'Can I cancel my subscription anytime?',
    answer: 'Yes, you can cancel your Pro subscription at any time. You\'ll continue to have access until the end of your billing period.',
  },
  {
    question: 'Is my data secure?',
    answer: 'Absolutely. We use bank-level encryption and never share your personal data or portfolio information with third parties.',
  },
  {
    question: 'What\'s included in the free plan?',
    answer: 'The free plan includes 10 stock searches per day, 3 sentiment analyses, a basic watchlist with 10 stocks, and access to the market movers dashboard.',
  },
];

export const FAQ: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <section className="py-20 lg:py-28 bg-ink-50">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <p className="text-body-sm font-medium text-accent mb-3">FAQ</p>
          <h2 className="text-display-sm lg:text-display-md text-ink-900">
            Common questions
          </h2>
        </div>

        {/* FAQ List */}
        <div className="space-y-3">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="bg-white rounded-xl border border-ink-200/50 overflow-hidden"
            >
              <button
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
                className="flex items-center justify-between w-full px-6 py-4 text-left"
              >
                <span className="font-medium text-body-md text-ink-900 pr-4">
                  {faq.question}
                </span>
                <svg
                  className={`w-5 h-5 text-ink-400 flex-shrink-0 transition-transform duration-200 ${openIndex === index ? 'rotate-180' : ''
                    }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {openIndex === index && (
                <div className="px-6 pb-4">
                  <p className="text-body-sm text-ink-500 leading-relaxed">
                    {faq.answer}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQ;
