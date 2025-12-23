'use client';

// =============================================================================
// FAQ SECTION
// =============================================================================
// Frequently asked questions with accordion
//
// Location: frontend/components/landing/FAQ.tsx
//
// =============================================================================

import React, { useState } from 'react';

const faqs = [
  {
    question: 'What is Caveray?',
    answer: 'Caveray is an AI-powered financial intelligence platform that helps investors analyze market sentiment, track insider trading, and make more informed investment decisions. We aggregate data from social media, news sources, and SEC filings to give you a comprehensive view of any stock.',
  },
  {
    question: 'How does the sentiment analysis work?',
    answer: 'Our sentiment analysis uses advanced natural language processing (NLP) models including FinBERT and VADER to analyze text from sources like StockTwits, Twitter/X, Reddit, and financial news. The AI classifies sentiment as bullish, bearish, or neutral, and provides an overall score for each stock.',
  },
  {
    question: 'Is my data secure?',
    answer: 'Absolutely. We use industry-standard encryption for all data in transit and at rest. We never share your personal information or portfolio data with third parties. Your investment data is only visible to you.',
  },
  {
    question: 'Can I cancel my subscription anytime?',
    answer: 'Yes, you can cancel your Pro subscription at any time. You\'ll continue to have access to Pro features until the end of your billing period. We also offer a 7-day money-back guarantee if you\'re not satisfied.',
  },
  {
    question: 'What\'s the difference between Free and Pro?',
    answer: 'The Free plan gives you limited daily searches and analyses, basic portfolio tracking, and 7-day history. Pro unlocks unlimited usage, Quant Lab for backtesting strategies, export features, email alerts, 90-day history, and priority support.',
  },
  {
    question: 'Where does your data come from?',
    answer: 'We aggregate data from multiple sources including real-time stock prices, SEC filings (Forms 3, 4, 13F), social media platforms (StockTwits, Twitter/X, Reddit), financial news outlets, and company financial statements. All data is updated in real-time or near real-time.',
  },
  {
    question: 'Do you provide investment advice?',
    answer: 'No, Caveray is an informational platform only. We provide data, analysis, and insights, but we do not provide personalized investment advice. All investment decisions should be made based on your own research and, if needed, consultation with a qualified financial advisor.',
  },
  {
    question: 'How accurate is the insider trading data?',
    answer: 'Our insider trading data comes directly from SEC Form 4 filings, which executives are required to file within 2 business days of a transaction. We parse and display this data as soon as it\'s available, typically within hours of filing.',
  },
];

const FAQItem: React.FC<{
  question: string;
  answer: string;
  isOpen: boolean;
  onToggle: () => void;
}> = ({ question, answer, isOpen, onToggle }) => {
  return (
    <div className="border-b border-border-light last:border-b-0">
      <button
        onClick={onToggle}
        className="w-full py-6 flex items-center justify-between text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-navy-500 focus-visible:ring-offset-2 rounded-lg"
        aria-expanded={isOpen}
      >
        <span className="font-heading font-semibold text-heading-sm text-navy-900 pr-8">
          {question}
        </span>
        <span
          className={`
            flex-shrink-0 w-8 h-8 rounded-full
            flex items-center justify-center
            transition-all duration-fast
            ${isOpen ? 'bg-terra-500 text-white rotate-180' : 'bg-cream-100 text-navy-500'}
          `}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>
      <div
        className={`
          overflow-hidden transition-all duration-normal
          ${isOpen ? 'max-h-96 pb-6' : 'max-h-0'}
        `}
      >
        <p className="text-body-md text-neutral-600 pr-16">
          {answer}
        </p>
      </div>
    </div>
  );
};

export const FAQ: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const handleToggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="py-20 lg:py-28 bg-cream-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <p className="text-overline text-terra-500 uppercase tracking-widest mb-4">
            FAQ
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-4">
            Frequently asked questions
          </h2>
          <p className="text-body-lg text-neutral-600">
            Everything you need to know about Caveray. Can&apos;t find what you&apos;re looking for?{' '}
            <a href="/contact" className="text-terra-500 hover:text-terra-600 underline">
              Contact our team
            </a>
            .
          </p>
        </div>

        {/* FAQ accordion */}
        <div className="bg-white rounded-2xl border border-border-light p-2 lg:p-4">
          {faqs.map((faq, index) => (
            <FAQItem
              key={index}
              question={faq.question}
              answer={faq.answer}
              isOpen={openIndex === index}
              onToggle={() => handleToggle(index)}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQ;
