'use client';

// =============================================================================
// FAQ SECTION - FIXED VERSION
// =============================================================================

import React, { useState } from 'react';

const faqs = [
  {
    question: 'What is Caveray and how does it work?',
    answer: 'Caveray is an AI-powered financial intelligence platform that analyzes market sentiment, tracks insider trading, and provides comprehensive stock analysis. We aggregate data from social media, news, SEC filings, and financial reports to give you actionable insights.',
  },
  {
    question: 'Is there a free plan available?',
    answer: 'Yes! Our free plan includes stock search, basic sentiment analysis, portfolio tracking for up to 5 positions, and 7-day analysis history. No credit card required to get started.',
  },
  {
    question: "What's included in the Pro plan?",
    answer: 'Pro unlocks unlimited searches and analyses, full Quant Lab access for backtesting strategies, export capabilities, email alerts, 90-day history, and priority support. It\'s designed for serious investors who want comprehensive tools.',
  },
  {
    question: 'How accurate is the sentiment analysis?',
    answer: 'Our AI models achieve 98% accuracy in sentiment classification. We continuously train on financial language patterns and validate against market movements. The sentiment scores aggregate thousands of data points for reliable insights.',
  },
  {
    question: 'Can I cancel my subscription anytime?',
    answer: 'Absolutely. You can cancel your Pro subscription at any time, and you\'ll continue to have access until the end of your billing period. We also offer a 7-day money-back guarantee for annual plans.',
  },
  {
    question: 'What data sources do you use?',
    answer: 'We analyze data from Twitter/X, Reddit, financial news outlets, SEC filings, earnings transcripts, and proprietary market data feeds. All sources are processed in real-time to provide the most current insights.',
  },
];

const FAQItem: React.FC<{
  question: string;
  answer: string;
  isOpen: boolean;
  onToggle: () => void;
}> = ({ question, answer, isOpen, onToggle }) => {
  return (
    <div className="border-b border-navy-100/50 last:border-b-0">
      <button onClick={onToggle} className="w-full flex items-center justify-between gap-4 py-6 text-left group">
        <span className="font-heading font-semibold text-heading-sm text-navy-900 group-hover:text-navy-700 transition-colors">
          {question}
        </span>
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300 ${isOpen ? 'bg-navy-900 text-white rotate-0' : 'bg-navy-100 text-navy-600 group-hover:bg-navy-200'}`}>
          {isOpen ? (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          )}
        </div>
      </button>

      <div className={`overflow-hidden transition-all duration-300 ease-out ${isOpen ? 'max-h-96 opacity-100 pb-6' : 'max-h-0 opacity-0'}`}>
        <p className="text-body-md text-navy-600/80 leading-relaxed pr-12">
          {answer}
        </p>
      </div>
    </div>
  );
};

export const FAQ: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  return (
    <section className="py-24 lg:py-32 bg-white">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <p className="text-body-sm font-semibold text-terra-500 uppercase tracking-widest mb-4">
            FAQ
          </p>
          <h2 className="font-display text-display-md text-navy-900">
            Frequently asked questions
          </h2>
        </div>

        <div className="bg-cream-50/50 rounded-2xl border border-navy-100/30 p-2">
          <div className="bg-white rounded-xl px-6">
            {faqs.map((faq, index) => (
              <FAQItem
                key={index}
                question={faq.question}
                answer={faq.answer}
                isOpen={openIndex === index}
                onToggle={() => setOpenIndex(openIndex === index ? null : index)}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default FAQ;
