'use client';

// =============================================================================
// FAQ DARK - Dark Theme FAQ Section
// =============================================================================
// FAQ section for the dark-themed Financial OS landing page
//
// Location: frontend/components/landing/FAQDark.tsx
// =============================================================================

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const faqs = [
    {
        question: 'What makes Caveray different from other financial platforms?',
        answer: 'Caveray is designed as a Financial Operating System, not just a data provider. We combine real-time sentiment analysis, insider trading intelligence, quantitative backtesting, and portfolio stress-testing in one unified platform—all powered by Python with institutional-grade analytics.',
    },
    {
        question: 'Is my financial data secure?',
        answer: 'Absolutely. We use bank-level encryption (AES-256) for all data at rest and TLS 1.3 for data in transit. We never sell your data, and your portfolio information is only accessible to you.',
    },
    {
        question: 'Can I use Caveray for algorithmic trading?',
        answer: 'Caveray\'s Quant Lab allows you to build, backtest, and analyze trading strategies. While we provide powerful analytics, we don\'t currently offer direct brokerage integration for live trading.',
    },
    {
        question: 'What data sources do you use?',
        answer: 'We aggregate data from SEC filings (Form 4 for insider trades), Congressional disclosures, social media sentiment, news articles, and market data providers to give you a comprehensive view of each security.',
    },
    {
        question: 'Is there a free trial?',
        answer: 'Yes! All paid plans include a 14-day free trial. You can also use our Free tier indefinitely with limited daily searches and analyses.',
    },
    {
        question: 'Can I cancel my subscription anytime?',
        answer: 'Yes, you can cancel your subscription at any time. There are no long-term contracts or cancellation fees.',
    },
];

const FAQItem: React.FC<{
    question: string;
    answer: string;
    isOpen: boolean;
    onClick: () => void;
}> = ({ question, answer, isOpen, onClick }) => {
    return (
        <motion.div
            initial={false}
            className="border-b border-obsidian-800 last:border-0"
        >
            <button
                onClick={onClick}
                className="w-full flex items-center justify-between gap-4 py-6 text-left"
            >
                <span className="text-lg font-medium text-white">{question}</span>
                <motion.div
                    animate={{ rotate: isOpen ? 45 : 0 }}
                    transition={{ duration: 0.2 }}
                    className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-obsidian-800 text-obsidian-400"
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v12M6 12h12" />
                    </svg>
                </motion.div>
            </button>
            <AnimatePresence initial={false}>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                    >
                        <p className="pb-6 text-obsidian-400 leading-relaxed">
                            {answer}
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export const FAQDark: React.FC = () => {
    const [openIndex, setOpenIndex] = useState<number | null>(0);

    return (
        <section className="py-24 lg:py-32 bg-obsidian-950">
            <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <p className="text-sm font-medium text-electric-400 mb-3 tracking-wide uppercase">
                        FAQ
                    </p>
                    <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4 tracking-tight">
                        Frequently asked questions
                    </h2>
                    <p className="text-lg text-obsidian-400">
                        Everything you need to know about Caveray.
                    </p>
                </motion.div>

                {/* FAQ List */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 }}
                    className="bg-obsidian-900/50 backdrop-blur-lg border border-obsidian-800 rounded-3xl p-6 lg:p-8"
                >
                    {faqs.map((faq, index) => (
                        <FAQItem
                            key={index}
                            question={faq.question}
                            answer={faq.answer}
                            isOpen={openIndex === index}
                            onClick={() => setOpenIndex(openIndex === index ? null : index)}
                        />
                    ))}
                </motion.div>

                {/* Contact CTA */}
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.2 }}
                    className="text-center mt-12"
                >
                    <p className="text-obsidian-400 mb-4">
                        Still have questions?
                    </p>
                    <a
                        href="mailto:support@caveray.com"
                        className="inline-flex items-center gap-2 text-electric-400 font-medium hover:text-electric-300 transition-colors"
                    >
                        Contact our team
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                    </a>
                </motion.div>
            </div>
        </section>
    );
};

export default FAQDark;
