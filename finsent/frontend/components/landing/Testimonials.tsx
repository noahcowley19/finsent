'use client';

import React from 'react';

// =============================================================================
// TESTIMONIALS - User reviews
// =============================================================================

const testimonials = [
  {
    quote: "Caveray has completely changed how I research stocks. The sentiment analysis is incredibly accurate.",
    author: "Sarah K.",
    role: "Retail Investor",
  },
  {
    quote: "The Quant Lab alone is worth the Pro subscription. I've built strategies I never thought possible.",
    author: "Michael T.",
    role: "Day Trader",
  },
  {
    quote: "Finally, a tool that makes institutional-level analysis accessible to individual investors.",
    author: "Jennifer R.",
    role: "Financial Advisor",
  },
];

export const Testimonials: React.FC = () => {
  return (
    <section className="py-20 lg:py-28 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <p className="text-body-sm font-medium text-accent mb-3">Testimonials</p>
          <h2 className="text-display-sm lg:text-display-md text-ink-900 mb-4">
            Loved by investors
          </h2>
        </div>

        {/* Testimonials Grid */}
        <div className="grid md:grid-cols-3 gap-6">
          {testimonials.map((testimonial, index) => (
            <div
              key={index}
              className="p-6 bg-ink-50 rounded-2xl border border-transparent hover:border-ink-200/50 transition-colors"
            >
              <p className="text-body-md text-ink-700 mb-6 leading-relaxed">
                "{testimonial.quote}"
              </p>
              <div>
                <p className="font-semibold text-body-sm text-ink-900">
                  {testimonial.author}
                </p>
                <p className="text-body-xs text-ink-500">
                  {testimonial.role}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;
