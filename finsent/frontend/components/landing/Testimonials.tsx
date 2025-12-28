'use client';

import React from 'react';
import { ScrollReveal } from '@/components/ui';

// =============================================================================
// TESTIMONIALS SECTION - Glass Cards with Quote Styling
// =============================================================================

const testimonials = [
  {
    quote: "Caveray has completely transformed how I approach stock analysis. The sentiment insights are incredibly accurate.",
    author: "Sarah Chen",
    role: "Portfolio Manager",
    company: "Horizon Capital",
  },
  {
    quote: "The Quant Lab feature alone is worth the subscription. Being able to backtest my strategies has saved me countless hours.",
    author: "Michael Torres",
    role: "Independent Trader",
    company: "Self-employed",
  },
  {
    quote: "Finally, a platform that combines all the tools I need in one place. The insider trading tracker is a game-changer.",
    author: "Emily Roberts",
    role: "Investment Analyst",
    company: "Sterling Investments",
  },
];

export const Testimonials: React.FC = () => {
  return (
    <section className="py-24 lg:py-32 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <ScrollReveal>
          <div className="text-center mb-16">
            <p className="text-sm font-medium text-electric-500 mb-3 tracking-wide uppercase">
              Testimonials
            </p>
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-obsidian-900 mb-4 tracking-tightest">
              Loved by investors
            </h2>
            <p className="text-lg text-obsidian-500 max-w-xl mx-auto">
              See what our users have to say about their experience with Caveray.
            </p>
          </div>
        </ScrollReveal>

        {/* Testimonials Grid */}
        <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
          {testimonials.map((testimonial, index) => (
            <ScrollReveal key={testimonial.author} delay={index * 100}>
              <div className="relative p-8 rounded-2xl bg-cream-50/80 backdrop-blur-lg border border-cream-200/50 shadow-glass hover:shadow-glass-lg transition-all duration-300 hover:-translate-y-1 h-full flex flex-col">
                {/* Quote Icon */}
                <svg
                  className="w-10 h-10 text-electric-500/20 mb-4"
                  fill="currentColor"
                  viewBox="0 0 32 32"
                >
                  <path d="M10 8c-3.3 0-6 2.7-6 6v10h10V14H6c0-2.2 1.8-4 4-4V8zm14 0c-3.3 0-6 2.7-6 6v10h10V14h-8c0-2.2 1.8-4 4-4V8z" />
                </svg>

                {/* Quote */}
                <p className="text-obsidian-700 leading-relaxed mb-6 flex-1">
                  "{testimonial.quote}"
                </p>

                {/* Author */}
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-electric-400 to-electric-600 flex items-center justify-center text-white font-semibold text-sm">
                    {testimonial.author.charAt(0)}
                  </div>
                  <div>
                    <p className="font-semibold text-obsidian-900 text-sm">
                      {testimonial.author}
                    </p>
                    <p className="text-xs text-obsidian-500">
                      {testimonial.role} at {testimonial.company}
                    </p>
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

export default Testimonials;
