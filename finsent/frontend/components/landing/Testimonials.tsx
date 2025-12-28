'use client';

// =============================================================================
// TESTIMONIALS SECTION - FIXED VERSION
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';

const testimonials = [
  {
    quote: "Caveray's sentiment analysis helped me spot the Tesla rally before it happened. The AI picks up on trends I would have missed reading forums manually.",
    author: 'Sarah Chen',
    role: 'Retail Investor',
    rating: 5,
  },
  {
    quote: "The insider trading tracker is invaluable. When I saw multiple executives buying shares of a small-cap, I followed suit and made a 40% return in three months.",
    author: 'Michael Torres',
    role: 'Day Trader',
    rating: 5,
  },
  {
    quote: "I manage portfolios for clients and Caveray saves me hours of research every week. The financial analysis tools are comprehensive yet easy to understand.",
    author: 'Jennifer Walsh',
    role: 'Financial Advisor',
    rating: 5,
  },
  {
    quote: "As someone new to investing, Caveray made it easy to understand what's happening with my stocks. The sentiment scores are especially helpful.",
    author: 'David Kim',
    role: 'Software Engineer',
    rating: 5,
  },
];

export const Testimonials: React.FC = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  const nextSlide = useCallback(() => {
    setCurrentIndex((prev) => (prev + 1) % testimonials.length);
  }, []);

  const prevSlide = useCallback(() => {
    setCurrentIndex((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  }, []);

  useEffect(() => {
    if (!isAutoPlaying) return;
    const timer = setInterval(nextSlide, 5000);
    return () => clearInterval(timer);
  }, [isAutoPlaying, nextSlide]);

  return (
    <section className="py-24 lg:py-32 bg-navy-900 overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <p className="text-body-sm font-semibold text-terra-400 uppercase tracking-widest mb-4">
            Testimonials
          </p>
          <h2 className="font-display text-display-md lg:text-display-lg text-white mb-4">
            Trusted by investors worldwide
          </h2>
          <p className="text-body-lg text-white/50">
            See what our users have to say about their experience with Caveray.
          </p>
        </div>

        <div className="relative max-w-4xl mx-auto" onMouseEnter={() => setIsAutoPlaying(false)} onMouseLeave={() => setIsAutoPlaying(true)}>
          <div className="relative bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl rounded-3xl p-8 lg:p-12 border border-white/10">
            <div className="absolute top-8 right-8 text-6xl text-terra-500/20 font-serif leading-none">
              &ldquo;
            </div>

            <div className="flex gap-1 mb-8">
              {[...Array(testimonials[currentIndex].rating)].map((_, i) => (
                <svg key={i} className="w-6 h-6 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
              ))}
            </div>

            <blockquote className="text-xl lg:text-2xl text-white font-light leading-relaxed mb-10 min-h-[100px]">
              {testimonials[currentIndex].quote}
            </blockquote>

            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-full bg-gradient-to-br from-terra-400 to-pink-500 flex items-center justify-center text-white text-lg font-semibold shadow-lg shadow-terra-500/20">
                {testimonials[currentIndex].author.charAt(0)}
              </div>
              <div>
                <p className="font-heading font-semibold text-white text-lg">
                  {testimonials[currentIndex].author}
                </p>
                <p className="text-body-sm text-white/50">
                  {testimonials[currentIndex].role}
                </p>
              </div>
            </div>
          </div>

          <button onClick={prevSlide} className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 lg:-translate-x-16 w-12 h-12 rounded-full bg-white/10 backdrop-blur-sm text-white flex items-center justify-center hover:bg-white/20 transition-all duration-200 border border-white/10" aria-label="Previous testimonial">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button onClick={nextSlide} className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 lg:translate-x-16 w-12 h-12 rounded-full bg-white/10 backdrop-blur-sm text-white flex items-center justify-center hover:bg-white/20 transition-all duration-200 border border-white/10" aria-label="Next testimonial">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>

          <div className="flex justify-center gap-3 mt-10">
            {testimonials.map((_, index) => (
              <button key={index} onClick={() => setCurrentIndex(index)} className={`h-2 rounded-full transition-all duration-300 ${index === currentIndex ? 'w-10 bg-gradient-to-r from-terra-500 to-pink-500' : 'w-2 bg-white/20 hover:bg-white/40'}`} aria-label={`Go to testimonial ${index + 1}`} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonials;
