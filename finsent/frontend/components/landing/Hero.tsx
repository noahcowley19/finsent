'use client';

// =============================================================================
// HERO SECTION - REDESIGNED
// =============================================================================
// Premium landing page hero with animated mesh gradient background
// Inspired by Lovable.com with clean typography and glassmorphism
//
// Location: frontend/components/landing/Hero.tsx
// =============================================================================

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { HiArrowRight, HiSearch, HiSparkles } from 'react-icons/hi';
import { MeshGradient } from '@/components/ui/MeshGradient';

export const Hero: React.FC = () => {
  return (
    <section className="relative min-h-[100vh] flex items-center overflow-hidden">
      {/* Animated Mesh Gradient Background */}
      <MeshGradient className="z-0" />

      {/* Content */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <div className="flex flex-col items-center text-center">
          {/* Announcement Badge */}
          <div
            className="
              inline-flex items-center gap-2 
              px-4 py-2 mb-8
              bg-white/70 backdrop-blur-md
              rounded-full 
              border border-white/50
              shadow-sm
              animate-fade-in
            "
          >
            <HiSparkles className="w-4 h-4 text-terra-500" />
            <span className="text-body-sm text-navy-700 font-medium">
              AI-Powered Financial Intelligence
            </span>
            <HiArrowRight className="w-4 h-4 text-navy-400" />
          </div>

          {/* Main Headline */}
          <h1
            className="
              font-display 
              text-display-lg md:text-display-xl lg:text-[5rem]
              text-navy-900 
              mb-6 
              max-w-4xl
              leading-[1.1]
              animate-fade-in-up
            "
            style={{ animationDelay: '100ms' }}
          >
            Make Smarter{' '}
            <span className="relative inline-block">
              <span className="relative z-10 text-transparent bg-clip-text bg-gradient-to-r from-terra-500 via-pink-500 to-purple-500">
                Investment
              </span>
            </span>
            {' '}Decisions
          </h1>

          {/* Subheadline */}
          <p
            className="
              text-body-lg lg:text-xl
              text-navy-600/80
              mb-12
              max-w-2xl
              leading-relaxed
              animate-fade-in-up
            "
            style={{ animationDelay: '200ms' }}
          >
            Analyze market sentiment, track insider trading, and discover opportunities
            before the crowd. All in one powerful platform.
          </p>

          {/* CTA Container - Glass Card */}
          <div
            className="
              w-full max-w-xl
              p-2
              bg-white/80 backdrop-blur-xl
              rounded-2xl
              border border-white/60
              shadow-xl shadow-navy-900/5
              animate-fade-in-up
            "
            style={{ animationDelay: '300ms' }}
          >
            <div className="flex flex-col sm:flex-row gap-2">
              <Link
                href="/search"
                className="
                  flex-1 flex items-center justify-center gap-2
                  px-6 py-4
                  bg-cream-50 hover:bg-cream-100
                  text-navy-600
                  rounded-xl
                  transition-all duration-200
                  group
                "
              >
                <HiSearch className="w-5 h-5 text-navy-400 group-hover:text-navy-600 transition-colors" />
                <span className="font-medium">Search any stock...</span>
              </Link>
              <Link
                href="/signup"
                className="
                  flex items-center justify-center gap-2
                  px-8 py-4
                  bg-navy-900 hover:bg-navy-800
                  text-white
                  font-semibold
                  rounded-xl
                  transition-all duration-200
                  hover:scale-[1.02]
                  shadow-lg shadow-navy-900/20
                "
              >
                Get Started
                <HiArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </div>

          {/* Trust Indicators */}
          <div
            className="
              flex flex-col sm:flex-row items-center gap-4 sm:gap-8 
              mt-12
              animate-fade-in-up
            "
            style={{ animationDelay: '400ms' }}
          >
            <div className="flex items-center gap-2 text-navy-500/70">
              <svg className="w-5 h-5 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span className="text-body-sm">Free to start</span>
            </div>
            <div className="flex items-center gap-2 text-navy-500/70">
              <svg className="w-5 h-5 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span className="text-body-sm">No credit card required</span>
            </div>
            <div className="flex items-center gap-2 text-navy-500/70">
              <svg className="w-5 h-5 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span className="text-body-sm">Cancel anytime</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div
            className="
              grid grid-cols-3 gap-8 lg:gap-16
              mt-20 pt-12
              border-t border-navy-200/30
              animate-fade-in-up
            "
            style={{ animationDelay: '500ms' }}
          >
            <div className="text-center">
              <p className="font-display text-display-sm lg:text-display-md text-navy-900 mb-1">50K+</p>
              <p className="text-body-sm text-navy-500/70">Stocks Analyzed</p>
            </div>
            <div className="text-center">
              <p className="font-display text-display-sm lg:text-display-md text-navy-900 mb-1">2M+</p>
              <p className="text-body-sm text-navy-500/70">Data Points</p>
            </div>
            <div className="text-center">
              <p className="font-display text-display-sm lg:text-display-md text-navy-900 mb-1">98%</p>
              <p className="text-body-sm text-navy-500/70">Accuracy</p>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-cream-50 to-transparent z-20" />
    </section>
  );
};

export default Hero;
