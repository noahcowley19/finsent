'use client';

// =============================================================================
// HERO SECTION - Financial Operating System
// =============================================================================
// Bold, terminal-inspired hero for the Financial Operating System landing
//
// Location: frontend/components/landing/HeroCommand.tsx
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

// =============================================================================
// ANIMATED TERMINAL CURSOR
// =============================================================================

const TerminalCursor: React.FC = () => (
    <motion.span
        animate={{ opacity: [1, 0] }}
        transition={{ duration: 0.8, repeat: Infinity, repeatType: 'reverse' }}
        className="inline-block w-3 h-8 bg-electric-400 ml-1 rounded-sm"
    />
);

// =============================================================================
// HERO COMPONENT
// =============================================================================

export const HeroCommand: React.FC = () => {
    return (
        <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-obsidian-950">
            {/* Background Effects */}
            <div className="absolute inset-0">
                {/* Grid pattern */}
                <div
                    className="absolute inset-0 opacity-20"
                    style={{
                        backgroundImage: `
              linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
            `,
                        backgroundSize: '60px 60px',
                    }}
                />

                {/* Radial glow */}
                <div
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] opacity-30"
                    style={{
                        background: 'radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%)',
                    }}
                />

                {/* Top-left light leak */}
                <div
                    className="absolute -top-1/4 -left-1/4 w-1/2 h-1/2 opacity-20"
                    style={{
                        background: 'radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, transparent 70%)',
                    }}
                />
            </div>

            {/* Content */}
            <div className="relative z-10 max-w-5xl mx-auto px-6 py-32 text-center">
                {/* Badge */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-obsidian-900/80 border border-obsidian-800 backdrop-blur-sm"
                >
                    <span className="w-2 h-2 rounded-full bg-success-400 animate-pulse" />
                    <span className="text-sm font-medium text-obsidian-300">
                        Python-Powered • Real-Time Analytics
                    </span>
                </motion.div>

                {/* Headline */}
                <motion.h1
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-6 leading-[1.05]"
                >
                    The Financial Operating System
                    <br />
                    <span className="text-gradient">for the Modern Quant</span>
                    <TerminalCursor />
                </motion.h1>

                {/* Subheadline */}
                <motion.p
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="text-lg lg:text-xl text-obsidian-400 max-w-2xl mx-auto mb-10"
                >
                    Institutional-grade tools. Zero cost. Track macro signals, decode insider trades,
                    backtest strategies, and stress-test your portfolio—all in one platform.
                </motion.p>

                {/* CTAs */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12"
                >
                    <Link
                        href="/signup"
                        className="group flex items-center gap-2 px-8 py-4 bg-electric-500 text-white font-semibold rounded-xl hover:bg-electric-600 transition-all duration-200 hover:-translate-y-1 hover:shadow-glow-lg"
                    >
                        Start Building Free
                        <svg className="w-5 h-5 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                    </Link>

                    <Link
                        href="/dashboard"
                        className="flex items-center gap-2 px-8 py-4 border border-obsidian-700 text-obsidian-300 font-medium rounded-xl hover:border-obsidian-600 hover:text-white transition-all duration-200"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Explore Dashboard
                    </Link>
                </motion.div>

                {/* Trust Indicators */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="flex flex-wrap items-center justify-center gap-8 text-sm text-obsidian-500"
                >
                    {[
                        { icon: '🔒', text: 'Bank-level encryption' },
                        { icon: '⚡', text: 'Real-time data feeds' },
                        { icon: '🎯', text: 'No credit card required' },
                    ].map((item) => (
                        <div key={item.text} className="flex items-center gap-2">
                            <span>{item.icon}</span>
                            <span>{item.text}</span>
                        </div>
                    ))}
                </motion.div>

                {/* Stats Bar */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 p-8 rounded-3xl bg-obsidian-900/50 border border-obsidian-800 backdrop-blur-sm"
                >
                    {[
                        { value: '50K+', label: 'Stocks Tracked' },
                        { value: '$2.1B', label: 'Insider Trades Analyzed' },
                        { value: '15M+', label: 'Backtests Run' },
                        { value: '99.9%', label: 'Uptime SLA' },
                    ].map((stat) => (
                        <div key={stat.label} className="text-center">
                            <p className="text-3xl lg:text-4xl font-bold text-white tracking-tight">{stat.value}</p>
                            <p className="text-sm text-obsidian-400 mt-1">{stat.label}</p>
                        </div>
                    ))}
                </motion.div>
            </div>

            {/* Bottom gradient fade */}
            <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-obsidian-950 to-transparent pointer-events-none" />
        </section>
    );
};

export default HeroCommand;
