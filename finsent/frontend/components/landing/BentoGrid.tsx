'use client';

// =============================================================================
// BENTO GRID - Live Widget Showcase
// =============================================================================
// A Bento Grid layout for the landing page showcasing live mini-widgets
//
// Location: frontend/components/landing/BentoGrid.tsx
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

// =============================================================================
// BENTO CARD BASE
// =============================================================================

interface BentoCardProps {
    children: React.ReactNode;
    href?: string;
    className?: string;
    span?: 'normal' | 'wide' | 'tall' | 'large';
}

const BentoCard: React.FC<BentoCardProps> = ({
    children,
    href,
    className = '',
    span = 'normal',
}) => {
    const spanClasses = {
        normal: '',
        wide: 'md:col-span-2',
        tall: 'md:row-span-2',
        large: 'md:col-span-2 md:row-span-2',
    };

    const cardContent = (
        <motion.div
            whileHover={{ scale: 1.02, y: -4 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className={`
        relative overflow-hidden h-full
        rounded-3xl p-6 lg:p-8
        bg-obsidian-900/60 backdrop-blur-lg
        border border-obsidian-800/50
        hover:border-electric-500/30
        transition-colors duration-300
        group
        ${spanClasses[span]}
        ${className}
      `}
        >
            {/* Gradient overlay on hover */}
            <div
                className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                style={{
                    background: 'linear-gradient(135deg, rgba(59,130,246,0.05) 0%, transparent 50%)',
                }}
            />
            {children}
        </motion.div>
    );

    if (href) {
        return (
            <Link href={href} className={`block ${spanClasses[span]}`}>
                {cardContent}
            </Link>
        );
    }

    return <div className={spanClasses[span]}>{cardContent}</div>;
};

// =============================================================================
// MACRO PREVIEW WIDGET
// =============================================================================

const MacroPreview: React.FC = () => {
    // Mock recession probability data
    const recessionProb = 32;
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference - (recessionProb / 100) * circumference;

    return (
        <BentoCard href="/economic" span="wide">
            <div className="flex items-center justify-between h-full">
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-3">
                        <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                        <span className="text-xs font-medium text-obsidian-400 uppercase tracking-wider">
                            Macro Command
                        </span>
                    </div>
                    <h3 className="text-2xl lg:text-3xl font-bold text-white mb-2">
                        Recession Probability
                    </h3>
                    <p className="text-obsidian-400 text-sm lg:text-base mb-4">
                        Real-time probability based on yield curve inversion and leading indicators.
                    </p>
                    <div className="flex items-center gap-2 text-electric-400 text-sm font-medium group-hover:gap-3 transition-all">
                        Explore Macro Dashboard
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                    </div>
                </div>

                {/* Gauge */}
                <div className="relative w-32 h-32 lg:w-40 lg:h-40 flex-shrink-0">
                    <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                        <circle
                            cx="50"
                            cy="50"
                            r="45"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="8"
                            className="text-obsidian-800"
                        />
                        <motion.circle
                            cx="50"
                            cy="50"
                            r="45"
                            fill="none"
                            stroke="url(#gaugeGradient)"
                            strokeWidth="8"
                            strokeLinecap="round"
                            strokeDasharray={circumference}
                            initial={{ strokeDashoffset: circumference }}
                            animate={{ strokeDashoffset }}
                            transition={{ duration: 1.5, ease: 'easeOut' }}
                        />
                        <defs>
                            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#22C55E" />
                                <stop offset="50%" stopColor="#F59E0B" />
                                <stop offset="100%" stopColor="#EF4444" />
                            </linearGradient>
                        </defs>
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center rotate-0">
                        <span className="text-3xl lg:text-4xl font-bold text-white">{recessionProb}%</span>
                        <span className="text-xs text-obsidian-400">12M Forward</span>
                    </div>
                </div>
            </div>
        </BentoCard>
    );
};

// =============================================================================
// INSIDER PULSE WIDGET
// =============================================================================

const InsiderPulse: React.FC = () => {
    const convictionBuys = [
        { ticker: 'NVDA', name: 'Jensen Huang', role: 'CEO', amount: '$5.2M', time: '2h ago' },
        { ticker: 'AAPL', name: 'Tim Cook', role: 'CEO', amount: '$3.8M', time: '6h ago' },
        { ticker: 'TSLA', name: 'Robyn Denholm', role: 'Chair', amount: '$2.1M', time: '1d ago' },
    ];

    return (
        <BentoCard href="/insider">
            <div className="flex items-center gap-2 mb-4">
                <div className="w-2 h-2 rounded-full bg-success-400 animate-pulse" />
                <span className="text-xs font-medium text-obsidian-400 uppercase tracking-wider">
                    Insider Radar
                </span>
            </div>
            <h3 className="text-xl font-bold text-white mb-4">
                Conviction Buys
            </h3>

            <div className="space-y-3">
                {convictionBuys.map((buy, i) => (
                    <motion.div
                        key={buy.ticker}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="flex items-center gap-3 p-3 rounded-xl bg-obsidian-800/50"
                    >
                        <div className="w-10 h-10 rounded-lg bg-success-500/20 flex items-center justify-center text-success-400 font-bold text-sm">
                            {buy.ticker}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white truncate">{buy.name}</p>
                            <p className="text-xs text-obsidian-500">{buy.role}</p>
                        </div>
                        <div className="text-right">
                            <p className="text-sm font-semibold text-success-400">{buy.amount}</p>
                            <p className="text-xs text-obsidian-500">{buy.time}</p>
                        </div>
                    </motion.div>
                ))}
            </div>

            <div className="flex items-center gap-2 text-electric-400 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                View All Signals
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
            </div>
        </BentoCard>
    );
};

// =============================================================================
// QUANT LAB HOOK WIDGET
// =============================================================================

const QuantLabHook: React.FC = () => {
    const metrics = [
        { label: 'Sharpe Ratio', value: '2.47', color: 'text-success-400' },
        { label: 'CAGR', value: '24.3%', color: 'text-success-400' },
        { label: 'Max Drawdown', value: '-8.2%', color: 'text-coral-400' },
        { label: 'Win Rate', value: '68%', color: 'text-electric-400' },
    ];

    return (
        <BentoCard href="/quant-lab">
            <div className="flex items-center gap-2 mb-4">
                <div className="w-2 h-2 rounded-full bg-electric-400 animate-pulse" />
                <span className="text-xs font-medium text-obsidian-400 uppercase tracking-wider">
                    Quant Lab
                </span>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">
                RSI Mean Reversion
            </h3>
            <p className="text-xs text-obsidian-500 mb-4">5Y Backtest • SPY</p>

            {/* Mini equity curve */}
            <div className="h-16 flex items-end gap-0.5 mb-4">
                {[40, 45, 42, 55, 60, 58, 65, 70, 68, 75, 80, 78, 85, 90, 88, 95, 100].map((h, i) => (
                    <motion.div
                        key={i}
                        initial={{ height: 0 }}
                        animate={{ height: `${h}%` }}
                        transition={{ delay: i * 0.05, duration: 0.3 }}
                        className="flex-1 bg-gradient-to-t from-electric-600 to-electric-400 rounded-t"
                    />
                ))}
            </div>

            <div className="grid grid-cols-2 gap-3">
                {metrics.map((metric) => (
                    <div key={metric.label} className="text-center p-2 rounded-lg bg-obsidian-800/50">
                        <p className={`text-lg font-bold ${metric.color}`}>{metric.value}</p>
                        <p className="text-xs text-obsidian-500">{metric.label}</p>
                    </div>
                ))}
            </div>

            <div className="flex items-center gap-2 text-electric-400 text-sm font-medium mt-4 group-hover:gap-3 transition-all">
                Build Your Strategy
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
            </div>
        </BentoCard>
    );
};

// =============================================================================
// PORTFOLIO PREVIEW WIDGET
// =============================================================================

const PortfolioPreview: React.FC = () => {
    return (
        <BentoCard href="/portfolio" span="wide">
            <div className="flex items-center gap-2 mb-4">
                <div className="w-2 h-2 rounded-full bg-coral-400 animate-pulse" />
                <span className="text-xs font-medium text-obsidian-400 uppercase tracking-wider">
                    My Portfolio
                </span>
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">
                Time Travel Analytics
            </h3>
            <p className="text-obsidian-400 text-sm mb-6">
                See how your portfolio would have performed under different historical scenarios.
            </p>

            {/* Blurred portfolio preview */}
            <div className="relative rounded-2xl overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-t from-obsidian-950 via-transparent to-transparent z-10" />
                <div className="blur-sm opacity-60">
                    <div className="grid grid-cols-3 gap-3 p-4 bg-obsidian-800/30 rounded-xl">
                        <div className="p-3 rounded-lg bg-obsidian-700/50">
                            <p className="text-2xl font-bold text-success-400">+24.3%</p>
                            <p className="text-xs text-obsidian-400">2008 Crisis</p>
                        </div>
                        <div className="p-3 rounded-lg bg-obsidian-700/50">
                            <p className="text-2xl font-bold text-coral-400">-12.1%</p>
                            <p className="text-xs text-obsidian-400">COVID Crash</p>
                        </div>
                        <div className="p-3 rounded-lg bg-obsidian-700/50">
                            <p className="text-2xl font-bold text-success-400">+18.7%</p>
                            <p className="text-xs text-obsidian-400">2022 Bear</p>
                        </div>
                    </div>
                </div>

                <div className="absolute inset-0 flex items-center justify-center z-20">
                    <div className="flex items-center gap-2 px-6 py-3 rounded-full bg-electric-500/90 text-white font-semibold shadow-glow-md">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                        Unlock with Free Account
                    </div>
                </div>
            </div>
        </BentoCard>
    );
};

// =============================================================================
// MAIN BENTO GRID
// =============================================================================

export const BentoGrid: React.FC = () => {
    return (
        <section className="py-24 lg:py-32 bg-obsidian-950">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Section Header */}
                <div className="text-center mb-16">
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-sm font-medium text-electric-400 uppercase tracking-wider mb-4"
                    >
                        Live from the Engine
                    </motion.p>
                    <motion.h2
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.1 }}
                        className="text-3xl lg:text-4xl font-bold text-white mb-4"
                    >
                        Institutional-Grade Analytics
                    </motion.h2>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                        className="text-lg text-obsidian-400 max-w-2xl mx-auto"
                    >
                        Not marketing fluff—real tools running real analysis. Click any widget to dive deeper.
                    </motion.p>
                </div>

                {/* Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0 }}
                        className="md:col-span-2"
                    >
                        <MacroPreview />
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.1 }}
                    >
                        <InsiderPulse />
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                    >
                        <QuantLabHook />
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.3 }}
                        className="md:col-span-2 lg:col-span-4"
                    >
                        <PortfolioPreview />
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default BentoGrid;
