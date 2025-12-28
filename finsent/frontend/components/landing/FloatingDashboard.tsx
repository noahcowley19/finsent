'use client';

import React, { useRef, useEffect, useState } from 'react';

// =============================================================================
// FLOATING DASHBOARD COMPONENT
// =============================================================================
// Hero section dashboard preview with:
// - 32px border-radius
// - 1px white semi-transparent inner border
// - Multi-layered diffuse shadow
// - Placeholder for screenshot
// - Self-drawing chart animations when entering viewport
// =============================================================================

export interface FloatingDashboardProps {
    className?: string;
}

export const FloatingDashboard: React.FC<FloatingDashboardProps> = ({
    className = '',
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        const element = containerRef.current;
        if (!element) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                    observer.disconnect();
                }
            },
            { threshold: 0.2, rootMargin: '-50px 0px' }
        );

        observer.observe(element);
        return () => observer.disconnect();
    }, []);

    return (
        <div
            ref={containerRef}
            className={`relative ${className}`}
        >
            {/* Outer glow */}
            <div
                className="absolute inset-0 opacity-50 blur-3xl"
                style={{
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.08) 100%)',
                    transform: 'scale(1.1)',
                }}
            />

            {/* Main dashboard window */}
            <div
                className="relative bg-white overflow-hidden"
                style={{
                    borderRadius: '2rem',
                    boxShadow: '0 50px 100px -20px rgba(0, 0, 0, 0.1), 0 30px 60px -15px rgba(0, 0, 0, 0.05)',
                }}
            >
                {/* Inner border stroke */}
                <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                        borderRadius: '2rem',
                        boxShadow: 'inset 0 0 0 1px rgba(255, 255, 255, 0.5)',
                    }}
                />

                {/* Dashboard header */}
                <div className="flex items-center gap-3 px-5 py-4 border-b border-obsidian-100/50 bg-cream-50/50">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-coral-400" />
                        <div className="w-3 h-3 rounded-full bg-amber-400" />
                        <div className="w-3 h-3 rounded-full bg-success-400" />
                    </div>
                    <div className="flex-1 flex items-center justify-center">
                        <div className="px-4 py-1.5 rounded-lg bg-cream-100 text-obsidian-400 text-xs font-medium">
                            app.caveray.com/dashboard
                        </div>
                    </div>
                    <div className="w-16" />
                </div>

                {/* Dashboard content */}
                <div className="p-6 bg-gradient-to-b from-cream-50 to-white">
                    {/* Sidebar + Main content layout */}
                    <div className="flex gap-6">
                        {/* Mini Sidebar */}
                        <div className="hidden md:flex flex-col gap-3 w-40">
                            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-obsidian-900 text-white text-xs font-medium">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                                </svg>
                                Dashboard
                            </div>
                            {['Sentiment', 'Portfolio', 'Screener', 'Watchlist'].map((item) => (
                                <div key={item} className="flex items-center gap-2 px-3 py-2 rounded-lg text-obsidian-500 text-xs">
                                    <div className="w-4 h-4 rounded bg-cream-200" />
                                    {item}
                                </div>
                            ))}
                        </div>

                        {/* Main Content */}
                        <div className="flex-1 space-y-4">
                            {/* Stats Row */}
                            <div className="grid grid-cols-3 gap-3">
                                <div className="p-4 rounded-xl bg-white border border-cream-200 shadow-sm">
                                    <p className="text-xs text-obsidian-400 mb-1">Total Value</p>
                                    <p className="text-lg font-semibold text-obsidian-900">$28,500</p>
                                    <p className="text-xs text-success-500 flex items-center gap-1 mt-1">
                                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                                        </svg>
                                        +12.4%
                                    </p>
                                </div>
                                <div className="p-4 rounded-xl bg-white border border-cream-200 shadow-sm">
                                    <p className="text-xs text-obsidian-400 mb-1">Sentiment</p>
                                    <p className="text-lg font-semibold text-obsidian-900">Bullish</p>
                                    <p className="text-xs text-coral-500 flex items-center gap-1 mt-1">
                                        78 score
                                    </p>
                                </div>
                                <div className="p-4 rounded-xl bg-white border border-cream-200 shadow-sm">
                                    <p className="text-xs text-obsidian-400 mb-1">Alerts</p>
                                    <p className="text-lg font-semibold text-obsidian-900">3 New</p>
                                    <p className="text-xs text-electric-500 flex items-center gap-1 mt-1">
                                        View all →
                                    </p>
                                </div>
                            </div>

                            {/* Chart placeholder */}
                            <div className="p-4 rounded-xl bg-white border border-cream-200 shadow-sm">
                                <div className="flex items-center justify-between mb-4">
                                    <p className="text-sm font-medium text-obsidian-900">Revenue Forecast</p>
                                    <div className="flex gap-2">
                                        <span className="px-2 py-1 rounded text-xs bg-cream-100 text-obsidian-500">1W</span>
                                        <span className="px-2 py-1 rounded text-xs bg-obsidian-900 text-white">1M</span>
                                        <span className="px-2 py-1 rounded text-xs bg-cream-100 text-obsidian-500">3M</span>
                                    </div>
                                </div>

                                {/* SVG Chart with drawing animation */}
                                <svg
                                    viewBox="0 0 400 120"
                                    className={`w-full h-24 ${isVisible ? 'chart-draw animate' : 'chart-draw'}`}
                                    fill="none"
                                >
                                    {/* Grid lines */}
                                    <path d="M0 30 L400 30" stroke="#E4E4E7" strokeWidth="1" opacity="0.5" />
                                    <path d="M0 60 L400 60" stroke="#E4E4E7" strokeWidth="1" opacity="0.5" />
                                    <path d="M0 90 L400 90" stroke="#E4E4E7" strokeWidth="1" opacity="0.5" />

                                    {/* Data line */}
                                    <path
                                        d="M0 90 L50 75 L100 80 L150 50 L200 55 L250 35 L300 40 L350 20 L400 25"
                                        stroke="#3B82F6"
                                        strokeWidth="2.5"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        style={{
                                            strokeDasharray: 1000,
                                            strokeDashoffset: isVisible ? 0 : 1000,
                                            transition: 'stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                        }}
                                    />

                                    {/* Gradient fill under line */}
                                    <defs>
                                        <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.2" />
                                            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
                                        </linearGradient>
                                    </defs>
                                    <path
                                        d="M0 90 L50 75 L100 80 L150 50 L200 55 L250 35 L300 40 L350 20 L400 25 L400 120 L0 120 Z"
                                        fill="url(#chartGradient)"
                                        style={{
                                            opacity: isVisible ? 1 : 0,
                                            transition: 'opacity 1s ease-out 0.5s',
                                        }}
                                    />
                                </svg>

                                <div className="flex justify-between text-xs text-obsidian-400 mt-2">
                                    <span>Jul</span>
                                    <span>Aug</span>
                                    <span>Sep</span>
                                    <span>Oct</span>
                                    <span>Nov</span>
                                    <span>Dec</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Floating accent elements */}
            <div
                className="absolute -bottom-4 -right-4 w-32 h-32 rounded-2xl bg-gradient-to-br from-electric-500/10 to-purple-500/10 blur-xl"
                style={{ transform: 'rotate(12deg)' }}
            />
        </div>
    );
};

export default FloatingDashboard;
